"""Deferred typing for split partitions: decide what a written table WAS after the run, not while
it is being written.

Capture sees a stage write csvs. It cannot see whether those writes partition anything until the
rest of the run exists, so emitting SplitManifest at write time forced the call early -- which is
why the fallback branch read the entry script's name (interpose.py `_stage == "make_splits"`).
Here the whole ledger is in hand, so the type is earned from the graph and the ledger keeps only
what was observed: a table was written, with these rows, at this path.

Identity is unchanged. Entity.id == content_hash and both types hash the same file bytes, so a
promoted node keeps its id and every edge already pointing at it stays correct.

What EARNS the type (a real split happened):
  conserved        sibling writes of one activity whose rows account exactly for a table that
                   activity read. Says a real split happened, nothing about which piece is which.
  library          a split fn produced the write; a capture-side observation carried on attributes,
                   since a post-hoc graph cannot see that train_test_split was called.

What LABELS a piece once earned:
  gradient_update  the table's samples reached a TrainingStep -- the only signal that says WHICH
                   partition trained the model. Never promotes on its own: the source table is
                   reachable by derivation from the training input too, so promoting on it alone
                   retypes the whole dataset as a split.

`filename` is deliberately NOT a tier. The path stem names the partition (DESIGN: val/test stay
filename-derived on purpose) but never earns the promotion -- a convention someone chose is not
evidence that a split happened. A pipeline whose splits meet no tier keeps plain DatasetMetadata:
a thinner graph, never a wrong one.

The pass also ARBITRATES, not just types. Two stages can describe the same file: the split stage
proves the rows account for a source table, while a later stage that merely read train.csv knows
only the stem and claims `filename` -- honestly, since the proof lives in the split process's
memory and dies with it. The projector merges by last-wins, so without arbitration the later,
weaker claim silently wins. Re-deriving the evidence here works because it is a property of the
DATA (row counts, edges), not of a moment in time, so an existing claim is overwritten only when
the graph proves something strictly stronger (schema.nodes.EVIDENCE_RANK).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from residue._env import env
from residue.schema import nodes as node_mod
from residue.schema.nodes import EVIDENCE_RANK
from residue.schema.types import EdgeRelation, Role
from residue.store.graph import Graph

_LIBRARY_HINT = "split_fn"   # attributes key capture sets when a split fn produced the write


def _indices(graph: Graph):
    """Generic PROV indices -- relations only, no node-type assumptions."""
    generated: dict[str, list[str]] = {}   # activity -> entities it generated
    used_input: dict[str, set[str]] = {}   # activity -> entities it consumed as DATA
    derived: dict[str, list[str]] = {}     # entity   -> entities it derives from
    for e in graph.edge_list:
        r = e.RELATION
        if r == EdgeRelation.WAS_GENERATED_BY:
            generated.setdefault(e.target, []).append(e.source)
        elif r == EdgeRelation.USED and getattr(e, "role", None) == Role.INPUT:
            used_input.setdefault(e.source, set()).add(e.target)
        elif r == EdgeRelation.WAS_DERIVED_FROM:
            derived.setdefault(e.source, []).append(e.target)
    return generated, used_input, derived


def _rows(graph: Graph, nid: str) -> Optional[int]:
    """Row count of a table node, whichever type carries it -- so conservation still fires on a
    ledger whose partitions were already typed at write time (legacy, or a re-promoted graph)."""
    n = graph.nodes.get(nid)
    if isinstance(n, node_mod.DatasetMetadata):
        return n.num_rows
    if isinstance(n, node_mod.SplitManifest):
        return n.num_samples
    return None


def _trained(graph: Graph, used_input, derived) -> set[str]:
    """Entities whose content reached a TrainingStep. Walks back through derivation because a
    training input is normally a TransformedDataset several derivations above the written csv."""
    seen: set[str] = set()
    stack: list[str] = []
    for aid, node in graph.nodes.items():
        if isinstance(node, node_mod.TrainingStep):
            stack.extend(used_input.get(aid, ()))
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(derived.get(nid, ()))
    return seen


def _conserved(graph: Graph, generated, used_input) -> tuple[set[str], set[str]]:
    """Row conservation: writes of one activity accounting exactly for a table it read.

    Returns (partitions, sources). Sources are returned so the caller can refuse to promote them:
    the source table is reachable by derivation from the training input, so `gradient_update`
    would otherwise sweep it up and call the whole dataset a split.
    """
    partitions: set[str] = set()
    sources: set[str] = set()
    for aid, outs in generated.items():
        pieces = [n for n in outs if _rows(graph, n) is not None]
        if len(pieces) < 2:
            continue
        for sid in used_input.get(aid, ()):
            src_rows = _rows(graph, sid)
            if src_rows is None:
                continue
            part = [n for n in pieces if n != sid]   # a stage may rewrite its own source table
            if len(part) < 2:
                continue
            if sum(_rows(graph, n) for n in part) == src_rows:
                partitions.update(part)
                sources.add(sid)
                break
    return partitions, sources


def _split_name(path: Optional[str]) -> Optional[str]:
    return Path(path).stem.lower() if path else None


def promote_splits(graph: Graph) -> int:
    """Type and re-evidence split partitions in place; returns how many nodes changed.

    Two cases, one rule -- keep the best-evidenced claim about a content hash. A neutral
    DatasetMetadata becomes a SplitManifest when the graph earns it; an existing SplitManifest is
    rewritten only when the graph's evidence outranks the one capture recorded, so a strong
    capture-time claim is never downgraded.

    RESIDUE_NO_PROMOTE=1 skips the pass, so a caller can measure this layer's effect on a
    projection by loading the same ledger twice."""
    if env("NO_PROMOTE"):
        return 0
    generated, used_input, derived = _indices(graph)
    conserved, sources = _conserved(graph, generated, used_input)
    trained = _trained(graph, used_input, derived) - sources

    promoted = 0
    for nid, node in list(graph.nodes.items()):
        is_split = isinstance(node, node_mod.SplitManifest)
        if not (is_split or isinstance(node, node_mod.DatasetMetadata)) or nid in sources:
            continue
        if nid in conserved:
            evidence = "conserved"
        elif node.attributes.get(_LIBRARY_HINT):
            evidence = "library"
        else:
            continue                      # nothing earned it -- stays whatever capture said
        name = _split_name(node.path) or (node.split if is_split else None)
        if nid in trained:                 # earned piece + reached training -> which one is train
            evidence, name = "gradient_update", "train"
        if is_split and EVIDENCE_RANK[evidence] <= EVIDENCE_RANK.get(node.split_evidence, 0):
            continue                      # capture's claim is already at least this strong
        graph.nodes[nid] = node_mod.SplitManifest(   # dict keeps the slot: log order preserved
            content_hash=node.content_hash,
            split=name,
            num_samples=node.num_samples if is_split else node.num_rows,
            num_patients=node.num_patients if is_split else None,
            path=node.path,
            split_evidence=evidence,
            attributes=dict(node.attributes),
        )
        promoted += 1
    return promoted
