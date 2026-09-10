"""Project the append-only event log into the in-memory PROV graph it denotes.

The log is a flat record stream with churn: entity nodes are re-emitted on every
reference and activities are emitted twice (enter with end_ns=None, exit with end_ns
set). This collapses that to a graph — nodes deduped by id (LAST record wins, so an
activity's exit record overwrites the enter record's None end_ns), edges deduped by key.

Everything downstream (validate.py, the visualizer, later diff.py) consumes a Graph and
should never have to think about the on-disk record churn — that is this module's whole job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from residue.schema import edges as edge_mod
from residue.schema import nodes as node_mod
from residue.schema.edges import Edge
from residue.schema.nodes import Node
from residue.schema.types import DerivationType, EdgeRelation, Role
from residue.store.log import read_events


@dataclass
class Graph:
    """Deduped nodes (by id) and edges (by key). Insertion order is log order, which is
    causal order (an activity is logged before the outputs that reference it)."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def add(self, event: dict) -> None:
        if event.get("node_kind"):           # node payloads carry node_kind; edges don't
            node = node_mod.from_dict(event)
            self.nodes[node.id] = node        # last wins: exit (end_ns set) overwrites enter
        else:
            edge = edge_mod.from_dict(event)
            self.edges[edge.key] = edge

    @property
    def edge_list(self) -> list[Edge]:
        return list(self.edges.values())

    @property
    def node_list(self) -> list[Node]:
        return list(self.nodes.values())


def project(events: Iterable[dict]) -> Graph:
    graph = Graph()
    for event in events:
        graph.add(event)
    return graph


def _link_dataset_membership(graph: Graph, members_dir: Path) -> None:
    """Content-addressed subset join (Slice 15). A preprocessing stage writes the whole processed store
    as one DatasetVersion, but training reads a per-sample subset and fingerprints it (as a SampleManifest)
    by decoded-pixel hash -- so the two never share an id and the store is left orphaned in the graph.
    Each side instead publishes its file-BYTE-hash set to a sidecar; a subset match (read-set of the run's
    SampleManifest contained in a DatasetVersion's store-set) means 'the files this run read all came from
    that store'. Subset, not equality, because a run legitimately reads only part of the store.

    Where the matched store attaches: the pixels it holds are the CONTENT the model's input is made of,
    and that input is produced by a chain of activities (the transform pipeline). So the store is wired in
    as an INPUT of the HEAD of that chain -- it flows raw -> processed -> through the transforms -> into
    the consumed data as ONE lineage, beside the split identity the head already consumes (which samples).
    Without it the pixels would bypass the transforms and reconnect only at the output, reading as a
    parallel dataset. The threading is purely structural: it walks generic PROV relations off the
    SampleManifest and names no transform, dataset, or node type beyond the two the sidecars identify.
    Two-step fallback keeps the link when a run doesn't fit: no transform chain -> the consumed entity
    derives straight from the store; no consumed entity at all (a pipeline with no transform capture) ->
    the store hangs off the SampleManifest leaf, as before. Silent no-op when the sidecars are absent."""
    if not members_dir.exists():
        return

    def load_set(node_id: str):
        f = members_dir / f"{node_id}.txt"
        return set(f.read_text().split()) if f.exists() else None

    stores = {n.id: s for n in graph.node_list
              if isinstance(n, node_mod.DatasetVersion) and (s := load_set(n.id))}
    if not stores:
        return

    # Generic edge indices -- relations only, no node-type assumptions beyond the sidecar-identified ones.
    genby: dict[str, list[str]] = {}       # entity   -> activities that generated it
    informed: dict[str, list[str]] = {}    # activity -> earlier activities it followed (out wasInformedBy)
    used_input: dict[str, set[str]] = {}   # activity -> entities it consumed as DATA (role INPUT)
    derived: dict[str, list[str]] = {}     # entity   -> entities it derives from
    for e in graph.edge_list:
        r = e.RELATION
        if r == EdgeRelation.WAS_GENERATED_BY:
            genby.setdefault(e.source, []).append(e.target)
        elif r == EdgeRelation.WAS_INFORMED_BY:
            informed.setdefault(e.source, []).append(e.target)
        elif r == EdgeRelation.USED and getattr(e, "role", None) == Role.INPUT:
            used_input.setdefault(e.source, set()).add(e.target)
        elif r == EdgeRelation.WAS_DERIVED_FROM:
            derived.setdefault(e.source, []).append(e.target)

    def is_store(nid: str) -> bool:
        return isinstance(graph.nodes.get(nid), node_mod.DatasetVersion)

    def followed(start: str) -> set[str]:
        """Activities reachable from `start` along outgoing wasInformedBy -- its producer history. The
        spine points to EARLIER activities, so this walks backward and stays small (the model's own
        training steps point INTO the chain, not out of it)."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            for b in informed.get(stack.pop(), ()):
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen

    def add(edge) -> None:
        graph.edges[edge.key] = edge

    for n in graph.node_list:
        if not isinstance(n, node_mod.SampleManifest):
            continue
        read = load_set(n.id)
        if not read:
            continue
        consumed = [t for t in derived.get(n.id, ()) if not is_store(t)]  # the data this manifest was
        for dv_id, store in stores.items():
            if not read <= store:
                continue
            threaded = False
            for d in consumed:
                bases = {t for t in derived.get(d, ()) if t != dv_id and not is_store(t)}  # which-samples
                if not bases:
                    continue
                producers = set()          # d's producing activities: its generator + that chain's history
                for a in genby.get(d, ()):
                    producers.add(a)
                    producers |= followed(a)
                chain = [a for a in producers if used_input.get(a, set()) & bases]  # the transform ops
                heads = [a for a in chain if not (followed(a) & set(chain))]        # where content enters
                for h in heads:
                    add(edge_mod.Used(source=h, target=dv_id, role=Role.INPUT))
                    threaded = True
            if threaded:
                continue
            for src in (consumed or [n.id]):   # fallback: store -> consumed data, else the manifest leaf
                add(edge_mod.WasDerivedFrom(source=src, target=dv_id,
                                            derivation_type=DerivationType.PREPROCESSING))


def load(path: str | Path) -> Graph:
    """Read a residue.jsonl ledger and project it to a Graph."""
    from residue.store.promote import promote_splits  # local: promote imports Graph from here

    path = Path(path)
    graph = project(read_events(path))
    _link_dataset_membership(graph, path.parent / "_dataset_members")
    promote_splits(graph)
    return graph
