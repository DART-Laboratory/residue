"""Structural diff of two provenance graphs

Compares a suspect run against a reference (clean) run and surfaces where they diverge
without replay. Every node in the two ledgers is compared, edges are omitted.

Node identifiers are specific to node type (e.g., Entity ids are content hashes), 
with some identifiers being run-scoped (Activity ids are run_id + counter), so 
they never match across two runs. thus, we align nodes by a logical key, and only
then compare content (node atrributes).

"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from residue.schema.nodes import Activity, Agent, Entity, Node
from residue.store.graph import Graph, load


_KEY_FIELDS = ("epoch", "split")


def _key_value(node: Node, name: str) -> Any:
    """A positional field, wherever it lives — first-class field or open attributes dict."""
    v = getattr(node, name, None)
    if v is None:
        v = (getattr(node, "attributes", None) or {}).get(name)
    return v if isinstance(v, (str, int, float, bool)) else None


def logical_key(node: Node) -> tuple:
    """Step-1 key: type + whatever positional fields the node declares. Not necessarily
    unique on its own; _slot_index refines it until it is."""
    parts: list = [type(node).__name__]
    for name in _KEY_FIELDS:
        v = _key_value(node, name)
        if v is not None:
            parts.append(f"{name}={v}")
    return tuple(parts)


def _prov_links(graph: Graph) -> tuple[dict[str, str], dict[str, list[str]]]:
    """WasGeneratedBy runs entity(source) -> activity(target). Index it both ways."""
    made_by: dict[str, str] = {}
    made: dict[str, list[str]] = defaultdict(list)
    for e in graph.edge_list:
        if type(e).__name__ == "WasGeneratedBy":
            made_by[e.source] = e.target
            made[e.target].append(e.source)
    return made_by, made


def _borrowed_ident(node: Node, graph: Graph, made_by, made) -> Optional[str]:
    """Positional identity borrowed from one PROV hop away.

    Two node types are anonymous in isolation but pinned by their neighbours: a
    CheckpointWrite declares nothing (its epoch belongs to the ModelCheckpoint it
    generated), and an EvaluationResult knows its split but not which epoch produced it
    (its generating Evaluation does). Borrowing beats an ordinal because it survives a run
    that writes a different *number* of checkpoints."""
    neighbours: list[str] = []
    if isinstance(node, Activity):
        neighbours = made.get(node.id, [])
    elif isinstance(node, Entity):
        a = made_by.get(node.id)
        neighbours = [a] if a else []

    marks = []
    for nid in neighbours:
        other = graph.nodes.get(nid)
        if other is None:
            continue
        for name in ("epoch", "split"):
            v = _key_value(other, name)
            if v is not None:
                marks.append(f"{name}={v}")
    return "via:" + ",".join(sorted(set(marks))) if marks else None


@dataclass
class SlotIndex:
    """Bijective node -> slot mapping for one graph, plus the accounting that proves it."""
    slots: dict[tuple, Node] = field(default_factory=dict)
    corroborations: dict[tuple, list[Node]] = field(default_factory=lambda: defaultdict(list))
    total_nodes: int = 0

    @property
    def accounted(self) -> int:
        return len(self.slots) + sum(len(v) for v in self.corroborations.values())


def _slot_index(graph: Graph) -> SlotIndex:
    """Assign every node a slot, refining shared keys until the mapping is one-to-one.

    One deliberate exception to one-node-one-slot: a slot can carry several content hashes
    within a single run — the artifact a stage generated, plus re-hashes that later stages
    log as Used inputs. Those re-hashes are corroboration of the same slot, not rival slots;
    promoting them would manufacture added/removed pairs out of ordinary re-reads. The
    generated node is the slot's identity (a plain last-one-wins dict let a later orphan
    re-hash mask a real upstream fork), and the re-hashes are still counted as accounted-for.
    """
    made_by, made = _prov_links(graph)
    generated = set(made_by)

    idx = SlotIndex(total_nodes=len(graph.node_list))
    groups: dict[tuple, list[Node]] = defaultdict(list)
    for n in graph.node_list:                          # node_list is in log (causal) order
        groups[logical_key(n)].append(n)

    for key, members in groups.items():
        if len(members) == 1:
            idx.slots[key] = members[0]
            continue

        # entity re-hashes: exactly one node in the slot was generated this run, the rest
        # are later re-reads of it. Keep the generated one; record the others alongside.
        made_here = [n for n in members if n.id in generated]
        if len(made_here) == 1 and all(isinstance(n, Entity) for n in members):
            idx.slots[key] = made_here[0]
            idx.corroborations[key] += [n for n in members if n is not made_here[0]]
            continue

        # genuinely distinct instances sharing a key: refine by neighbourhood, then ordinal.
        refined: dict[tuple, list[Node]] = defaultdict(list)
        for n in members:
            b = _borrowed_ident(n, graph, made_by, made)
            refined[key + ((b,) if b else ())].append(n)
        for rkey, rmembers in refined.items():
            if len(rmembers) == 1:
                idx.slots[rkey] = rmembers[0]
            else:
                for i, n in enumerate(rmembers):
                    idx.slots[rkey + (f"#{i}",)] = n
    return idx

 
_CONTEXT_TYPES = {"Code", "EnvDigest", "Agent", "User", "Device", "SoftwareAgent"}
_TAIL_FIELDS = {
    ("ModelCheckpoint", "content"),
    ("ModelCheckpoint", "is_best"),
    ("ModelCheckpoint", "filename"),
    ("CheckpointWrite", "call_site"),
    ("EvaluationResult", "content"),
    ("EvaluationResult", "metrics"),
}

_FIELD_PROJECTION = {("ModelCheckpoint", "path"): lambda v: str(v).rsplit("/", 1)[0]}

_TRAJECTORY_TYPES = {"ModelCheckpoint", "CheckpointWrite", "EvaluationResult",
                     "Evaluation", "TrainingStep"}
_SEVERITY = {"spine": 3, "tail": 2, "context": 1}

_SKIP_FIELDS = {"content_hash", "activity_id", "agent_id", "start_ns", "end_ns", "pid", "attributes"}

# Run-scoped substrings. A checkpoint path embeds the run id (RUN_ID=date +%Y_%m_%d_%H%M%S),
# so it differs between ANY two runs and says nothing about either
_RUN_ID_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{6}")


def _normalize(v: Any) -> Any:
    if isinstance(v, str):
        return _RUN_ID_RE.sub("<RUN_ID>", v)
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize(x) for k, x in v.items()}
    return v


def _bucket_field(type_name: str, field_name: str) -> str:
    if type_name in _CONTEXT_TYPES:
        return "context"
    if (type_name, field_name) in _TAIL_FIELDS:
        return "tail"
    return "spine"


# ---- diff result -------------------------------------------------------------

def _trunc(v: Any, width: int) -> str:
    s = str(v)
    return s if len(s) <= width else s[:width] + "…"


def _freeze(v: Any) -> str:
    """Order-insensitive comparison key for a list element (elements are often nested lists)."""
    try:
        return json.dumps(v, sort_keys=True)
    except TypeError:
        return repr(v)


def _render_value_delta(ref: Any, sus: Any, width: int) -> str:
    """What changed, not both values side by side. A truncated head-to-head is useless for the
    fields that matter most here: ModelArchitecture.modules is ~180 entries whose first 60 chars
    are identical between a clean and a backdoored run, so the substituted layer never shows.
    Lists render as their element-wise difference, dicts as their per-key difference."""
    if isinstance(ref, list) and isinstance(sus, list):
        rk, sk = [_freeze(x) for x in ref], [_freeze(x) for x in sus]
        rs, ss = set(rk), set(sk)
        removed = [x for x, k in zip(ref, rk) if k not in ss]
        added = [x for x, k in zip(sus, sk) if k not in rs]
        if not removed and not added:
            return f"reordered, same {len(sus)} element(s)"
        parts = []
        for sign, items in (("−", removed), ("+", added)):
            if not items:
                continue
            shown = ", ".join(_trunc(x, 44) for x in items[:5])
            more = f" …+{len(items) - 5} more" if len(items) > 5 else ""
            parts.append(f"{sign}{len(items)}: {shown}{more}")
        return "   ".join(parts)

    if isinstance(ref, dict) and isinstance(sus, dict):
        parts = [f"{k}: {_trunc(ref.get(k), 30)} → {_trunc(sus.get(k), 30)}"
                 for k in sorted(set(ref) | set(sus)) if ref.get(k) != sus.get(k)]
        more = f"   …+{len(parts) - 5} more key(s)" if len(parts) > 5 else ""
        return "   ".join(parts[:5]) + more

    return f"{_trunc(ref, width)} → {_trunc(sus, width)}"


@dataclass
class FieldDelta:
    """One field that differs between the two runs' occupants of the same slot."""
    name: str
    ref: Any
    suspect: Any
    bucket: str

    def render(self, width: int = 60) -> str:
        body = _render_value_delta(self.ref, self.suspect, width)
        return f"{self.name}: {body}"


@dataclass
class NodeDelta:
    """One divergence between the two runs, carrying enough to explain and rank it."""
    key: tuple
    kind: str                       # "added" | "removed" | "modified"
    rank: int                       # causal depth (root cause = smallest)
    ref: Optional[Node] = None      # the clean-side node (None for added)
    suspect: Optional[Node] = None  # the suspect-side node (None for removed)
    bucket: str = "spine"           # worst bucket among its field deltas
    field_deltas: list[FieldDelta] = field(default_factory=list)

    @property
    def node(self) -> Node:
        return self.suspect or self.ref

    @property
    def type_name(self) -> str:
        return type(self.node).__name__

    @property
    def ident(self) -> str:
        return "/".join(str(p) for p in self.key[1:]) or "—"

    def spine_fields(self) -> list[FieldDelta]:
        return [f for f in self.field_deltas if f.bucket == "spine"]


@dataclass
class GraphDiff:
    deltas: list[NodeDelta] = field(default_factory=list)
    unchanged: int = 0
    ref_index: Optional[SlotIndex] = None
    sus_index: Optional[SlotIndex] = None

    def _in(self, bucket: str) -> list[NodeDelta]:
        # Causal rank ascending is the ONLY ordering claim this module makes. Within a rank the
        # secondary sort is alphabetical — reproducible output, explicitly not a strength ranking.
        # Rank ties are the normal case, not an edge case: the config entities read at setup
        # (ModelArchitecture, Objective, HyperparameterSet) all sit at rank 0, and picking one of
        # them as "the" root would present a tie-break as a finding.
        return sorted((d for d in self.deltas if d.bucket == bucket),
                      key=lambda d: (d.rank, d.type_name, d.ident))

    def tiers(self) -> list[tuple[int, list[NodeDelta]]]:
        """Spine divergences grouped into rank tiers, earliest first."""
        by_rank: dict[int, list[NodeDelta]] = defaultdict(list)
        for d in self.divergences:
            by_rank[d.rank].append(d)
        return [(r, by_rank[r]) for r in sorted(by_rank)]

    @property
    def divergences(self) -> list[NodeDelta]:
        """Spine divergences only, earliest-first — the actual findings."""
        return self._in("spine")

    @property
    def context(self) -> list[NodeDelta]:
        """Ambient-identity diffs (code/env/agent): surfaced, never scored as root cause."""
        return self._in("context")

    @property
    def tail(self) -> list[NodeDelta]:
        return self._in("tail")

    @property
    def earliest_tier(self) -> list[NodeDelta]:
        """The candidate set: every divergence at the shallowest causal depth. Plural by
        design — nothing here is more root than anything else beside it."""
        t = self.tiers()
        return t[0][1] if t else []

    @property
    def root_cause(self) -> Optional[NodeDelta]:
        """A single root ONLY when the earliest tier has exactly one member. When several
        divergences tie for shallowest, there is no single root and this returns None rather
        than breaking the tie — read earliest_tier instead."""
        tier = self.earliest_tier
        return tier[0] if len(tier) == 1 else None

    def coverage(self) -> str:
        """The totality proof: every node on both sides landed somewhere."""
        parts = []
        for label, idx in (("reference", self.ref_index), ("suspect", self.sus_index)):
            if idx is None:
                continue
            extra = sum(len(v) for v in idx.corroborations.values())
            tag = f" (+{extra} re-hash)" if extra else ""
            ok = "OK" if idx.accounted == idx.total_nodes else f"LEAK {idx.total_nodes - idx.accounted}"
            parts.append(f"{label}: {idx.accounted}/{idx.total_nodes} nodes in {len(idx.slots)} slots{tag} [{ok}]")
        return "  |  ".join(parts)


# rank = longest dependency chain behind a node; the smallest-rank divergence is the root.

def causal_rank(graph: Graph) -> dict[str, int]:
    preds: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edge_list:
        if e.source in preds and e.target in graph.nodes:
            preds[e.source].append(e.target)     # source depends on target

    rank: dict[str, int] = {}

    def visit(nid: str, stack: frozenset) -> int:
        if nid in rank:
            return rank[nid]
        if nid in stack:                         # cycle guard (shouldn't happen in a DAG)
            return 0
        r = 0
        for p in preds.get(nid, ()):
            r = max(r, visit(p, stack | {nid}) + 1)
        rank[nid] = r
        return r

    for nid in graph.nodes:
        visit(nid, frozenset())
    return rank


def _compare(ref: Node, suspect: Node, literal: bool) -> list[FieldDelta]:
    """Every field of the slot's two occupants, whatever node kind they are.

    Declared dataclass fields AND the open attributes dict, because that is where the
    forensic payload of an Activity lives — a TrainingStep has no content hash, so
    `batch_sequence`, `forward_inputs_unmatched`, `dataloader_origin` and the rest are
    reachable no other way. Entities additionally report the hash fork itself, which is the
    ground truth of divergence even when no declared field explains it (the content is in
    the hashed bytes)."""
    t = type(suspect).__name__
    norm = (lambda v: v) if literal else _normalize
    out: list[FieldDelta] = []

    for f in fields(suspect):
        if f.name in _SKIP_FIELDS:
            continue
        rv, sv = norm(getattr(ref, f.name, None)), norm(getattr(suspect, f.name, None))
        proj = _FIELD_PROJECTION.get((t, f.name))
        if proj and rv is not None and sv is not None:
            rv, sv = proj(rv), proj(sv)
        if rv != sv:
            out.append(FieldDelta(f.name, rv, sv, _bucket_field(t, f.name)))

    ra = norm(getattr(ref, "attributes", None) or {})
    sa = norm(getattr(suspect, "attributes", None) or {})
    for k in sorted(set(ra) | set(sa)):
        if ra.get(k) != sa.get(k):
            out.append(FieldDelta(k, ra.get(k), sa.get(k), _bucket_field(t, k)))

    if isinstance(suspect, Entity) and ref.id != suspect.id:
        out.append(FieldDelta("content", ref.id[:12], suspect.id[:12], _bucket_field(t, "content")))
    return out


def _node_bucket(node: Node, deltas: list[FieldDelta]) -> str:
    if not deltas:
        return "spine"
    if type(node).__name__ in _CONTEXT_TYPES:
        return "context"
    return max((d.bucket for d in deltas), key=lambda b: _SEVERITY[b])


def _structural_bucket(node: Node) -> str:
    """Bucket for a node that exists on one side only. There are no field deltas to read, so
    the verdict rests on the type: a per-epoch artifact that simply wasn't emitted (or was
    emitted extra) reflects a different training curve, not a different pipeline."""
    t = type(node).__name__
    if t in _CONTEXT_TYPES:
        return "context"
    if t in _TRAJECTORY_TYPES:
        return "tail"
    return "spine"


def diff(reference: Graph, suspect: Graph, literal: bool = False) -> GraphDiff:
    """Align reference (clean) vs suspect by logical slot; report every field that diverges."""
    ref_idx, sus_idx = _slot_index(reference), _slot_index(suspect)
    ref_rank, sus_rank = causal_rank(reference), causal_rank(suspect)

    out = GraphDiff(ref_index=ref_idx, sus_index=sus_idx)
    for key in sorted(ref_idx.slots.keys() | sus_idx.slots.keys(), key=lambda k: tuple(map(str, k))):
        r, s = ref_idx.slots.get(key), sus_idx.slots.get(key)
        if r is not None and s is not None:
            fds = _compare(r, s, literal)
            if fds:
                out.deltas.append(NodeDelta(key, "modified", sus_rank.get(s.id, 0), ref=r,
                                            suspect=s, bucket=_node_bucket(s, fds),
                                            field_deltas=fds))
            else:
                out.unchanged += 1
        elif s is not None:
            out.deltas.append(NodeDelta(key, "added", sus_rank.get(s.id, 0), suspect=s,
                                        bucket=_structural_bucket(s)))
        else:
            out.deltas.append(NodeDelta(key, "removed", ref_rank.get(r.id, 0), ref=r,
                                        bucket=_structural_bucket(r)))
    return out


# ---- reporting ---------------------------------------------------------------

# shape fields worth echoing even when identical — the "same shape, different content" story
# (e.g. label_flip: num_samples/num_patients unchanged, content forks).
_SHAPE_FIELDS = ("num_samples", "num_patients", "num_files", "resolution")


def _shape_note(delta: NodeDelta) -> str:
    if delta.kind != "modified":
        return ""
    same = []
    for f in _SHAPE_FIELDS:
        rv = getattr(delta.ref, f, None)
        if rv is not None and rv == getattr(delta.suspect, f, None):
            same.append(f"{f}={rv}")
    return ("unchanged shape: " + "  ".join(same)) if same else ""


_EPOCH_IDENT = re.compile(r"epoch=(\d+)$")
_EPOCH_ANY = re.compile(r"epoch=(\d+)")     # epoch anywhere in a composite/borrowed ident


def _ident_span(members: list[NodeDelta]) -> str:
    """Name every instance in a folded row, contracting a contiguous epoch run to a range."""
    epochs = []
    for m in members:
        mo = _EPOCH_IDENT.fullmatch(m.ident)
        if not mo:
            epochs = None
            break
        epochs.append(int(mo.group(1)))
    if epochs:
        epochs.sort()
        if epochs[-1] - epochs[0] + 1 == len(epochs):
            return f"epoch={epochs[0]}..{epochs[-1]}"
        return "epoch=" + ",".join(str(e) for e in epochs)
    return ", ".join(m.ident for m in members)


def _epoch_span(members: list[NodeDelta]) -> str:
    """Which epochs a folded row covers, contracted to ranges. Unlike _ident_span this scrapes
    `epoch=N` from ANYWHERE in the ident, because the per-epoch tail nodes carry composite keys
    (an EvaluationResult's is `split=val/via:epoch=7,split=val`) that _ident_span renders in full
    — 13 of those on one line is why the tail was summarised rather than enumerated."""
    eps = sorted({int(m) for d_ in members for m in _EPOCH_ANY.findall(d_.ident)})
    if not eps:
        return _ident_span(members)
    runs: list[list[int]] = [[eps[0], eps[0]]]
    for e in eps[1:]:
        if e == runs[-1][1] + 1:
            runs[-1][1] = e
        else:
            runs.append([e, e])
    parts = [str(a) if a == b else f"{a}..{b}" for a, b in runs]
    return "epoch=" + (",".join(parts[:6]) + ("…" if len(parts) > 6 else ""))


def _fold_deltas(deltas: list[NodeDelta], fold: bool, fields_of) -> list[list[NodeDelta]]:
    """Group repeats of one finding across instances, ordered rank-ascending. `fields_of` picks
    which field deltas make two findings 'the same', so the spine can fold on its spine fields
    while the tail folds on all of its own."""
    groups: dict[tuple, list[NodeDelta]] = defaultdict(list)
    for delta in deltas:
        gk = ((delta.type_name, delta.kind, tuple(f.name for f in fields_of(delta)))
              if fold else (delta.key,))
        groups[gk].append(delta)
    return sorted(groups.values(),
                  key=lambda ms: (min(m.rank for m in ms), ms[0].type_name, ms[0].ident))


def _tail_breakdown(d: GraphDiff, fold: bool) -> list[str]:
    """Enumerate the tail instead of counting it. Two unlike things land in this bucket and the
    one-line summary conflates them: a MODIFIED slot (present in both runs, trained values differ
    — expected on any rerun) and an added/removed one (only one run emitted the artifact at all,
    which is a run-length difference, not a value difference). A reader who sees `CheckpointWrite`
    in a line about 'weights/metrics' reasonably concludes a checkpoint's contents differed, when
    in fact one run wrote checkpoints the other never wrote."""
    out: list[str] = []
    for label, kinds in (("values differ, same slot on both sides", ("modified",)),
                         ("present on one side only — run length / trajectory", ("added", "removed"))):
        members = [t for t in d.tail if t.kind in kinds]
        if not members:
            continue
        out.append(f"      {label}:")
        for group in _fold_deltas(members, fold, lambda x: x.field_deltas):
            head = min(group, key=lambda m: m.rank)
            where = head.ident if len(group) == 1 else _epoch_span(group)
            count = "" if len(group) == 1 else f"  [{len(group)}]"
            what = (", ".join(f.name for f in head.field_deltas)
                    or ("suspect only" if head.kind == "added" else "reference only"))
            out.append(f"        {head.kind.upper():8s} {head.type_name}({where}){count}   {what}")
    return out


def format_report(d: GraphDiff, fold: bool = True, show_all: bool = True,
                  tail_verbose: bool = False) -> str:
    div = d.divergences
    tiers = d.tiers()
    head = f"provenance diff: {len(div)} spine divergence(s) in {len(tiers)} rank tier(s), {d.unchanged} slots identical"
    lines = [head, f"  coverage — {d.coverage()}"]
    if tiers:
        r, members = tiers[0]
        lines.append(f"  earliest divergence at rank {r}: {len(members)} co-equal candidate(s)"
                     + ("" if len(members) == 1 else " — no ordering within a tier"))
    lines.append("")
    if not div:
        lines.append("  no spine divergence: every compared slot is identical on data lineage.")

    earliest = tiers[0][0] if tiers else None
    for members in _fold_deltas(div, fold, lambda x: x.spine_fields()):
        head_delta = min(members, key=lambda m: m.rank)
        lo, hi = min(m.rank for m in members), max(m.rank for m in members)
        span = f"rank {lo}" if lo == hi else f"rank {lo}..{hi}"
        tag = "EARLIEST" if lo == earliest else "downstream"
        where = head_delta.ident if len(members) == 1 else _ident_span(members)
        count = "" if len(members) == 1 else f"  [{len(members)} instances]"
        lines.append(f"  [{tag}] {head_delta.kind.upper()} {head_delta.type_name}({where})  {span}{count}")
        for fd in head_delta.spine_fields():
            lines.append(f"      {fd.render()}")
        note = _shape_note(head_delta)
        if note:
            lines.append(f"      {note}")

    for delta in d.context:
        lines += ["", f"  [context] {delta.kind.upper()} {delta.type_name} — not a data fork; recorded, not scored"]
        # a different *committed* code is a real ship-it event; uncommitted churn is just editing.
        if delta.type_name == "Code" and delta.kind == "modified":
            rc, sc = getattr(delta.ref, "git_commit", None), getattr(delta.suspect, "git_commit", None)
            if rc != sc:
                lines.append(f"      committed code changed: {str(rc)[:8]} → {str(sc)[:8]}  (worth a look)")
            else:
                lines.append(f"      same commit {str(sc)[:8]}, working-tree edits only")
        elif show_all:
            for fd in delta.field_deltas[:4]:
                lines.append(f"      {fd.render(40)}")

    if d.tail:
        types = sorted({t.type_name for t in d.tail})
        lines += ["", f"  {len(d.tail)} trained-tail divergence(s) ({', '.join(types)}) — "
                      "weights/metrics, expected to differ across any rerun"]
        if tail_verbose:
            lines += _tail_breakdown(d, fold)
    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Structural diff of two provenance ledgers.")
    ap.add_argument("reference", type=Path, help="clean / baseline residue.jsonl")
    ap.add_argument("suspect", type=Path, help="suspect residue.jsonl")
    ap.add_argument("--no-fold", dest="fold", action="store_false",
                    help="print every instance separately instead of folding repeats onto one "
                         "line (folding is presentation only; every instance is named either way)")
    ap.add_argument("--literal", action="store_true",
                    help="do not normalise run-scoped substrings (run ids in paths)")
    ap.add_argument("--tail-verbose", action="store_true",
                    help="enumerate the trained-tail divergences (split by modified vs "
                         "present-on-one-side) instead of reporting only their count")
    args = ap.parse_args()

    d = diff(load(args.reference), load(args.suspect), literal=args.literal)
    print(format_report(d, fold=args.fold, tail_verbose=args.tail_verbose))


if __name__ == "__main__":
    main()
