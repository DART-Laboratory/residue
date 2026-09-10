"""Render a projected PROV graph to Graphviz DOT, then to SVG/PNG via the system `dot`.

The whole point is readability at scale (DESIGN: the raw graph is edge-dense because every
activity re-attaches its full context — the agents and the code/env/hyperparameters that are
*constant across the run*). So by default we COLLAPSE that constant context: the shared nodes
go in one "Run context" box and their per-activity fan-out edges (wasAssociatedWith, and the
intent-role `used` edges) are suppressed. What remains is the lineage backbone — the activity
spine (wasInformedBy) and the data lineage (used-input, wasGeneratedBy, wasDerivedFrom) — which
is the story you actually read. Pass full=True to draw every edge (the hairball, for debugging).

Three more projections keep that backbone legible as runs grow: the training loop
is a repeated motif, so `fold` elides its middle epochs into one summary node (head/tail epochs
stay expanded); `slice_from` renders only the lineage of one node (its ancestors — what produced
it — and/or descendants — what it reached); and `verbose` toggles the per-edge labels and node
attribute detail that are signal on a small graph but noise on a big one (off by default).

"DOT" is Graphviz's text format; we emit it and pipe it to /usr/bin/dot, so there is no python
graphviz dependency.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

from residue.analysis.diff import (_CONTEXT_TYPES, _TRAJECTORY_TYPES, causal_rank, diff)
from residue.schema.nodes import Activity, Agent, Entity
from residue.schema.types import EdgeRelation, Role
from residue.store.graph import Graph, load

# Shape always encodes PROV kind, in every palette — colour is then free to encode something else.
_SHAPES = {"entity": "box", "activity": "ellipse", "agent": "house"}

# Backbone edge styling: (color, style, label). The label is only drawn when verbose.
_EDGE_STYLE = {
    EdgeRelation.WAS_INFORMED_BY:   ("#444444", "bold",   ""),        # the temporal spine
    EdgeRelation.WAS_GENERATED_BY:  ("#1f6feb", "solid",  "genBy"),
    EdgeRelation.WAS_DERIVED_FROM:  ("#888888", "dashed", ""),        # label = derivation_type
    EdgeRelation.USED:              ("#2da44e", "solid",  "uses"),
}

_MONO_EDGE_STYLE = {r: ("#000000", "solid", "") for r in _EDGE_STYLE}


# ---- palettes ----------------------------------------------------------------
# What node COLOUR encodes is a choice, so it is a named palette rather than a constant.
#   kind    - PROV kind (entity/activity/agent). The original scheme; still the default.
#   bucket  - the diff taxonomy (spine / context / trajectory), so a figure states the
#             classification instead of the caption having to. Context is grey on purpose:
#             it is the ambient stuff excluded from ranking, and grey says so.
#   rca     - one message only: which nodes are root-cause candidates and how deep. Everything
#             else is plain black-on-white so nothing competes with the red. The red FADES with
#             causal depth (earliest tier strongest) because darker reads as more important, and
#             the earliest tier is the finding — a deepening ramp would emphasise the consequences.
#   bw      - print/greyscale-safe: meaning carried by shape and border, not hue.
# Tiers are banded to three, not one shade per rank: ranks run into double digits and a 12-step
# ramp would imply a precision the ranking does not claim.

@dataclass(frozen=True)
class Palette:
    """fills/borders are keyed by *class*, which `classify` decides how to compute."""
    classify: str                                   # "kind" | "bucket" | "rca"
    fills: dict[str, str]
    borders: dict[str, tuple[str, int]]             # class -> (colour, penwidth); "" = dot default
    edges: dict = None                              # None -> _EDGE_STYLE
    legend: tuple[tuple[str, str], ...] = ()        # (class, caption) rows drawn under the graph title
    fold_fill: str = "#ffd27f"
    best_outline: str = "#1f78b4"
    flag_red: str = "#cc0000"
    flag_amber: str = "#e08a00"
    table_border: str = "#999999"
    link_divergent: bool = False                    # colour an edge whose BOTH ends are divergent

    def style(self, cls: str) -> tuple[str, str, int]:
        fill = self.fills.get(cls, self.fills.get("_default", "#ffffff"))
        colour, width = self.borders.get(cls, self.borders.get("_default", ("", 0)))
        return fill, colour, width


PALETTES: dict[str, Palette] = {
    "kind": Palette(
        classify="kind",
        fills={"entity": "#cfe8ff", "activity": "#ffe0b3", "agent": "#cff3d0"},
        borders={"_default": ("", 0)},
    ),
    "bucket": Palette(
        classify="bucket",
        fills={"spine": "#cfe0f5", "tail": "#e4d6ef", "context": "#e8e8e8"},
        borders={"spine": ("#2166ac", 2), "tail": ("#762a83", 2),
                 "context": ("#909090", 1)},
        legend=(("spine", "spine"), ("tail", "trajectory"), ("context", "context")),
        fold_fill="#cfe0f5",       # the elided epochs are training steps: spine, like what they replace
    ),
    "rca": Palette(
        classify="rca",
        # red is reserved for ranked candidates. A trajectory consequence is grey, not red-outlined:
        # any red reads as "finding" at a glance, and these are explicitly NOT candidates — grey is
        # the same "excluded from ranking" signal the bucket palette gives context.
        fills={"plain": "#ffffff", "tier0": "#e8433c", "tier1": "#f28c86", "tierN": "#fbd5d2",
               "touched": "#d9d9d9"},
        borders={"plain": ("#000000", 1), "tier0": ("#8f0f16", 3),
                 "tier1": ("#c22b28", 2), "tierN": ("#e08a86", 1), "touched": ("#606060", 1)},
        edges=_MONO_EDGE_STYLE,
        legend=(("tier0", "earliest rank tier — candidates"), ("tier1", "next tier"),
                ("tierN", "downstream divergence"), ("plain", "identical to reference")),
        fold_fill="#ffffff", best_outline="#000000", table_border="#000000",
        link_divergent=True,
    ),
    "bw": Palette(
        classify="kind",
        # true two-tone: no greys at all, so kind rides entirely on shape (box/ellipse/house), which
        # every palette already encodes. fold_fill and best_outline are overridden because their
        # defaults are an orange and a blue that would be the only colour on the page.
        fills={"entity": "#ffffff", "activity": "#ffffff", "agent": "#ffffff"},
        borders={"_default": ("#000000", 1)},
        edges=_MONO_EDGE_STYLE,
        fold_fill="#ffffff", best_outline="#000000", table_border="#000000",
    ),
}

def _type_bucket(node) -> str:
    """Fallback bucket for a node with no diff to consult. Type-level only: whether a *modified*
    node is spine or tail is a field-level verdict that needs the reference run, so a single-ledger
    render approximates, and `--vs` supplies the real thing."""
    t = type(node).__name__
    if t in _CONTEXT_TYPES or node.KIND == "agent":
        return "context"
    return "tail" if t in _TRAJECTORY_TYPES else "spine"


def _node_class(node, palette: Palette, buckets: dict, tiers: dict, with_tail: bool = False) -> str:
    if palette.classify == "bucket":
        return buckets.get(node.id) or _type_bucket(node)
    if palette.classify == "rca":
        return _rca_class(node.id, buckets, tiers, with_tail)
    return node.KIND


def _rca_class(nid: str, buckets: dict, tiers: dict, with_tail: bool) -> str:
    """Red is reserved for ranked candidates; a trajectory consequence goes grey instead, so the
    blast radius is visible without 100+ nodes competing with the handful that are the finding."""
    t = tiers.get(nid)
    if t is None:
        return "touched" if with_tail and buckets.get(nid) == "tail" else "plain"
    return "tier0" if t == 0 else "tier1" if t == 1 else "tierN"


def diff_overlays(reference: Graph, suspect: Graph) -> tuple[dict, dict, set]:
    """Per-node overlays read off a real diff: {id -> bucket}, {id -> tier index}, and the ids
    present only in the suspect. Keyed on the suspect graph, which is the one being drawn —
    reference-only (removed) nodes have nothing to paint and are reported by diff.py instead."""
    d = diff(reference, suspect)
    buckets, added = {}, set()
    for delta in d.deltas:
        if delta.suspect is None:
            continue
        buckets[delta.suspect.id] = delta.bucket
        if delta.kind == "added":
            added.add(delta.suspect.id)
    tiers = {m.suspect.id: i for i, (_, members) in enumerate(d.tiers())
             for m in members if m.suspect is not None}
    return buckets, tiers, added


_FOLD_ID = "__training_fold__"   # synthetic node standing in for the elided epochs


def _short(node_id: str) -> str:
    return node_id[:8] if len(node_id) > 8 else node_id


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _verbose_detail(node) -> str:
    """Extra identifying fields appended to the label only in verbose mode."""
    cls = type(node).__name__
    parts: list[str] = []
    if cls == "HyperparameterSet":
        for k in ("learning_rate", "batch_size", "epochs", "seed"):
            v = getattr(node, k, None)
            if v is not None:
                parts.append(f"{k}={v}")
    elif cls == "EvaluationResult":
        metrics = getattr(node, "metrics", None) or {}
        display = [("F1", "f1_macro"), ("AUROC", "auroc_macro"), ("Accuracy", "accuracy")]
        for label, key in display:
            if key in metrics:
                v = metrics[key]
                parts.append(f"{label}={v:.3f}" if isinstance(v, float) else f"{label}={v}")
    elif cls == "ModelCheckpoint":
        if getattr(node, "path", None):
            parts.append(Path(node.path).name)
    elif cls == "TransformationConfig":
        for k in ("size", "interpolation", "library", "version"):
            v = getattr(node, k, None)
            if v is not None:
                parts.append(f"{k}={v}")
    elif cls == "Objective":
        if getattr(node, "terminal_op", None):
            parts.append(f"grad_fn={node.terminal_op}")
    elif cls == "NormalizationStats":
        for k in ("mean", "std"):
            v = getattr(node, k, None)
            if v:
                parts.append(f"{k}={v}")
    return "\\n".join(parts)


# --detail=all: dump every recorded field rather than the curated shortlist above. Structural
# ids and timestamps stay out (they identify the record, not the run); hashes are shortened.
_DETAIL_SKIP = {"id", "activity_id", "agent_id", "start_ns", "end_ns", "attributes"}


def _fmt_value(v, width: int = 46) -> str:
    if isinstance(v, float):
        s = f"{v:.6g}"
    elif isinstance(v, (list, tuple)):
        s = f"[{len(v)}] " + ", ".join(str(x) for x in v[:3]) + (" …" if len(v) > 3 else "")
    elif isinstance(v, dict):
        s = "{" + ", ".join(f"{k}: {x}" for k, x in list(v.items())[:3]) + ("…}" if len(v) > 3 else "}")
    else:
        s = str(v)
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return s if len(s) <= width else s[:width] + "…"


def _all_detail(node) -> str:
    """Every field the node actually carries -- declared dataclass fields and the open attributes
    dict alike, since that is where an Activity's forensic payload lives (cf. diff._compare)."""
    parts: list[str] = []
    for f in fields(node):
        if f.name in _DETAIL_SKIP:
            continue
        v = getattr(node, f.name, None)
        if v is None or v == [] or v == {} or v == "":
            continue
        parts.append(f"{f.name}={_fmt_value(v[:12] if 'hash' in f.name and isinstance(v, str) else v)}")
    for k, v in sorted((getattr(node, "attributes", None) or {}).items()):
        if v is None or v == [] or v == {} or v == "":
            continue
        parts.append(f"{k}={_fmt_value(v)}")
    return "\\n".join(parts)


def _loss_name(node) -> str:
    """Human-readable loss from the Objective's autograd terminal op: strip the `Backward0` suffix
    PyTorch appends (BinaryCrossEntropyWithLogitsBackward0 -> BinaryCrossEntropyWithLogits)."""
    op = getattr(node, "terminal_op", None)
    return re.sub(r"Backward\d*$", "", op) if op else ""


def _node_label(node, *, verbose: bool = False, detail: str = "curated") -> str:
    """Type on the first line; the most identifying field on the second."""
    t = type(node).__name__
    ident = ""
    if isinstance(node, Activity):
        if getattr(node, "epoch", None) is not None:
            ident = f"epoch {node.epoch}"
        elif getattr(node, "operation", None):     # PreprocessingOp: the transform name
            ident = str(node.operation)
        elif getattr(node, "split", None):
            ident = str(node.split)
    elif isinstance(node, Agent):
        ident = (getattr(node, "name", None) or getattr(node, "hostname", None)
                  or getattr(node, "version", None) or "")
    elif isinstance(node, Entity):
        ep = getattr(node, "epoch", None)
        if t == "TransformationConfig":        # the transform params are the identity here, not the hash
            bits = [str(b) for b in (getattr(node, "interpolation", None),
                                     f"{node.size}²" if getattr(node, "size", None) else None) if b]
            ident = " · ".join(bits + [_short(node.id)])
        elif t == "TransformedDataset":         # the data the model consumes, after the transform chain
            nt = getattr(node, "num_transforms", None)
            ident = f"{nt} ops · {_short(node.id)}" if nt is not None else _short(node.id)
        elif t == "Objective":                  # the loss, named from its autograd terminal op
            ident = _loss_name(node) or _short(node.id)
        elif getattr(node, "split", None):
            ident = f"{node.split} · {_short(node.id)}"
        elif ep is not None:
            ident = f"epoch {ep} · {_short(node.id)}"
        else:
            ident = _short(node.id)
    label = f"{t}\\n{ident}" if ident else t
    extra = _all_detail(node) if detail == "all" else (_verbose_detail(node) if verbose else "")
    if extra:
        label += f"\\n{extra}"
    return label


def _is_hidden(edge) -> bool:
    """The constant-context fan-out: association + intent. Suppressed when collapsing."""
    if edge.RELATION == EdgeRelation.WAS_ASSOCIATED_WITH:
        return True
    if edge.RELATION == EdgeRelation.USED and getattr(edge, "role", None) == Role.INTENT:
        return True
    return False


def _context_ids(graph: Graph, kept: set[str]) -> set[str]:
    """Nodes whose every incident edge (within `kept`) is hidden-context — i.e. they ONLY ever
    appear as shared run context (agents, code, env, hyperparameters), never in the data lineage.
    Derived from edges, not hardcoded by type, so new intent entities collapse automatically."""
    incident: dict[str, list] = {nid: [] for nid in kept}
    for e in graph.edge_list:
        if e.source in kept and e.target in kept:
            incident[e.source].append(e)
            incident[e.target].append(e)
    return {nid for nid, es in incident.items() if es and all(_is_hidden(e) for e in es)}


def _shortcut_derivations(graph: Graph) -> set[tuple[str, str]]:
    """The (source, target) pairs of WasDerivedFrom edges that a genBy+used path already expresses:
    source wasGeneratedBy some activity that used target. Only THESE are safe to drop under
    --no-redundancy -- a derivation edge that is a node's sole link (e.g. a finalize-emitted
    SampleManifest, which has no generating activity) has no such shortcut and must stay, else the
    node floats."""
    genby: dict[str, set[str]] = {}   # entity -> activities that generated it
    used: dict[str, set[str]] = {}    # activity -> entities it used
    for e in graph.edge_list:
        if e.RELATION == EdgeRelation.WAS_GENERATED_BY:
            genby.setdefault(e.source, set()).add(e.target)
        elif e.RELATION == EdgeRelation.USED:
            used.setdefault(e.source, set()).add(e.target)
    out: set[tuple[str, str]] = set()
    for e in graph.edge_list:
        if e.RELATION != EdgeRelation.WAS_DERIVED_FROM:
            continue
        if any(e.target in used.get(a, ()) for a in genby.get(e.source, ())):
            out.add((e.source, e.target))
    return out


def _resolve_id(graph: Graph, ref: str) -> str:
    """Map a (possibly truncated) node id to the full id, erroring on miss/ambiguity."""
    if ref in graph.nodes:
        return ref
    matches = [nid for nid in graph.nodes if nid.startswith(ref)]
    if not matches:
        raise SystemExit(f"no node id starts with {ref!r}")
    if len(matches) > 1:
        raise SystemExit(f"{ref!r} is ambiguous: {', '.join(_short(m) for m in matches[:6])} ...")
    return matches[0]


def _slice_ids(graph: Graph, seed: str, direction: str) -> set[str]:
    """Reachable node ids from `seed`. Edges point effect->cause (newer->older, used, informedBy),
    so following source->target walks toward ANCESTORS (what produced the seed); the reverse walks
    toward DESCENDANTS (what the seed reached). Traverses every edge, incl. hidden context, so a
    lineage slice still pulls in the hyperparameters/code/env/agents that produced the seed."""
    fwd: dict[str, list[str]] = {}
    rev: dict[str, list[str]] = {}
    for e in graph.edge_list:
        fwd.setdefault(e.source, []).append(e.target)
        rev.setdefault(e.target, []).append(e.source)

    def reach(adj: dict[str, list[str]]) -> set[str]:
        seen, stack = {seed}, [seed]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    if direction == "ancestors":
        return reach(fwd)
    if direction == "descendants":
        return reach(rev)
    return reach(fwd) | reach(rev)


def _epoch_of(graph: Graph) -> dict[str, int]:
    """Per-node epoch index. TrainingStep/ModelCheckpoint carry `epoch` directly; Evaluation carries
    it in `attributes` (the eval episode pins the epoch that produced it). CheckpointWrite inherits it
    from the TrainingStep it wasInformedBy, and EvaluationResult from the Evaluation that generated it
    -- so a per-epoch eval episode folds into the training loop alongside the epoch it scored, rather
    than hanging out beside the fold node. (HyperparameterSet.epochs is plural — a count, not an
    index — so reading the singular `epoch` attribute deliberately skips it.)"""
    ep: dict[str, int] = {}
    for nid, n in graph.nodes.items():
        e = getattr(n, "epoch", None)
        if e is None:
            e = (getattr(n, "attributes", None) or {}).get("epoch")
        if e is not None:
            ep[nid] = e
    for e in graph.edge_list:
        src = type(graph.nodes.get(e.source)).__name__
        if e.target not in ep:
            continue
        if e.RELATION == EdgeRelation.WAS_INFORMED_BY and src == "CheckpointWrite":
            ep[e.source] = ep[e.target]
        elif e.RELATION == EdgeRelation.WAS_GENERATED_BY and src == "EvaluationResult":
            ep[e.source] = ep[e.target]
    return ep


def _final_checkpoint_id(graph: Graph, kept: set[str]) -> str | None:
    """The run's shipped artifact: the last-saved best.pt ModelCheckpoint (preferred over periodic
    ones, then highest epoch). Kept visible even when its epoch lands in the folded band."""
    best: tuple[tuple[bool, int], str] | None = None
    for nid in kept:
        n = graph.nodes[nid]
        if type(n).__name__ != "ModelCheckpoint":
            continue
        path = getattr(n, "path", "") or ""
        key = (path.endswith("best.pt"), getattr(n, "epoch", None) or -1)
        if best is None or key > best[0]:
            best = (key, nid)
    return best[1] if best else None


def _fold_ids(graph: Graph, kept: set[str], head: int, tail: int) -> tuple[set[str], dict | None]:
    """Epoch-indexed nodes whose epoch falls in the elided middle band, plus a spec describing the
    summary node. Keeps the first `head` and last `tail` epochs expanded; folds only when >=2
    epochs would collapse (folding one saves nothing). The run's final checkpoint is always kept
    expanded -- it's the shipped artifact, so it hangs off the fold node rather than vanishing in."""
    ep = {nid: v for nid, v in _epoch_of(graph).items() if nid in kept}
    epochs = sorted(set(ep.values()))
    keep = set(epochs[:head]) | (set(epochs[len(epochs) - tail:]) if tail else set())
    fold = [e for e in epochs if e not in keep]
    if len(fold) < 2:
        return set(), None
    ids = {nid for nid, v in ep.items() if v in fold}
    ids.discard(_final_checkpoint_id(graph, kept))   # keep the shipped checkpoint visible
    return ids, {"count": len(fold), "lo": min(fold), "hi": max(fold)}


# ---- flagging: signal rules ---------------------------------------------------
# One rule per captured provenance signal (per slice), NOT per attack -- so the set grows with what we
# CAPTURE, not with the attack list, and one rule catches every attack that trips the same signal. Each
# rule reads a node and returns a short reason if it looks tampered, else None. No clean run needed.
# Add a rule here as each slice's signal becomes detectable (next: Slice 11 order-vs-seed mismatch).

# A stock single loss's autograd graph terminates in its own loss op (NllLoss/BCEWithLogits/Mse/...).
# When the terminal op is instead one of these arithmetic combinators, two or more subgraphs were joined
# into one objective -- an observed structural fact, whatever produced it. (Mul/Div left out: a legit
# weighted single loss is a bare Mul, not a multi-term combine.)
_LOSS_COMBINATOR_PREFIXES = ("AddBackward", "SubBackward", "MeanBackward", "SumBackward",
                             "StackBackward", "CatBackward")


def _rule_composite_objective(node) -> str | None:
    """Slice 12: an Objective whose terminal op is an arithmetic combinator = the loss joins two or more
    terms rather than being one loss op. The reason states only that observed structure, not a cause."""
    if type(node).__name__ != "Objective":
        return None
    op = getattr(node, "terminal_op", None) or ""
    return f"loss combines ≥2 terms ({op})" if op.startswith(_LOSS_COMBINATOR_PREFIXES) else None


def _rule_extra_backward(node) -> str | None:
    """Slice 12: >1 backward pass per optimizer step. The reason states the observed count only -- it
    does not name a technique (could be gradient balancing, accumulation, a multi-step update, ...)."""
    if type(node).__name__ != "TrainingStep":
        return None
    n = node.attributes.get("backward_per_step") or 0
    return f"{n} backward passes/step" if n > 1 else None


def _rule_forward_input_mutated(node) -> str | None:
    """Slice 13: a forward whose input differs from the batch the loader handed over = the batch was
    rewritten in-loop between loader and model(). The reason states only that observed fact -- the edit
    could be an in-loop poison (CCBA) or a benign GPU-side transform (mixup/cutmix), so it stays amber."""
    if type(node).__name__ != "TrainingStep":
        return None
    n = node.attributes.get("forward_inputs_unmatched") or 0
    return f"{n} batch(es) differ from what the loader produced" if n > 0 else None


_FLAG_RULES = [_rule_composite_objective, _rule_extra_backward, _rule_forward_input_mutated]


# ---- signature dictionary (AV-style) -----------------------------------------
# A known-attack layer stacked on the per-signal tells above. A signature names a combination of
# tells that a known attack produces, with a causal constraint: the tells must fire on nodes a
# direct edge joins (a Poirot-style graph signature, not a flat bag of events). Trade-off, by design:
# precise on the enumerated attacks, blind to novel ones and to variants that trip only some tells
# (a blind-backdoor variant weighting the two losses by a fixed constant instead of MGDA fires
# composite-objective ALONE, so it never matches the pair below and stays amber). Grow the list as
# attacks are catalogued.

@dataclass(frozen=True)
class Signature:
    name: str            # the known pattern this combination indicates
    tells: tuple         # per-signal rule fns that must ALL fire...
    linked: bool = True  # ...on nodes forming one edge-connected region


_SIGNATURES = [
    Signature(name="blind-backdoor pattern (composite objective + linked extra backwards)",
              tells=(_rule_composite_objective, _rule_extra_backward)),
]


def _one_component(nodes: list[str], adj: dict[str, set[str]]) -> bool:
    """True if every node in `nodes` reaches every other using only direct edges between them."""
    ns = set(nodes)
    if len(ns) < 2:
        return False
    seen = {next(iter(ns))}
    stack = list(seen)
    while stack:
        for nb in adj.get(stack.pop(), ()):
            if nb in ns and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == ns


def flag_suspicious(graph: Graph) -> dict[str, tuple[str, str]]:
    """Map node id -> (severity, reason). No clean run needed. Two layers:
      amber  a lone per-signal tell fired here (composite objective, extra backwards, ...). Each is
             individually common in benign pipelines (multi-task losses trip the first, gradient
             accumulation the second), so alone it means only 'worth a look' -- an honest observation.
      red    these nodes match a known signature in _SIGNATURES: every tell of the signature fired,
             on nodes a direct edge joins. The node still shows its OWN observed tell (not the pattern
             name) -- red only marks that the tells co-occur and are linked, a stronger but still
             name-free signal. The signature's name is surfaced separately by the RCA, never asserted
             on the node."""
    hits: dict[object, list[str]] = {rule: [] for rule in _FLAG_RULES}
    reasons: dict[str, str] = {}
    for n in graph.node_list:
        for rule in _FLAG_RULES:
            reason = rule(n)
            if reason:
                hits[rule].append(n.id)
                reasons[n.id] = reason
                break

    adj: dict[str, set[str]] = {}
    for e in graph.edge_list:
        adj.setdefault(e.source, set()).add(e.target)
        adj.setdefault(e.target, set()).add(e.source)

    out: dict[str, tuple[str, str]] = {}
    for sig in _SIGNATURES:
        if any(not hits.get(t) for t in sig.tells):
            continue                                        # a required tell never fired
        matched = [nid for t in sig.tells for nid in hits[t]]
        if sig.linked and not _one_component(matched, adj):
            continue                                        # tells fired but aren't wired together
        for nid in matched:
            out[nid] = ("red", reasons[nid])   # the node's own observed tell, never the pattern name

    for nid, reason in reasons.items():                     # tells not claimed by a signature -> amber
        out.setdefault(nid, ("amber", reason))
    return out


# ---- flag-path RCA (BackTracker-style backward tracing) -----------------------
# Localizes each finding WITHOUT a clean run: group the flagged nodes into causally-connected
# components, and within each the earliest node (smallest causal_rank = closest to the pipeline's
# inputs) is the root cause; the rest are its downstream consequences. The root's captured call_site
# names WHERE it entered; the backward_call_site across the finding corroborate with the actual code that
# ran (this is where blind_backdoor.py surfaces). The signature NAME (red) is a SEPARATE interpretive
# label -- the root cause itself is the observed node + source site, never the name.

def _first_tell(node) -> str | None:
    """The observed signal on a node, independent of any signature that later claimed it."""
    for rule in _FLAG_RULES:
        r = rule(node)
        if r:
            return r
    return None


def _matched_signature(graph: Graph, comp: list[str]) -> str | None:
    """The name of the signature a red component matches -- every one of its tells fired on some node
    in the component. Recomputed here (not read off the nodes) so the pattern name lives ONLY in the
    RCA, as a separate interpretive label, and is never asserted on the graph nodes themselves."""
    nodes = [graph.nodes[nid] for nid in comp]
    for sig in _SIGNATURES:
        if all(any(rule(n) for n in nodes) for rule in sig.tells):
            return sig.name
    return None


@dataclass
class Finding:
    severity: str            # "red" | "amber"
    signature: str | None    # interpretive name (red matches only); None for a lone amber tell
    root_id: str             # earliest-in-causal-order flagged node = root cause
    root_tell: str           # the observed signal on the root (honest fact, not the name)
    root_call_site: object   # source location(s) captured on the root, if any
    root_rank: int
    consequences: list       # other flagged node ids in the finding, causal order


def flag_rca(graph: Graph, flagged: dict[str, tuple[str, str]] | None = None) -> list[Finding]:
    """Root-cause the flag-path findings by backward tracing. One Finding per causally-connected
    group of flagged nodes (earliest = root, rest = consequences), most-severe / earliest first."""
    flagged = flagged if flagged is not None else flag_suspicious(graph)
    if not flagged:
        return []
    rank = causal_rank(graph)

    adj: dict[str, set[str]] = {}                    # undirected adjacency over flagged nodes only
    for e in graph.edge_list:
        if e.source in flagged and e.target in flagged:
            adj.setdefault(e.source, set()).add(e.target)
            adj.setdefault(e.target, set()).add(e.source)

    seen: set[str] = set()
    findings: list[Finding] = []
    for start in flagged:
        if start in seen:
            continue
        comp, stack = [start], [start]               # walk the connected component
        seen.add(start)
        while stack:
            for nb in adj.get(stack.pop(), ()):
                if nb not in seen:
                    seen.add(nb); comp.append(nb); stack.append(nb)
        comp.sort(key=lambda nid: rank.get(nid, 0))  # earliest (shallowest) = root cause
        root = comp[0]
        sev, _ = flagged[root]
        node = graph.nodes[root]
        findings.append(Finding(
            severity=sev, signature=_matched_signature(graph, comp) if sev == "red" else None,
            root_id=root, root_tell=_first_tell(node) or "",
            root_call_site=(node.attributes or {}).get("call_site"),
            root_rank=rank.get(root, 0), consequences=comp[1:]))
    findings.sort(key=lambda f: (f.severity != "red", f.root_rank))
    return findings


def _innermost(call_site) -> str | None:
    """The first (innermost) frame of a captured call_site, which may be a list of frames or a str."""
    if not call_site:
        return None
    return call_site[0] if isinstance(call_site, (list, tuple)) else str(call_site)


def _code_sites(graph: Graph, ids: list) -> list[str]:
    """Every source location the finding's nodes recorded -- call_site frames + backward_call_site keys --
    deduped in first-seen order. The observed 'what code ran', where injected code surfaces."""
    sites: list[str] = []
    for nid in ids:
        attrs = graph.nodes[nid].attributes or {}
        cs = attrs.get("call_site")
        frames = list(cs) if isinstance(cs, (list, tuple)) else ([cs] if cs else [])
        for s in frames + list((attrs.get("backward_call_site") or {}).keys()):
            if s and s not in sites:
                sites.append(s)
    return sites


def format_rca(graph: Graph, findings: list[Finding] | None = None) -> str:
    """Human-readable flag-path RCA report."""
    if findings is None:
        findings = flag_rca(graph)
    if not findings:
        return "flag-path RCA: no signals fired."
    lines = [f"flag-path RCA: {len(findings)} finding(s)", ""]
    for f in findings:
        head = f"[{f.severity.upper()}]" + (f" {f.signature}" if f.signature else "")
        root = graph.nodes[f.root_id]
        lines.append(f"  {head}")
        lines.append(f"    ROOT CAUSE  {type(root).__name__}  rank={f.root_rank}")
        lines.append(f"      tell: {f.root_tell}")
        site = _innermost(f.root_call_site)
        if site:
            lines.append(f"      formed at: {site}")
        if f.consequences:
            kinds = ", ".join(sorted({type(graph.nodes[c]).__name__ for c in f.consequences}))
            lines.append(f"    {len(f.consequences)} consequence(s): {kinds}")
        sites = _code_sites(graph, [f.root_id, *f.consequences])
        if sites:
            lines.append("    code that ran in this finding:")
            lines += [f"      - {s}" for s in sites]
        lines.append("")
    return "\n".join(lines).rstrip()


def _legend_row(pal: Palette, span: int, legend=None) -> str:
    """A key for what colour means, hosted in the run-context box so it costs no layout."""
    legend = pal.legend if legend is None else legend
    if not legend:
        return ""
    swatches = "".join(
        f'<TD BGCOLOR="{pal.fills[c]}" COLOR="{pal.borders.get(c, ("", 0))[0] or "#000000"}"'
        f' FIXEDSIZE="TRUE" WIDTH="20" HEIGHT="12"></TD>'
        f'<TD BORDER="0" ALIGN="LEFT"><FONT POINT-SIZE="11">{_html_escape(cap)}</FONT></TD>'
        for c, cap in legend)
    return (f'<TR><TD COLSPAN="{span}" BORDER="0">'
            f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="4" CELLPADDING="3">'
            f'<TR>{swatches}</TR></TABLE></TD></TR>')


def to_dot(graph: Graph, *, full: bool = False, fold: bool = False, head: int = 1, tail: int = 1,
           slice_from: str | None = None, direction: str = "ancestors",
           verbose: bool = False, no_redundancy: bool = False, mono: bool = False,
           rankdir: str = "TB", title: str | None = None,
           flagged: dict[str, tuple[str, str]] | None = None,
           palette: str | Palette = "kind", buckets: dict[str, str] | None = None,
           tiers: dict[str, int] | None = None, added: set[str] | None = None,
           with_tail: bool = False, compact: bool = False, detail: str = "curated") -> str:
    flagged = flagged or {}   # {node id -> (severity, reason)}; red = signature match, amber = lone tell
    pal = palette if isinstance(palette, Palette) else PALETTES[palette]
    buckets, tiers, added = buckets or {}, tiers or {}, added or set()
    legend = pal.legend
    if with_tail and pal.classify == "rca":
        legend = legend[:-1] + (("touched", "trajectory consequence — not ranked"),) + legend[-1:]
    kept = _slice_ids(graph, _resolve_id(graph, slice_from), direction) if slice_from else set(graph.nodes)
    context = set() if full else _context_ids(graph, kept)
    folded, fold_spec = _fold_ids(graph, kept, head, tail) if fold else (set(), None)

    # The fold hides instances, not findings: if any elided epoch diverged, the stand-in inherits
    # the earliest tier among them, so a folded band can never render as clean.
    fold_tiers = [tiers[nid] for nid in folded if nid in tiers]
    fold_tail = [nid for nid in folded if nid not in tiers and buckets.get(nid) == "tail"]
    if fold_tiers:
        tiers = {**tiers, _FOLD_ID: min(fold_tiers)}

    def remap(nid: str) -> str:
        return _FOLD_ID if nid in folded else nid

    lines = [
        "digraph provenance {",
        f"  rankdir={rankdir};",        # TB = upright portrait; LR = left-to-right (wide for long chains)
        '  fontname="Helvetica"; fontsize=10;',   # the graph label (run-context box) inherits this
        "  node [style=filled, fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=8];",
    ]
    if compact:
        # squeeze whitespace only: separation between ranks/nodes, never the type or the box around
        # it. A figure shrunk by shrinking its labels just moves the problem to the printed page.
        lines += ["  ranksep=0.16; nodesep=0.12;"]

    best_ckpt = _final_checkpoint_id(graph, kept)   # the shipped best.pt: badge it as the run's output

    def node_stmt(node) -> str:
        shape = _SHAPES[node.KIND]
        fill, outline, width = pal.style(_node_class(node, pal, buckets, tiers, with_tail))
        label = _node_label(node, verbose=verbose, detail=detail)
        style = "filled"   # suspect-only nodes were dashed here; solid reads better in print
        if node.id == best_ckpt:
            label += "\\n★ best"
            if node.id not in tiers and buckets.get(node.id) != "tail":
                # badge the run's output, but never at the cost of hiding that it diverged
                outline, width = pal.best_outline, max(width, 2)
        if node.id in flagged:
            severity, reason = flagged[node.id]
            label += f"\\n⚠ {reason}"
            outline = pal.flag_red if severity == "red" else pal.flag_amber
            width = 3
        extra = f', penwidth={width}, color="{outline}"' if outline else ""
        if detail == "all":     # a field dump reads as a left-flush block, not centred lines
            label = label.replace("\\n", "\\l") + "\\l"
        return f'  "{node.id}" [label="{label}", shape={shape}, style={style}, fillcolor="{fill}"{extra}];'

    # Backbone nodes in the main graph; constant context boxed off in a cluster.
    for node in graph.node_list:
        if node.id in kept and node.id not in context and node.id not in folded:
            lines.append(node_stmt(node))

    if fold_spec:
        rng = f"{fold_spec['lo']}" if fold_spec["lo"] == fold_spec["hi"] else f"{fold_spec['lo']}–{fold_spec['hi']}"
        fill, extra, note = pal.fold_fill, "", ""
        if pal.classify == "rca" and (_FOLD_ID in tiers or fold_tail):
            cls = _rca_class(_FOLD_ID, {_FOLD_ID: "tail"} if fold_tail else {}, tiers, with_tail)
            fill, colour, width = pal.style(cls)
            extra = f', color="{colour}", penwidth={width}'
            n = len(fold_tiers) or len(fold_tail)
            note = f"\\n{n} diverge"
        lines.append(
            f'  "{_FOLD_ID}" [label="Training loop\\n×{fold_spec["count"]} epochs ({rng}){note}",'
            f' shape=ellipse, style="filled", fillcolor="{fill}"{extra}, peripheries=2];')

    if context:
        # One plaintext node holding an HTML table: a self-contained horizontal strip whose width is
        # its own, not borrowed from the main graph's rank columns (the old invisible-chain trick rode
        # those columns, so mono's wide edge labels stretched the box). Layout-independent and compact.
        ctx_ids = list(context)
        cells = []
        for nid in ctx_ids:
            node = graph.nodes[nid]
            fill, _, _ = pal.style(_node_class(node, pal, buckets, tiers))
            parts = _node_label(node, verbose=verbose, detail=detail).split("\\n")
            inner = f"<B>{_html_escape(parts[0])}</B>" + "".join(
                f"<BR/>{_html_escape(p)}" for p in parts[1:])
            cells.append(f'<TD BGCOLOR="{fill}">{inner}</TD>')
        n = len(cells)
        title_row = (f'<TR><TD COLSPAN="{n}" BORDER="0"><FONT POINT-SIZE="18"><B>'
                     f'{_html_escape(title)}</B></FONT></TD></TR>') if title else ""
        table = (
            f'<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="4" CELLPADDING="3" COLOR="{pal.table_border}">'
            f'{title_row}'                               # figure headline, spanning the box
            # these nodes are collapsed precisely because every edge they have is run-context
            # fan-out; the caption says so instead of drawing 100+ identical edges.
            f'<TR><TD COLSPAN="{n}" BORDER="0"><FONT POINT-SIZE="11">'
            f'Attached to all activities</FONT></TD></TR>'
            f'<TR>{"".join(cells)}</TR>'
            f'{_legend_row(pal, n, legend)}'
            f'<TR><TD COLSPAN="{n}" BORDER="0" HEIGHT="8"></TD></TR></TABLE>>')  # spacer: gap to graph
        # graph-level label (not a node): Graphviz centers it at the top, out of the node layout
        lines += ['  labelloc="t";', '  labeljust="c";', f'  label={table};']
    elif title:
        # no context box to host it (e.g. --full or a narrow slice), so the title takes the label slot itself
        lines += ['  labelloc="t";', '  labeljust="c";',
                  f'  label=<<FONT POINT-SIZE="18"><B>{_html_escape(title)}</B></FONT>>;']

    shortcut = _shortcut_derivations(graph) if no_redundancy else set()
    seen_edges: set[tuple] = set()
    for e in graph.edge_list:
        if e.source not in kept or e.target not in kept:
            continue
        if not full and _is_hidden(e):
            continue
        if (e.source, e.target) in shortcut:
            continue                         # entity-to-entity shortcut a genBy+used path already draws
        src, tgt = remap(e.source), remap(e.target)
        if src == tgt:                       # an edge wholly inside the folded band
            continue
        color, style, label = (pal.edges or _EDGE_STYLE).get(e.RELATION, ("#000000", "solid", ""))
        derivation = getattr(e, "derivation_type", None)
        if mono or pal.edges:                # all-black; relation name only when verbose (else bare, compact)
            color, style = "#000000", "solid"
            label = str(e.RELATION) if verbose else ""
        elif e.RELATION == EdgeRelation.WAS_DERIVED_FROM and derivation:
            label = str(derivation)          # derivation_type kept even when terse — it's the signal
        elif not verbose:
            label = ""                       # drop the repeated genBy/uses/input noise at scale
        elif e.RELATION == EdgeRelation.USED and getattr(e, "role", None) == Role.INPUT:
            label = "input"
        if flagged.get(src, ("", ""))[0] == "red" and flagged.get(tgt, ("", ""))[0] == "red":
            color, style = pal.flag_red, "bold"  # edge inside a signature-matched region -> red
        if pal.link_divergent and src in tiers and tgt in tiers:
            # the divergence chain itself, so the eye follows candidate -> consequence
            color, style = pal.borders["tier1"][0], "bold"
        sig = (src, tgt, e.RELATION, label)
        if sig in seen_edges:
            continue
        seen_edges.add(sig)
        attrs = f'color="{color}", style={style}'
        if label:
            attrs += f', label="{label}", fontcolor="{color}"'
        lines.append(f'  "{src}" -> "{tgt}" [{attrs}];')

    lines.append("}")
    return "\n".join(lines)


def _default_title(jsonl: Path) -> str:
    """The run's name for the figure headline. Runs live at <run>/residue/residue.jsonl
    (or flat <run>/residue.jsonl), so the run dir name is the timestamp/label we want."""
    run = (jsonl.parent.parent.name if jsonl.parent.name in ("residue", "provenance")
           else jsonl.parent.name)
    return f"Run {run}"


def render(jsonl: str | Path, out: str | Path | None = None, *, fmt: str = "svg", **kw) -> Path:
    """Project `jsonl` and write a rendered graph next to it (or to `out`). Returns the path.
    Extra keyword args (full, fold, head, tail, slice_from, direction, verbose, title) pass to to_dot.
    title=None -> default 'Run <run-dir>'; title='' -> no title."""
    if shutil.which("dot") is None:
        raise RuntimeError("graphviz `dot` not found on PATH (install graphviz or the system pkg)")
    jsonl = Path(jsonl)
    out = Path(out) if out else jsonl.with_suffix(f".{fmt}")
    if kw.get("title") is None:                  # unset -> derive from path; '' stays '' (suppressed)
        kw["title"] = _default_title(jsonl)
    graph = load(jsonl)
    if kw.pop("flag", False):                    # run the signal rules, paint the hits red
        kw["flagged"] = flag_suspicious(graph)
    ref = kw.pop("vs", None)                     # diff against a clean run -> real buckets/tiers
    if ref:
        kw["buckets"], kw["tiers"], kw["added"] = diff_overlays(load(ref), graph)
    dot_src = to_dot(graph, **kw)
    subprocess.run(["dot", f"-T{fmt}", "-o", str(out)], input=dot_src, text=True, check=True)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Render a residue.jsonl ledger to a graph image.")
    p.add_argument("jsonl", help="path to residue.jsonl")
    p.add_argument("-o", "--out", help="output path (default: <jsonl>.svg)")
    p.add_argument("-f", "--format", default="svg", help="dot output format (svg, png, pdf)")
    p.add_argument("--full", action="store_true",
                   help="draw every edge incl. agent/intent fan-out (no context collapse)")
    p.add_argument("--fold", action="store_true",
                   help="collapse the middle epochs of the training loop into one summary node")
    p.add_argument("--head", type=int, default=1, help="epochs to keep expanded at the start (--fold)")
    p.add_argument("--tail", type=int, default=1, help="epochs to keep expanded at the end (--fold)")
    p.add_argument("--from", dest="slice_from", metavar="NODE_ID",
                   help="render only the lineage of this node (id or unique prefix)")
    p.add_argument("--direction", choices=("ancestors", "descendants", "both"), default="ancestors",
                   help="slice direction: ancestors=what produced it (default), descendants=what it reached")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show per-edge labels and node attribute detail (off by default)")
    p.add_argument("--detail", choices=("curated", "all"), default="curated",
                   help="node label contents: curated=type + identifying field (+ the -v shortlist), "
                        "all=every recorded field, declared and open-attribute alike")
    p.add_argument("--no-redundancy", action="store_true",
                   help="drop redundant entity-to-entity derivation edges")
    p.add_argument("--rankdir", choices=("TB", "LR"), default="TB",
                   help="layout direction: TB=upright portrait (default), LR=left-to-right (wide)")
    p.add_argument("--compact", action="store_true",
                   help="tighten rank/node separation and drop a font point; same nodes, smaller figure")
    p.add_argument("--palette", choices=tuple(PALETTES), default="kind",
                   help="what node colour encodes: kind=PROV kind (default), bucket=spine/"
                        "trajectory/context, rca=causal-rank tiers in red on black-and-white, "
                        "bw=print-safe greyscale")
    p.add_argument("--vs", metavar="REFERENCE.jsonl",
                   help="clean ledger to diff against; supplies real buckets and rank tiers "
                        "(without it, --palette bucket falls back to node type and rca has "
                        "nothing to paint)")
    p.add_argument("--with-tail", dest="with_tail", action="store_true",
                   help="outline trajectory divergences (weights/metrics) too, so the blast radius "
                        "shows; fill stays reserved for ranked candidates")
    p.add_argument("--mono", action="store_true",
                   help="draw all edges black; with -v, label each with its PROV relation (used/wasGeneratedBy/...)")
    p.add_argument("--title", default=None,
                   help='figure title (default: "Run <run-dir name>"; pass --title "" to omit)')
    p.add_argument("--flag", action="store_true",
                   help="run the signal rules and paint flagged nodes/edges red (no clean run needed)")
    p.add_argument("--rca", action="store_true",
                   help="print flag-path root-cause analysis (backward trace over flagged nodes) to stdout")
    args = p.parse_args(argv)
    if args.rca:
        print(format_rca(load(args.jsonl)))
    out = render(args.jsonl, args.out, fmt=args.format, full=args.full, fold=args.fold,
                 head=args.head, tail=args.tail, slice_from=args.slice_from,
                 direction=args.direction, verbose=args.verbose, no_redundancy=args.no_redundancy,
                 mono=args.mono, rankdir=args.rankdir, title=args.title, compact=args.compact,
                 flag=args.flag or args.rca, detail=args.detail,
                 palette=args.palette, vs=args.vs, with_tail=args.with_tail)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
