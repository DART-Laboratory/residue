"""Structural validation: is this a well-formed PROV graph?

Looks *inward only* — graph vs. its own schema. It never compares a hash against
the artifact on disk, nor one run against another; that is attack detection and
lives in analysis/ A passing graph means "structurally coherent," never "this run 
is clean": a poisoned-but-well-formed graph passes here by design.

Operates on typed Node/Edge objects, so it runs on any partial run "as captured,"
independent of capture/store code — the objects may come from a live run,
from_dict() off a log, or a hand-built fixture; validate can't tell.

Two tiers (integrity vs completeness):
  violations  — corruption / instrumentation bugs; caller decides if fatal.
  incomplete  — expected gaps in a truncated run (e.g. an activity with a start
                but no end). Informational; a run killed mid-epoch still passes.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from residue.schema.edges import Edge
from residue.schema.nodes import Activity, Entity, Node


class Code(StrEnum):
    DANGLING_SOURCE = "dangling_source"      # edge source id not among nodes
    DANGLING_TARGET = "dangling_target"
    BAD_SOURCE_KIND = "bad_source_kind"      # endpoint exists but wrong node KIND
    BAD_TARGET_KIND = "bad_target_kind"
    ENTITY_NO_HASH = "entity_no_hash"        # entity identity missing (DESIGN 4.2)
    ACTIVITY_NO_START = "activity_no_start"  # an activity that never began = corruption
    REVERSED_CLOCK = "reversed_clock"        # end_ns < start_ns
    ACTIVITY_NO_END = "activity_no_end"      # incomplete tier: still running / truncated


@dataclass(kw_only=True)
class Issue:
    code: Code
    ref: str      # node id or edge key the issue is about
    detail: str


@dataclass(kw_only=True)
class Report:
    violations: list[Issue] = field(default_factory=list)
    incomplete: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no integrity violations. Incomplete markers do not fail a graph."""
        return not self.violations


def validate(nodes: Iterable[Node], edges: Iterable[Edge]) -> Report:
    """Check structural integrity of a node/edge collection. Never raises."""
    report = Report()
    by_id: dict[str, Node] = {}
    for node in nodes:
        by_id[node.id] = node  # content-addressed entities dedup naturally; last wins
        _check_node(node, report)
    for edge in edges:
        _check_edge(edge, by_id, report)
    return report


def _check_node(node: Node, report: Report) -> None:
    if isinstance(node, Entity) and not node.content_hash:
        report.violations.append(Issue(
            code=Code.ENTITY_NO_HASH, ref=node.id, detail="entity has no content hash"))

    if isinstance(node, Activity):
        if node.start_ns is None:
            report.violations.append(Issue(
                code=Code.ACTIVITY_NO_START, ref=node.id, detail="activity has no start_ns"))
        if node.end_ns is None:
            report.incomplete.append(Issue(
                code=Code.ACTIVITY_NO_END, ref=node.id,
                detail="activity has no end_ns (still running / truncated)"))
        elif node.start_ns is not None and node.end_ns < node.start_ns:
            report.violations.append(Issue(
                code=Code.REVERSED_CLOCK, ref=node.id,
                detail=f"end_ns {node.end_ns} < start_ns {node.start_ns}"))


def _check_edge(edge: Edge, by_id: dict[str, Node], report: Report) -> None:
    kind = type(edge).__name__
    src = by_id.get(edge.source)
    if src is None:
        report.violations.append(Issue(
            code=Code.DANGLING_SOURCE, ref=edge.key,
            detail=f"{kind} source {edge.source!r} not among nodes"))
    elif src.KIND not in type(edge).SOURCE_KINDS:
        report.violations.append(Issue(
            code=Code.BAD_SOURCE_KIND, ref=edge.key,
            detail=f"{kind} source is {src.KIND}, expected {', '.join(type(edge).SOURCE_KINDS)}"))

    tgt = by_id.get(edge.target)
    if tgt is None:
        report.violations.append(Issue(
            code=Code.DANGLING_TARGET, ref=edge.key,
            detail=f"{kind} target {edge.target!r} not among nodes"))
    elif tgt.KIND not in type(edge).TARGET_KINDS:
        report.violations.append(Issue(
            code=Code.BAD_TARGET_KIND, ref=edge.key,
            detail=f"{kind} target is {tgt.KIND}, expected {', '.join(type(edge).TARGET_KINDS)}"))
