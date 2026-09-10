"""A Recorder turns one pipeline step into a typed PROV fragment. Inside an `activity(...)`
scope it auto-emits, so the caller never hand-builds an edge:

    Activity node          (start_ns on enter, end_ns on exit)
    Used(act -> input)     role=INPUT   for each data input
    Used(act -> config)    role=INTENT  for each config/param entity
    WasGeneratedBy(out -> act)          for each produced entity
    WasDerivedFrom(out -> input)        lineage, tagged with derivation_type
    WasInformedBy(act -> prev)          temporal spine across the run
    WasAssociatedWith(act -> agent)     for each run agent

Events go to an EventSink (append-only)

Two deliberate choices made:
  - The activity node is emitted twice: on enter with end_ns=None, on exit with end_ns
    set. The graph projector merges by id (last wins), so a crash mid-step leaves an
    activity with no end_ns -> validate.py marks it `incomplete`, exactly as intended.
  - Attribution (entity -> agent) is NOT auto-emitted: it is inferable transitively
    (generatedBy . associatedWith), so emitting it would just bloat the log.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Protocol

from residue.schema.edges import (
    Used, WasAssociatedWith, WasDerivedFrom, WasGeneratedBy, WasInformedBy,
)
from residue.schema.nodes import Activity, Agent, Entity, Node
from residue.schema.types import DerivationType, Role


class EventSink(Protocol):
    """Minimal append-only sink. store/log.py will implement this; a list suffices now."""
    def append(self, event: dict) -> None: ...


class Recorder:
    def __init__(
        self,
        sink: EventSink,
        *,
        agents: Iterable[Agent] = (),
        run_id: Optional[str] = None,
    ):
        self._sink = sink
        self.run_id = run_id or uuid.uuid4().hex
        self._agents = tuple(agents)
        self._counter = 0
        self._prev_activity_id: Optional[str] = None  # WasInformedBy spine
        self._emitted: set[str] = set()  # node ids / edge keys already logged (dedup)

    # ---- emission primitives -------------------------------------------------

    def _emit_node(self, node: Node) -> None:
        self._sink.append(node.to_dict())  # nodes re-emit by design (id merge); see below

    def _node_once(self, node: Node) -> None:
        """Emit a node at most once, keyed by id."""
        if node.id not in self._emitted:
            self._emitted.add(node.id)
            self._sink.append(node.to_dict())

    def _emit_edge(self, edge) -> None:
        if edge.key not in self._emitted:
            self._emitted.add(edge.key)
            self._sink.append(edge.to_dict())

    def entity(self, ent: Entity) -> Entity:
        """Log an entity node once (content-addressed -> dedup by id). Returns it for chaining."""
        self._node_once(ent)
        return ent

    # ---- the chokepoint scope ------------------------------------------------

    @contextmanager
    def activity(
        self,
        activity: Activity,
        *,
        inputs: Iterable[Entity] = (),
        config: Iterable[Entity] = (),
        derivation: Optional[DerivationType] = None,
        informed_by: Optional[str] = None,
        chain: bool = True,
        observed_span: Optional[tuple[int, int]] = None,
    ) -> Iterator["_Scope"]:
        """Run a pipeline step as one PROV fragment. `inputs` are data (role=INPUT),
        `config` are params/intent (role=INTENT); produced entities are declared via the
        yielded scope's .generated(). `derivation` tags the lineage edges from each output
        back to each input. `informed_by` (or auto `chain`) wires the temporal spine.

        `observed_span` is (first_ns, last_ns) of the work this activity describes, for steps
        emitted at teardown rather than around the work: without it the window is the emit
        instant, so the activity claims a sub-millisecond point in the middle of a stage that
        ran for minutes. Wall-clock (time.time_ns) on both ends, which is what makes the
        window comparable to an external tracer's timestamps."""
        if not activity.activity_id:
            activity.activity_id = self._next_activity_id()
        activity.start_ns = observed_span[0] if observed_span else time.time_ns()
        # Single chokepoint every Activity passes through. Namespaced pid (an OS tracer's
        # `processId`, not `hostProcessId`) -- identical outside a pid namespace, diverges inside one.
        activity.pid = os.getpid()

        self._emit_node(activity)  # enter: end_ns is still None (supports `incomplete`)

        inputs = tuple(inputs)
        config = tuple(config)
        for ent in inputs:
            self.entity(ent)
            self._emit_edge(Used(source=activity.id, target=ent.id, role=Role.INPUT))
        for cfg in config:
            self.entity(cfg)
            self._emit_edge(Used(source=activity.id, target=cfg.id, role=Role.INTENT))

        target = informed_by or (self._prev_activity_id if chain else None)
        if target is not None:
            self._emit_edge(WasInformedBy(source=activity.id, target=target))

        for agent in self._agents:
            self._node_once(agent)
            self._emit_edge(WasAssociatedWith(source=activity.id, target=agent.id))

        scope = _Scope(activity, inputs, derivation)
        try:
            yield scope
        finally:
            if scope.end_override is not None:      # observed end, stamped from inside the block
                activity.end_ns = scope.end_override
            else:
                activity.end_ns = observed_span[1] if observed_span else time.time_ns()
            self._emit_node(activity)  # exit: same id, now with end_ns (projector: last wins)
            for out in scope._produced:
                self.entity(out)
                self._emit_edge(WasGeneratedBy(source=out.id, target=activity.id))
                for src in inputs:
                    self._emit_edge(WasDerivedFrom(
                        source=out.id, target=src.id, derivation_type=derivation))
            self._prev_activity_id = activity.id

    def _next_activity_id(self) -> str:
        self._counter += 1
        return f"{self.run_id}:{self._counter}"


class _Scope:
    """Handle yielded inside an activity() block; collects the entities it produces."""

    def __init__(self, activity: Activity, inputs: tuple[Entity, ...],
                 derivation: Optional[DerivationType]):
        self.activity = activity
        self._inputs = inputs
        self._derivation = derivation
        self._produced: list[Entity] = []
        self.end_override: Optional[int] = None

    def ends_at(self, ns: int) -> None:
        """Stamp the end at an observed instant instead of at scope exit. `observed_span` cannot
        serve here: it is passed at open time, and an activity held open across later work (an epoch
        closed by the NEXT epoch's first optimizer.step) only learns its true end afterwards. Without
        this the window runs to the exit instant and swallows whatever ran in between."""
        self.end_override = ns

    def generated(self, entity: Entity) -> Entity:
        """Declare an entity produced by this activity. Edges are wired on scope exit."""
        self._produced.append(entity)
        return entity
