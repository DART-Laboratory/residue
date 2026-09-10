"""PROV-DM edges: every relationship between two nodes is a directional
statement (source = subject, target = object), referenced by node id.

Endpoint kinds (source -> target) double as the validation spec:

  used               activity -> entity
  wasGeneratedBy     entity   -> activity
  wasDerivedFrom     entity   -> entity     (+ derivation_type)
  wasInformedBy      activity -> activity
  wasAssociatedWith  activity -> agent
  wasAttributedTo    entity   -> agent

Identity: edges are content-addressed by `key` = hash(relation, endpoints,
attributes), so the same statement asserted across runs dedups to one edge
"""

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Optional

from residue.schema.types import DerivationType, EdgeRelation, NodeKind, Role


@dataclass(kw_only=True)
class Edge:
    """Common spine: directional source/target ids + open attributes + serialization."""

    source: str   # id of the subject node
    target: str   # id of the target node
    attributes: dict[str, Any] = field(default_factory=dict)

    RELATION:     ClassVar[str]                  = "edge"
    SOURCE_KINDS: ClassVar[tuple[NodeKind, ...]] = ()   # node KINDs allowed at source
    TARGET_KINDS: ClassVar[tuple[NodeKind, ...]] = ()   # node KINDs allowed at target

    @property
    def key(self) -> str:
        """Content-addressed edge id; identical statements dedup across runs."""
        parts = {f.name: getattr(self, f.name) for f in fields(self)}
        parts["relation"] = self.RELATION
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the append-only event log. Inverse: from_dict()."""
        out: dict[str, Any] = {
            "edge_kind": "edge",
            "relation": self.RELATION,
            "type": type(self).__name__,
            "key": self.key,
        }
        for f in fields(self):
            if f.name == "attributes":
                continue
            out[f.name] = getattr(self, f.name)
        out["attributes"] = self.attributes
        return out


@dataclass(kw_only=True)
class Used(Edge):
    """Activity consumed an entity. role separates input data from intent/config."""

    role: Optional[Role] = None   # input vs intent (config) — see DESIGN 4.5
    RELATION = EdgeRelation.USED
    SOURCE_KINDS = (NodeKind.ACTIVITY,)
    TARGET_KINDS = (NodeKind.ENTITY,)


@dataclass(kw_only=True)
class WasGeneratedBy(Edge):
    """Entity was produced by an activity."""

    RELATION = EdgeRelation.WAS_GENERATED_BY
    SOURCE_KINDS = (NodeKind.ENTITY,)
    TARGET_KINDS = (NodeKind.ACTIVITY,)


@dataclass(kw_only=True)
class WasDerivedFrom(Edge):
    """Entity derives from another entity; derivation_type says how (lineage edge)."""

    derivation_type: Optional[DerivationType] = None
    RELATION = EdgeRelation.WAS_DERIVED_FROM
    SOURCE_KINDS = (NodeKind.ENTITY,)
    TARGET_KINDS = (NodeKind.ENTITY,)


@dataclass(kw_only=True)
class WasInformedBy(Edge):
    """Activity was sequenced/triggered by another activity."""

    RELATION = EdgeRelation.WAS_INFORMED_BY
    SOURCE_KINDS = (NodeKind.ACTIVITY,)
    TARGET_KINDS = (NodeKind.ACTIVITY,)


@dataclass(kw_only=True)
class WasAssociatedWith(Edge):
    """Activity was carried out by / on behalf of an agent."""

    RELATION = EdgeRelation.WAS_ASSOCIATED_WITH
    SOURCE_KINDS = (NodeKind.ACTIVITY,)
    TARGET_KINDS = (NodeKind.AGENT,)


@dataclass(kw_only=True)
class WasAttributedTo(Edge):
    """Entity is ascribed to an agent."""

    RELATION = EdgeRelation.WAS_ATTRIBUTED_TO
    SOURCE_KINDS = (NodeKind.ENTITY,)
    TARGET_KINDS = (NodeKind.AGENT,)


# Registry + deserialization (log -> edge, for rebuilding the graph)

EDGE_TYPES: dict[str, type[Edge]] = {
    cls.__name__: cls
    for cls in (
        Used, WasGeneratedBy, WasDerivedFrom,
        WasInformedBy, WasAssociatedWith, WasAttributedTo,
    )
}


def from_dict(data: dict[str, Any]) -> Edge:
    """Rebuild an edge from its serialized form (inverse of Edge.to_dict)."""
    payload = dict(data)
    type_name = payload.pop("type")
    payload.pop("edge_kind", None)
    payload.pop("relation", None)
    payload.pop("key", None)  # derived, not a constructor field
    return EDGE_TYPES[type_name](**payload)
