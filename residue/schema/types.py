"""Controlled vocabulary: the closed sets of string values that nodes.py and
edges.py reference. This module is the *leaf* of the schema — it imports nothing
from the schema, so nodes/edges/validate/capture/analysis can all import it
without cycles.

Scope note: only closed sets that analysis branches on live here. Descriptive
free-text (hostname, path, gpu_model) stays str. `operation` (resize/normalize)
and `interpolation` are deferred — real fields, but no consumer branches on them
yet; promote when one does.
"""

from enum import StrEnum


class NodeKind(StrEnum):
    """The three PROV-DM node kinds. Edges declare legal endpoint kinds with these;
    every concrete node carries its KIND. (The abstract base's "node" sentinel is
    never a real kind, so it is intentionally absent.)"""

    ENTITY = "entity"
    ACTIVITY = "activity"
    AGENT = "agent"


class EdgeRelation(StrEnum):

    USED = "used"
    WAS_GENERATED_BY = "wasGeneratedBy"
    WAS_DERIVED_FROM = "wasDerivedFrom"
    WAS_INFORMED_BY = "wasInformedBy"
    WAS_ASSOCIATED_WITH = "wasAssociatedWith"
    WAS_ATTRIBUTED_TO = "wasAttributedTo"


class DerivationType(StrEnum):

    SPLIT = "split"
    PREPROCESSING = "preprocessing"
    AUGMENTATION = "augmentation"
    TRAINING = "training"
    EVALUATION = "evaluation"


class Role(StrEnum):

    INPUT = "input"
    INTENT = "intent"
