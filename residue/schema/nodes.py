"""PROV-DM has exactly three node kinds — Entity, Activity, Agent — and every
relationship between them is an *edge* (see edges.py)

Each concrete schema type is its own small dataclass declaring its first-class 
fields, and inherits an open `attributes` dict for fields we have not pinned 
down yet. 

Identity:
  Entity   id = content_hash     (content-addressed; see DESIGN 4.2)
  Activity id = activity_id      (run-scoped: run_id + counter)
  Agent    id = agent_id         (stable: username, gpu model, library version)

"""

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Optional

from residue.schema.types import NodeKind


# Base classes — the three PROV-DM node kinds

@dataclass(kw_only=True)
class Node:
    attributes: dict[str, Any] = field(default_factory=dict)

    KIND: ClassVar[str] = "node"

    @property
    def id(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the append-only event log. Inverse: from_dict()."""
        out: dict[str, Any] = {
            "node_kind": self.KIND,
            "type": type(self).__name__,
            "id": self.id,
        }
        for f in fields(self):
            if f.name == "attributes":
                continue
            out[f.name] = getattr(self, f.name)
        out["attributes"] = self.attributes
        return out


@dataclass(kw_only=True)
class Entity(Node):
    """A thing that exists and has a content hash. id == content_hash."""

    content_hash: str
    KIND: ClassVar[NodeKind] = NodeKind.ENTITY

    @property
    def id(self) -> str:
        return self.content_hash


@dataclass(kw_only=True)
class Activity(Node):
    """A thing that happens, with wall-clock start/end in nanoseconds."""

    activity_id: str
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None
    pid: Optional[int] = None            # process that ran it; pairs with an OS tracer's pid
    KIND: ClassVar[NodeKind] = NodeKind.ACTIVITY

    @property
    def id(self) -> str:
        return self.activity_id


@dataclass(kw_only=True)
class Agent(Node):
    """Who or what is responsible. Stable id, no content hash."""

    agent_id: str
    KIND: ClassVar[NodeKind] = NodeKind.AGENT

    @property
    def id(self) -> str:
        return self.agent_id


# Entities  (content_hash is the identity; see notes on each for what it hashes)

@dataclass(kw_only=True)
class RawDataset(Entity):
    """Original on-disk dataset. content_hash = manifest of per-file hashes."""

    num_files: Optional[int] = None
    external_anchor: Optional[str] = None  # e.g. published NIH checksum


@dataclass(kw_only=True)
class DatasetVersion(Entity):
    """Post-transformation snapshot. content_hash = manifest of per-file hashes."""

    num_files: Optional[int] = None
    resolution: Optional[int] = None


@dataclass(kw_only=True)
class DatasetMetadata(Entity):
    """Source metadata table (e.g. NIH Data_Entry_2017.csv) the splits are computed from.
    content_hash = hash of the file bytes."""

    path: Optional[str] = None
    num_rows: Optional[int] = None


@dataclass(kw_only=True)
class SplitManifest(Entity):
    """One train/val/test partition listing. content_hash = hash of the split csv bytes.
    Patient-level counts are the leakage-relevant unit (splits are patient-disjoint)."""

    split: Optional[str] = None          # train | val | test
    num_samples: Optional[int] = None
    num_patients: Optional[int] = None
    path: Optional[str] = None           # where it was observed on disk; NOT what content_hash covers
    # HOW the `split` label was earned, so a name-derived label is never mistaken for a derived fact:
    #   gradient_update  these samples TRAINED the model -- their batches were observed going through
    #                    a forward pass, a backward pass, and an optimizer step. The ONLY evidence
    #                    that says which partition is the training set: an eval pass runs the forward
    #                    half and stops, so nothing else in a pipeline looks like this.
    #   conserved        these partitions account exactly for a source table. Says a real split
    #                    happened; says NOTHING about which piece is which -- the name is still a stem.
    #   library          a split fn (train_test_split / random_split) named it
    #   filename         the csv's file stem, i.e. a convention someone chose -- NOT evidence
    split_evidence: Optional[str] = None


# How strong each claim is, so the strongest wins wherever two are made about the same content.
# One table, two readers: capture ranks claims WITHIN a process, and store/promote.py ranks them
# ACROSS processes when the projector merges them. `stage_name` is no longer produced (capture
# stopped inferring splits from the entry script's name) but stays here to rank old ledgers.
EVIDENCE_RANK = {"gradient_update": 3, "conserved": 2, "library": 2, "filename": 1, "stage_name": 1}


@dataclass(kw_only=True)
class SampleManifest(Entity):
    """A hash of every individual sample the model actually consumed this run.
    content_hash = hash of the sorted {index -> sample hash} map. This is finer-grained than
    SplitManifest, which only hashes the label csv: here a single poisoned image changes the hash
    even when its label and every file-level checksum look clean -- the runtime, content-level witness
    at the point of consumption. Samples are hashed pre-transform so random augmentation doesn't change
    the result run to run. Built by observing the reads training already performs (transform-tee on the
    pre-transform sample, per-worker sidecars merged at finalize) rather than a synthetic dataset pass,
    so it fingerprints the actual read path (dataloader/sampler/wrappers) and adds no side-effect I/O."""

    split: Optional[str] = None
    num_samples: Optional[int] = None


@dataclass(kw_only=True)
class TransformationConfig(Entity):
    """Params for one transformation invocation. content_hash = hash of these params."""

    size: Optional[int] = None
    interpolation: Optional[str] = None  # e.g., bilinear
    library: Optional[str] = None
    version: Optional[str] = None


@dataclass(kw_only=True)
class TransformSpec(Entity):
    """One op in a transform/augment pipeline (e.g. RandomAffine, ColorJitter). Full applied params
    live in `attributes`; content_hash = hash of {operation, order, params}. Each op is its own node,
    so a swapped/edited/reordered transform forks exactly this node — the under-studied transform
    attack surface (interpolation, sneaked-in flips, perturbed jitter) made first-class provenance data."""

    operation: Optional[str] = None      # the transform class name, e.g. "RandomAffine"
    order: Optional[int] = None          # position in the pipeline; sequence is semantically load-bearing


@dataclass(kw_only=True)
class TransformedDataset(Entity):
    """The data as the model actually consumes it after the full transform pipeline (any mix of
    augmentation + normalization; either may be absent). No materialized bytes — transforms run
    per-batch in memory — so identity = hash of {base dataset, ordered transform-op hashes, seed}. A
    tampered transform or a swapped base forks this node, and the whole train/eval tail forks with it."""

    base: Optional[str] = None           # the DatasetVersion id it derives from
    seed: Optional[int] = None
    num_transforms: Optional[int] = None


@dataclass(kw_only=True)
class HyperparameterSet(Entity):
    """Training hyperparameters. content_hash = hash of these params."""

    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    seed: Optional[int] = None  # training RNG seed (init + shuffle + augmentation)


@dataclass(kw_only=True)
class NormalizationStats(Entity):
    """Per-channel normalize mean/std applied at train+eval. content_hash = hash_norm_stats(...),
    a tolerant (rounded) hash so the config-declared stats and a fresh recomputation that agree
    dedup to one node — see decision: norm attestation."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)


@dataclass(kw_only=True)
class ModelArchitecture(Entity):
    """The model's structure, independent of trained values. content_hash = hash of the
    {param name -> shape} map, so it is DETERMINISTIC across runs even though weights are not.
    A swapped backbone, an inserted/edited layer, or a backdoor head forks this node -- and
    unlike ModelCheckpoint (nondeterministic -> diff 'tail'), it sits on the scored spine, so a
    model swap surfaces as a root cause instead of hiding in the whole-repo Code hash."""

    name: Optional[str] = None
    num_parameters: Optional[int] = None


@dataclass(kw_only=True)
class ModelCheckpoint(Entity):
    """Saved model state. content_hash = file hash."""

    epoch: Optional[int] = None
    path: Optional[str] = None
    framework_version: Optional[str] = None


@dataclass(kw_only=True)
class EvaluationResult(Entity):
    """Metrics output. content_hash = hash of the metrics dict."""

    split: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(kw_only=True)
class Code(Entity):
    """Script or library file. content_hash = file/tree hash."""

    path: Optional[str] = None
    git_commit: Optional[str] = None
    dirty: Optional[bool] = None


@dataclass(kw_only=True)
class Objective(Entity):
    """The training objective the gradient descends. content_hash = a depth-limited fingerprint of the
    loss tensor's autograd-op graph -- so it forks when the objective's structure changes (a single-path
    NllLoss vs an AddBackward summing a main + backdoor loss, the blind-backdoor tell). Read off the
    autograd graph, not the loss source, so it's invariant to WHERE the loss is computed (a separate
    module, the model's forward, or the train loop) -- the diffuse surface those attacks hide in."""

    terminal_op: Optional[str] = None       # loss tensor grad_fn class (NllLossBackward0, AddBackward0, ...)
    graph_fingerprint: Optional[str] = None  # hash of the loss autograd graph's op-type structure


@dataclass(kw_only=True)
class EnvDigest(Entity):
    """Container image hash + pip-freeze hash + CUDA version."""

    image_hash: Optional[str] = None
    pip_freeze_hash: Optional[str] = None
    cuda_version: Optional[str] = None
    python_version: Optional[str] = None


# Activities  (run-scoped id + wall-clock start_ns/end_ns)

@dataclass(kw_only=True)
class Register(Activity):
    """Trust-root ingress: admit a pre-existing artifact and bind its hash at time T."""

    anchor: Optional[str] = None


@dataclass(kw_only=True)
class PreprocessingOp(Activity):
    """A single preprocessing operation (resize, normalize)."""

    operation: Optional[str] = None


@dataclass(kw_only=True)
class TransformOp(Activity):
    """One op in an online train/eval transform pipeline (RandomAffine, ColorJitter, ToTensor,
    Normalize ...), the visible step the data flows through. `operation` is the transform class;
    applied params live in `attributes`; `seed` pins any stochastic op. The tamper signal lives on
    the content-addressed TransformedDataset the chain produces, not on this run-scoped activity."""

    operation: Optional[str] = None
    seed: Optional[int] = None


@dataclass(kw_only=True)
class DataSplit(Activity):
    """Train/validation/test partitioning. Reproducibility pinned by seed.

    `evidence` says how we know a split happened -- `conserved` when the partitions were checked to
    account for a source table, `library` when a split fn was observed. (`stage_name`, inferring a
    split from the entry script's name, was removed: it read a convention, not the data, and made
    capture depend on recognising the pipeline.) `coverage` and `partitions_disjoint` are the MEASUREMENT
    behind `conserved`, kept as values rather than collapsed to a boolean: a run whose coverage
    drops below 1.0, or whose partitions stop being disjoint, is itself a finding."""

    seed: Optional[int] = None
    evidence: Optional[str] = None
    coverage: Optional[float] = None            # |union(partitions)| / |source|
    partitions_disjoint: Optional[bool] = None


@dataclass(kw_only=True)
class TrainingStep(Activity):
    """One training epoch."""

    epoch: Optional[int] = None


@dataclass(kw_only=True)
class CheckpointWrite(Activity):
    """Serialization of a checkpoint."""


@dataclass(kw_only=True)
class Evaluation(Activity):
    """Evaluation pass over a test set."""

    split: Optional[str] = None


# Agents  (stable id, no content hash)

@dataclass(kw_only=True)
class User(Agent):
    """Researcher initiating the run."""

    name: Optional[str] = None


@dataclass(kw_only=True)
class Device(Agent):
    """Compute node and GPU model."""

    hostname: Optional[str] = None
    gpu_model: Optional[str] = None


@dataclass(kw_only=True)
class SoftwareAgent(Agent):
    """Framework version responsible for an activity (vs EnvDigest, the recorded entity)."""

    framework: Optional[str] = None
    version: Optional[str] = None


# Registry + deserialization  (log -> node, for rebuilding the graph)

NODE_TYPES: dict[str, type[Node]] = {
    cls.__name__: cls
    for cls in (
        # entities
        RawDataset, DatasetVersion, DatasetMetadata, SplitManifest, SampleManifest,
        TransformationConfig, TransformSpec, TransformedDataset, HyperparameterSet,
        NormalizationStats, ModelArchitecture, ModelCheckpoint, EvaluationResult, Code, Objective, EnvDigest,
        # activities
        Register, PreprocessingOp, TransformOp, DataSplit, TrainingStep,
        CheckpointWrite, Evaluation,
        # agents
        User, Device, SoftwareAgent,
    )
}

# back-compat: ledgers written before the rename serialize this entity as "NormalizedDataset".
NODE_TYPES["NormalizedDataset"] = TransformedDataset


def from_dict(data: dict[str, Any]) -> Node:
    """Rebuild a node from its serialized form (inverse of Node.to_dict)."""
    payload = dict(data)
    type_name = payload.pop("type")
    payload.pop("node_kind", None)
    payload.pop("id", None)  # id is a derived property, not a constructor field
    return NODE_TYPES[type_name](**payload)
