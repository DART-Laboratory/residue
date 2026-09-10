"""Transparent capture by patching torch: `init()` installs hooks that turn torch
events into Recorder activity fragments, so a script gets provenance with no
hand-placed calls.

Slice 1 — Optimizer.step: HyperparameterSet, ModelArchitecture, TrainingStep.
Slice 2 — DataLoader.__iter__: epoch boundaries -> one TrainingStep per training epoch
          (chained on the time spine) + batch_size on the HyperparameterSet.
Slice 3 — Dataset identity: the train SplitManifest the model reads, as a Used input.
Slice 4 — Transforms: the dataset's Compose -> a TransformedDataset (the data actually
          trained on) derived from the split; changing any op forks it.
Slice 5 — torch.save: each checkpoint write -> a ModelCheckpoint (file hash) made by a
          CheckpointWrite, tied to the epoch that produced it. The saved object is matched to the
          model being trained by tensor-shape fingerprint, so a raw state_dict, a wrapped dict (any
          key), or a pickled nn.Module all resolve, and non-weight tensor dicts are rejected.
Slice 6 — split entry points (sklearn train_test_split, torch random_split): each call ->
          a DataSplit (seed) making one SplitManifest per partition. DataFrame partitions
          hash via df.to_csv (same as Slice 3) so the train partition matches across stages;
          works for any dataset, not tied to one pipeline.
Slice 7 — metric calls (sklearn.metrics scalar fns, torchmetrics.Metric.compute): each call's
          return value -> batched into one EvaluationResult per eval episode, made by an Evaluation
          tied to the producing epoch. The patch rebinds already-imported names (a sys.modules walk)
          so it fires regardless of import order; the scored split is the just-iterated loader's
          dataset, named for free when its hash matches a manifest an earlier slice labeled.
Slice 8 — file loads (pandas.read_csv): each load maps the loaded frame's content-hash -> the
          source file's stem (val.csv -> 'val'), so a split/eval manifest of that data inherits the
          name even when the split was made upstream and this process only reads it. Content-hash is
          the cross-stage join key; persisting the map for cross-RUN inheritance is the next step.
Slice 9 — file writes (DataFrame.to_csv) in the make_splits stage: each written partition -> a named
          SplitManifest, batched into one DataSplit derived from the source DatasetMetadata (the table
          read just before the writes). Hooks the WRITE, not the split fn, so it captures a custom/
          hand-rolled splitter that Slice 6's library hooks never see. The split.csv it writes hashes
          (df.to_csv scheme) identical to the train run's read of it -> cross-stage dedup, real edge.
Slice 10 — per-sample content: at the first epoch, hash every train sample once (transform turned off,
          so augmentation doesn't change the hash) -> one SampleManifest. A direct pass over the dataset,
          not a watch on the loader, so it's complete and order-independent. Catches per-image content
          poisoning (BadNets pixels, clean-label tampering) that the label-csv SplitManifest misses.
Slice 11 — batch order: tee BatchSampler.__iter__ to record the realized order of sample indices each
          epoch, plus the sampler/shuffle/seed. Three hashes on the TrainingStep tell apart the BRRR
          data-ordering attacks -- batch reorder, datapoint reshuffle, datapoint replacement -- with no
          clean run to diff against; the sampler+seed let a later check ask if the order matches the seed.
Slice 12 — objective/loss: tee Tensor.backward + autograd.grad to capture the training objective the
          gradient descends -- the loss tensor's terminal grad_fn (a stock NllLossBackward0 vs an
          AddBackward0 that sums several losses) fingerprinted over its autograd-op graph -> one
          Objective, an input to every epoch. Backward passes per optimizer step land on the TrainingStep;
          >1 is the multi-objective / MGDA tell. Catches loss code-poisoning (blind backdoors) with no
          clean run to diff: the composite loss and the extra backward passes are visible in one run.
Slice 13 — loader->forward integrity: at the first epoch, hash the content of each batch the real
          DataLoader yields, then a global forward-pre-hook hashes the batch entering model() and checks
          it was one the loader produced. A forward input the loader never emitted means the batch was
          rewritten in-loop between loader and model -> forward_inputs_unmatched on the TrainingStep.
          Catches in-loop batch poisoning (CCBA) that leaves disk, labels, objective and backward count
          all clean; also (honestly) trips on benign GPU-side batch edits (mixup/cutmix), so it's a
          lone amber tell, not a signature.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib
import os
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Optional

from residue.capture.hashing import hash_bytes, hash_canonical, hash_file
from residue.capture.session import Session, residue_session
from residue.capture.transforms import record_transform_chain
from residue.schema.nodes import (
    EVIDENCE_RANK,
    CheckpointWrite, DatasetMetadata, DatasetVersion, DataSplit, Evaluation, EvaluationResult,
    HyperparameterSet, ModelArchitecture, ModelCheckpoint, Objective, PreprocessingOp, RawDataset,
    SampleManifest, SplitManifest, TrainingStep,
)
from residue.schema.types import DerivationType
from residue.schema.edges import WasDerivedFrom

_WRAPPED = "_torchprov_wrapped"  # marker on a patched fn so installing twice is a no-op

# for call-site naming: skip frames in our own package and in the stdlib when finding user code
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the residue/ package
_STDLIB_DIR = sysconfig.get_paths().get("stdlib", "")

# module-global run state (one training process, one ledger)
_session: Optional[Session] = None
_session_cm = None              # the residue_session contextmanager, closed at exit
_stage = None                   # this process's stage name ("train", "make_splits", ...)
_installed = False
_originals: dict[tuple, Any] = {}  # (cls, attr) -> unpatched fn, for restore()

# run-derived entities (computed once, on the first optimizer step)
_hp = None
_arch = None
_intent: list = []
_param_shapes: dict = {}  # {tuple(param.shape): count} over optimized tensors, for checkpoint identity (Slice 5)

# per-epoch training state
_epoch = 0
_epoch_steps = 0
_pending_pass = False           # a DataLoader pass began; next step claims it as an epoch
_pending_batch_size = None      # batch_size of the loader that began the pending pass
_pending_dataset = None         # loader.dataset of the pending pass (Slice 3)
_pending_transform = None       # loader.dataset.transform (Compose) of the pending pass (Slice 4)
# image ingest/preprocessing state (Slice 15), keyed by basename so raw and processed sides align
_img_reads: dict = {}           # basename -> file hash of each raw image the stage read
_img_writes: dict = {}          # basename -> file hash of each image the stage wrote
_img_ops: dict = {}             # op name -> its params (resize target, interpolation, ...)
_img_resolution = None          # target edge length of the last resize seen
_img_call_site = None           # user-code frames of the first image write
_img_span = None                # (first_ns, last_ns) of observed image io/ops -> PreprocessingOp window
_img_read_paths: dict = {}      # basename -> abspath of each image READ, in observed order
_img_write_paths: dict = {}     # basename -> abspath of each image WRITTEN, in observed order

_split_node = None              # SplitManifest of the train data, computed once (Slice 3)
_train_input = None             # what the model trains on: TransformedDataset, else _split_node (Slice 4)
_lineage_done = False           # data lineage (split + transforms) recorded once; latches even when _split_node stays None (coarse base, no .df)
_train_cm = None                # the open recorder.activity() for the current epoch
_train_scope = None             # its scope, so _close_epoch can stamp the observed end
_last_step_ns = None            # wall clock of the most recent optimizer.step() (the epoch's real end)
_train_activity: Optional[TrainingStep] = None

# Slice 7 — evaluation episode state (a run of metric calls between two data passes)
_last_pass_dataset = None       # dataset of the most recent DataLoader pass (the scored split at eval)
_manifest_by_hash: dict = {}    # content_hash -> split label, so an eval split inherits a known name
_manifest_path_by_hash: dict = {}  # content_hash -> the path it was read from / written to (see _register_manifest)
_last_pass_start_ns = None      # when the most recent DataLoader pass began (start of an eval episode)
_eval_span = None               # (first_ns, last_ns) of the open episode -> Evaluation window
_eval_metrics: dict = {}        # {metric_name: value} accumulated for the open episode
_eval_manifest = None           # SplitManifest the open episode scored
_eval_epoch = None              # epoch index the open episode belongs to
_eval_informed_by = None        # TrainingStep id the open episode hangs off
_eval_call_site: list = []      # user-code frames of the first metric call in the episode
_eval_losses: dict = {}         # {loss_name: [sum, count]} per-batch losses, averaged into the episode

# Slice 10 — per-sample content (Option A: fingerprint the reads training actually does, not a
# shadow pass). The manifest is emitted at finalize from the observed reads, merged across worker
# processes; a run whose train loader we can't hook before it forks falls back to the shadow pass.
_sample_manifest = None          # emitted at finalize from observed reads (or the shadow fallback)
_capture_dataset = None          # the train Dataset instance we fingerprint (set before its loader forks)
_capture_transform_orig = None   # its original transform, so restore() can put it back
_reading_idx = None              # index __getitem__ is currently building (per process; set by the wrap)
_reading_file_hash = None        # file-byte hash of the image opened for the current idx (set at Image.open)
_sample_reads: dict = {}         # {idx: content_hash} observed in THIS process (main aggregates these)
_read_files: set = set()         # file-byte hashes of images read this run -> DatasetVersion membership join
_seen_idx: set = set()           # indices already hashed in THIS process -- hash once, skip re-reads
_sample_active = False           # recording on? True through epoch 1; off at the 2nd train pass (fork gate)
_sample_armed = False            # observe-capture wraps installed this run (else epoch 1 shadow-falls-back)
_sidecar_dir = None              # per-worker sidecar dir (set before fork, inherited by forked workers)
_sidecar_fh = None               # this worker's append handle (opened lazily, line-buffered)
_train_pass_seen = 0             # shuffled (train-convention) passes seen -> 1 arms capture, 2 disarms it

# Slice 11 — batch order + seed (BRRR: reorder / reshuffle / replace)
_current_order = None            # the batch-index list the live BatchSampler.__iter__ is teeing into
_epoch_order_ref = None          # the order buffer belonging to the open train epoch (hashed at close)
_pending_sampler = None          # sampler / shuffle / seed of the pending data pass

# Slice 12 — objective / loss (loss code-poisoning: blind backdoors, CCBA)
_objective = None                # Objective entity in force, re-checked every backward, a per-epoch input
_objective_seen: dict = {}       # content_hash -> Objective, so a loss that toggles back reuses its node
_backward_events = 0             # backward()/autograd.grad() calls since the last optimizer.step
_step_backward_max = 0           # most backward passes seen in any step of the open epoch (>1 = MGDA)
_step_backward_call_sites: list = []  # call sites of this step's backwards, folded into the epoch each step

# Slice 14 — per-sample loss scoring pass (whitebox data-ordering: BRRR loss-ranking; also coreset /
# curriculum / OHEM selection). A non-reduced (reduction='none') loss is a per-sample loss extraction;
# a RUN of them with no optimizer.step() between is a scoring pass the training loop never needs -- the
# model is being probed per-sample to rank/select data. We count the longest such run per epoch boundary.
_persample_run = 0               # consecutive per-sample (non-scalar) loss calls since the last step
_persample_samples = 0           # samples covered by that consecutive run
_scan_samples_max = 0            # largest such run (samples) seen this run -- a no-backward whitebox scan (batch_reordering)

# Slice 13 — loader->forward integrity (in-loop batch poisoning: CCBA)
_fwd_active = False               # armed through epoch 1 (same fork gate as sample capture)
_fwd_hook_handle = None          # RemovableHandle for the global forward-pre-hook (removed at disarm)
_loader_content: set = set()     # content hashes of the tensors in the batch the loader just yielded
_fwd_expect = False              # a loader batch was drawn; the next (outermost) forward should match it
_fwd_unmatched = 0               # forwards this run whose input the loader never emitted (folded per step)
_offloader_forwards = 0          # extra depth-0 forwards this pass on inputs the loader never produced (injected probes)
_loss_targets_offloader = 0      # loss evals this pass whose TARGET the loader never produced (attacker-relabelled copy)
_loss_call_site: list = []          # user-code frames where the objective's loss was COMPUTED (injected loss code resolves off the train loop, unlike call_site which is the backward site)
_loader_origin: list = []        # user-code frames that drove the train loader's iteration; an in-loop poisoner (CCBA) wrapping the loader resolves off the train loop
_loader_batch = None             # the clean batch (tensors) teed this step, kept for a per-row/region diff (ccba)
_alter_row_frac_max = 0.0        # max fraction of batch rows the model input altered vs the loader (in-flight, epoch 1)
_alter_region_frac_max = 0.0     # max fraction of an altered row's elements that differ (spatial confinement of the edit)
_alter_share_class = False       # any step where the altered rows shared a non-universal active class (label-selected)

# Slice 14 — forwards per optimizer step (extra in-loop forwards: blind_backdoor, memory_backdoor).
# Counts OUTERMOST train-mode forwards only: the global pre-hook fires for every submodule too, so
# depth gates it to the top-level model() call. Independent of Slice 13's epoch-1 arming -- this runs
# every epoch, like the backward counter it mirrors.
_fwd_count_handles: list = []    # RemovableHandles for the (pre, post) counting hook pair
_fwd_depth = 0                   # module nesting depth; 0 = the outermost forward
_forward_events = 0              # outermost forwards since the last optimizer step
_step_forward_max = 0            # largest per-step count seen this epoch
_model_ref = None                # the top-level model, grabbed at the first depth-0 forward, for _module_graph

# Slice 9 — write-boundary split state (make_splits stage)
_last_read = None               # (content_hash, num_rows, path) of the last read_csv this stage = split source
_split_writes: list = []        # _Frame records for the csv writes seen so far this stage
_stage_frames: list = []        # _Frame for every csv frame SEEN this stage (read or written) = source candidates
_split_source = None            # DatasetMetadata the current batch derives from (pinned at the first write)
_split_call_site: list = []     # user-code frames of the first write in the batch (names the DataSplit)
_split_span = None              # (first_ns, last_ns) of the observed source read + partition writes


def init(stage: Optional[str] = None, *, run_dir=None, repo_root: str = "."):
    """Begin transparent capture for this process. Opens the run ledger (emitting Code +
    Env), installs the torch hooks, and registers an atexit cleanup that closes the open
    epoch and the log. `stage` defaults to the entry script's name (make_splits.py ->
    'make_splits'), so the common case is one arg-free line near the top of a script --
    the mlflow.autolog()/wandb.init() convention. A second call to this is a no-op."""
    global _session, _session_cm, _installed, _stage
    if _installed:
        return _session

    import torch  # imported here so the package never hard-requires torch

    _stage = stage if stage is not None else _detect_stage()
    _session_cm = residue_session(_stage, run_dir=run_dir, repo_root=repo_root)
    _session = _session_cm.__enter__()
    _install_optimizer_hook(torch)
    _install_dataloader_hook(torch)
    _install_save_hook(torch)
    _install_split_hooks(torch)
    _install_sampler_hook(torch)
    _install_backward_hook(torch)
    _install_forward_counter(torch)
    _install_metric_hooks()
    _install_read_hooks()
    _install_write_hooks()
    _install_image_hooks()
    atexit.register(_finalize)
    _installed = True
    return _session


def current_log_dir() -> Optional[Path]:
    """The directory of the ledger this process is writing (where sidecars belong), or None if no
    capture session is live. The single source of truth for the run's output dir -- resolved by the
    session's precedence (RESIDUE_LOG > run_dir > default stamp), so callers never re-derive it."""
    return _session.log_path.parent if _session is not None else None


def _detect_stage() -> str:
    """Stage name from the entry script: scripts/make_splits.py -> 'make_splits' (the basename stem
    matches the stage names the hooks expect, e.g. Slice 9 gates on 'make_splits'). Falls back to
    'main' for a REPL / `python -c` with no script."""
    argv0 = sys.argv[0] if sys.argv else ""
    return os.path.splitext(os.path.basename(argv0))[0] or "main"


# ---- optimizer interposition (Slice 1) ---------------------------------------

def _install_optimizer_hook(torch) -> None:
    """Patch `step` on every Optimizer subclass that defines its own (SGD, Adam, ... )"""
    base = torch.optim.Optimizer
    for cls in {base, *_all_subclasses(base)}:
        fn = cls.__dict__.get("step")
        if fn is None or getattr(fn, _WRAPPED, False):
            continue
        _originals[(cls, "step")] = fn
        setattr(cls, "step", _wrap_step(fn))


def _wrap_step(orig):
    @functools.wraps(orig)
    def step(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)  # do the real update first
        try:
            _on_step(self)
        except Exception:
            pass  # capture must never break the training it observes
        return result

    setattr(step, _WRAPPED, True)
    return step


def _on_step(optimizer) -> None:
    """Per-batch callback. The first step reads the run's hyperparameters + architecture. A
    step right after a new data pass (_pending_pass) opens a fresh per-epoch TrainingStep,
    linked to the previous epoch on the time spine; otherwise it advances the current one."""
    global _hp, _arch, _intent, _split_node, _train_input, _train_cm, _train_activity
    global _train_scope, _last_step_ns
    global _epoch, _epoch_steps, _pending_pass, _lineage_done, _epoch_order_ref
    global _backward_events, _step_backward_max, _persample_run, _persample_samples, _scan_samples_max

    if _hp is None:  # first step of the run: batch_size is known by now (a pass preceded it)
        _hp = _record_hyperparams(optimizer)
        _arch = _record_arch(optimizer)
        _intent = [n for n in (_hp, _arch, _session.code, _session.env) if n is not None]

    if not _lineage_done and _pending_dataset is not None:  # Slice 3/4: data lineage, recorded once
        _split_node = _record_split(_pending_dataset)
        # Slice 4: record the transform chain off whatever base we can name. A .df gives the strong
        # content-hashed SplitManifest; without one (e.g. an image dataset) fall back to a coarse
        # dataset identity so the Compose is still captured -- the chain is dataset-agnostic.
        base = _split_node
        if base is None and _pending_transform is not None:
            base = _record_dataset_identity(_pending_dataset)
        if base is not None:
            _train_input = _record_transforms(base)
        _lineage_done = True  # latch: TransformOp is an activity, so re-running would flood the ledger

    if _pending_pass or _train_cm is None:  # a new training epoch begins
        _close_epoch()
        _epoch += 1
        _epoch_steps = 0
        _train_activity = TrainingStep(activity_id="", epoch=_epoch)
        _train_activity.attributes["call_site"] = _call_site()  # the training-loop line that drove this epoch
        if _epoch == 1 and not _sample_armed:  # Slice 10: no observe-capture armed -> shadow-pass fallback
            _capture_samples(_pending_dataset)
        _epoch_order_ref = _current_order  # Slice 11: the order buffer this epoch is filling
        _step_backward_max = 0             # Slice 12: reset the per-epoch backward tallies
        _train_activity.attributes["backward_call_site"] = {}
        if _pending_sampler is not None:
            _train_activity.attributes["sampler"] = _pending_sampler
        data = _train_input if _train_input is not None else _split_node
        inputs = [n for n in (data, _sample_manifest, _objective) if n is not None]  # Slice 12: + Objective
        _train_cm = _session.recorder.activity(
            _train_activity, inputs=inputs, config=_intent, derivation=DerivationType.TRAINING)
        _train_scope = _train_cm.__enter__()
        _pending_pass = False

    # AFTER the block above on purpose: _close_epoch() inside it must still see the PREVIOUS epoch's
    # last step. Stamping earlier would hand the old epoch this step's time -- the very overshoot
    # this exists to remove.
    _last_step_ns = time.time_ns()
    _epoch_steps += 1
    _train_activity.attributes["num_steps"] = _epoch_steps
    _fold_backward_step()  # Slice 12: this step's backward count + sites into the epoch
    _fold_forward_step()   # Slice 14: this step's outermost-forward count into the epoch
    # Slice 14: a per-sample-loss scan with no optimizer step between = a no-backward whitebox pass over
    # the data (batch_reordering ranks the dataset by the victim's own loss before epoch 2). Fold the
    # largest scan, then reset the per-step run. Written every step (0 on normal steps); the burst runs at
    # an epoch boundary OUTSIDE the epoch-1 armed window, so this is deliberately NOT gated on _fwd_active.
    if _persample_samples > _scan_samples_max:
        _scan_samples_max = _persample_samples
    _train_activity.attributes["persample_scan_samples"] = _scan_samples_max
    _persample_run = _persample_samples = 0
    if _fwd_active:  # Slice 13: armed (epoch-1) steps record the count even at 0 -- "check ran, found none"
        _train_activity.attributes["forward_inputs_unmatched"] = _fwd_unmatched  # no key at all = never ran
        _train_activity.attributes["forwards_not_from_loader"] = _offloader_forwards  # injected-probe forwards
        _train_activity.attributes["loss_targets_not_from_loader"] = _loss_targets_offloader  # attacker-relabelled loss
        _train_activity.attributes["altered_row_fraction"] = _alter_row_frac_max        # in-flight edit: row subset
        _train_activity.attributes["altered_region_fraction"] = _alter_region_frac_max  # in-flight edit: spatial extent
        _train_activity.attributes["altered_rows_share_class"] = _alter_share_class     # in-flight edit: label-selected
    if _loader_origin:  # code that drove this epoch's train loader (a loader-wrapping poisoner resolves off the loop)
        _train_activity.attributes["dataloader_origin"] = _loader_origin
    try:
        _train_activity.attributes["last_lr"] = _as_float(optimizer.param_groups[0]["lr"])
    except (IndexError, KeyError, TypeError):
        pass


# ---- dataloader interposition (Slice 2) --------------------------------------

def _install_dataloader_hook(torch) -> None:
    """Patch DataLoader.__iter__ to mark data-pass boundaries. A training epoch is a pass
    whose iteration is followed by optimizer.step() calls; val/test passes set the same
    pending flag but no step uses it (the next train pass resets it first), so they're
    safely ignored here."""
    cls = torch.utils.data.DataLoader
    fn = cls.__dict__.get("__iter__")
    if fn is not None and not getattr(fn, _WRAPPED, False):
        _originals[(cls, "__iter__")] = fn
        setattr(cls, "__iter__", _wrap_iter(fn))


def _wrap_iter(orig):
    @functools.wraps(orig)
    def __iter__(self):
        global _loader_origin
        try:
            _on_iter(self)
            # Who drove this iteration. On a train pass it's normally the trainer; a wrapper that
            # interposes on the loader (e.g. an in-loop batch poisoner) iterates the real loader from
            # inside its own __iter__, so its frame is on the stack here. Generic origin, no attack ref.
            if (_pending_sampler or {}).get("train_order"):
                _loader_origin = _call_site()
        except Exception:
            pass
        it = orig(self)
        # Slice 13: tee the real train loader's batches so the forward hook can check what model() gets
        # against what the loader produced. Only the armed (epoch-1) train pass; an in-loop poisoner
        # (CCBA) wraps THIS loader, so what we tee here is still the clean pre-poison batch.
        if _fwd_active and (_pending_sampler or {}).get("train_order"):
            return _tee_batches(it)
        return it

    setattr(__iter__, _WRAPPED, True)
    return __iter__


def _tee_batches(it):
    """Yield the loader's batches unchanged, recording each one's tensor content (current batch only)
    and arming the next forward to check against it."""
    global _loader_content, _fwd_expect, _loader_batch
    for batch in it:
        try:
            _loader_content = _batch_content(batch)
            _loader_batch = batch     # kept by reference for the per-row/region diff; the batch is alive anyway
            _fwd_expect = True
        except Exception:
            _loader_content = set()
            _loader_batch = None
        yield batch


def _effective_batch_size(loader) -> Optional[int]:
    """The batch size the pass actually uses. torch nulls DataLoader.batch_size whenever a
    batch_sampler is supplied (the two are mutually exclusive in its API) even though batches are
    still that size — the value just moves onto the batch_sampler. Reading the raw attribute would
    record None for any run that swaps the sampler, misstating the run's batch size as a
    hyperparameter change. Sampler substitution is evidence, but it belongs to the sampler, which
    Slice 11 records separately."""
    bs = getattr(loader, "batch_size", None)
    if bs is None:
        bs = getattr(getattr(loader, "batch_sampler", None), "batch_size", None)
    return bs


def _on_iter(loader) -> None:
    """A new data pass began: mark it pending so the next optimizer.step() claims it as an
    epoch, and remember this loader's batch_size for the HyperparameterSet. A pass also closes
    any open eval episode (Slice 7): metrics for the previous pass all arrive before this one
    starts, so a new pass means that episode is complete."""
    global _pending_pass, _pending_batch_size, _pending_dataset, _pending_transform
    global _last_pass_dataset, _pending_sampler, _last_pass_start_ns
    if _eval_metrics or _eval_losses:   # Slice 7: the prior episode is done -> emit it before this pass
        _flush_eval()
    _last_pass_start_ns = time.time_ns()   # an eval episode's work starts with the pass that feeds it
    _pending_pass = True
    _pending_batch_size = _effective_batch_size(loader)
    _pending_dataset = getattr(loader, "dataset", None)
    _pending_transform = getattr(_pending_dataset, "transform", None)  # the Compose (Slice 4)
    _last_pass_dataset = _pending_dataset  # the split a following metric call scores (Slice 7)
    _pending_sampler = _sampler_info(loader)  # Slice 11: sampler/shuffle/seed of this pass

    # Slice 10 (Option A): arm observe-real-reads capture here -- _on_iter runs BEFORE orig(self)
    # forks the loader's workers, so the wraps land in every worker. A shuffled pass is the
    # train-convention signal; the first arms, a second (next epoch's train pass) disarms so the
    # fresh epoch>=2 worker forks skip re-hashing what epoch 1 already fingerprinted.
    global _train_pass_seen, _sample_active, _sample_armed, _fwd_active, _fwd_expect, _loader_batch
    _fwd_expect = False   # Slice 13: a new pass invalidates any pending forward expectation
    if _pending_sampler.get("train_order") and _pending_dataset is not None:
        _train_pass_seen += 1
        if _train_pass_seen == 1:
            _sample_armed = _arm_sample_capture(_pending_dataset)
            _fwd_active = _install_forward_check()   # Slice 13: watch the loader->forward boundary
        elif _train_pass_seen >= 2:
            _sample_active = False
            _fwd_active = False
            _loader_batch = None   # stop holding the last epoch-1 batch once the tee disarms
            _remove_forward_check()


def _close_epoch() -> None:
    """Close the current epoch's TrainingStep (stamps end_ns + final num_steps, and moves the
    recorder's spine pointer so the next epoch links WasInformedBy back to this one)."""
    global _train_cm, _train_activity, _epoch_order_ref, _train_scope
    if _train_activity is not None and _epoch_order_ref:  # Slice 11: stamp the realized order
        try:
            _train_activity.attributes.update(_epoch_order_attrs(_epoch_order_ref))
        except Exception:
            pass
    _epoch_order_ref = None
    if _train_cm is not None:
        try:
            # the epoch ended at its LAST step, not here: _close_epoch runs from the next epoch's
            # first step, so the exit instant is a whole val pass + checkpoint write too late.
            if _train_scope is not None and _last_step_ns is not None:
                _train_scope.ends_at(_last_step_ns)
            _train_cm.__exit__(None, None, None)
        except Exception:
            pass
    _train_cm = _train_scope = None
    _train_activity = None


# ---- checkpoint interposition (Slice 5) --------------------------------------

def _install_save_hook(torch) -> None:
    """Patch torch.save (a module-level function, so we replace the attribute on the module)
    to notice checkpoint writes. Safe to run twice via the _WRAPPED marker, like the method patches."""
    _patch_module_fn(torch, "save", _wrap_save)


def _wrap_save(orig):
    @functools.wraps(orig)
    def save(obj, f, *args, **kwargs):
        t0 = time.time_ns()                     # the write starts here, not where we notice it
        result = orig(obj, f, *args, **kwargs)  # write the file first
        try:
            _on_save(obj, f, (t0, time.time_ns()))
        except Exception:
            pass  # capture must never break the save it observes
        return result

    setattr(save, _WRAPPED, True)
    return save


def _on_save(obj, f, span=None) -> None:
    """A torch.save landed. Recover the model weights from obj -- a raw state_dict, a wrapped
    checkpoint dict (weights nested under any key), or a pickled nn.Module -- and emit a
    ModelCheckpoint only when the saved tensor shapes match the architecture being trained (the
    _param_shapes fingerprint). That match is what makes this convention-agnostic: it ties the write
    to THIS run's model regardless of the wrapper key, and rejects non-weight tensor dicts (cached
    features, embeddings). Tied to the epoch that produced it (informed_by the open TrainingStep,
    derived from the data trained on). Epoch from a wrapper's 'epoch' key else the tracked epoch;
    'best' is read from the filename."""
    if _train_activity is None:  # no training context to tie the write to
        return
    if _extract_state_dict(obj) is None:  # not the model we're training (or unreadable)
        return
    path = _save_path(f)
    if path is None:
        return

    ep = obj.get("epoch") if isinstance(obj, dict) else None
    epoch = (ep + 1) if isinstance(ep, int) else (_epoch if _epoch > 0 else None)  # wrapper (0-based) else tracked
    ckpt = ModelCheckpoint(
        content_hash=hash_file(path),
        epoch=epoch,
        path=path,
        attributes={"filename": os.path.basename(path),
                    "is_best": os.path.basename(path) == "best.pt"})
    data = _train_input if _train_input is not None else _split_node
    inputs = [data] if data is not None else []
    # informed_by the producing epoch + chain=False so this side-write never moves the training
    # spine (the next epoch's _close_epoch resets the spine pointer to its TrainingStep anyway).
    with _session.recorder.activity(
            CheckpointWrite(activity_id="", attributes={"call_site": _call_site()}),
            inputs=inputs, config=_intent, derivation=DerivationType.TRAINING,
            informed_by=_train_activity.id, chain=False, observed_span=span) as scope:
        scope.generated(ckpt)


def _save_path(f) -> Optional[str]:
    """The filesystem path torch.save wrote to, or None for a file-like/buffer we can't re-read."""
    if isinstance(f, (str, bytes)):
        return os.fsdecode(f)
    if isinstance(f, os.PathLike):
        return os.fspath(f)
    return None


def _extract_state_dict(obj):
    """The model-weights dict inside a torch.save target, or None. Tries obj as a flat state_dict, a
    pickled nn.Module, and one level of dict nesting (wrapped checkpoints under any key); the first
    candidate whose tensor shapes match the tracked architecture wins -- so the model is found
    regardless of wrapper key and a non-weight tensor dict (features/embeddings) is rejected."""
    for cand in _state_dict_candidates(obj):
        if _matches_arch(cand):
            return cand
    return None


def _state_dict_candidates(obj):
    """Yield dicts that might be the model's state_dict: a pickled nn.Module's own state_dict, obj
    itself (a raw state_dict), and obj's one-level-nested dict values (wrapped checkpoints like
    {'model_state_dict': sd} / {'net': sd}). Best-effort; never raises."""
    import torch
    if isinstance(obj, torch.nn.Module):
        try:
            yield obj.state_dict()
        except Exception:
            pass
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            if isinstance(v, dict):
                yield v


def _matches_arch(cand) -> bool:
    """True if every optimized-parameter shape (the _param_shapes fingerprint) appears among cand's
    tensor values -- i.e. cand carries the model being trained. A state_dict is the model's params
    plus buffers, so the param shapes are a subset; extra buffer shapes are ignored. False before the
    arch is known (no first step) or for any dict missing the full model."""
    import torch
    if not _param_shapes or not isinstance(cand, dict):
        return False
    counts: dict = {}
    for v in cand.values():
        if isinstance(v, torch.Tensor):
            key = tuple(v.shape)
            counts[key] = counts.get(key, 0) + 1
    return all(counts.get(shape, 0) >= n for shape, n in _param_shapes.items())


# ---- split interposition (Slice 6) -------------------------------------------

def _install_split_hooks(torch) -> None:
    """Catch split activity for any dataset: patch the standard split entry points so any
    pipeline that splits through a known library records a DataSplit + one SplitManifest per
    partition, with NO per-dataset code. torch.utils.data.random_split always; sklearn's
    train_test_split if sklearn can be imported. Cross-validator .split() (yields indices ->
    won't match Slice 3's row hash) is left for later. NOTE: patches the function on its
    home module, so a `from sklearn.model_selection import train_test_split` bound BEFORE
    init() keeps the old name -- call init() ahead of those imports to be caught."""
    _patch_module_fn(torch.utils.data, "random_split", _wrap_random_split)
    try:
        import sklearn.model_selection as ms
    except Exception:
        return  # sklearn absent -> nothing to patch, train_test_split simply never fires
    _patch_module_fn(ms, "train_test_split", _wrap_train_test_split)


def _wrap_train_test_split(orig):
    @functools.wraps(orig)
    def train_test_split(*arrays, **kwargs):
        t0 = time.time_ns()
        result = orig(*arrays, **kwargs)  # do the real split first
        try:
            _on_train_test_split(arrays, kwargs, result, (t0, time.time_ns()))
        except Exception:
            pass  # capture must never break the split it observes
        return result

    setattr(train_test_split, _WRAPPED, True)
    return train_test_split


def _on_train_test_split(arrays, kwargs, result, span=None) -> None:
    """sklearn returns 2*len(arrays) outputs: result[2i]=array-i train, result[2i+1]=test. Emit
    one DataSplit making a train + test SplitManifest per input from a source manifest of that
    input. DataFrame partitions hash via df.to_csv (same as Slice 3) -> the train partition matches
    the SplitManifest the model later trains on. seed = random_state when it's an int."""
    if not arrays or _session is None:
        return
    seed = kwargs.get("random_state")
    attrs = {k: _canon(kwargs[k]) for k in ("test_size", "train_size", "shuffle") if k in kwargs}
    attrs["stratified"] = kwargs.get("stratify") is not None
    attrs["n_arrays"] = len(arrays)

    inputs, outs = [], []
    for i, arr in enumerate(arrays):
        src = _partition_manifest(arr, "source")
        if src is not None:
            inputs.append(src)
        outs += [n for n in (_partition_manifest(result[2 * i], "train"),
                             _partition_manifest(result[2 * i + 1], "test")) if n is not None]
    _emit_split(seed if isinstance(seed, int) else None, attrs, inputs, outs, span)


def _wrap_random_split(orig):
    @functools.wraps(orig)
    def random_split(dataset, lengths, *args, **kwargs):
        t0 = time.time_ns()
        result = orig(dataset, lengths, *args, **kwargs)
        try:
            _on_random_split(dataset, args, kwargs, result, (t0, time.time_ns()))
        except Exception:
            pass
        return result

    setattr(random_split, _WRAPPED, True)
    return random_split


def _on_random_split(dataset, args, kwargs, result, span=None) -> None:
    """torch random_split -> list of Subset. Each Subset becomes a SplitManifest hashed on its
    sorted index membership -- 'which samples landed in which split' (the stage-2 hide-in-split
    surface). Index identity does NOT match Slice 3's df hash; that's the honest cost of one
    content scheme covering any dataset. seed read off the generator if one is given."""
    if _session is None:
        return
    gen = kwargs.get("generator") or (args[0] if args else None)
    seed = None
    try:
        if gen is not None:
            seed = int(gen.initial_seed())
    except Exception:
        pass
    src = _partition_manifest(getattr(dataset, "df", None), "source")
    inputs = [src] if src is not None else []
    outs = []
    for i, sub in enumerate(result):
        idx = getattr(sub, "indices", None)
        if idx is None:
            continue
        members = sorted(int(j) for j in idx)
        node = SplitManifest(content_hash=hash_canonical({"indices": members}),
                             split=f"partition_{i}", num_samples=len(members),
                             split_evidence="library")
        outs.append(_register_manifest(node))
    _emit_split(seed, {"n_partitions": len(result)}, inputs, outs, span)


def _open_eval_span() -> None:
    """Grow the open episode's window. It STARTS at the data pass that fed the episode, not at the
    first metric call: the pass is where the split is actually read, and that io is the whole reason
    the window exists. Falls back to now if metrics arrive with no pass on record."""
    global _eval_span
    if _eval_span is None:
        _eval_span = (_last_pass_start_ns or time.time_ns(), time.time_ns())
    else:
        _eval_span = (_eval_span[0], time.time_ns())


def _observed_span(span) -> tuple:
    """Grow an observed (first_ns, last_ns) window to now. Wall clock, not monotonic: these windows
    exist to be comparable to an external tracer's event timestamps, and are the only thing that lets
    a teardown-emitted activity own the io it actually performed."""
    t = time.time_ns()
    return (t, t) if span is None else (span[0], t)


# How a split label was earned, strongest first (schema.nodes owns the table -- store/promote.py
# ranks the same claims across processes). A structural label must be able to OVERWRITE a filename
# one: the csv is read before training starts, so first-writer-wins would let the file stem beat the
# gradient evidence every time. This ranking only orders claims made INSIDE one process; claims from
# different stages meet in the projector, which is why promote.py arbitrates there too.
_EVIDENCE_RANK = EVIDENCE_RANK
_manifest_evidence_by_hash: dict = {}   # content_hash -> how its label in _manifest_by_hash was earned


def _claim_label(h: str, name: Optional[str], evidence: str) -> None:
    """Claim `name` as the split label for content `h`, keeping the best-evidenced claim. Ties keep
    the incumbent (first wins), so this is setdefault plus a priority override."""
    if not name or name == "source":
        return
    rank = _EVIDENCE_RANK.get(evidence, 0)
    if rank <= _EVIDENCE_RANK.get(_manifest_evidence_by_hash.get(h), 0) and h in _manifest_by_hash:
        return
    _manifest_by_hash[h] = name
    _manifest_evidence_by_hash[h] = evidence


def _row_fingerprint(csv_text: str) -> Optional[frozenset]:
    """The set of rows in a frame, as short per-row digests. Membership is what makes the split check
    a VERIFICATION rather than a search: two frames can share a row count by coincidence, but not a
    row set. In-process only -- never stored on a node, so no census key changes."""
    try:
        lines = csv_text.splitlines()[1:]          # drop the header
        return frozenset(hashlib.blake2b(l.encode(), digest_size=8).digest() for l in lines)
    except Exception:
        return None


class _Frame:
    """One csv frame we observed this stage: enough to test whether some writes partition it."""
    __slots__ = ("h", "name", "path", "nrows", "rows", "written")

    def __init__(self, h, name, path, nrows, rows, written):
        self.h, self.name, self.path = h, name, path
        self.nrows, self.rows, self.written = nrows, rows, written


def _note_frame(h, name, path, nrows, rows, written) -> "_Frame":
    """Remember a frame as a source candidate. Same content seen twice (written then re-read) is one
    candidate -- keep the first, which is the one that carries the write's path."""
    for f in _stage_frames:
        if f.h == h:
            return f
    fr = _Frame(h, name, path, nrows, rows, written)
    _stage_frames.append(fr)
    return fr


def _register_manifest(node: "SplitManifest") -> "SplitManifest":
    """Record a manifest's content_hash -> split label (Slice 7 names an eval split by hash match)
    and log it. Only labels worth inheriting are kept (a real split name, not a None/source tag).
    Also carries `path` across every record of the same content: a manifest is minted in several
    places (write boundary, loader dataset, partition), most of which never see a filename, and the
    projector's last-wins merge would otherwise drop the path the one path-aware site recorded."""
    label = getattr(node, "split", None)
    if label:
        _claim_label(node.content_hash, label,
                     getattr(node, "split_evidence", None) or "filename")
    if getattr(node, "path", None):
        _manifest_path_by_hash.setdefault(node.content_hash, node.path)
    else:
        node.path = _manifest_path_by_hash.get(node.content_hash)
    return _session.recorder.entity(node)


def _partition_manifest(obj, split: str) -> Optional["SplitManifest"]:
    """A SplitManifest for one partition, content-hashed so a DataFrame/Series hashes via df.to_csv
    (same as Slice 3 -> the train partition matches across stages) and any other array hashes its raw
    bytes. Returns None for a None obj or a shape we can't hash."""
    if obj is None:
        return None
    h, n = _hash_array_like(obj)
    if h is None:
        return None
    return _register_manifest(
        SplitManifest(content_hash=h, split=split, num_samples=n, split_evidence="library"))


def _hash_array_like(obj) -> tuple[Optional[str], Optional[int]]:
    """(content_hash, num_samples) for a split partition. DataFrame/Series -> to_csv bytes (same as
    Slice 3); ndarray/tensor -> tobytes; otherwise hash the materialized list. Best-effort:
    (None, None) when nothing fits."""
    n = len(obj) if hasattr(obj, "__len__") else None
    to_csv = getattr(obj, "to_csv", None)
    if callable(to_csv):
        try:
            return hash_bytes(obj.to_csv(index=False).encode()), n
        except Exception:
            return None, None
    tobytes = getattr(obj, "tobytes", None)
    if callable(tobytes):
        try:
            return hash_bytes(obj.tobytes()), n
        except Exception:
            return None, None
    try:
        return hash_canonical(_canon(list(obj))), n
    except Exception:
        return None, None


def _emit_split(seed, attrs, inputs, outs, span=None) -> None:
    """Record one DataSplit: `inputs` (the source partitions) derive `outs` (the split partitions),
    tagged SPLIT, with Code+Env as intent. No-op when nothing hashable came out. `span` is the real
    (start, end) of the splitter call -- both callers wrap a single function, so unlike the csv path
    in _flush_splits there is nothing to accumulate."""
    if not outs:
        return
    config = [n for n in (_session.code, _session.env) if n is not None]
    with _session.recorder.activity(
            DataSplit(activity_id="", seed=seed, attributes={**attrs, "call_site": _call_site()}),
            inputs=inputs, config=config, derivation=DerivationType.SPLIT,
            observed_span=span) as scope:
        for o in outs:
            scope.generated(o)


# ---- file-load interposition (Slice 8) ---------------------------------------

def _install_read_hooks() -> None:
    """Name splits by the file they're loaded from. Interpose pandas.read_csv: every load maps the
    loaded frame's content-hash -> the source path's stem, so a later split/eval manifest with that
    hash inherits the name with NO per-dataset code. This covers splits made upstream (a prior
    make_splits run) that this process only reads -- the train run never sees their split op, just
    their files. Rebinds already-imported `read_csv` references (import-order independent)."""
    try:
        import pandas as pd
    except Exception:
        return  # pandas absent -> file-named splits simply never register
    _patch_callable_everywhere(pd, "read_csv", _wrap_read_csv)


def _wrap_read_csv(orig):
    @functools.wraps(orig)
    def read_csv(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            _on_read_csv(args, kwargs, result)
        except Exception:
            pass  # capture must never break the load it observes
        return result

    setattr(read_csv, _WRAPPED, True)
    return read_csv


def _on_read_csv(args, kwargs, df) -> None:
    """Map a just-loaded frame's content-hash (the to_csv scheme Slices 3/6/7 share) to its file
    stem, so a manifest of that data inherits the name. setdefault: an authoritative name from a
    captured split op (Slice 6) wins over the filename. No-op for buffers/URLs or non-frames."""
    src = kwargs.get("filepath_or_buffer", args[0] if args else None)
    name = _path_stem(src)
    to_csv = getattr(df, "to_csv", None)
    if name is None or not callable(to_csv):
        return
    try:
        h = hash_bytes(df.to_csv(index=False).encode())
    except Exception:
        return
    _claim_label(h, name, "filename")
    _manifest_path_by_hash.setdefault(h, os.path.abspath(os.fspath(src)))
    n = len(df) if hasattr(df, "__len__") else None
    # a table this stage READ is a source candidate: in a full run the split's source is only ever
    # read, never written, so the candidate set cannot be the writes alone
    _note_frame(h, name, os.path.abspath(os.fspath(src)), n,
                _row_fingerprint(df.to_csv(index=False)), False)
    global _last_read, _split_span
    # the table read just before the writes is the split-source candidate. Tracked in every stage:
    # a stage that turns out not to write a partition just never reaches _flush_splits' emit path.
    _last_read = (h, n, os.fspath(src))
    _split_span = _observed_span(_split_span)  # reading the source table is work the DataSplit did


def _path_stem(src) -> Optional[str]:
    """Filename stem of a path-like source: 'a/b/val.csv' -> 'val'. None for buffers, URLs, or
    anything without a plain stem (so only real files become split names)."""
    if not isinstance(src, (str, os.PathLike)):
        return None
    stem = os.path.splitext(os.path.basename(os.fspath(src)))[0]
    return stem or None


# ---- file-write interposition (Slice 9) --------------------------------------

def _install_write_hooks() -> None:
    """Capture split partitions as they're WRITTEN. Interpose DataFrame.to_csv: in the make_splits
    stage each write to a path -> a named SplitManifest, batched (flushed at exit) into one DataSplit
    derived from the source table. Hooking the write -- not sklearn/torch split fns (Slice 6) -- is
    what catches a hand-rolled splitter; the written bytes hash (df.to_csv scheme) identical to a
    later read of the same file, so the partition dedups across stages into one node with a real
    edge. Gated to make_splits: a write's node TYPE is stage-specific (other stages are future
    slices), and the stage is the only signal that says 'these writes are splits'."""
    try:
        import pandas as pd
    except Exception:
        return  # pandas absent -> no DataFrame writes to capture
    _patch_method(pd.DataFrame, "to_csv", _wrap_to_csv)


def _wrap_to_csv(orig):
    @functools.wraps(orig)
    def to_csv(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        try:
            _on_to_csv(self, args, kwargs)
        except Exception:
            pass  # capture must never break the write it observes
        return result

    setattr(to_csv, _WRAPPED, True)
    return to_csv


def _on_to_csv(df, args, kwargs) -> None:
    """A frame was written: BUFFER it. We deliberately do not mint a node here, because at write time
    there is no way to tell a split partition from any other table a splitting stage happens to write
    (an intermediate subset, a filtered copy). That is decided in _flush_splits, once every frame this
    stage touched is known and the partition can be identified by CONSERVATION rather than by the
    stage's name. A path-less to_csv (returns a string, e.g. our own hashing) has no stem -> skipped,
    which also makes the hash call below non-recursive."""
    if _session is None:
        return
    path = kwargs.get("path_or_buf", args[0] if args else None)
    name = _path_stem(path)
    if name is None:
        return
    try:
        csv_text = df.to_csv(index=False)
        h = hash_bytes(csv_text.encode())
    except Exception:
        return
    _claim_label(h, name, "filename")
    _manifest_path_by_hash.setdefault(h, os.path.abspath(os.fspath(path)))
    fr = _note_frame(h, name, os.path.abspath(os.fspath(path)), len(df),
                     _row_fingerprint(csv_text), True)
    fr.written = True
    global _split_source, _split_call_site, _split_span
    if not _split_writes:                       # first write of the batch: snapshot source + name
        _split_source = _source_metadata(_last_read)
        _split_call_site = _call_site()
    if fr not in _split_writes:
        _split_writes.append(fr)
    _split_span = _observed_span(_split_span)


def _source_metadata(last_read) -> Optional["DatasetMetadata"]:
    """The DatasetMetadata a split batch derives from: the table read just before the writes. None
    when nothing was read this stage (then the DataSplit has no input)."""
    if last_read is None:
        return None
    h, num_rows, path = last_read
    return _session.recorder.entity(
        DatasetMetadata(content_hash=h, path=path, num_rows=num_rows))


def _find_partition(writes: list) -> tuple:
    """Identify which buffered writes actually PARTITION one of the frames this stage saw.

    Membership makes this a check, not a search: a write is a piece of `src` iff its rows are a
    subset of src's rows, so the candidate pieces are read straight off the data -- there is no
    subset-sum to solve and no combination to guess. A partition is accepted only when the pieces are
    pairwise disjoint and together account for src exactly.

    Returns (source_frame, [partition frames], coverage, disjoint) or (None, [], None, None)."""
    best = (None, [], None, None)
    for src in _stage_frames:
        if src.rows is None or not src.nrows:
            continue
        pieces = [w for w in writes if w is not src and w.rows is not None and w.nrows
                  and w.rows <= src.rows]
        if len(pieces) < 2:                      # one piece is a copy or a subset, not a split
            continue
        union = frozenset().union(*[p.rows for p in pieces])
        disjoint = sum(len(p.rows) for p in pieces) == len(union)
        coverage = len(union) / len(src.rows)
        if coverage == 1.0 and disjoint and len(pieces) > len(best[1]):
            best = (src, pieces, coverage, disjoint)
    return best


def _flush_splits() -> None:
    """Emit the buffered csv writes, tagged SPLIT. seed is None -- transparent capture can't see the
    splitter's seed (the write boundary is below it). Names the activity by the first write's call
    site. No-op when nothing was written.

    One path: writes must be VERIFIED to partition a frame this stage saw. That frame is the source
    (not merely the last table read) and only those writes are partitions -- any other write is a
    derived table and is emitted as DatasetMetadata, so a real artifact still appears in the graph
    without being called a split. Writes that partition nothing are not a split, whatever the entry
    script is called: the stage-name fallback is gone, so capture no longer needs to recognise the
    pipeline it is observing."""
    global _split_writes, _split_source, _split_call_site, _split_span, _stage_frames
    writes = _split_writes
    _split_writes = []
    if not writes:
        _stage_frames = []
        return
    source, pieces, coverage, disjoint = _find_partition(writes)
    call_site, span = _split_call_site, _split_span
    _split_source, _split_call_site, _split_span = None, [], None

    if source is not None:
        evidence = "conserved"
        src_node = _session.recorder.entity(
            DatasetMetadata(content_hash=source.h, path=source.path, num_rows=source.nrows))
        for w in writes:                        # written, but not a piece of the split
            if w is not source and w not in pieces:
                _session.recorder.entity(
                    DatasetMetadata(content_hash=w.h, path=w.path, num_rows=w.nrows))
    else:
        _stage_frames = []
        return                                   # nothing shows these writes partition anything

    # Deferred typing: a write enters the ledger as the table it observably IS. Whether those tables
    # partition anything is a property of the whole run, so promote_splits (store/promote.py) assigns
    # SplitManifest at projection time, where the evidence exists. The DataSplit activity below still
    # records what this stage saw (evidence/coverage/disjoint) -- that is an observation, not a type.
    outs = [_session.recorder.entity(
                DatasetMetadata(content_hash=p.h, path=p.path, num_rows=p.nrows))
            for p in pieces]
    inputs = [src_node] if src_node is not None else []
    config = [n for n in (_session.code, _session.env) if n is not None]
    with _session.recorder.activity(
            DataSplit(activity_id="", seed=None, evidence=evidence, coverage=coverage,
                      partitions_disjoint=disjoint, attributes={"call_site": call_site}),
            inputs=inputs, config=config, derivation=DerivationType.SPLIT,
            observed_span=span) as scope:
        for o in outs:
            scope.generated(o)
    _stage_frames = []


# ---- per-sample interposition (Slice 10) -------------------------------------

def _capture_samples(dataset) -> None:
    """FALLBACK path: hash every train sample once via a direct pass (transform off) into one
    SampleManifest. Used only when observe-real-reads couldn't be armed before the loader forked
    (no transform to tee, or a non-shuffled train loader). The primary path (_arm_sample_capture)
    fingerprints the reads training already does, so no shadow pass runs. Forks iff sample CONTENT
    changes, catching per-image poisoning the label-csv SplitManifest misses."""
    global _sample_manifest
    if dataset is None or not hasattr(dataset, "__len__"):
        return
    try:
        n = len(dataset)
    except Exception:
        return
    orig = type(dataset).__getitem__
    items = []
    for i in range(n):
        try:
            items.append((i, _hash_sample(_raw_sample(dataset, i, orig))))
        except Exception:
            pass  # a sample we can't hash is skipped, never fatal
    if not items:
        return
    _sample_manifest = _session.recorder.entity(SampleManifest(
        content_hash=hash_canonical({"samples": items}),
        split="train", num_samples=len(items),
        attributes={"granularity": "per-sample", "pre_transform": True}))


def _raw_sample(dataset, index, orig):
    """The sample with the dataset's transform turned off, so augmentation randomness doesn't move
    its hash. Falls back to the plain item when there's no .transform to disable."""
    t = getattr(dataset, "transform", None)
    if t is None:
        return orig(dataset, index)
    try:
        dataset.transform = None
        return orig(dataset, index)
    finally:
        dataset.transform = t


def _hash_sample(sample) -> str:
    """A content hash of one sample -- a tuple/list (input, label), a dict of fields, or a bare input.
    Each value goes through _content_of so a PIL image / tensor hashes its RAW BYTES; hashing a dict
    whole would str() the PIL image, which embeds a memory address and is nondeterministic."""
    if isinstance(sample, dict):
        parts = [_content_of(v) for _, v in sorted(sample.items(), key=lambda kv: str(kv[0]))]
    elif isinstance(sample, (tuple, list)):
        parts = [_content_of(p) for p in sample]
    else:
        parts = [_content_of(sample)]
    return hash_canonical(parts)


def _content_of(obj):
    tobytes = getattr(obj, "tobytes", None)  # PIL Image / ndarray
    if callable(tobytes):
        try:
            return hash_bytes(tobytes())
        except Exception:
            pass
    numpy = getattr(obj, "numpy", None)       # torch Tensor
    if callable(numpy):
        try:
            detach = getattr(obj, "detach", None)  # a CUDA/grad tensor must come to host before .numpy()
            t = obj.detach().cpu() if callable(detach) else obj
            return hash_bytes(t.numpy().tobytes())
        except Exception:
            pass
    return _canon(obj)


# ---- loader->forward integrity (Slice 13) ------------------------------------

def _collect_tensor_hashes(obj, out: set) -> None:
    """Add the content hash of every tensor/ndarray leaf reachable in `obj` (dict / list / tuple /
    bare tensor) to `out`. Whole-tensor granularity: poisoning any row forks the batch tensor's hash."""
    if callable(getattr(obj, "numpy", None)):        # a tensor / ndarray leaf
        h = _content_of(obj)
        if isinstance(h, str):
            out.add(h)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_tensor_hashes(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_tensor_hashes(v, out)


def _batch_content(batch) -> set:
    """Content hashes of every tensor leaf in one loader batch."""
    out: set = set()
    _collect_tensor_hashes(batch, out)
    return out


def _tensor_leaves(obj, out: list) -> None:
    """Collect every tensor/ndarray leaf reachable in `obj` (dict / list / tuple / bare) in order."""
    if callable(getattr(obj, "numpy", None)) and hasattr(obj, "shape"):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _tensor_leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _tensor_leaves(v, out)


def _rows_share_class(label, row_changed) -> bool:
    """Do the altered rows share an active class the whole batch does NOT? A label-SELECTED edit (ccba
    stamps rows already carrying the target class) leaves the altered rows homogeneous on a class that
    isn't universal; an arbitrary edit does not. Excluding batch-universal classes avoids firing on a
    label every row happens to carry. Handles [B] class-index and [B, C] multi-hot. Attack-agnostic."""
    changed = label[row_changed]
    if changed.shape[0] == 0:
        return False
    if label.dim() == 1:                              # class-index: one class across all changed rows...
        uniq = changed.unique()
        if uniq.numel() != 1:
            return False
        return bool((label == uniq[0]).sum() < label.shape[0])   # ...and not the whole batch's class
    active_all_changed = (changed == 1).all(dim=0)    # multi-hot: a class active in every changed row...
    active_all_batch = (label == 1).all(dim=0)        # ...but not active in every batch row
    return bool((active_all_changed & (~active_all_batch)).any())


def _measure_row_region(model_input) -> None:
    """The model input diverged from the teed loader batch (an in-flight edit). Measure it row/region-wise
    from tensors already in flight -- no replay: which fraction of rows changed (a subset, not the whole
    batch?), how confined the per-row change is (a small patch vs the whole image?), and whether the
    changed rows were label-selected. Keeps the max across the epoch-1 steps so clean=0 and a poisoner>0."""
    global _alter_row_frac_max, _alter_region_frac_max, _alter_share_class
    if _loader_batch is None or not hasattr(model_input, "shape"):
        return
    a = model_input.detach().to("cpu")
    leaves: list = []
    _tensor_leaves(_loader_batch, leaves)
    clean = next((t for t in leaves if tuple(t.shape) == tuple(a.shape)), None)   # the loader's version of this input
    if clean is None:
        return
    clean = clean.detach().to("cpu")
    if a.dtype != clean.dtype:
        clean = clean.to(a.dtype)
    b = a.shape[0]
    diff = (a != clean).reshape(b, -1)
    row_changed = diff.any(dim=1)                     # [B] which rows differ
    n = int(row_changed.sum())
    if n == 0:
        return
    _alter_row_frac_max = max(_alter_row_frac_max, n / b)
    region = float(diff.float().mean(dim=1)[row_changed].mean())   # mean fraction of a changed row's elements edited
    _alter_region_frac_max = max(_alter_region_frac_max, region)
    label = next((t for t in leaves if t is not clean and t.shape and t.shape[0] == b and t.dim() <= 2), None)
    if label is not None:
        _alter_share_class = _alter_share_class or _rows_share_class(label.detach().to("cpu"), row_changed)


def _install_forward_check() -> bool:
    """Register a global forward-pre-hook that checks each model() input against the loader batch. It's
    global (fires for every module) so it needs no handle on the user's model -- the outermost module's
    hook fires before its children, so the first fire after a drawn batch is the top-level forward."""
    global _fwd_hook_handle
    if _fwd_hook_handle is not None:
        return True
    try:
        import torch
        _fwd_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(_on_forward_pre)
        return True
    except Exception:
        return False


def _remove_forward_check() -> None:
    global _fwd_hook_handle
    if _fwd_hook_handle is not None:
        try:
            _fwd_hook_handle.remove()
        except Exception:
            pass
        _fwd_hook_handle = None


def _on_forward_pre(module, args):
    """The first forward after a batch was drawn: does model()'s input trace to a tensor the loader
    produced? If none of its tensor leaves match the drawn batch, the batch was rewritten between loader
    and model (an in-loop poison like CCBA, or a benign GPU-side edit). Count it; the fold onto the
    epoch's TrainingStep happens at the next optimizer step (this fires before the step opens it)."""
    global _fwd_expect, _fwd_unmatched
    if not _fwd_expect or not args:
        return
    _fwd_expect = False
    seen: set = set()
    _collect_tensor_hashes(args[0], seen)
    if seen and not (seen & _loader_content):
        _fwd_unmatched += 1
        try:
            _measure_row_region(args[0])   # per-row/region characterization of the in-flight edit (ccba)
        except Exception:
            pass  # capture must never break the training it observes


# ---- forwards per optimizer step (Slice 14) ----------------------------------

def _install_forward_counter(torch) -> None:
    """Count outermost train-mode forwards. Slice 13 inspects only the FIRST forward after a batch is
    drawn (was the batch rewritten); this counts how MANY there were, catching extra in-loop forwards
    on inputs the loader never produced. Global hooks fire for every submodule, so depth gates to the
    top-level call; the post-hook pairs with the pre-hook to unwind it."""
    global _fwd_count_handles
    m = torch.nn.modules.module
    try:
        _fwd_count_handles = [m.register_module_forward_pre_hook(_count_forward_pre),
                              m.register_module_forward_hook(_count_forward_post)]
    except Exception:
        _fwd_count_handles = []  # capture must never break the training it observes


def _count_forward_pre(module, args):
    global _fwd_depth, _forward_events, _model_ref, _offloader_forwards
    # A criterion is an nn.Module too and is called at depth 0 like the model, so depth alone would
    # double every step. Parameters are what separates them: a loss carries none, a model always does.
    if _fwd_depth == 0 and module.training and next(module.parameters(), None) is not None:
        _forward_events += 1                 # eval passes run no optimizer step -- don't fold into one
        if _model_ref is None:
            _model_ref = module              # the top-level model; fingerprinted by _module_graph at arch-record
        # a further depth-0 forward in the same step, on inputs the loader never produced, is an injected
        # probe (memory backdoor queries the model with synthetic index triggers). Only checked while the
        # epoch-1 tee keeps _loader_content current (Slice 13 arming); the first forward is the real batch.
        if _fwd_active and _forward_events > 1 and args:
            seen: set = set()
            _collect_tensor_hashes(args[0], seen)
            if seen and not (seen & _loader_content):
                _offloader_forwards += 1
    _fwd_depth += 1


def _count_forward_post(module, args, output):
    global _fwd_depth
    if _fwd_depth > 0:
        _fwd_depth -= 1


def _fold_forward_step() -> None:
    """At each optimizer step, roll this step's forwards into the open epoch: keep the largest per-step
    count (>1 = an extra in-loop forward). Mirrors _fold_backward_step; reset per step."""
    global _forward_events, _step_forward_max, _fwd_depth
    if _forward_events > _step_forward_max:
        _step_forward_max = _forward_events
    _train_activity.attributes["forwards_per_step"] = _step_forward_max
    _forward_events = 0
    _fwd_depth = 0  # a forward that raised would leave this unbalanced; the step boundary re-zeroes it


# ---- observe-real-reads capture (Slice 10, Option A) -------------------------

class _TransformTee:
    """Wraps a dataset's transform. Its argument on each call is the sample's PRE-transform content
    (the image straight off disk), so hashing it here fingerprints content WITHOUT the augmentation
    randomness the post-transform tensor carries -- and without a shadow pass, since it fires on the
    reads the loader already does. Delegates to the real transform unchanged."""

    def __init__(self, orig):
        self._orig = orig

    def __call__(self, x):
        try:
            _record_read(_reading_idx, x)
        except Exception:
            pass  # capture must never break the read it observes
        return self._orig(x) if self._orig is not None else x

    def __getattr__(self, name):  # let Compose introspection (.transforms) fall through to the real one
        return getattr(self._orig, name)


def _arm_sample_capture(dataset) -> bool:
    """Install observe-real-reads on `dataset` before its loader forks workers: a class-level
    __getitem__ wrap that stamps the index being built, and a tee on the transform that hashes each
    sample's pre-transform content. Returns False (caller shadow-falls-back) when there's no transform
    to tee -- the seam we read the pre-transform sample off."""
    global _capture_dataset, _capture_transform_orig, _sidecar_dir, _sample_active
    if getattr(dataset, "transform", None) is None:
        return False
    _capture_dataset = dataset
    _capture_transform_orig = dataset.transform
    dataset.transform = _TransformTee(dataset.transform)
    _patch_method(type(dataset), "__getitem__", _wrap_getitem)  # class-level; gated to this instance
    _sidecar_dir = _session.log_path.parent / "_sample_reads"
    try:
        _sidecar_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _sample_active = True
    return True


def _wrap_getitem(orig):
    """Stamp the index __getitem__ is building so the transform tee can key its hash by it. Gated to
    _capture_dataset so a same-class eval dataset (different instance) is untouched. Works in workers:
    each is a fork whose dataset object IS the forked _capture_dataset."""
    @functools.wraps(orig)
    def __getitem__(self, index):
        global _reading_idx, _reading_file_hash
        target = self is _capture_dataset
        if target:
            _reading_idx = index
            _reading_file_hash = None
        try:
            return orig(self, index)
        finally:
            if target:
                _reading_idx = None

    setattr(__getitem__, _WRAPPED, True)
    return __getitem__


def _worker_id():
    """This process's DataLoader worker id, or None in the main process."""
    try:
        import torch.utils.data as tud
        info = tud.get_worker_info()
        return None if info is None else info.id
    except Exception:
        return None


def _record_read(idx, x) -> None:
    """One real read observed: hash its pre-transform content once. Main-process reads aggregate in
    memory; worker reads append to a per-process sidecar the main process merges at finalize."""
    if idx is None or not _sample_active or idx in _seen_idx:
        return
    _seen_idx.add(idx)
    h = _content_of(x)
    fh = _reading_file_hash  # byte hash of the file this idx opened (None if it decoded from memory)
    if _worker_id() is None:
        _sample_reads[idx] = h
        if fh is not None:
            _read_files.add(fh)
    else:
        _sidecar_write(idx, h, fh)


def _sidecar_write(idx, h, fh=None) -> None:
    """Append '<idx>\\t<sample-hash>\\t<file-hash>' to this worker's sidecar (line-buffered, so it
    survives the worker's teardown). One file per pid -> no concurrent writers share a handle. The
    third column is the byte hash of the file the idx opened (blank if it decoded from memory)."""
    global _sidecar_fh
    if _sidecar_dir is None:
        return
    if _sidecar_fh is None:
        _sidecar_fh = open(_sidecar_dir / f"reads_{os.getpid()}.jsonl", "a", buffering=1)
    _sidecar_fh.write(f"{idx}\t{h}\t{fh or ''}\n")


def _emit_sample_manifest():
    """At finalize: merge main-process reads with every worker sidecar into one {idx: hash}, and emit
    the SampleManifest, derived from the train data node. Nothing observed (never armed) -> None, and
    the shadow fallback has already emitted its own manifest."""
    global _sample_manifest
    merged = dict(_sample_reads)
    files = set(_read_files)
    if _sidecar_dir is not None and _sidecar_dir.exists():
        for f in sorted(_sidecar_dir.glob("reads_*.jsonl")):
            try:
                for line in f.read_text().splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[1]:
                        merged[int(parts[0])] = parts[1]
                    if len(parts) >= 3 and parts[2]:
                        files.add(parts[2])
            except Exception:
                pass
    if not merged:
        return None
    items = [[i, merged[i]] for i in sorted(merged)]
    node = _session.recorder.entity(SampleManifest(
        content_hash=hash_canonical({"samples": items}),
        split="train", num_samples=len(items),
        attributes={"granularity": "per-sample", "pre_transform": True, "source": "observed-reads"}))
    data = _train_input if _train_input is not None else _split_node
    if data is not None:
        try:
            _session.recorder._emit_edge(
                WasDerivedFrom(source=node.id, target=data.id, derivation_type=None))
        except Exception:
            pass
    # Slice 15 join: publish the file-byte read-set so graph.load() can link this manifest to the
    # DatasetVersion store whose file set contains it (read-set subset of store-set).
    _write_members_sidecar(node.id, files)
    _sample_manifest = node
    return node


# ---- batch-order interposition (Slice 11) ------------------------------------

def _install_sampler_hook(torch) -> None:
    """Record the realized batch order. BatchSampler yields its index lists in the MAIN process --
    even when the loader uses workers -- so wrapping its __iter__ tees the exact order the model saw."""
    _patch_method(torch.utils.data.BatchSampler, "__iter__", _wrap_batchsampler_iter)


def _wrap_batchsampler_iter(orig):
    @functools.wraps(orig)
    def __iter__(self):
        global _current_order
        buf: list = []
        _current_order = buf  # eager: set before the first batch is drawn, so the epoch can grab it

        def gen():
            for batch in orig(self):
                try:
                    buf.append(tuple(int(i) for i in batch))
                except Exception:
                    buf.append(())
                yield batch

        return gen()

    setattr(__iter__, _WRAPPED, True)
    return __iter__


def _sampler_info(loader) -> dict:
    """The loader's sampler kind + shuffle flag + seed (off the loader/sampler generator, else the
    global RNG seed) -- the 'declared' order, to later check the realized order against."""
    import torch
    # The real schedule driver: torch wraps the sampler in a batch_sampler internally, and a loader
    # rebuilt with an injected batch_sampler (BRRR) hides its sampler there while loader.sampler falls
    # back to a plain SequentialSampler. Read the inner one so the injected sampler isn't masked.
    sampler = getattr(getattr(loader, "batch_sampler", None), "sampler", None) or getattr(loader, "sampler", None)
    gen = getattr(loader, "generator", None) or getattr(sampler, "generator", None)
    try:
        seed = int(gen.initial_seed()) if gen is not None else int(torch.initial_seed())
    except Exception:
        seed = None
    return {"sampler": type(sampler).__name__,
            "shuffle": type(sampler).__name__ == "RandomSampler",
            # train-pass signal for arming epoch-1 capture: a train pass shuffles (RandomSampler) OR uses
            # a custom sampler; only eval/val uses the stock SequentialSampler. Don't assume RandomSampler
            # -- an injected sampler (batch_reordering) or a foreign pipeline's WeightedRandomSampler still
            # counts as a train pass. Separate from `shuffle` (a scored field that means shuffle-requested).
            # KNOWN LIMITATION (heuristic): a val loader with a non-Sequential sampler (e.g. DistributedSampler
            # under DDP) is misread as a train pass. This never FALSE-FLAGS -- a clean pass tees the batch the
            # model then receives, so forward_inputs_unmatched stays 0 (matches anchor); worst case is a missed
            # arm if val precedes train. FUTURE WORK: detect the train pass by the pass an optimizer.step()
            # follows (the real definition, already tracked for epoch attribution) instead of the sampler type.
            "train_order": type(sampler).__name__ != "SequentialSampler",
            "origin": f"{type(sampler).__module__}:{type(sampler).__qualname__}",  # source location of the schedule-driving class
            "seed": seed}


def _epoch_order_attrs(order) -> dict:
    """Four hashes off the realized order that tell apart the BRRR variants, each discarding more
    detail than the last. `order` is the list of per-batch index tuples the epoch consumed. Read the
    LOOSEST one that differs: samples_used -> replacement; else batch_grouping -> datapoint reshuffle;
    else batch_contents_sequence -> batch reorder; else only batch_sequence -> within-batch reorder."""
    batches = [b for b in order if b]
    flat = [i for b in batches for i in b]
    per_batch = [hash_canonical(sorted(b)) for b in batches]  # each batch, ignoring internal order
    return {
        "batch_sequence": hash_canonical(flat),                        # exact sample sequence, within + across batches
        "batch_contents_sequence": hash_canonical(per_batch),          # batch order, ignoring within-batch order
        "batch_grouping": hash_canonical(sorted(per_batch)),           # which batches exist, ignoring batch order
        "samples_used": hash_canonical(sorted(flat)),                  # which indices appear, with counts
        "num_batches": len(batches),
    }


# ---- backward / objective interposition (Slice 12) ---------------------------

def _install_backward_hook(torch) -> None:
    """Tee the two ways a gradient is taken: Tensor.backward (the common path) and autograd.grad
    (the per-task / MGDA path blind backdoors use). Both build the run's Objective and count as one
    backward pass toward the per-step total."""
    _patch_method(torch.Tensor, "backward", _wrap_backward)
    _patch_callable_everywhere(torch.autograd, "grad", _wrap_autograd_grad)


def _wrap_backward(orig):
    @functools.wraps(orig)
    def backward(self, *args, **kwargs):
        try:
            _on_backward(self, is_objective=True)
        except Exception:
            pass  # capture must never break the training it observes
        return orig(self, *args, **kwargs)

    setattr(backward, _WRAPPED, True)
    return backward


def _wrap_autograd_grad(orig):
    @functools.wraps(orig)
    def grad(outputs, *args, **kwargs):
        try:
            _on_backward(outputs, is_objective=False)  # an auxiliary grad probe (MGDA): count, don't define
        except Exception:
            pass
        return orig(outputs, *args, **kwargs)

    setattr(grad, _WRAPPED, True)
    return grad


def _on_backward(loss, *, is_objective: bool) -> None:
    """A backward pass fired: count it and note where it was invoked (so MGDA's extra passes, which can
    come from injected code rather than the train loop, are attributable). The Objective is built from
    the first Tensor.backward -- the loss the optimizer descends -- not from autograd.grad probes on
    sub-losses (which is_objective=False marks)."""
    global _backward_events
    _backward_events += 1
    site = _call_site()
    if site:
        _step_backward_call_sites.append(site[0])  # innermost user frame: the line that ran this backward
    if is_objective:
        _note_objective(loss)


def _note_objective(loss) -> None:
    """Re-fingerprint the objective on EVERY backward, forking a new Objective entity when the loss
    graph changes shape. Fingerprinting once per run would miss a loss that only turns malicious partway
    through training -- the duty-cycled case Bagdasaryan & Shmatikov S7.2 says a defender has to check
    every iteration, and the shape their own threshold T produces. The walk is bounded (_graph_ops,
    ~12us), and an unchanged hash returns on a string compare, so a clean run still yields exactly one
    Objective node and an unchanged projection. A fork is attributed at epoch granularity: the epoch's
    Used edge is wired at epoch open, so one that happens mid-epoch is edged to the NEXT epoch."""
    global _objective
    grad_fn = getattr(loss, "grad_fn", None)
    if grad_fn is None:
        return                                    # no graph (requires_grad off) -- nothing to identify
    ops = _graph_ops(grad_fn)
    key = hash_canonical(ops)
    if _objective is not None and _objective.content_hash == key:
        return                                    # the common path: same objective as the last backward
    node = _objective_seen.get(key) or _record_objective(loss, ops=ops)
    if node is not None:
        _objective_seen[key] = node
        _objective = node


def _fold_backward_step() -> None:
    """At each optimizer step, roll this step's backwards into the open epoch: keep the largest
    per-step count (>1 = multi-objective / MGDA) and tally the passes per call site. Then reset the
    per-step buffers. Called after the epoch is open, so _train_activity is set."""
    global _backward_events, _step_backward_max
    if _backward_events > _step_backward_max:
        _step_backward_max = _backward_events
    _train_activity.attributes["backward_per_step"] = _step_backward_max
    sites = _train_activity.attributes["backward_call_site"]
    for s in _step_backward_call_sites:
        sites[s] = sites.get(s, 0) + 1
    _backward_events = 0
    _step_backward_call_sites.clear()


def _record_objective(loss, ops: Optional[list] = None) -> Optional["Objective"]:
    """Identity of the objective from the loss tensor's autograd graph: the terminal op + a
    fingerprint of the op-type structure. Read off the graph, so it doesn't matter where the loss
    was computed. None if the loss carries no grad_fn (e.g. requires_grad off). `ops` lets the
    per-backward check pass the walk it already did rather than repeating it."""
    grad_fn = getattr(loss, "grad_fn", None)
    if grad_fn is None:
        return None
    if ops is None:
        ops = _graph_ops(grad_fn)
    node = Objective(
        content_hash=hash_canonical(ops),
        terminal_op=type(grad_fn).__name__,
        graph_fingerprint=hash_canonical(ops),
        # the distinct op-type names in the loss graph -- graph_fingerprint hashes these away, but the
        # names are what tell a reconstruction term (L1/MSE over pixels) from a relabel term (CE over
        # labels); exposed so the op composition is legible, not just a fork hash.
        # call_site = where .backward() was driven from (the train loop, same clean or attacked);
        # loss_call_site = where the loss VALUE was computed (injected loss code resolves off the loop).
        attributes={"call_site": _call_site(), "loss_call_site": _loss_call_site,
                    "loss_ops": sorted({name for _, name in ops})})
    return _session.recorder.entity(node)


def _graph_ops(grad_fn, max_depth: int = 8, max_nodes: int = 256) -> list:
    """The loss autograd graph as a bounded list of (depth, op-name), walked breadth-first from the
    terminal op. Bounds keep a deep graph cheap; the shape is what forks the Objective (one loss path
    vs several merging at an Add)."""
    out: list = []
    frontier = [(grad_fn, 0)]
    while frontier and len(out) < max_nodes:
        fn, depth = frontier.pop(0)
        if fn is None or depth > max_depth:
            continue
        out.append([depth, type(fn).__name__])
        for nxt, _ in getattr(fn, "next_functions", ()):  # (next_fn, input_index) pairs
            frontier.append((nxt, depth + 1))
    return out


# ---- metric interposition (Slice 7) ------------------------------------------

# sklearn.metrics functions that return a single scalar score. Array/curve returners
# (confusion_matrix, roc_curve, precision_recall_curve, classification_report) are left
# unpatched -- they aren't a metric value -- and the scalar guard skips them anyway.
_SKLEARN_METRICS = (
    "f1_score", "accuracy_score", "roc_auc_score", "precision_score", "recall_score",
    "average_precision_score", "log_loss",
)

# torch.nn.functional loss fns -- the shared bottom every nn loss module (and functional call)
# reaches: BCEWithLogitsLoss.forward -> binary_cross_entropy_with_logits, etc. Patching here catches
# the loss VALUE on every split (train/val/test), even under no_grad where the Objective can't form.
_LOSS_FNS = (
    "binary_cross_entropy_with_logits", "binary_cross_entropy", "cross_entropy", "nll_loss",
    "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "kl_div",
)


def _install_metric_hooks() -> None:
    """Patch the metric calls everyone uses so their return values become EvaluationResults.
    sklearn.metrics scalar fns + torchmetrics.Metric.compute (base class -> every subclass) +
    torch.nn.functional loss fns. All patched with _patch_callable_everywhere: the names are usually
    bound (`from sklearn.metrics import f1_score`) at MODULE-IMPORT time, before init() runs, so a
    plain module-attribute patch would miss them -- we rebind every already-imported reference."""
    try:
        import sklearn.metrics as skm
    except Exception:
        skm = None
    if skm is not None:
        for name in _SKLEARN_METRICS:
            _patch_callable_everywhere(skm, name, _wrap_metric_fn)
    try:
        import torch.nn.functional as F
        for name in _LOSS_FNS:
            _patch_callable_everywhere(F, name, _wrap_loss_fn)
    except Exception:
        pass
    try:
        import torchmetrics
    except Exception:
        return  # torchmetrics absent -> class-based metrics simply never fire
    _patch_method(torchmetrics.Metric, "compute", _wrap_tm_compute)


def _wrap_metric_fn(orig):
    """sklearn metric: run it, attribute its scalar return to the open eval episode. Named by the
    function (f1_score, roc_auc_score, ...); the `average` kwarg disambiguates micro vs macro."""
    @functools.wraps(orig)
    def metric(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            _on_metric(orig.__name__, result, kwargs.get("average"))
        except Exception:
            pass  # capture must never break the metric it observes
        return result

    setattr(metric, _WRAPPED, True)
    return metric


def _wrap_tm_compute(orig):
    """torchmetrics: Metric.compute() returns a tensor; name by the metric class (MulticlassF1Score
    ...). Non-scalar tensors (per-class vectors) are skipped by the scalar guard in _on_metric."""
    @functools.wraps(orig)
    def compute(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        try:
            val = result.item() if hasattr(result, "item") else result
            _on_metric(type(self).__name__, val, None)
        except Exception:
            pass
        return result

    setattr(compute, _WRAPPED, True)
    return compute


def _metric_key(name: str, average) -> str:
    """A stable key for one metric inside the episode's dict. `average` (sklearn's micro/macro/...)
    splits f1_score etc. into distinct entries; a numeric suffix breaks any remaining collision
    (e.g. per_class_auroc calling roc_auc_score once per class)."""
    key = f"{name}_{average}" if isinstance(average, str) else name
    if key not in _eval_metrics:
        return key
    i = 2
    while f"{key}_{i}" in _eval_metrics:
        i += 1
    return f"{key}_{i}"


def _wrap_loss_fn(orig):
    """torch.nn.functional loss: run it, fold its scalar into the open episode's per-batch loss mean.
    Named by the functional (binary_cross_entropy_with_logits, ...). reduction='none' returns a
    non-scalar and is skipped by the guard in _on_loss -- it isn't a single loss value."""
    @functools.wraps(orig)
    def loss_fn(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            _on_loss(orig.__name__, result)
            _check_loss_target(args, kwargs)
        except Exception:
            pass  # capture must never break the training it observes
        return result

    setattr(loss_fn, _WRAPPED, True)
    return loss_fn


def _check_loss_target(args, kwargs) -> None:
    """A loss was computed against some target. If that target isn't one the loader produced, an
    attacker-chosen label was optimized against (blind_backdoor relabels a triggered copy to the
    backdoor class while leaving the real batch's labels alone). Target-side mirror of the
    _offloader_forwards input-side check; every F.* loss takes the target as its 2nd positional arg.
    Only meaningful while the epoch-1 tee keeps _loader_content current, and only inside a training
    step: _forward_events>=1 means a train-mode forward already ran this step (val forwards don't
    increment it), so val-loss targets against the stale train tee aren't counted."""
    global _loss_targets_offloader
    if not _fwd_active or _forward_events < 1:
        return
    target = kwargs.get("target")
    if target is None and len(args) > 1:
        target = args[1]
    if target is None:
        return
    seen: set = set()
    _collect_tensor_hashes(target, seen)
    if seen and not (seen & _loader_content):
        _loss_targets_offloader += 1


def _on_loss(name: str, value) -> None:
    """A loss was computed (once per batch). Keep a running sum+count per loss name so the episode
    carries the epoch's MEAN loss, mirroring the avg_loss users log. A loss can open the episode
    (pinning only the split -- epoch/informed_by are left for the end-of-epoch metrics, which see the
    correct epoch); if metrics never come, they stay None."""
    global _eval_manifest, _eval_call_site, _loss_call_site
    if _session is None:
        return
    t = getattr(value, "numel", None)
    if t is None or value.numel() != 1:   # non-scalar (reduction='none') or not a tensor
        _note_persample_loss(value)       # Slice 14: a per-sample loss extraction, not one loss value
        return
    v = _as_float(value.item())
    if v is None:
        return
    # Where this scalar loss was COMPUTED. The objective (first train backward) reads the most recent
    # value: for a forked loss it resolves into the injected loss code (attacks/*.py), for a clean run
    # into the trainer's loss_fn call. A generic code-origin fact, not gated on any attack.
    _loss_call_site = _call_site()
    if not _eval_metrics and not _eval_losses:   # a loss opened the episode: pin the split now
        _eval_manifest = _eval_split_manifest(_last_pass_dataset)
        _eval_call_site = _call_site()
    # deliberately NO _open_eval_span() here. A per-batch training loss would anchor the episode to
    # the TRAIN pass, giving the epoch's mean-loss node a window that straddles two TrainingSteps and
    # steals their io. The mean loss is a value RECORDED DURING the epoch; the epoch's window already
    # covers that time. Only _on_metric -- a real scored pass -- earns a span.
    s = _eval_losses.get(name)
    if s is None:
        _eval_losses[name] = [v, 1]
    else:
        s[0] += v
        s[1] += 1


def _note_persample_loss(value) -> None:
    """Slice 14: a non-reduced loss (reduction='none') was computed = per-sample loss extraction. Count
    consecutive such calls since the last optimizer.step(); _on_step resets the run each step, so a run
    that grows large is a scoring pass with no training between (the loop probes the model per-sample).
    Benign focal/OHEM losses reduce+backward every step, so their run never exceeds ~1 -- the discriminator
    is the consecutive length, not the mere presence of a per-sample loss."""
    global _persample_run, _persample_samples
    if getattr(value, "numel", None) is None:
        return
    _persample_run += 1
    try:
        _persample_samples += int(value.shape[0])
    except (AttributeError, IndexError, TypeError):
        _persample_samples += int(value.numel())


def _on_metric(name: str, value, average) -> None:
    """A metric returned. Scalar values pile into the open eval episode (opening one if needed):
    the episode pins the epoch + the scored split + the call site at its FIRST metric, then
    accumulates until a new data pass or process exit flushes it into one EvaluationResult. The
    data pass is the sole boundary on purpose -- every metric read off ONE forward pass (aggregate
    scores AND a per-class helper reusing the same logits) is the same evaluation and belongs in
    one node, so a change of computing function no longer forks (that double-counted a single pass).
    _metric_key suffixes any name collision the merge produces (roc_auc_score, roc_auc_score_2 ...)."""
    global _eval_manifest, _eval_epoch, _eval_informed_by, _eval_call_site
    if _session is None:
        return
    v = _as_float(value)
    if v is None:           # array/curve/None return -> not a scalar metric, skip
        return
    if not _eval_metrics:   # first metric of a new episode: snapshot its context
        _eval_manifest = _eval_split_manifest(_last_pass_dataset)
        _eval_epoch = _epoch if _epoch > 0 else None
        _eval_informed_by = _train_activity.id if _train_activity is not None else None
        _eval_call_site = _call_site()
    _open_eval_span()
    _eval_metrics[_metric_key(name, average)] = v


def _eval_split_manifest(dataset) -> Optional["SplitManifest"]:
    """The SplitManifest for the split a metric scored: hash the just-iterated loader's dataset.df
    (same df.to_csv scheme as Slice 3, so it dedups with that split's node). The name is inherited
    when the hash matches a manifest an earlier slice labeled (else None -- the val/test naming
    heuristic is deferred). df-less datasets aren't hashed here (would materialize images)."""
    df = getattr(dataset, "df", None)
    if df is None:
        return None
    try:
        h = hash_bytes(df.to_csv(index=False).encode())
    except Exception:
        return None
    return _register_manifest(
        SplitManifest(content_hash=h, split=_manifest_by_hash.get(h), num_samples=len(df),
                      split_evidence=_manifest_evidence_by_hash.get(h)))


def _flush_eval() -> None:
    """Emit the open eval episode as one Evaluation activity + one EvaluationResult entity, then
    reset. EvaluationResult is content-addressed over the COMPLETE metrics dict, so it's minted
    once here, not per metric. informed_by the producing epoch + chain=False, so eval never moves
    the training spine (mirrors CheckpointWrite)."""
    global _eval_metrics, _eval_manifest, _eval_epoch, _eval_informed_by, _eval_call_site
    global _eval_losses, _eval_span
    if not _eval_metrics and not _eval_losses:
        return
    metrics, manifest = dict(_eval_metrics), _eval_manifest
    for name, (tot, cnt) in _eval_losses.items():   # per-batch losses -> the epoch's mean
        metrics[name] = tot / cnt if cnt else tot
    split = getattr(manifest, "split", None) if manifest is not None else None
    # reset before emitting (capture must be reentrant-safe)
    _eval_metrics, _eval_manifest, _eval_losses = {}, None, {}
    span, _eval_span = _eval_span, None
    result = EvaluationResult(
        content_hash=hash_canonical({"metrics": metrics, "split": split}),
        split=split, metrics=metrics)
    inputs = [manifest] if manifest is not None else []
    with _session.recorder.activity(
            Evaluation(activity_id="", split=split,
                       attributes={"epoch": _eval_epoch, "call_site": _eval_call_site}),
            inputs=inputs, config=_intent, derivation=DerivationType.TRAINING,
            informed_by=_eval_informed_by, chain=False, observed_span=span) as scope:
        scope.generated(result)
    _eval_epoch = _eval_informed_by = None
    _eval_call_site = []


# ---- entity derivation -------------------------------------------------------

def _record_hyperparams(optimizer) -> HyperparameterSet:
    """Live hyperparameters off param_groups[0]. lr/weight_decay/optimizer/batch_size are
    top-level (batch_size from the DataLoader hook); optimizer-specific keys (momentum,
    betas, eps, ...) go in attributes. epochs / seed stay None (can't see them yet)."""
    param_group = optimizer.param_groups[0] if optimizer.param_groups else {}
    lr = _as_float(param_group.get("lr"))
    wd = _as_float(param_group.get("weight_decay"))
    name = type(optimizer).__name__
    bs = _pending_batch_size
    extra = {k: _canon(v) for k, v in param_group.items()
             if k not in ("params", "lr", "weight_decay")}
    hp = HyperparameterSet(
        content_hash=hash_canonical(
            {"learning_rate": lr, "weight_decay": wd, "optimizer": name,
             "batch_size": bs, "extra": extra}),
        learning_rate=lr, weight_decay=wd, optimizer=name, batch_size=bs, attributes=extra)
    return _session.recorder.entity(hp)


def _record_arch(optimizer) -> ModelArchitecture:
    """Architecture identity. content_hash spans the optimized-tensor shapes AND, when the live model was
    grabbed at the first forward (_model_ref), a module-graph fingerprint -- module types by name, buffer
    names+shapes, and per-module forward-pre-hook counts. The graph is what catches non-parametric edits a
    shape-only fingerprint misses: an architectural backdoor swaps a layer for one with no new params
    (a type change at a fixed name), adds trigger state as buffers, and wires an extra input edge as a
    forward-pre-hook. Still records the param-shape multiset in _param_shapes (Slice 5 checkpoint identity),
    which stays shapes-only so a saved state_dict keeps matching."""
    global _param_shapes
    shapes, n_params = [], 0
    for grp in optimizer.param_groups:
        for p in grp.get("params", []):
            shapes.append(list(p.shape))
            n_params += int(p.numel())
    _param_shapes = {}
    for s in shapes:
        key = tuple(s)
        _param_shapes[key] = _param_shapes.get(key, 0) + 1
    graph = _module_graph(_model_ref)
    # content_hash spans structure only: param shapes + arch_name + module graph + buffers. The
    # forward-pre-hook count is EXPOSED as an attribute but NOT folded in -- a benign read-only
    # input-grabbing hook (blind/memory backdoors attach one) is a runtime attachment, not a
    # structural change, so architecture_unchanged must not fork on it. An arch backdoor is still
    # caught: it changes module types + buffers, which stay in the hash.
    if graph is None:
        payload = {"shapes": shapes}
    else:
        payload = {"shapes": shapes, "graph": {k: v for k, v in graph.items() if k != "forward_pre_hooks"}}
    attrs = {"fingerprint": "param-shapes" if graph is None else "param-shapes+module-graph",
             "named": graph is not None}
    if graph is not None:
        attrs["arch_name"] = graph["arch_name"]     # top-level module class, read off the live model
        attrs["modules"] = graph["modules"]         # (name, type) per submodule, so the diff names WHAT changed
        attrs["buffers"] = graph["buffers"]         # (name, shape) per buffer; non-weight trigger state surfaces here
        attrs["forward_pre_hooks"] = graph["forward_pre_hooks"]   # instance pre-hook count; a wired input edge shows up
    arch = ModelArchitecture(
        content_hash=hash_canonical(payload),
        num_parameters=n_params,
        attributes=attrs)
    return _session.recorder.entity(arch)


def _module_graph(model) -> Optional[dict]:
    """Structural fingerprint of the live model, catching non-parametric edits param-shapes miss. Best-
    effort: capture must never break training, so any failure yields None (falls back to shapes-only)."""
    if model is None:
        return None
    try:
        modules = [[name, type(m).__name__] for name, m in model.named_modules()]
        buffers = [[name, list(b.shape)] for name, b in model.named_buffers()]
        hooks = sum(len(getattr(m, "_forward_pre_hooks", {})) for _, m in model.named_modules())
        return {"arch_name": type(model).__name__, "modules": modules, "buffers": buffers,
                "forward_pre_hooks": hooks}
    except Exception:
        return None


def _record_split(dataset) -> Optional["SplitManifest"]:
    """Slice 3 (Option A): identity of the after-split partition the model trains on, read from
    the dataset's loaded label table (loader.dataset.df). split="train" by construction -- the
    claimed pass IS the train loader. content_hash = hash of the table's csv form (CONTENT, not
    file bytes -> won't match the explicit hash_file SplitManifest; Slice 6 hashes make_splits'
    output the same way to reconnect). Works for any dataset: only the content hash + row count,
    NO dataset-specific columns (e.g. patient id). Assumes the dataset exposes a pandas .df, the
    structural assumption Option A accepts."""
    df = getattr(dataset, "df", None)
    if df is None:
        return None
    try:
        csv_text = df.to_csv(index=False)
        node = SplitManifest(
            content_hash=hash_bytes(csv_text.encode()),
            split="train", num_samples=len(df), split_evidence="gradient_update")
    except Exception:
        return None
    return _register_manifest(node)


def _record_dataset_identity(dataset) -> Optional["RawDataset"]:
    """Coarse base for the transform chain when the dataset has no content-hashable .df (e.g. an
    image dataset like CIFAR10). Identity = {class name, length} only -- no pixel materialization, so
    it's weaker than the .df SplitManifest (no cross-stage content join) but still forks if the
    dataset class or size changes, and gives Slice 4 a node for the TransformedDataset to derive from.
    The 'coarse' attribute marks it as the weak-identity path for downstream (diff) to read."""
    try:
        n = len(dataset) if hasattr(dataset, "__len__") else None
    except Exception:
        n = None
    cls = type(dataset).__name__
    return _session.recorder.entity(RawDataset(
        content_hash=hash_canonical({"dataset_class": cls, "num_samples": n}),
        num_files=n, attributes={"dataset_class": cls, "identity": "coarse"}))


def _record_transforms(base):
    """Slice 4: if the dataset carries a transform pipeline (a Compose or a single transform),
    emit it as the visible TransformOp chain producing one TransformedDataset derived from `base`
    (the split) -- that result, not the raw split, is what the model trains on. Returns `base`
    unchanged when there's no transform. seed=None (RNG not visible from this hook yet) and
    norm_node=None (the precomputed NormalizationStats lives in another process) -- Normalize then
    hashes on its own mean/std params, still forking on tamper. Reuses record_transform_chain as-is;
    the Compose comes off dataset.transform, the same Option-A read as _record_split."""
    compose = _pending_transform
    if compose is None:  # bare single transform vs Compose handled by record_transform_chain
        return base
    try:
        return record_transform_chain(
            _session.recorder, base, compose,
            seed=None, code=_session.code, env=_session.env, call_site=_call_site())
    except Exception:
        return base


# ---- image ingest / preprocessing interposition (Slice 15) -------------------

# Adapter rows -- (module, attr) per library entry point, so supporting a new reader is a row, not
# new plumbing. PIL is this pipeline; skimage.io + pydicom are what torchxrayvision (the open-source
# stand-in for a real chest-X-ray product) reads with; cv2 is the other common one. A library that
# isn't installed is skipped, so none of these become dependencies.
_IMAGE_READERS = (("skimage.io", "imread"), ("imageio", "imread"), ("imageio.v2", "imread"),
                  ("cv2", "imread"), ("pydicom", "dcmread"), ("pydicom.filereader", "dcmread"))
_IMAGE_WRITERS = (("skimage.io", "imsave"), ("imageio", "imwrite"), ("imageio.v2", "imwrite"),
                  ("cv2", "imwrite"))
_IMAGE_OPS = (("skimage.transform", "resize"), ("cv2", "resize"))


def _install_image_hooks() -> None:
    """Slice 15: the raw -> processed image boundary, the one stage no torch hook can see.

    Reads and writes are hashed as FILE BYTES, not in-memory pixels: that way the DatasetVersion a
    preprocessing stage writes hashes identical to a later run's read of the same files, so the two
    dedup into one node with a real edge -- the same cross-stage join the read_csv/to_csv slices use.
    In-memory pixel hashes would never meet. Emits nothing unless writes were seen (_flush_image_stage)."""
    for mod_path, attr in _IMAGE_READERS:
        _patch_image_fn(mod_path, attr, _wrap_image_read)
    for mod_path, attr in _IMAGE_WRITERS:
        _patch_image_fn(mod_path, attr, _wrap_image_write)
    for mod_path, attr in _IMAGE_OPS:
        _patch_image_fn(mod_path, attr, _wrap_image_op)
    try:  # PIL: open is a module fn, save/resize are Image methods
        from PIL import Image as pil_image
    except Exception:
        return
    _patch_module_fn(pil_image, "open", _wrap_image_read)
    _patch_method(pil_image.Image, "save", _wrap_image_write)
    _patch_method(pil_image.Image, "resize", _wrap_image_op)


def _patch_image_fn(mod_path: str, attr: str, wrap) -> None:
    """Patch one adapter row, skipping any library that isn't installed."""
    try:
        mod = importlib.import_module(mod_path)
    except Exception:
        return
    _patch_module_fn(mod, attr, wrap)


def _wrap_image_read(orig):
    @functools.wraps(orig)
    def reader(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            _on_image_file(_img_reads, args[0] if args else None)
        except Exception:
            pass  # capture must never break the load it observes
        return result

    setattr(reader, _WRAPPED, True)
    return reader


def _wrap_image_write(orig):
    @functools.wraps(orig)
    def writer(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:  # arg 0 is the path, except on the bound Image.save where self shifts it to arg 1
            dest = args[1] if _is_pil_image(args[0] if args else None) else (args[0] if args else None)
            _on_image_file(_img_writes, kwargs.get("fp", dest), record_site=True)
        except Exception:
            pass
        return result

    setattr(writer, _WRAPPED, True)
    return writer


def _wrap_image_op(orig):
    name = f"{getattr(orig, '__module__', '?')}.{getattr(orig, '__qualname__', orig.__name__)}"

    @functools.wraps(orig)
    def op(*args, **kwargs):
        try:
            _on_image_op(name, args, kwargs)
        except Exception:
            pass
        return orig(*args, **kwargs)

    setattr(op, _WRAPPED, True)
    return op


def _is_pil_image(obj) -> bool:
    return type(obj).__module__.startswith("PIL.") and hasattr(obj, "save")


def _on_image_file(bucket: dict, src, record_site: bool = False) -> None:
    """Fingerprint one image file the stage just read or wrote. Keyed by basename so the raw and
    processed sides line up per image; a re-read of the same path overwrites, so the last state
    on disk is what the manifest reports."""
    global _img_call_site, _reading_file_hash, _img_span
    if not isinstance(src, (str, os.PathLike)):
        return  # buffers / file objects / arrays: no file identity to record
    path = os.fspath(src)
    _img_span = _observed_span(_img_span)
    h = hash_file(path)
    base = os.path.basename(path)
    bucket[base] = h
    # The manifest is basename-keyed (raw and processed line up per image), so the full path is
    # dropped here -- and it is the only thing that can fuse these manifest-identity entities to an
    # OS tracer's file events. Keep it beside the hash; dicts preserve insertion = write order.
    (_img_read_paths if bucket is _img_reads else _img_write_paths)[base] = os.path.abspath(path)
    # a read inside an observed __getitem__ is the file that idx consumed: hand its byte hash to
    # _record_read so the run's file-byte read-set can be subset-matched to the DatasetVersion store.
    if bucket is _img_reads and _sample_active:
        _reading_file_hash = h
    if record_site and _img_call_site is None:
        _img_call_site = _call_site()


_INTERP_KEY = {"PIL.": "resample", "cv2.": "interpolation", "skimage.": "order"}


def _on_image_op(name: str, args, kwargs) -> None:
    """Record a resize-style op and its shape/interpolation params. Params are the attack surface
    here (an interpolation swap silently rewrites every image), so they are captured per op name;
    the last call wins, since a preprocessing loop applies the same op to every file. Interpolation
    is read positionally as well as by keyword -- PIL and cv2 are both routinely called that way,
    and missing it would drop the very field this hook exists to watch."""
    global _img_resolution, _img_span
    _img_span = _observed_span(_img_span)
    rest = args[1:]  # arg 0 is the image (or self)
    target = next((a for a in rest if isinstance(a, (int, tuple, list))), None)
    params = {"target": list(target) if isinstance(target, (tuple, list)) else target}
    key = next((v for p, v in _INTERP_KEY.items() if name.startswith(p)), None)
    if key and key not in kwargs:
        pos = next((a for a in rest if a is not target and isinstance(a, int)), None)
        if pos is not None:
            kwargs = {**kwargs, key: pos}
    for k, v in kwargs.items():
        decoded = _img_param(_interp_name(name, k, v))
        if decoded is not None:
            params[k] = decoded
    _img_ops[name] = params
    dims = [d for d in (target if isinstance(target, (tuple, list)) else [target]) if isinstance(d, int)]
    if dims:
        _img_resolution = max(dims)


def _interp_name(op: str, key: str, v):
    """Resolve an interpolation constant to its name. PIL and cv2 expose these as plain ints, not
    enums, so a captured 'resample: 2' is unreadable in a diff -- and this is precisely the param an
    interpolation swap changes. Falls back to the raw value for anything unrecognized."""
    if not isinstance(v, int) or isinstance(v, bool):
        return v
    try:
        if key == "resample" and op.startswith("PIL."):
            from PIL import Image as pil_image
            return pil_image.Resampling(v).name
        if key == "interpolation" and op.startswith("cv2."):
            import cv2
            return next((n for n in dir(cv2) if n.startswith("INTER_") and getattr(cv2, n) == v), v)
    except Exception:
        pass
    return v


def _img_param(v):
    """Scalar-ish params only -- an array is dropped. Enums stringify BEFORE the int check: PIL's
    resample is an IntEnum, and 'Resampling.BILINEAR' is what makes an interpolation swap legible
    in a diff, where a bare 2 is not."""
    if v is None or isinstance(v, (bool, float, str)):
        return v
    if hasattr(v, "name") and hasattr(v, "value"):  # enum (PIL Resampling, cv2 interpolation flags)
        return str(v)
    return v if isinstance(v, int) else None


def _write_members_sidecar(node_id: str, hashes) -> None:
    """Persist a node's file-byte-hash membership set beside the ledger (one hash per line). Too big to
    inline as a node attribute at dataset scale (100k+ files), so the graph loader reads it back to run
    the DatasetVersion<->SampleManifest subset join. Best-effort: a failure here never breaks a run."""
    members = sorted(set(hashes))
    if not members:
        return
    try:
        d = _session.log_path.parent / "_dataset_members"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{node_id}.txt").write_text("\n".join(members))
    except Exception:
        pass


def _write_paths_sidecar(node_id: str, bucket: dict, paths: dict) -> None:
    """Persist `<content_hash>\\t<abspath>` per file, in OBSERVED ORDER, beside the ledger. The
    companion to _write_members_sidecar, which is hash-SORTED and therefore cannot be aligned against
    anything: a manifest-identity entity (RawDataset, DatasetVersion) is one node over thousands of
    files and carries no path, so this file is what lets a cross-layer join match our content to an
    OS tracer's paths. Best-effort; a failure here never breaks a run."""
    if not bucket:
        return
    try:
        d = _session.log_path.parent / "_dataset_paths"
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"{h}\t{paths[b]}" for b, h in bucket.items() if b in paths]
        if lines:
            (d / f"{node_id}.tsv").write_text("\n".join(lines))
    except Exception:
        pass


def _flush_image_stage() -> None:
    """Emit the raw -> processed chain, but ONLY if this stage WROTE images: a stage that materializes
    images is a preprocessing stage by definition, whatever its script happens to be called. Gating on
    writes rather than on a stage name keeps it convention-agnostic. A read-only stage (a training run
    whose Dataset decodes in __getitem__, e.g. torchxrayvision) is left to Slice 10's SampleManifest,
    which already fingerprints exactly the samples training consumed."""
    if not _img_writes:
        return
    raw = _session.recorder.entity(RawDataset(
        content_hash=hash_canonical(sorted(_img_reads.items())),
        num_files=len(_img_reads),
        attributes={"identity": "file-manifest"})) if _img_reads else None
    ops = sorted(_img_ops.items())
    config = [n for n in (_session.code, _session.env) if n is not None]
    with _session.recorder.activity(
            PreprocessingOp(activity_id="",
                            operation="->".join(n for n, _ in ops) or None,
                            attributes={"ops": {n: p for n, p in ops}, "call_site": _img_call_site}),
            inputs=[raw] if raw is not None else [], config=config,
            derivation=DerivationType.PREPROCESSING, observed_span=_img_span) as scope:
        dv = scope.generated(DatasetVersion(
            content_hash=hash_canonical(sorted(_img_writes.items())),
            num_files=len(_img_writes), resolution=_img_resolution))
    # publish the store's file-byte set so a later read stage's SampleManifest can subset-match into it
    _write_members_sidecar(dv.id, _img_writes.values())
    # ...and the hash->path map in write order, which is what a cross-layer join needs (see the
    # sidecar's docstring). Both ends: the processed store AND the raw files it was built from.
    _write_paths_sidecar(dv.id, _img_writes, _img_write_paths)
    if raw is not None:
        _write_paths_sidecar(raw.id, _img_reads, _img_read_paths)


# ---- teardown ----------------------------------------------------------------

def _finalize() -> None:
    """Close the open epoch and the ledger. Registered via atexit so the one-line init()
    needs no enclosing `with`."""
    try:
        _flush_eval()   # Slice 7: the final eval episode (e.g. the test pass) has no later pass to flush it
    except Exception:
        pass
    try:
        _flush_splits()  # Slice 9: emit the make_splits batch as one DataSplit
    except Exception:
        pass
    try:
        _flush_image_stage()  # Slice 15: raw -> processed chain, if this stage materialized images
    except Exception:
        pass
    _close_epoch()
    _remove_forward_check()   # Slice 13: drop the global forward-pre-hook if the run ended in epoch 1
    try:
        _emit_sample_manifest()  # Slice 10 (Option A): merge observed reads (+ worker sidecars) -> manifest
    except Exception:
        pass
    if _session_cm is not None:
        try:
            _session_cm.__exit__(None, None, None)
        except Exception:
            pass


def restore() -> None:
    """Undo the patches and reset state (for tests / repeated in-process runs). Does not
    reopen the ledger."""
    global _installed, _stage, _hp, _arch, _intent, _split_node, _train_input, _epoch, _epoch_steps
    global _pending_pass, _pending_batch_size, _pending_dataset, _pending_transform
    global _train_cm, _train_activity, _last_pass_dataset, _eval_span, _last_pass_start_ns
    global _train_scope, _last_step_ns
    global _eval_metrics, _eval_manifest, _eval_epoch, _eval_informed_by, _eval_call_site
    global _eval_losses
    global _last_read, _split_writes, _stage_frames, _split_source, _split_call_site, _param_shapes, _lineage_done
    global _split_span
    global _sample_manifest, _current_order, _epoch_order_ref, _pending_sampler
    global _objective, _objective_seen, _backward_events, _step_backward_max
    global _persample_run, _persample_samples, _scan_samples_max
    global _capture_dataset, _capture_transform_orig, _reading_idx, _reading_file_hash
    global _sample_reads, _read_files, _seen_idx
    global _sample_active, _sample_armed, _sidecar_dir, _sidecar_fh, _train_pass_seen
    global _fwd_active, _fwd_expect, _loader_content, _fwd_unmatched, _offloader_forwards
    global _loss_targets_offloader, _loader_batch, _loss_call_site, _loader_origin
    global _alter_row_frac_max, _alter_region_frac_max, _alter_share_class
    global _fwd_count_handles, _fwd_depth, _forward_events, _step_forward_max, _model_ref
    global _img_reads, _img_writes, _img_ops, _img_resolution, _img_call_site, _img_span
    global _img_read_paths, _img_write_paths
    _img_reads, _img_writes, _img_ops = {}, {}, {}
    _img_read_paths, _img_write_paths = {}, {}
    _img_resolution = _img_call_site = _img_span = None
    _remove_forward_check()   # Slice 13: unregister the global forward-pre-hook
    _fwd_active = _fwd_expect = False
    _loader_content = set()
    _fwd_unmatched = 0
    _offloader_forwards = 0
    _loss_targets_offloader = 0
    _loss_call_site = []
    _loader_origin = []
    _loader_batch = None
    _alter_row_frac_max = _alter_region_frac_max = 0.0
    _alter_share_class = False
    for h in _fwd_count_handles:   # Slice 14: unregister the counting pair
        try:
            h.remove()
        except Exception:
            pass
    _fwd_count_handles = []
    _fwd_depth = _forward_events = _step_forward_max = 0
    _model_ref = None
    if _capture_dataset is not None and _capture_transform_orig is not None:
        try:
            _capture_dataset.transform = _capture_transform_orig  # undo the tee
        except Exception:
            pass
    if _sidecar_fh is not None:
        try:
            _sidecar_fh.close()
        except Exception:
            pass
    for (cls, name), fn in _originals.items():
        setattr(cls, name, fn)
    _originals.clear()
    _installed = False
    _stage = None
    _hp = _arch = _split_node = _train_input = None
    _param_shapes = {}
    _lineage_done = False
    _intent = []
    _epoch = _epoch_steps = 0
    _pending_pass = False
    _pending_batch_size = None
    _pending_dataset = None
    _pending_transform = None
    _train_cm = _train_activity = _train_scope = _last_step_ns = None
    _last_pass_dataset = None
    _manifest_by_hash.clear()
    _manifest_path_by_hash.clear()
    _eval_metrics = {}
    _eval_span = _last_pass_start_ns = None
    _eval_manifest = _eval_epoch = _eval_informed_by = None
    _eval_call_site = []
    _eval_losses = {}
    _last_read = _split_source = _split_span = None
    _split_writes = []
    _stage_frames = []
    _manifest_evidence_by_hash.clear()
    _split_call_site = []
    _sample_manifest = None
    _capture_dataset = _capture_transform_orig = _reading_idx = _reading_file_hash = None
    _sample_reads = {}
    _read_files = set()
    _seen_idx = set()
    _sample_active = _sample_armed = False
    _sidecar_dir = _sidecar_fh = None
    _train_pass_seen = 0
    _current_order = _epoch_order_ref = _pending_sampler = None
    _objective = None
    _objective_seen = {}
    _backward_events = _step_backward_max = 0
    _persample_run = _persample_samples = _scan_samples_max = 0
    _step_backward_call_sites.clear()


# ---- helpers -----------------------------------------------------------------

def _patch_module_fn(module, name: str, wrap) -> None:
    """Replace a module-level function with `wrap(orig)`, saving the original for restore().
    Safe to run twice via the _WRAPPED marker; no-op if the attribute is missing."""
    fn = getattr(module, name, None)
    if fn is None or getattr(fn, _WRAPPED, False):
        return
    _originals[(module, name)] = fn
    setattr(module, name, wrap(fn))


def _patch_method(cls, attr: str, wrap) -> None:
    """Replace a class method with `wrap(orig)`, saving the original for restore(). Reads the attr
    off the class (inherited is fine for a base like torchmetrics.Metric). _WRAPPED-guarded."""
    fn = getattr(cls, attr, None)
    if fn is None or getattr(fn, _WRAPPED, False):
        return
    _originals[(cls, attr)] = fn
    setattr(cls, attr, wrap(fn))


def _patch_callable_everywhere(home_module, name: str, wrap) -> None:
    """Patch a function AND every already-imported reference to it. A plain `setattr(home_module,
    name, ...)` can't reach a `from home_module import name` binding made before we ran (the name
    was copied into that module's namespace). So we wrap once, then walk sys.modules and rebind any
    attribute that still points at the original object. Order-independent; runs once at init().
    Each (module, attr) rebind is recorded in _originals so restore() undoes all of them."""
    orig = getattr(home_module, name, None)
    if orig is None or getattr(orig, _WRAPPED, False):
        return
    wrapped = wrap(orig)
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            members = vars(mod)
        except Exception:
            continue
        for attr, val in list(members.items()):
            if val is orig:
                try:
                    setattr(mod, attr, wrapped)
                    _originals[(mod, attr)] = orig
                except Exception:
                    pass


def _all_subclasses(cls) -> set:
    subs = set(cls.__subclasses__())
    for s in tuple(subs):
        subs |= _all_subclasses(s)
    return subs


def _call_site(limit: int = 3) -> list:
    """Short trace of the user-code frames that triggered the current hook: up to `limit`
    'file:line (func)' entries, innermost first. Skips our own package and library/stdlib
    frames so each event is named by the user's line, not torch's. [] when none is found
    (e.g. driven straight from a REPL)."""
    out: list = []
    frame = sys._getframe(1)  # start at the hook callback that called us
    while frame is not None and len(out) < limit:
        name = frame.f_code.co_filename
        if _is_user_frame(name):
            out.append(f"{os.path.basename(name)}:{frame.f_lineno} ({frame.f_code.co_name})")
        frame = frame.f_back
    return out


def _is_user_frame(filename: str) -> bool:
    """True for a frame in the user's own code: not our package, not an installed library or the
    stdlib, not a synthetic <...> frame (e.g. <stdin>, <frozen importlib...>)."""
    if not filename or filename.startswith("<"):
        return False
    path = os.path.abspath(filename)
    if path.startswith(_PKG_DIR):
        return False
    if os.sep + "site-packages" + os.sep in path or os.sep + "dist-packages" + os.sep in path:
        return False
    if _STDLIB_DIR and path.startswith(_STDLIB_DIR):
        return False
    return True


def _as_float(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def _canon(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
