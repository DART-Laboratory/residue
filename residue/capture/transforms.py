"""Introspect a torchvision `Compose` into ordered, content-hashed `TransformSpec` nodes without
hooking into core/preprocessing/transforms.py (the clean library). The script that builds the
pipeline hands the Compose here; we read each op's *applied* params straight off the object, so a
tampered transform (swapped interpolation, a sneaked-in flip, perturbed jitter) forks exactly that
op's node. See DESIGN and the transformation-attack-surface focus.

We read `vars(op)` rather than `repr(op)`: torchvision's repr omits fields like `interpolation` and
`fill` — the very image-scaling attack surface — so repr would hash blind to them.
"""

from __future__ import annotations

from typing import Any

from residue.capture.hashing import hash_canonical
from residue.schema.nodes import TransformedDataset, TransformSpec, TransformOp
from residue.schema.types import DerivationType

# nn.Module bookkeeping, not data-semantics params — drop so it never perturbs identity.
_DROP = {"training"}


def _canon(v: Any) -> Any:
    """Coerce a param value to a stable, JSON-canonicalizable form (enums/tuples -> str/list), so
    both the content hash and the stored attributes round-trip cleanly through from_dict."""
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)                      # InterpolationMode and friends


def _params(op: Any) -> dict[str, Any]:
    return {k: _canon(v) for k, v in vars(op).items()
            if k not in _DROP and not k.startswith("_")}


def _as_ops(transform: Any) -> list[Any]:
    """The ordered op list of a transform pipeline, duck-typed so we never import torchvision.
    A Compose-like object exposes `.transforms`; a bare single transform (e.g. ToTensor()) is
    its own one-op pipeline; anything else (None, non-callable) contributes nothing."""
    ops = getattr(transform, "transforms", None)
    if ops is not None:
        return list(ops)
    return [transform] if callable(transform) else []


def describe_pipeline(compose: Any) -> list[TransformSpec]:
    """One TransformSpec per op in the pipeline, in order. content_hash spans
    {operation, order, params} — order included so a reordering is itself a fork."""
    specs: list[TransformSpec] = []
    for i, op in enumerate(_as_ops(compose)):
        operation = type(op).__name__
        params = _params(op)
        content_hash = hash_canonical({"operation": operation, "order": i, "params": params})
        specs.append(TransformSpec(
            content_hash=content_hash, operation=operation, order=i, attributes=params))
    return specs


def record_transform_chain(rec, base, compose, *, seed, code, env, norm_node=None, call_site=None):
    """Emit the pipeline as a chain of visible TransformOp activities (RandomAffine -> ColorJitter
    -> ToTensor -> Normalize), wired into the run's temporal spine, and return the one content-
    addressed TransformedDataset they produce (the data the model consumes). The result's id spans
    {base, ordered op-hashes, seed}, so tampering ANY op forks it and everything trained on it.

    Normalize is special-cased: its op-hash is the existing NormalizationStats id (its tolerant 4dp
    hash), so a recomputed-stats node that agrees still dedups — see decision: norm attestation.
    Each op visibly reads/writes data only at the ends (first op uses `base`, last generates the
    result, so the result derives from `base`); the middle ops carry their params and sit on the
    spine. Returns `base` unchanged if the pipeline is empty. `call_site` (if given) is stamped on
    every op's attributes -- it's the line where the pipeline was observed, not where it was built."""
    specs = describe_pipeline(compose)
    if not specs:
        return base

    recipe = [
        (norm_node.id if (s.operation == "Normalize" and norm_node is not None) else s.content_hash)
        for s in specs
    ]
    result = TransformedDataset(
        content_hash=hash_canonical({"base": base.id, "recipe": recipe, "seed": seed}),
        base=base.id, seed=seed, num_transforms=len(specs))

    last = len(specs) - 1
    for i, spec in enumerate(specs):
        is_norm = spec.operation == "Normalize"
        attrs = dict(spec.attributes)
        if call_site is not None:
            attrs["call_site"] = call_site
        op = TransformOp(activity_id="", operation=spec.operation, seed=seed, attributes=attrs)
        cfg = ([norm_node] if (is_norm and norm_node is not None) else []) + [code, env]
        inputs = [base] if i in (0, last) else []      # data enters at the first op, exits at the last
        with rec.activity(op, inputs=inputs, config=cfg,
                          derivation=DerivationType.AUGMENTATION) as scope:
            if i == last:
                scope.generated(result)                # result wasDerivedFrom base (last op's input)
    return result
