"""Content-addressing: turn an artifact into the hash that is its identity.

Three strategies, one per artifact class:
  single blob   checkpoint                 -> hash_file       (full file bytes)
  file tree     dataset directory          -> manifest_hash   (Merkle, localizing)
  parsed value  config / params / metrics  -> hash_canonical  (canonicalized content)

Two deliberately different sensitivities:
  - datasets hash *raw bytes* — we want to catch ANY change.
  - configs hash *canonicalized parsed content* (sorted keys, normalized types) so a
    reformat / comment edit is not a false positive, but a value change is.

This module is the only part of the schema/capture stack that reads disk. The hashes
it returns are passed into Entity(content_hash=...) at capture time; the schema layer
itself never touches the filesystem (keeps validate.py and Edge.key cheap).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

_ALGO = "sha256"
_CHUNK = 1 << 20  # reads 1 MiB chunks rather than an entire checkpoint at a time 


def hash_bytes(data: bytes) -> str:
    return hashlib.new(_ALGO, data).hexdigest()


def hash_file(path: str | Path) -> str:
    """Full content hash of a single blob (e.g. a checkpoint). Streamed, not loaded whole."""
    h = hashlib.new(_ALGO)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_canonical(obj: Any) -> str:
    """Hash of canonicalized parsed content: insensitive to key order / formatting,
    sensitive to values. For configs, hyperparameters, metrics dicts. Same canonical
    scheme as Edge.key, so identical values dedup to the same id across runs."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(blob.encode())


def manifest(root: str | Path, pattern: str = "**/*") -> dict[str, str]:
    """Walks every file under root and calls hash_file(p)
    
    Returns a dictionary mapping the relative path of the file to that specific file's hash.
    Example: {"images/cat.png": "a1b2c3...", "images/dog.png": "x9y8z7..."}
    """
    root = Path(root)
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    return {str(p.relative_to(root)): hash_file(p) for p in files}


def manifest_hash(root: str | Path, pattern: str = "**/*") -> tuple[dict[str, str], str]:
    """Calls manifest (getting the per-file map m), then runs hash_canonical(m) over that entire dictionary to produce one hash representing the whole dataset."""
    m = manifest(root, pattern)
    return m, hash_canonical(m)


def hash_norm_stats(mean, std, precision: int = 4) -> str:
    """Hashes the mean and std rounded to 4 decimals, so stats that match to that precision get the same id (and anything differing more forks)"""
    r = lambda v: [round(float(x), precision) for x in v]
    return hash_canonical({"mean": r(mean), "std": r(std)})
