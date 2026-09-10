"""Capture the execution environment as an EnvDigest entity.

We record the digest: pip-freeze hash, CUDA/python versions, user, device, 
container image — not the full pip freeze. The hash identifies drift, and the
stored versions explain it.

"""

import os
import subprocess
import sys

from residue._env import env as _env
from residue.capture.hashing import hash_bytes, hash_canonical
from residue.schema.nodes import EnvDigest


def _pip_freeze_hash() -> str | None:
    """Hash the sorted dependency set. Sorted so install-order churn isn't a false drift."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    lines = sorted(line.strip() for line in out.splitlines() if line.strip())
    return hash_bytes("\n".join(lines).encode())


def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.version.cuda  # None on a CPU-only build


def capture_env() -> EnvDigest:
    """Snapshot the current environment into an EnvDigest."""
    pip_hash = _pip_freeze_hash()
    cuda = _cuda_version()
    python = ".".join(map(str, sys.version_info[:3]))
    image = _env("IMAGE_HASH")  # set by the container runtime; None bare-metal/SLURM

    content_hash = hash_canonical({
        "pip_freeze_hash": pip_hash,
        "cuda_version": cuda,
        "python_version": python,
    })

    return EnvDigest(
        content_hash=content_hash,
        image_hash=image,        # recorded for forensics, not folded into identity
        pip_freeze_hash=pip_hash,
        cuda_version=cuda,
        python_version=python,
    )
