"""Bootstrap for the capture layer: wire one process into the run's ledger.

A run spans several processes (download -> split -> preprocess -> train, each its own
`python` invocation) but writes ONE hash-chained log. This module is what makes those
processes agree on the same ledger and identities, so the scripts never hand-wire the
sink/recorder/agents themselves.

Where the log lives, in precedence order (each env var also answers to its old PROV_ name):
  RESIDUE_LOG     full path to the run's JSONL. Set once by the orchestrator so every
                  process appends to the same ledger (the multi-process pipeline).
  run_dir arg     a caller's run directory -> <run_dir>/residue/residue.jsonl, so the
                  ledger sits beside that run's checkpoints/config/logs (solo `train.py`).
  default         a fresh outputs/residue/<stamp>/ ledger.
  RESIDUE_RUN_ID  optional override for the run id (else derived from the dir name).

Per-process the run id is suffixed with the `stage` ("train", "preprocess", ...) before it
reaches the Recorder, because the Recorder's activity counter resets each process: without
the suffix two stages would both mint `<run>:1`. Activity ids therefore read `<run>:train:1`;
diff.py masks the run id when aligning graphs (DESIGN 9.1), so the suffix is free.

Emits the run's Code + EnvDigest entities once and returns them so each hook can pass them
as `config` (role=INTENT) to its activity — that is what ties every step to the code/env
that produced it.
"""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from residue._env import env
from residue.capture.env import capture_env
from residue.capture.hashing import hash_canonical
from residue.capture.recorder import Recorder
from residue.schema.nodes import Code, Device, EnvDigest, SoftwareAgent, User
from residue.store.log import HashChainedLog

_DEFAULT_ROOT = Path("outputs/residue")


# ---- code identity (git commit + dirty) --------------------------------------

def _git(args: list[str], repo_root: str | Path) -> Optional[str]:
    """Run a git command best-effort; None if git/repo is unavailable."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def capture_code(repo_root: str | Path = ".") -> Code:
    """Identity of the running code. content_hash = hash of (commit, working-tree diff):
    a clean tree is its commit; a dirty tree folds in `git diff HEAD`, so two different
    uncommitted states get distinct ids (dirty=True is the tamper-relevant signal). Untracked
    files are flagged dirty but not hashed into identity (diff HEAD only covers tracked)."""
    commit = _git(["rev-parse", "HEAD"], repo_root)
    commit = commit.strip() if commit else None
    status = _git(["status", "--porcelain"], repo_root)
    dirty = bool(status.strip()) if status is not None else None
    diff = _git(["diff", "HEAD"], repo_root) if dirty else None
    content_hash = hash_canonical({"commit": commit, "diff": diff})
    return Code(content_hash=content_hash, path=str(repo_root), git_commit=commit, dirty=dirty)


# ---- agents (stable identities for the run) ----------------------------------

def _build_agents() -> list:
    try:
        username = getpass.getuser()
    except Exception:
        username = os.getenv("USER") or "unknown"

    host = socket.gethostname()
    gpu = None
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return [
        User(agent_id=f"user:{username}", name=username),
        Device(agent_id=f"device:{host}", hostname=host, gpu_model=gpu),
        SoftwareAgent(
            agent_id=f"software:pytorch:{torch_version}",
            framework="pytorch", version=torch_version,
        ),
    ]


# ---- run/ledger resolution ---------------------------------------------------

def _resolve(run_dir: Optional[str | Path]) -> tuple[Path, str]:
    """Pick the ledger path and run id (see module docstring for precedence)."""
    run_id_env = env("RUN_ID")

    env_log = env("LOG")
    if env_log:
        log_path = Path(env_log)
        return log_path, (run_id_env or log_path.parent.name)

    if run_dir is not None:
        run_dir = Path(run_dir)
        return run_dir / "residue" / "residue.jsonl", (run_id_env or run_dir.name)

    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = _DEFAULT_ROOT / f"{stamp}_{uuid.uuid4().hex[:8]}"
    return run_dir / "residue.jsonl", (run_id_env or run_dir.name)


# ---- the session -------------------------------------------------------------

@dataclass
class Session:
    recorder: Recorder
    code: Code         # pass as config to activities (role=INTENT)
    env: EnvDigest     # pass as config to activities (role=INTENT)
    run_id: str
    log_path: Path


@contextmanager
def residue_session(
    stage: str, *, run_dir: Optional[str | Path] = None, repo_root: str | Path = ".",
) -> Iterator[Session]:
    """Open this process's slice of the run ledger. `stage` namespaces activity ids and
    must be unique per process in a pipeline run ("preprocess", "split", "train", ...).
    `run_dir` places the ledger beside that run's other outputs (ignored if RESIDUE_LOG is set)."""
    log_path, run_id = _resolve(run_dir)

    log = HashChainedLog(log_path, label=f"{run_id}:{stage}")
    recorder = Recorder(sink=log, agents=_build_agents(), run_id=f"{run_id}:{stage}")

    code = recorder.entity(capture_code(repo_root))
    env = recorder.entity(capture_env())

    try:
        yield Session(recorder=recorder, code=code, env=env, run_id=run_id, log_path=log_path)
    finally:
        log.close()
