"""Append-only, hash-chained event log — the tamper-evident ledger the Recorder writes to.

Each line is one event (a node or edge to_dict()) wrapped in a three-field envelope:

    {... event payload ..., "prev": <hash of prev line>, "hash": <hash of THIS line>, "mac": <tag>}

`hash` covers the payload AND `prev`, so every record commits to its predecessor: editing
or deleting any past line breaks the `prev` link of the next and every `hash` after it.

The chain alone only proves *internal consistency*: our threat model gives the adversary control
of the training code, so they own this process and could rebuild the whole chain from record 0,
or splice a forged record in and recompute every hash after it. `mac` closes that, using the
classic forward-secure audit log (Schneier & Kelsey, "Cryptographic Support for Secure Logs on
Untrusted Machines", USENIX Sec '98): a per-segment key K is drawn at open, each record is tagged
with HMAC(K, hash), and K is then evolved K <- SHA256(K) and its predecessor overwritten. An
adversary who takes over at record i holds only K_i, and the one-way evolution puts every earlier
key permanently out of reach — so they can append lies from the moment their code runs, but they
cannot alter or delete anything already recorded.

K_0 is written out of band at open (see emit_key) and NEVER stored in the ledger — a key the
adversary can read back is no key at all. The verifier needs that line; see verify_chain.

Implements the Recorder's EventSink protocol (just `append(event)`), so the Recorder never sees
the envelope: chaining is computed here, in the one writer chokepoint. The reader half
(strip_envelope) hands graph.py back the bare payload, so schema.from_dict() is none the wiser.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

from residue._env import env
from residue.capture.hashing import hash_canonical

_ENVELOPE = ("prev", "hash", "mac")  # fields this layer adds; stripped before schema.from_dict
_KEY_PREFIX = "RESIDUE-KEY"
_KEY_PREFIXES = (_KEY_PREFIX, "PROV-KEY")   # read both; ledgers predate the rename
_GENESIS = "-"                       # stands in for prev=None at the head of a fresh ledger


def _record_hash(record: dict) -> str:
    """Hash of a record with `prev` set but `hash`/`mac` not yet present. Same canonical
    scheme (sorted keys) as everywhere else, so it's reproducible across machines/runs."""
    return hash_canonical(record)


def _tag(key: bytes, record_hash: str) -> str:
    """MAC over the record hash — which already covers payload + prev, so tagging it tags all."""
    return hmac.new(key, record_hash.encode(), hashlib.sha256).hexdigest()


_key_sink_override: Optional[str] = None


def set_key_sink(path: Optional[str]) -> None:
    """Choose where K_0 goes, in-process. Deliberately not an env var: the code under capture
    inherits the environment and could read the path back out of it."""
    global _key_sink_override
    _key_sink_override = path


def default_key_path(run_id: str) -> Path:
    """~/.residue-keys/<run_id>.txt. Derived from the run id alone so every stage of one run
    appends to one file and the reporter can find it without being told where to look."""
    return Path.home() / ".residue-keys" / f"{run_id or 'unknown'}.txt"


def emit_key(label: str, start: str, key: bytes) -> None:
    """Hand K_0 to whoever launched the run, out of band -- never into the ledger, and never
    derived from the ledger path: a key sitting next to the log it authenticates protects nothing.

    Default is ~/.residue-keys/<run_id>.txt, which trades some secrecy for the keys surviving at
    all: a predictable path is one the training process can also compute, but a key nobody kept
    authenticates nothing. RESIDUE_KEY_SINK overrides the location, and "-" sends it to stderr --
    on a tty or pipe the job cannot read back or unwrite what it already flushed, which is the
    stronger channel when the property has to hold against the code under capture.
    """
    line = f"{_KEY_PREFIX} {label} {start} {key.hex()}"
    sink = _key_sink_override or env("KEY_SINK")
    if sink == "-":
        print(line, file=sys.stderr, flush=True)
        return
    path = Path(sink) if sink else default_key_path(label.split(":")[0])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"{_KEY_PREFIX} -> {path}", file=sys.stderr, flush=True)
    except OSError:                     # unwritable home / read-only fs: stderr beats losing it
        print(line, file=sys.stderr, flush=True)


class HashChainedLog:
    """Append-only sink over a single JSONL file. Safe to reopen across separate script
    runs: it reads the existing tail hash on open and keeps the chain going, so the whole
    pipeline (download -> ... -> train, each its own process) forms one continuous ledger.

    Each open starts a new MAC *segment* with its own K_0 (a pipeline of N processes therefore
    emits N key lines); the key line carries the chain hash the segment starts from, which is
    what lets the verifier tell where one segment's keys stop applying and the next's begin."""

    def __init__(self, path: str | Path, label: str = "-"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash: Optional[str] = self._read_last_hash()
        # mutable buffer so evolution can overwrite the old key in place; CPython gives no
        # guarantee against copies, so this is best-effort erasure, not a wipe.
        self._key = bytearray(os.urandom(32))
        emit_key(label, self._last_hash or _GENESIS, bytes(self._key))
        self._f = open(self.path, "a", encoding="utf-8")

    def _read_last_hash(self) -> Optional[str]:
        """Tail hash of an existing log, or None for a fresh/empty one. Reads the whole file;
        fine at our scale. A torn trailing line (crash mid-write) is not auto-repaired."""
        if not self.path.exists():
            return None
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if last is None:
            return None
        return json.loads(last)["hash"]

    def append(self, event: dict) -> None:
        """Wrap one event in the chain envelope and write it. flush() each line so a crash
        keeps every event that completed — the 'incomplete' activity story relies on this."""
        record = {**event, "prev": self._last_hash} # link to previous line's hash
        record["hash"] = _record_hash(record)       # hash of this record (payload + prev)
        record["mac"] = _tag(bytes(self._key), record["hash"])
        self._key[:] = hashlib.sha256(self._key).digest()  # evolve; this record's key is now gone
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        self._last_hash = record["hash"]

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "HashChainedLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- reader half (for graph.py and human inspection) -------------------------

def strip_envelope(record: dict) -> dict:
    """Drop the chain fields, returning the bare event payload schema.from_dict() expects."""
    return {k: v for k, v in record.items() if k not in _ENVELOPE}


def read_records(path: str | Path) -> Iterator[dict]:
    """Yield raw chained records (envelope included), in log order."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_events(path: str | Path) -> Iterator[dict]:
    """Yield bare event payloads (envelope stripped) — the graph.py entry point."""
    for record in read_records(path):
        yield strip_envelope(record)


def read_keys(path: str | Path) -> dict[str, bytes]:
    """Parse `RESIDUE-KEY <label> <start> <hex>` lines into {start_hash: K_0}. Takes the run's
    stderr/console capture as-is — other output is ignored, so you can point it at a job log."""
    keys: dict[str, bytes] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0] in _KEY_PREFIXES:
                keys[parts[2]] = bytes.fromhex(parts[3])
    return keys


def verify_chain(path: str | Path, keys: Optional[dict[str, bytes]] = None) -> None:
    """Walk the log, recomputing each hash and checking each `prev` link. Raises
    ValueError at the first break (the line index pins where tampering starts).

    With `keys` (from read_keys) it also re-derives the forward-secure MAC key per record and
    checks every tag, which is what actually catches a full-chain rebuild — the hash checks
    alone verify a forged chain just as happily as an honest one."""
    prev = None
    key: Optional[bytearray] = None
    for i, record in enumerate(read_records(path)):
        stored = record.get("hash")
        if record.get("prev") != prev:
            raise ValueError(f"chain break at record {i}: prev={record.get('prev')!r}, expected {prev!r}")
        recomputed = _record_hash({k: v for k, v in record.items() if k not in ("hash", "mac")})
        if recomputed != stored:
            raise ValueError(f"hash mismatch at record {i}: stored={stored!r}, recomputed={recomputed!r}")

        if keys is not None:
            k0 = keys.get(prev or _GENESIS)
            if k0 is not None:                      # a new segment starts here
                key = bytearray(k0)
            if key is None:
                raise ValueError(f"no key covers record {i}: unverified segment starting at {prev!r}")
            if "mac" not in record:
                raise ValueError(f"record {i} has no mac: written by a pre-MAC capture, or stripped")
            if not hmac.compare_digest(record["mac"], _tag(bytes(key), stored)):
                raise ValueError(f"mac mismatch at record {i}: record was forged or rewritten")
            key[:] = hashlib.sha256(key).digest()

        prev = stored


def dump(path: str | Path) -> None:
    """Pretty-print payloads (envelope hidden) for eyeballing during dev."""
    for event in read_events(path):
        print(json.dumps(event, indent=2, default=str))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Verify a provenance ledger's hash chain and MACs.")
    ap.add_argument("ledger")
    ap.add_argument("--keys", help="file holding the run's PROV-KEY line(s) (its stderr/job log)")
    args = ap.parse_args()

    keys = read_keys(args.keys) if args.keys else None
    verify_chain(args.ledger, keys)
    n = sum(1 for _ in read_records(args.ledger))
    print(f"OK: {n} records, chain intact" + (f", {len(keys)} keyed segment(s) authenticated"
                                              if keys else " (chain only — pass --keys to authenticate)"))
