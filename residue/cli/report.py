"""One post-hoc report for a finished run: is the evidence trustworthy, is it complete,
what diverged from a clean reference, and what the ledger actually observed.

Four sections, in the order an investigator needs them:

  [1] LEDGER INTEGRITY   was any record altered/deleted after it was written
                         (residue.store.log.verify_chain, + MACs when a key file exists)
  [2] TERMINATION        did the run finish, or was the ledger cut mid-write
  [3] ROOT CAUSE         residue.analysis.diff, verbatim (needs a reference ledger)
  [4] OBSERVED           the flat projection of what this run captured

Reads only finished artifacts -- it never instruments a run, so it is re-runnable over any
ledger that already exists, and section 3 can be added later once a reference run is chosen.

`observed` is a parameter rather than a fixed projection: a caller scoring several tools against
one field list passes its own vocabulary, and gets the same report shape back.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from residue.analysis.diff import diff, format_report
from residue.analysis.observe import ABSENT, observe
from residue.schema.validate import Code, validate
from residue.store.graph import load
from residue.store.log import default_key_path, read_keys, verify_chain

WIDTH = 96
VALUE_WIDTH = 118       # section 4 prints values verbatim; give the 64-char hashes room to fit
_RULE = "=" * WIDTH

# both names are accepted everywhere a ledger is located: ledgers written before the rename are
# still the reference runs an investigation diffs against.
_LEDGER_NAMES = ("residue.jsonl", "provenance.jsonl")
_KEY_NAMES = ("residue_keys.txt", "prov_keys.txt")


# ---- locating things ---------------------------------------------------------

def resolve_ledger(target: str | Path) -> Path:
    """Accept a run dir, its residue/ (or provenance/) dir, or the ledger itself."""
    p = Path(target)
    if p.is_file():
        return p
    for sub in ("", "residue", "provenance"):
        for name in _LEDGER_NAMES:
            cand = p / sub / name if sub else p / name
            if cand.is_file():
                return cand
    raise SystemExit(f"no ledger ({' or '.join(_LEDGER_NAMES)}) under {p}")


def run_name(ledger: Path) -> str:
    """The run's name for the report headline: the ledger's own dir, unless that dir is just the
    tool's subfolder inside a run dir, in which case the run dir is the name worth printing."""
    return (ledger.parent.parent.name if ledger.parent.name in ("residue", "provenance")
            else ledger.parent.name)


def find_keys(ledger: Path, explicit: Optional[Path]) -> tuple[Optional[dict], Optional[Path]]:
    """Where K_0 landed: what the caller points us at, else the default ~/.residue-keys/<run>.txt
    that emit_key writes, else the conventional spots -- checked but not assumed, since a key beside
    the log it authenticates protects nothing."""
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = [default_key_path(run_name(ledger))]
        candidates += [d / n for d in (ledger.parent, ledger.parent.parent) for n in _KEY_NAMES]
    for c in candidates:
        if c.is_file() and (keys := read_keys(c)):
            return keys, c
    return None, None


# ---- [1] integrity -----------------------------------------------------------

def section_integrity(ledger: Path, keys: Optional[dict], keyfile: Optional[Path]) -> tuple[list[str], bool]:
    lines, ok = [], True
    n = macked = 0
    for line in open(ledger, encoding="utf-8"):
        if line.strip():
            n += 1
            macked += '"mac"' in line

    try:
        verify_chain(ledger, keys)
        chain_ok = True
    except json.JSONDecodeError:                  # torn final line: a crash, not a tamper
        return ["  chain: UNVERIFIED — last line truncated, walk stops there (see [2])"], False
    except ValueError as e:
        chain_ok, ok = False, False
        lines.append(f"  chain: TAMPERED — {e}")

    if chain_ok:
        lines.append(f"  chain: INTACT — {n} records, no node altered or deleted")

    # hashes alone verify a rebuilt chain as happily as an honest one; say which one we did.
    if keys and chain_ok:
        lines.append(f"  macs : AUTHENTICATED — {len(keys)} segment(s) against {keyfile}")
    elif macked == 0:
        lines.append("  macs : none in this ledger (pre-MAC capture) — hash chain only")
    else:
        lines.append(f"  macs : {macked}/{n} tagged but UNCHECKED (no K_0 file; pass --keys) — "
                     "chain proves internal consistency only")
    return lines, ok


# ---- [2] termination ---------------------------------------------------------

def section_termination(graph) -> tuple[list[str], bool]:
    """Early termination has one tell: the Recorder writes each activity twice (open, then close
    with end_ns), so a run that was killed leaves activities with no end_ns. A ledger whose last
    line is truncated didn't even survive to load()."""
    lines, ok = [], True
    if graph is None:
        return ["  CUT SHORT — final record truncated, the process died mid-write"], False

    report = validate(graph.node_list, graph.edge_list)
    open_acts = [i for i in report.incomplete if i.code == Code.ACTIVITY_NO_END]
    acts = [n for n in graph.node_list if hasattr(n, "activity_id")]
    stages = sorted({a.activity_id.split(":")[-2] for a in acts if a.activity_id.count(":") >= 2})

    if open_acts:
        ok = False
        lines.append(f"  CUT SHORT — {len(open_acts)}/{len(acts)} activities never closed: "
                     + ", ".join(i.ref for i in open_acts[:3])
                     + (f" …+{len(open_acts) - 3}" if len(open_acts) > 3 else ""))
    else:
        lines.append(f"  CLOSED CLEANLY — all {len(acts)} activities closed  |  "
                     f"stages: {', '.join(stages) or '—'}")
    if report.violations:
        ok = False
        lines.append(f"  {len(report.violations)} structural violation(s): "
                     + ", ".join(sorted({str(v.code) for v in report.violations})))
    return lines, ok


# ---- [3] root cause ----------------------------------------------------------

def section_rca(ref: Optional[Path], graph, literal: bool, tail_verbose: bool = False) -> list[str]:
    if graph is None:
        return ["  OMITTED — ledger truncated, no graph to diff"]
    if ref is None:
        return ["  OMITTED — no reference run given (--ref <clean run>); RCA is differential"]
    body = format_report(diff(load(ref), graph, literal=literal), tail_verbose=tail_verbose)
    return ["  reference: " + str(ref), ""] + ["  " + ln for ln in body.splitlines()]


# ---- [4] observed ------------------------------------------------------------

def _wrap(s: str, width: int) -> list[str]:
    """Never elide -- long values wrap. break_long_words=False keeps a hash or a path whole
    even when it overruns the column."""
    return textwrap.wrap(s, max(width, 24), break_long_words=False,
                         break_on_hyphens=False) or [s]


def _annotate(lines: list[str], note: str, width: int) -> list[str]:
    """Note rides the last value line when it fits, else gets its own."""
    if len(lines[-1]) + len(note) <= width:
        return lines[:-1] + [lines[-1] + note]
    return lines + [note.lstrip()]


def _render_observed_value(v: Any, width: int) -> list[str]:
    """One field as one or more lines, verbatim -- a report an investigator has to cross-check
    against observed.json isn't evidence. Keyed values (per-epoch, or per-op for the transform
    params) collapse ONLY when every value is identical, which loses nothing; the key count and
    distinct count stay, so a collapse is visible. When they differ, every key is printed."""
    if isinstance(v, dict) and v:
        items = list(v.items())
        label = "epoch" if all(str(k).isdigit() for k, _ in items) else "key"
        vals = [json.dumps(x, default=str) for _, x in items]
        if len(items) == 1:                       # epoch-gated capture (Slice 13 arms epoch 1 only)
            return _annotate(_wrap(vals[0], width), f"   [{label} {items[0][0]} only]", width)
        if len(set(vals)) == 1:
            return _annotate(_wrap(vals[0], width),
                             f"   [same for all {len(items)} {label}s]", width)
        out = [f"[{len(items)} {label}s, {len(set(vals))} distinct]"]
        for (k, _), val in zip(items, vals):
            wrapped = _wrap(f"{k} = {val}", width - 2)
            out += ["  " + wrapped[0]] + ["    " + w for w in wrapped[1:]]
        return out
    return _wrap(json.dumps(v, default=str), width)


def section_observed(observed: dict) -> list[str]:
    covered = sum(1 for v in observed.values() if v != ABSENT)
    lines = [f"  {covered}/{len(observed)} fields observed", ""]
    pad = max(len(k) for k in observed) if observed else 0
    for k, v in observed.items():
        val = _render_observed_value(v, VALUE_WIDTH - pad - 4)
        lines.append(f"  {k.ljust(pad)}  {val[0]}")
        lines += [" " * (pad + 4) + ln for ln in val[1:]]
    return lines


# ---- assembly ----------------------------------------------------------------

def build_report(ledger: Path, ref: Optional[Path] = None, keys_file: Optional[Path] = None,
                 literal: bool = False, tail_verbose: bool = False,
                 observed: Optional[dict] = None) -> tuple[str, bool]:
    """`observed` overrides the projection in section 4 -- pass a fixed vocabulary's {key: value}
    to report against that instead of the ledger's own field set."""
    try:
        graph = load(ledger)
    except json.JSONDecodeError:      # torn final line — the run died mid-write; [2] says so
        graph = None
    keys, keyfile = find_keys(ledger, keys_file)

    note = ""
    if observed is None:
        if graph is None:
            observed, note = {}, "(skipped — ledger is truncated)"
        else:
            observed = observe(ledger)

    integrity, ok_i = section_integrity(ledger, keys, keyfile)
    termination, ok_t = section_termination(graph)

    out = [
        _RULE,
        f" Full Provenance Report — {run_name(ledger)}",
        f" ledger    : {ledger}  ({ledger.stat().st_size // 1024} KB)",
        f" generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        _RULE, "",
        "[1] Integrity of Provenance Ledger", *integrity, "",
        "[2] Ledger Termination", *termination, "",
        "[3] Root Cause Analysis",
        *section_rca(ref, graph, literal, tail_verbose), "",
        "[4] Summary of Observed Nodes for Forensic Analysis" + (f" {note}" if note else ""),
        *section_observed(observed), "",
        _RULE,
        f" summary: ledger integrity {'OK' if ok_i else 'FAILED'} | ledger termination "
        f"{'OK' if ok_t else 'FAILED'} | rca included {'YES' if ref else 'NO (no reference run supplied)'}",
        _RULE,
    ]
    return "\n".join(out), (ok_i and ok_t)
