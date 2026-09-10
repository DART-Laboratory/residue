"""`residue` command line.

    residue run    <script.py> [args...]   capture a run (the script is not modified)
    residue report <run>                   integrity + termination + RCA + what was observed
    residue graph  <ledger>                render the ledger to SVG/PNG (needs graphviz `dot`)
    residue diff   <reference> <suspect>   structural diff of two ledgers
    residue verify <ledger>                hash-chain and MAC check alone
    residue observe <ledger>               flat {key: value} projection -> observed.json

Every subcommand except `run` reads finished artifacts only, so they are re-runnable over any
ledger that already exists.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

LEDGER_NAME = "residue.jsonl"


def _resolve_out(out: str) -> Path:
    """--out takes either a file or a directory: a path ending in .jsonl is the ledger itself,
    anything else is a directory to put the default-named ledger in."""
    p = Path(out)
    path = p if p.suffix == ".jsonl" else p / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cmd_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="residue run", description="Run a script under capture.")
    ap.add_argument("-o", "--out", metavar="PATH",
                    help=f"ledger file, or a directory to write {LEDGER_NAME} into "
                         "(default: outputs/<stamp>/)")
    ap.add_argument("--keys-to", metavar="PATH", default=None,
                    help="where to write the run's K_0 (default: ~/.residue-keys/<run>.txt)")
    ap.add_argument("--graph", action="store_true",
                    help="render the ledger once the run finishes (needs graphviz `dot`)")
    ap.add_argument("script", help="the script to run, unmodified")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed through to the script")
    a = ap.parse_args(argv)

    # --out wins over the environment: an exported RESIDUE_LOG silently overriding an explicit flag
    # is the wrong way round for a CLI, even though that is the precedence init() applies.
    if a.out:
        os.environ["RESIDUE_LOG"] = str(_resolve_out(a.out))

    if a.keys_to:
        from residue.store.log import set_key_sink
        set_key_sink(a.keys_to)

    # atexit runs handlers in REVERSE registration order, so registering the render before init()
    # puts it after init()'s _finalize -- the ledger is closed and every activity ended by then.
    ledger: dict[str, Path] = {}
    if a.graph:
        import atexit
        atexit.register(_render_after_run, ledger)

    from residue import init
    sys.argv = [a.script, *a.args]      # before init(), so the stage is detected from the target
    try:
        session = init()
    except ModuleNotFoundError as e:    # capture hooks torch; reading a ledger never does
        if e.name != "torch":
            raise
        print("residue run needs torch in this environment (capture hooks it).\n"
              "  pip install torch      — or 'pip install residue[torch]'\n"
              "Reading a ledger (report/diff/verify/observe/graph) does not need it.",
              file=sys.stderr)
        return 1
    ledger["path"] = session.log_path
    runpy.run_path(a.script, run_name="__main__")
    return 0


def _render_after_run(ledger: dict) -> None:
    """--graph, at exit. Never lets a rendering problem (no graphviz, say) look like a failed run:
    the ledger is already written and `residue graph` can retry any time."""
    path = ledger.get("path")
    if path is None:
        return
    try:
        from residue.analysis.viz import render
        print(f"wrote {render(path)}", file=sys.stderr)
    except Exception as e:
        print(f"graph skipped: {e}", file=sys.stderr)


def _cmd_report(argv: list[str]) -> int:
    from residue.cli.report import build_report, resolve_ledger

    ap = argparse.ArgumentParser(prog="residue report",
                                 description="Post-hoc report for a finished run.")
    ap.add_argument("target_run", help="run dir, its residue/ dir, or the ledger itself")
    ap.add_argument("--ref", default=None, help="clean reference run (enables section 3)")
    ap.add_argument("--keys", default=None, help="file holding the run's RESIDUE-KEY line(s) (default: ~/.residue-keys/<run>.txt)")
    ap.add_argument("--literal", action="store_true", help="diff without run-id normalisation")
    ap.add_argument("--tail-verbose", action="store_true",
                    help="enumerate the trained-tail divergences in section 3 instead of counting them")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default: residue_report.txt beside the ledger)")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the report (section 4 is long; by default it only goes to the file)")
    a = ap.parse_args(argv)

    ledger = resolve_ledger(a.target_run)
    text, ok = build_report(ledger,
                            ref=resolve_ledger(a.ref) if a.ref else None,
                            keys_file=Path(a.keys) if a.keys else None,
                            literal=a.literal, tail_verbose=a.tail_verbose)
    out = Path(a.out) if a.out else ledger.parent / "residue_report.txt"
    out.write_text(text + "\n")
    if a.stdout:
        print(text)
    print(f"report -> {out}" + ("" if ok else "   [!] integrity/termination check FAILED"))
    return 0 if ok else 1


def _cmd_graph(argv: list[str]) -> int:
    from residue.analysis.viz import main as viz_main
    viz_main(argv)
    return 0


def _cmd_diff(argv: list[str]) -> int:
    from residue.analysis.diff import diff, format_report
    from residue.store.graph import load

    ap = argparse.ArgumentParser(prog="residue diff",
                                 description="Structural diff of two ledgers.")
    ap.add_argument("reference", type=Path, help="clean / baseline ledger")
    ap.add_argument("suspect", type=Path, help="suspect ledger")
    ap.add_argument("--no-fold", dest="fold", action="store_false",
                    help="print every instance separately instead of folding repeats onto one line")
    ap.add_argument("--literal", action="store_true",
                    help="do not normalise run-scoped substrings (run ids in paths)")
    ap.add_argument("--tail-verbose", action="store_true",
                    help="enumerate the trained-tail divergences instead of reporting their count")
    a = ap.parse_args(argv)

    d = diff(load(a.reference), load(a.suspect), literal=a.literal)
    print(format_report(d, fold=a.fold, tail_verbose=a.tail_verbose))
    return 0


def _cmd_verify(argv: list[str]) -> int:
    from residue.cli.report import find_keys
    from residue.store.log import read_keys, read_records, verify_chain

    ap = argparse.ArgumentParser(prog="residue verify",
                                 description="Verify a ledger's hash chain and MACs.")
    ap.add_argument("ledger", help="the residue.jsonl to check")
    ap.add_argument("--keys", help="file holding the run's RESIDUE-KEY line(s) "
                                   "(default: ~/.residue-keys/<run>.txt)")
    a = ap.parse_args(argv)

    keys, keyfile = find_keys(Path(a.ledger), Path(a.keys) if a.keys else None)
    try:
        verify_chain(a.ledger, keys)
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    n = sum(1 for _ in read_records(a.ledger))
    print(f"OK: {n} records, chain intact"
          + (f", {len(keys)} keyed segment(s) authenticated against {keyfile}" if keys
             else " (chain only — no key file found; pass --keys to authenticate)"))
    return 0


def _cmd_observe(argv: list[str]) -> int:
    from residue.analysis.observe import ABSENT, write_observed

    ap = argparse.ArgumentParser(prog="residue observe",
                                 description="Flat {key: value} projection of a ledger.")
    ap.add_argument("ledger", help="the residue.jsonl to project")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default: observed.json beside the ledger)")
    a = ap.parse_args(argv)

    written = write_observed(a.ledger, out=a.out)
    import json
    obs = json.loads(written.read_text())
    covered = sum(1 for v in obs.values() if v != ABSENT)
    print(f"observed {covered}/{len(obs)} fields -> {written}")
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "report": _cmd_report,
    "graph": _cmd_graph,
    "diff": _cmd_diff,
    "verify": _cmd_verify,
    "observe": _cmd_observe,
}


def main(argv: list[str] | None = None) -> None:
    # Split by hand rather than with subparsers: `residue run train.py --epochs 5` has to hand the
    # target's own flags through untouched, which a subparser would try to claim.
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        raise SystemExit(0)
    if argv[0] in ("-V", "--version"):
        from importlib.metadata import PackageNotFoundError, version
        try:
            print(version("residue"))
        except PackageNotFoundError:
            print("unknown (not installed)")
        raise SystemExit(0)

    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"residue: unknown command {cmd!r}\n", file=sys.stderr)
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(_COMMANDS[cmd](rest))


if __name__ == "__main__":
    main()
