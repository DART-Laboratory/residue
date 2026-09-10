"""Flatten a finished ledger to {key: value} -- what this run actually captured.

Keys are `Type.field`. With no `keys` argument the key set is derived from the ledger itself: every
node type present, its declared fields and its open `attributes` keys. Pass `keys` to project onto a
fixed vocabulary instead -- that is how a benchmark scores several tools against one field list, and
why the key set is a parameter rather than a constant in here.

ABSENT = the key was asked for and this run recorded nothing for it.
"""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable, Optional

ABSENT = "ABSENT"

# identity and envelope: present on every node, so they say nothing about what was captured.
# pid stays -- it is what pairs an activity with an OS tracer's process tree.
_SKIP = {"attributes", "activity_id", "agent_id", "start_ns", "end_ns"}


def _by_type(graph) -> dict[str, list]:
    out: dict[str, list] = {}
    for n in graph.node_list:
        out.setdefault(type(n).__name__, []).append(n)
    return out


def _field(node, name: str) -> Any:
    """A node's field: first-class attr, else the open `attributes` dict, else None."""
    v = getattr(node, name, None)
    if v is None:
        v = getattr(node, "attributes", {}).get(name)
    return v


def _collapse(values: list) -> Any:
    """Drop Nones; the lone value if constant across nodes, else the distinct list (order preserved)."""
    seen: list = []
    for v in values:
        if v is not None and v not in seen:
            seen.append(v)
    if not seen:
        return ABSENT
    return seen[0] if len(seen) == 1 else seen


def _extract(key: str, nbt: dict) -> Any:
    typ, _, name = key.partition(".")
    nodes = nbt.get(typ, [])
    return _collapse([_field(n, name) for n in nodes]) if nodes else ABSENT


def default_keys(nbt: dict) -> list[str]:
    """The ledger's own vocabulary: for each node type present, its declared fields plus whatever
    open attribute keys its instances carry. Ordered by type, then declared before open, so the
    summary reads in schema order rather than hash order."""
    keys: list[str] = []
    for typ, nodes in nbt.items():
        seen: set[str] = set()
        for f in dataclass_fields(type(nodes[0])):
            if f.name not in _SKIP and f.name not in seen:
                seen.add(f.name)
                keys.append(f"{typ}.{f.name}")
        for n in nodes:
            for name in getattr(n, "attributes", {}):
                if name not in seen:
                    seen.add(name)
                    keys.append(f"{typ}.{name}")
    return keys


def observe(ledger: str | Path, keys: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Project the ledger at `ledger` onto `keys` -> {key: observed value | ABSENT}."""
    from residue.store.graph import load
    nbt = _by_type(load(ledger))
    return {key: _extract(key, nbt) for key in (default_keys(nbt) if keys is None else keys)}


def write_observed(ledger: str | Path, keys: Optional[Iterable[str]] = None,
                   out: str | Path | None = None) -> Path:
    """Write observed.json (next to the ledger by default)."""
    ledger = Path(ledger)
    out = Path(out) if out else ledger.parent / "observed.json"
    out.write_text(json.dumps(observe(ledger, keys), indent=2, default=str))
    return out
