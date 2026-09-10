"""Environment lookup that still answers to the pre-rename names.

RESIDUE_<NAME> first, then PROV_<NAME>, so a pipeline whose scripts export the old names keeps
working across the rename instead of silently falling back to defaults.
"""

from __future__ import annotations

import os
from typing import Optional


def env(name: str) -> Optional[str]:
    return os.getenv(f"RESIDUE_{name}") or os.getenv(f"PROV_{name}")
