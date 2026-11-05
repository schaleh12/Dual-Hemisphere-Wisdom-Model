"""Pytest configuration ensuring tests run consistently from either the top-level root or `LMFT-Architecture/`."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ARCH_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ARCH_ROOT.parent / "LMFT-Training"

for p in (ARCH_ROOT / "src", ARCH_ROOT, TRAIN_ROOT, TRAIN_ROOT / "experiments"):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)


def pytest_sessionstart(session) -> None:
    os.chdir(ARCH_ROOT)
