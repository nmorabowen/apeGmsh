"""Shared fixtures for the MPM particle-set tests (ADR 0134 D1, D2)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

#: The fork's fixtures and validators, vendored byte-identical by slice M2-S0 (#2979).
MPM_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mpm"
_VENDORED = ("mpm_contracts", "ladruno_format", "ladruno_basis")


@pytest.fixture(scope="module")
def contracts() -> Iterator[ModuleType]:
    """The vendored ``mpm_contracts`` module, with ``sys.path``/``sys.modules`` restored.

    Module scope, not session: the bare names must be gone before any later module runs
    (``tests/test_import_namespacing.py`` refuses a test module left under a bare name).

    The copies patch ``sys.path`` from their own location, so they are loaded by putting
    their directory on ``sys.path``, never by editing them (``tests/fixtures/mpm/README.md``).
    """
    saved_path = list(sys.path)
    saved_mods = {n: sys.modules.get(n) for n in _VENDORED}
    for n in _VENDORED:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(MPM_FIXTURES))
    try:
        yield importlib.import_module("mpm_contracts")
    finally:
        sys.path[:] = saved_path
        for n, m in saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
