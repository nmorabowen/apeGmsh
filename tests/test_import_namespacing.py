"""The tests tree may never register a BARE top-level module.

``tests/opensees/`` has the same name as the ``opensees`` extension the
OpenSees install ships. pytest derives a test module's importable name by
walking UP from the file while ``__init__.py`` exists, so without
``tests/__init__.py`` the walk stops at ``tests/`` and every module under
``tests/opensees/`` is imported as bare ``opensees.<...>`` — binding
``sys.modules["opensees"]`` to ``tests/opensees/__init__.py``.

``--import-mode=importlib`` does NOT prevent this. pytest >= 8.1 resolves
the package root from the ``__init__.py`` chain in importlib mode too;
the import MODE governs how the module object is created, not what it is
called.

What that cost: this venv's ``_ladruno_opensees_boot`` .pth aliases
``openseespy`` / ``openseespy.opensees`` to whatever ``import opensees``
yields, resolved lazily on first use. Once the tests package won that
name, the alias resolved to it process-wide — so ``openseespy.opensees``
became a package with no ``wipe``, ``pytest.importorskip`` happily
returned it instead of skipping, and every live-OpenSees test in the
same process either failed or silently skipped. Measured on
``origin/main`` d09d3c7d: 67 failed and 43 spuriously skipped in one
process, all green when the same files ran alone.

The fix is one empty ``tests/__init__.py``, which re-roots the whole tree
under ``tests.``. These are the gates that keep it there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def test_tests_dir_is_a_package() -> None:
    """The one file the whole invariant rests on.

    Non-vacuous even when this file runs alone, which the sys.modules
    gate below is not.
    """
    assert (TESTS_DIR / "__init__.py").exists(), (
        "tests/__init__.py is missing. Without it pytest names every "
        "module under tests/<pkg>/ as bare '<pkg>.…', and tests/opensees/ "
        "then shadows the real OpenSees module for the whole process."
    )


def test_no_test_module_is_registered_under_a_bare_name() -> None:
    """Every imported module living under tests/ must be named ``tests.*``.

    Strongest in a full-suite run, where hundreds of test modules are
    already in ``sys.modules`` by the time this executes.
    """
    offenders = []
    for name, mod in list(sys.modules.items()):
        origin = getattr(mod, "__file__", None)
        if not origin:
            continue
        try:
            path = Path(origin).resolve()
        except (OSError, ValueError):
            continue
        if not path.is_relative_to(TESTS_DIR):
            continue
        if name == "tests" or name.startswith("tests."):
            continue
        offenders.append(f"{name} -> {path}")

    assert not offenders, (
        "these test modules are registered under a bare top-level name, "
        "where they can shadow a real dependency:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_openseespy_still_resolves_to_a_real_backend() -> None:
    """The consequence check — the thing that actually broke.

    ``importorskip`` returns whatever is importable under the name, so a
    shadowed ``openseespy.opensees`` passes the skip gate and then fails
    on the first ``ops.wipe()``. Assert the module the suite depends on
    is the backend and not a directory of tests.
    """
    ops = pytest.importorskip(
        "openseespy.opensees", reason="no OpenSees backend in this env",
    )
    origin = getattr(ops, "__file__", None)
    assert not (origin and Path(origin).resolve().is_relative_to(TESTS_DIR)), (
        f"openseespy.opensees resolved to the tests tree ({origin})"
    )
    for verb in ("wipe", "model", "node", "element"):
        assert hasattr(ops, verb), (
            f"openseespy.opensees has no {verb!r} — it is not an OpenSees "
            f"backend (resolved to {origin})."
        )
