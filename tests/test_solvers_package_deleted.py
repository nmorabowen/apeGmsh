"""Phase 8.8 — verify ``apeGmsh.solvers`` is fully gone.

After the multi-phase Phase 8 untangle (8.1 through 8.7) every name
that used to live under ``apeGmsh.solvers`` was relocated to its
canonical home elsewhere — broker records under ``apeGmsh.mesh``,
bridge primitives under ``apeGmsh.opensees``, recorder spec types
under ``apeGmsh.results.spec``. Phase 8.8 deletes the package
entirely.

These tests verify the Phase 8 acceptance criteria from
https://github.com/nmorabowen/apeGmsh/blob/a9f8b670df700dd5b1a9c40f5c7d1f25dd1d400b/architecture/phase-8-untangle.md (pinned at a9f8b670):

- ``apeGmsh/solvers/`` directory is empty / deleted.
- ``git grep \"from apeGmsh.solvers\" src/`` returns zero matches.
- ``git grep \"from apeGmsh.solvers\" tests/`` returns zero matches.

The filesystem check (``solvers/`` directory absence) is the
authoritative one — it works in any environment regardless of
which apeGmsh install Python happens to resolve. The import
check is a belt-and-suspenders sanity gate that fires once the
deletion lands on every install path.
"""
from __future__ import annotations

from pathlib import Path

import apeGmsh


def test_solvers_directory_does_not_exist() -> None:
    """``apeGmsh/solvers/`` directory was deleted in Phase 8.8."""
    pkg_root = Path(apeGmsh.__file__).resolve().parent
    solvers = pkg_root / "solvers"
    assert not solvers.exists(), (
        f"apeGmsh/solvers/ should have been deleted in Phase 8.8 "
        f"but exists at {solvers}. Either the deletion regressed or "
        f"the editable install is pointing at a pre-8.8 checkout."
    )


def test_no_src_imports_from_apeGmsh_solvers() -> None:
    """No file under ``src/apeGmsh/`` imports from ``apeGmsh.solvers``."""
    # The test file lives at ``tests/test_solvers_package_deleted.py``;
    # walk up to repo root, then into ``src/apeGmsh``.
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src" / "apeGmsh"
    assert src_root.is_dir(), (
        f"expected src tree at {src_root}; layout changed?"
    )

    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        if "from apeGmsh.solvers" in text or "import apeGmsh.solvers" in text:
            offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        "Phase 8 acceptance criteria: zero `apeGmsh.solvers` imports "
        f"in src/. Offenders: {offenders}"
    )


def test_no_test_imports_from_apeGmsh_solvers() -> None:
    """No file under ``tests/`` imports from ``apeGmsh.solvers``."""
    tests_root = Path(__file__).resolve().parent
    offenders: list[str] = []
    for py in tests_root.rglob("*.py"):
        if py.name == "test_solvers_package_deleted.py":
            # This very file mentions the path as strings; not an import.
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "from apeGmsh.solvers" in text or "import apeGmsh.solvers" in text:
            offenders.append(str(py.relative_to(tests_root.parent)))
    assert not offenders, (
        "Phase 8 acceptance criteria: zero `apeGmsh.solvers` imports "
        f"in tests/. Offenders: {offenders}"
    )
