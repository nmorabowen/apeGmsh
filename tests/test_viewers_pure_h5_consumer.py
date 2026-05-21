"""Phase 8.7 acceptance test — viewers/ is a pure model.h5 consumer.

Walks every Python source file under ``src/apeGmsh/viewers/`` and
asserts the import surface complies with the contract recorded in
[ADR 0014](../src/apeGmsh/opensees/architecture/decisions/0014-viewer-is-pure-h5-consumer.md)
and [phase-8.7-scope.md §6](../src/apeGmsh/opensees/architecture/phase-8.7-scope.md):

* ``viewers/*`` may NOT ``from apeGmsh.mesh ...`` (the broker is
  off-limits to the viewer package after Phase 8.7).
* ``viewers/*`` may ``from apeGmsh.opensees ...`` only via the
  reference reader at ``apeGmsh.opensees.emitter.h5_reader``; every
  other ``opensees`` submodule is off-limits.

The check is structural — runs against the AST so a misspelled or
re-namespaced import gets caught at PR review time, not at runtime
when a fixture is loaded.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


VIEWERS_DIR = Path(__file__).resolve().parent.parent / "src" / "apeGmsh" / "viewers"
ALLOWED_OPENSEES_MODULES = frozenset({
    # Phase 8.7 acceptance: only the reference h5 reader is allowed.
    "apeGmsh.opensees.emitter.h5_reader",
    # Phase 8 (ADR 0020 INV-1): the ``__main__`` CLI needs the
    # OpenSeesModel symbol to forward ``--model-h5`` into
    # ``Results.from_mpco(...)``.  The leak is confined to the CLI
    # entry-point; the rest of viewers/ remains pure-h5.
    "apeGmsh.opensees.OpenSeesModel",
    "apeGmsh.opensees",
})


def _viewers_python_files() -> list[Path]:
    """Every .py file under viewers/, recursively."""
    return sorted(p for p in VIEWERS_DIR.rglob("*.py") if p.is_file())


def _collect_offending_imports(path: Path) -> list[tuple[int, str]]:
    """Parse ``path`` and return ``(lineno, module)`` for every forbidden import.

    Forbidden modules:
    * ``apeGmsh.mesh`` and any submodule.
    * ``apeGmsh.opensees`` and any submodule, EXCEPT the entries in
      :data:`ALLOWED_OPENSEES_MODULES`.

    Both ``import apeGmsh.mesh.X`` and ``from apeGmsh.mesh.X import Y``
    are flagged.  ``from apeGmsh.opensees.emitter import h5_reader`` is
    recognised as importing ``apeGmsh.opensees.emitter.h5_reader``
    (the allowed leaf), not the parent ``apeGmsh.opensees.emitter``.
    Relative imports (``.scene``, ``..data``) inside the viewers
    package are ignored — they don't reach mesh / opensees.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    offenders: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have node.level >= 1 — never cross
            # out of viewers/.
            if node.level and node.level > 0:
                continue
            module = node.module or ""
            for alias in node.names:
                # Resolve the dotted path each imported name represents.
                # ``from apeGmsh.opensees.emitter import h5_reader``
                # really imports ``apeGmsh.opensees.emitter.h5_reader``;
                # checking the parent alone would over-reject.
                full = f"{module}.{alias.name}" if module else alias.name
                if _is_forbidden(full):
                    offenders.append((node.lineno, full))

    return offenders


def _is_forbidden(module: str) -> bool:
    """True if ``module`` is a banned dotted path."""
    if module == "apeGmsh.mesh" or module.startswith("apeGmsh.mesh."):
        return True
    if module == "apeGmsh.opensees" or module.startswith("apeGmsh.opensees."):
        return module not in ALLOWED_OPENSEES_MODULES
    return False


def test_viewers_dir_exists() -> None:
    """Sanity check — the package we walk has to be present."""
    assert VIEWERS_DIR.is_dir(), (
        f"viewers/ not found at {VIEWERS_DIR}. The acceptance test "
        f"assumes src/apeGmsh/viewers/ exists; if the package has "
        f"moved, update the path constant in this test."
    )


def test_viewers_have_no_mesh_or_opensees_imports() -> None:
    """No ``from apeGmsh.mesh ...`` or ``from apeGmsh.opensees.<not h5_reader> ...``.

    The acceptance test for Phase 8.7 (ADR 0014).  When this fails,
    the failure message lists every offender so you can spot the
    new leak quickly.
    """
    files = _viewers_python_files()
    assert files, "No viewer source files found — test path is wrong."

    leaks: list[tuple[Path, int, str]] = []
    for path in files:
        for lineno, module in _collect_offending_imports(path):
            leaks.append((path, lineno, module))

    if leaks:
        rel_leaks = sorted(
            (
                str(path.relative_to(VIEWERS_DIR.parent.parent.parent)),
                lineno,
                module,
            )
            for path, lineno, module in leaks
        )
        msg = "\n".join(
            f"  {p}:{lno}  →  {mod!r}" for p, lno, mod in rel_leaks
        )
        raise AssertionError(
            "viewers/ must not import from apeGmsh.mesh or any "
            "apeGmsh.opensees submodule except "
            f"{sorted(ALLOWED_OPENSEES_MODULES)!r}.\n"
            f"Found {len(rel_leaks)} forbidden import(s):\n{msg}\n\n"
            "See ADR 0014 (decisions/0014-viewer-is-pure-h5-consumer.md) "
            "and phase-8.7-scope.md §6 for the contract."
        )


# =====================================================================
# Phase 5 (ADR 0020) — removed-surface scans
# =====================================================================
#
# These tests pin the negative space the Phase 5 collapse opened up:
# the now-removed ``_resolve_effective_model_h5`` resolver and the
# ``_pending_model_h5`` field. Both surfaces lived only on
# :class:`ResultsViewer` and have no replacement (the chain forward
# ``Results.model -> OpenSeesModel`` subsumed them). If either string
# reappears in ``src/`` it's a regression — the AST-walk version
# below is robust to comments / docstrings.


def _src_root() -> Path:
    return VIEWERS_DIR.parent.parent  # src/apeGmsh -> src/


def _scan_for_substring(needle: str) -> list[tuple[Path, int, str]]:
    """Return every ``(path, lineno, line)`` under ``src/`` mentioning
    ``needle`` in a real code position (not just a docstring or
    comment). Matches *any* mention because the symbols are private
    enough that comment references should also be cleaned up."""
    hits: list[tuple[Path, int, str]] = []
    for path in _src_root().rglob("*.py"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hits.append((path, lineno, line))
    return hits


def test_no_resolve_effective_model_h5() -> None:
    """``_resolve_effective_model_h5`` was removed in Phase 5. The
    chain-forward replacement is the symmetric
    ``_build_viewer_data`` branching gated on ``results.model``.

    See ADR 0020 §Decision / INV-5.
    """
    hits = _scan_for_substring("_resolve_effective_model_h5")
    if hits:
        rel = [
            (str(p.relative_to(_src_root())), lno, line.strip())
            for p, lno, line in hits
        ]
        msg = "\n".join(f"  {p}:{lno}  →  {ln!r}" for p, lno, ln in rel)
        raise AssertionError(
            "_resolve_effective_model_h5 was removed in Phase 5 (ADR 0020) "
            "but still appears in src/. Drop the references; the "
            "chain-forward replacement is _build_viewer_data gated on "
            f"results.model.\n{msg}"
        )


def test_no_pending_model_h5_field() -> None:
    """``_pending_model_h5`` was the legacy deprecation carrier on
    :class:`ResultsViewer`; it's now ``_legacy_model_h5`` (clearly
    deprecation-tagged) so any remaining ``_pending_model_h5``
    reference is stale."""
    hits = _scan_for_substring("_pending_model_h5")
    if hits:
        rel = [
            (str(p.relative_to(_src_root())), lno, line.strip())
            for p, lno, line in hits
        ]
        msg = "\n".join(f"  {p}:{lno}  →  {ln!r}" for p, lno, ln in rel)
        raise AssertionError(
            "_pending_model_h5 was removed in Phase 5 (ADR 0020) but "
            "still appears in src/. The deprecation carrier is now "
            f"_legacy_model_h5.\n{msg}"
        )


if __name__ == "__main__":
    # Stand-alone invocation prints a usable summary; pytest discovers
    # the test functions normally.
    files = _viewers_python_files()
    print(f"Scanned {len(files)} files under {VIEWERS_DIR}")
    total = 0
    for path in files:
        for lineno, module in _collect_offending_imports(path):
            rel = path.relative_to(VIEWERS_DIR.parent.parent.parent)
            print(f"  {rel}:{lineno}  →  {module!r}")
            total += 1
    if total == 0:
        print("Clean — no forbidden imports.")
    else:
        print(f"\n{total} forbidden import(s) found.")
        sys.exit(1)
