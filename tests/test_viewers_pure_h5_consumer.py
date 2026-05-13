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
