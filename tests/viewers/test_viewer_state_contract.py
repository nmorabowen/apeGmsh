"""ADR 0056 V2 — viewer state & event contract AST guards.

Machine-enforces INV-5 of
[ADR 0056](../../src/apeGmsh/opensees/architecture/decisions/0056-viewer-state-and-event-contract.md):
in ``src/apeGmsh/viewers/ui/**`` no code may render, flip render
artifacts, or import a render backend directly — UI code calls owner
mutators and fires dispatcher events; the reconciler (the dispatcher's
pumps + the RenderBackend implementations) is the only artifact writer
and the dispatcher the only caller of ``render()``.

Three guards, in the established AST-guard pattern
(``test_diagrams_pure_no_pyvista.py`` / ``test_scene_ir_pure.py`` /
``test_viewers_pure_h5_consumer.py``):

* **G-RENDER**   — no ``<expr>.render(...)`` call expressions.
* **G-ARTIFACT** — no ``SetVisibility`` / ``set_layer_visible`` /
  ``SetPickable`` / ``add_mesh`` / ``remove_actor`` calls. Baseline is
  ZERO — this guard is a hard gate from day one.
* **G-IMPORT**   — no ``pyvista`` / ``vtk*`` / ``pyvistaqt`` imports
  and no imports of ``apeGmsh.viewers.backends`` (absolute or
  relative).

Allowlists are per-file violation COUNTS, enumerated below with the
reason each entry survives. The count is a ratchet: an allowlisted
file may go DOWN (update the number when it does) but never up, and a
file not listed fails on its first violation. Adding or raising an
entry requires citing ADR 0056 and a reason in the comment — an
allowlist that only grows is the failure mode this test exists to
prevent.

Guard scope grows in lockstep with adoption (ADR 0056 Part 5): V3
widens to ``mesh_viewer.py`` + ``overlays/``, V4 to
``model_viewer.py``. Widen ``GUARDED_DIRS``/``GUARDED_FILES`` then.
"""
from __future__ import annotations

import ast
from pathlib import Path

UI_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
    / "ui"
)

# ── Allowlists — (filename relative to ui/) -> max violation count ──
#
# G-RENDER: the window HOST's control-layer renders. ViewerWindow is
# the Qt shell shared by all three viewers; its camera-preset /
# parallel-projection / fit-view / theme-refresh handlers render the
# interactor directly because they mutate camera/window state that no
# owner-model field represents (yet). Ratified as the durable
# allowlist in ADR 0056 Part 5; revisit if camera state ever becomes
# owned view state.
_RENDER_ALLOW: dict[str, int] = {
    "viewer_window.py": 5,
}

# G-ARTIFACT: ZERO baseline — hard gate.
_ARTIFACT_ALLOW: dict[str, int] = {}

# G-IMPORT: viewer_window.py is the Qt shell that CONSTRUCTS the
# QtInteractor render widget — the one ui/ module that must touch
# pyvistaqt by definition (line ~50, lazy import) — and it applies
# pyvista-level theme defaults at window construction (line ~250).
# Same control-layer carve-out as its renders. (Caught by this guard
# on first run: the pyvistaqt import was invisible to the grep-based
# baseline — the AST guard is already stricter than the survey.)
_IMPORT_ALLOW: dict[str, int] = {
    "viewer_window.py": 2,
}

_ARTIFACT_NAMES = frozenset({
    "SetVisibility",
    "set_layer_visible",
    "SetPickable",
    "add_mesh",
    "remove_actor",
})

_FORBIDDEN_IMPORT_ROOTS = frozenset({
    "pyvista", "pyvistaqt", "vtk", "vtkmodules",
})


def _ui_files() -> list[Path]:
    return sorted(p for p in UI_DIR.rglob("*.py") if p.is_file())


def _attr_calls(tree: ast.AST, names: frozenset[str]) -> list[tuple[int, str]]:
    """All ``<expr>.<name>(...)`` call sites whose attribute is in ``names``."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
        ):
            hits.append((node.lineno, node.func.attr))
    return hits


def _backend_imports(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    hits.append((node.lineno, alias.name))
                elif alias.name.startswith("apeGmsh.viewers.backends"):
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and node.level > 0:
                # Relative: ``from ..backends import ...`` from ui/ is
                # a backends import too.
                if module.split(".", 1)[0] == "backends":
                    hits.append((node.lineno, f"{'.' * node.level}{module}"))
                continue
            root = module.split(".", 1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                hits.append((node.lineno, module))
            elif module.startswith("apeGmsh.viewers.backends"):
                hits.append((node.lineno, module))
    return hits


def _check(
    guard: str,
    allow: dict[str, int],
    collect,
) -> None:
    files = _ui_files()
    assert files, f"No ui/ source files found — {guard} path is wrong."

    failures: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = collect(tree)
        rel = path.relative_to(UI_DIR).as_posix()
        budget = allow.get(rel, 0)
        if len(hits) > budget:
            detail = ", ".join(f"line {ln}: {what}" for ln, what in hits)
            failures.append(
                f"  {rel}: {len(hits)} violation(s) (allowlisted: {budget})"
                f" — {detail}"
            )
        elif hits and len(hits) < budget:
            failures.append(
                f"  {rel}: allowlist says {budget} but only {len(hits)} "
                f"remain — ratchet the {guard} allowlist down (ADR 0056)."
            )
    if failures:
        raise AssertionError(
            f"{guard} (ADR 0056 INV-5) violated — UI code must route "
            "through owner mutators + dispatcher events, never touch "
            "render artifacts directly:\n" + "\n".join(failures)
        )


def test_ui_dir_exists() -> None:
    assert UI_DIR.is_dir(), (
        f"ui/ not found at {UI_DIR}; update the path constant if the "
        "package moved."
    )


def test_g_render_no_direct_renders_in_ui() -> None:
    _check(
        "G-RENDER", _RENDER_ALLOW,
        lambda tree: _attr_calls(tree, frozenset({"render"})),
    )


def test_g_artifact_no_actor_flag_calls_in_ui() -> None:
    _check(
        "G-ARTIFACT", _ARTIFACT_ALLOW,
        lambda tree: _attr_calls(tree, _ARTIFACT_NAMES),
    )


def test_g_import_no_backend_imports_in_ui() -> None:
    _check("G-IMPORT", _IMPORT_ALLOW, _backend_imports)
