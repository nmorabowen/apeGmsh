"""ADR 0056 V2/V3 — viewer state & event contract AST guards.

Machine-enforces INV-5 of
[ADR 0056](../../src/apeGmsh/opensees/architecture/decisions/0056-viewer-state-and-event-contract.md):
in the guarded scopes no code may render, flip render artifacts, or
import a render backend directly — UI code calls owner mutators and
fires dispatcher events; the reconciler (the dispatcher's pumps + the
RenderBackend implementations) is the only artifact writer and the
dispatcher the only caller of ``render()``.

Three guards, in the established AST-guard pattern
(``test_diagrams_pure_no_pyvista.py`` / ``test_scene_ir_pure.py`` /
``test_viewers_pure_h5_consumer.py``):

* **G-RENDER**   — no ``<expr>.render(...)`` call expressions.
* **G-ARTIFACT** — no ``SetVisibility`` / ``set_layer_visible`` /
  ``SetPickable`` / ``add_mesh`` / ``remove_actor`` calls.
* **G-IMPORT**   — no ``pyvista`` / ``vtk*`` / ``pyvistaqt`` imports
  and no imports of ``apeGmsh.viewers.backends`` (absolute or
  relative).

Scope grows in lockstep with adoption (ADR 0056 Part 5): V2 guarded
``ui/**``; V3 added ``mesh_viewer.py`` + ``overlays/**`` when the
mesh viewer joined the dispatcher. V4 adds ``model_viewer.py``.

Allowlists are per-file violation COUNTS, enumerated below with the
reason each entry survives. The count is a two-way ratchet: an
allowlisted file may go DOWN (the test then demands the number be
updated) but never up, and a file not listed fails on its first
violation. Adding or raising an entry requires citing ADR 0056 and a
reason in the comment — an allowlist that only grows is the failure
mode this test exists to prevent.

Note on the V3 scopes: ``mesh_viewer.py`` legitimately CONTAINS that
viewer's reconciler (the overlay-rebuild pump bodies and the
dispatcher's render binding), and ``overlays/`` are artifact-drawing
helpers by nature — their allowlisted counts are reconciler code, not
bypasses. The ratchet's job here is to catch NEW call sites appearing
outside the designated ones; the burn-down direction is extracting
the artifact code behind the SceneLayer seam (ADR 0042) over time.
"""
from __future__ import annotations

import ast
from pathlib import Path

VIEWERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
)

# Scopes guarded so far (viewers/-relative).
_GUARDED_DIRS = ("ui", "overlays")
_GUARDED_FILES = ("mesh_viewer.py", "model_viewer.py")

# ── Allowlists — (path relative to viewers/) -> max violation count ─
#
# G-RENDER:
# * ui/viewer_window.py — the window HOST's control-layer renders
#   (camera presets / parallel projection / fit-view / theme refresh).
#   Ratified durable in ADR 0056 Part 5; revisit if camera state ever
#   becomes owned view state.
# * mesh_viewer.py — 1 is the dispatcher's own render binding
#   (``render=lambda: plotter.render()``, the ONE render path); the
#   other 9 are V3-out-of-scope subsystems (labels ×3, wireframe,
#   edges, dim filter, prefs point-size, hover recolor, selection
#   recolor) — burn down as those subsystems join the contract.
# * overlays/* — self-rendering overlay helpers (clip plane, measure,
#   origin markers, local axes, tangent/normal, prefs callbacks);
#   artifact-drawing by nature, pre-dispatcher. Burn down at V4+.
_RENDER_ALLOW: dict[str, int] = {
    "ui/viewer_window.py": 5,
    "mesh_viewer.py": 10,
    # model_viewer.py — 1 is the dispatcher's render binding; the
    # other 7 are V4-out-of-scope subsystems (dim filter, labels,
    # prefs point-size + pick-color, scene rebuild, hover recolor,
    # selection recolor). The 8 call-site mutator renders + the
    # on_changed render subscriber were deleted at V4.
    "model_viewer.py": 8,
    "overlays/clip_plane_overlay.py": 5,
    "overlays/local_axes_overlay.py": 1,
    "overlays/measure_overlay.py": 3,
    "overlays/mesh_tangent_normal_overlay.py": 1,
    "overlays/origin_markers_overlay.py": 2,
    "overlays/pref_helpers.py": 3,
    "overlays/tangent_normal_overlay.py": 1,
}

# G-ARTIFACT:
# * ui/** — ZERO baseline, hard gate (V2).
# * mesh_viewer.py — the overlay-rebuild pump bodies (add_mesh /
#   remove_actor for loads, mass, boundary, constraints), the label
#   togglers, and the dim-filter SetVisibility. Designated reconciler
#   + V3-out-of-scope subsystems; counts ratchet down as artifact code
#   moves behind the SceneLayer seam.
# * overlays/* — artifact-drawing helpers by nature.
_ARTIFACT_ALLOW: dict[str, int] = {
    "mesh_viewer.py": 14,
    # model_viewer.py — label-actor teardown + the _rebuild_scene
    # actor swap (its designated post-geometry-mutation reconciler).
    "model_viewer.py": 3,
    "overlays/glyph_helpers.py": 2,
    "overlays/local_axes_overlay.py": 2,
    "overlays/measure_overlay.py": 2,
    "overlays/mesh_tangent_normal_overlay.py": 7,
    "overlays/origin_markers_overlay.py": 1,
    "overlays/probe_overlay.py": 8,
    "overlays/tangent_normal_overlay.py": 7,
}

# G-IMPORT:
# * ui/viewer_window.py — constructs the QtInteractor (lazy import)
#   and applies pyvista theme defaults: the host's job by definition.
# * mesh_viewer.py / overlays/* — pyvista/numpy mesh construction for
#   the overlay glyphs (reconciler-side); burn down behind the
#   SceneLayer seam.
_IMPORT_ALLOW: dict[str, int] = {
    "ui/viewer_window.py": 2,
    "mesh_viewer.py": 4,
    "overlays/clip_plane_overlay.py": 1,
    "overlays/constraint_overlay.py": 1,
    "overlays/glyph_helpers.py": 1,
    "overlays/local_axes_overlay.py": 1,
    "overlays/measure_overlay.py": 1,
    "overlays/mesh_tangent_normal_overlay.py": 1,
    "overlays/moment_glyph.py": 1,
    "overlays/origin_markers_overlay.py": 1,
    "overlays/probe_overlay.py": 1,
    "overlays/tangent_normal_overlay.py": 1,
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


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for d in _GUARDED_DIRS:
        files.extend(
            p for p in (VIEWERS_DIR / d).rglob("*.py") if p.is_file()
        )
    for f in _GUARDED_FILES:
        p = VIEWERS_DIR / f
        if p.is_file():
            files.append(p)
    return sorted(files)


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
                # Relative: ``from ..backends import ...`` is a
                # backends import too.
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
    files = _guarded_files()
    assert files, f"No guarded source files found — {guard} path is wrong."

    failures: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = collect(tree)
        rel = path.relative_to(VIEWERS_DIR).as_posix()
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


def test_guarded_scope_exists() -> None:
    assert (VIEWERS_DIR / "ui").is_dir(), (
        f"ui/ not found under {VIEWERS_DIR}; update the path constants "
        "if the package moved."
    )
    assert (VIEWERS_DIR / "mesh_viewer.py").is_file()


def test_g_render_no_direct_renders() -> None:
    _check(
        "G-RENDER", _RENDER_ALLOW,
        lambda tree: _attr_calls(tree, frozenset({"render"})),
    )


def test_g_artifact_no_actor_flag_calls() -> None:
    _check(
        "G-ARTIFACT", _ARTIFACT_ALLOW,
        lambda tree: _attr_calls(tree, _ARTIFACT_NAMES),
    )


def test_g_import_no_backend_imports() -> None:
    _check("G-IMPORT", _IMPORT_ALLOW, _backend_imports)
