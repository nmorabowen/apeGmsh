"""ADR 0056 V2/V3 — viewer state & event contract AST guards.

Machine-enforces INV-5 of
[ADR 0056](../../src/apeGmsh/opensees/architecture/decisions/0056-viewer-state-and-event-contract.md):
in the guarded scopes no code may render, flip render artifacts, or
import a render backend directly — UI code calls owner mutators and
fires dispatcher events; the reconciler (the dispatcher's pumps + the
RenderBackend implementations) is the only artifact writer and the
dispatcher the only caller of ``render()``.

Four guards, in the established AST-guard pattern
(``test_diagrams_pure_no_pyvista.py`` / ``test_scene_ir_pure.py`` /
``test_viewers_pure_h5_consumer.py``):

* **G-RENDER**   — no ``<expr>.render(...)`` call expressions.
* **G-ARTIFACT** — no ``SetVisibility`` / ``set_layer_visible`` /
  ``SetPickable`` / ``add_mesh`` / ``remove_actor`` calls.
* **G-IMPORT**   — no ``pyvista`` / ``vtk*`` / ``pyvistaqt`` imports
  and no imports of ``apeGmsh.viewers.backends`` (absolute or
  relative).
* **G-ACTORS**   — nothing outside a diagram reads its ``_actors``.
  Unlike the three above it scans ALL of ``viewers/**`` with a hard
  zero and no allowlist; see its section at the bottom of this file.

Scope grows in lockstep with adoption (ADR 0056 Part 5): V2 guarded
``ui/**``; V3 added ``mesh_viewer.py`` + ``overlays/**`` when the
mesh viewer joined the dispatcher. V4 adds ``model_viewer.py``.
V5 adds ``results_viewer.py`` (ADR 0084 D5), and with it
``_pump_set.py`` + ``_session_apply.py`` (ADR 0084 D7 / PR 7) — the
units carved OUT of ``results_viewer._show_impl``. Guarding the
extraction targets is not optional: an unguarded new module would
reopen the exact hole D5 closed, since "move it to a helper file" is
the cheapest way to launder a direct render past the ratchet.

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
_GUARDED_FILES = (
    "mesh_viewer.py",
    "model_viewer.py",
    "results_viewer.py",
    # ADR 0084 D7 / PR 7 — the PumpSet + session-apply units extracted
    # from ``results_viewer._show_impl``. Measured at extraction:
    # 0 / 0 / 0 on all three guards, so they carry NO allowlist entry
    # and fail on their first violation. That is deliberate: these two
    # files are the reconciler's new home, and the whole point of the
    # seam is that the pumps talk to diagrams and scene point arrays,
    # never to actors or a backend. The guarded call sites in
    # ``results_viewer.py`` were all outside the moved code, which is
    # why its 9 / 15 / 7 budgets did NOT ratchet down in this PR.
    "_pump_set.py",
    "_session_apply.py",
)

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
    # ui/_bg_toggle_gear.py — floating background-toggle button (white/dark);
    # two one-shot display-mode renders, same pattern as viewer_window.py.
    "ui/_bg_toggle_gear.py": 2,
    # mesh_viewer.py — +1 for _toggle_nodes (same pattern as the
    # existing _toggle_wireframe/_toggle_edges/_on_mesh_filter callbacks;
    # ADR 0056 V3 out-of-scope until SceneLayer seam lands).
    "mesh_viewer.py": 11,
    # model_viewer.py — 1 is the dispatcher's render binding; the
    # other 7 are V4-out-of-scope subsystems (dim filter, labels,
    # prefs point-size + pick-color, scene rebuild, hover recolor,
    # selection recolor, _toggle_pg_color). The 8 call-site mutator
    # renders + the on_changed render subscriber were deleted at V4.
    "model_viewer.py": 8,
    # results_viewer.py — the results reconciler lives in this file
    # (ADR 0056 / ADR 0084 D5), so it enters the guard with a measured
    # budget, same precedent as mesh_viewer.py. Reconciler-legitimate:
    # 1 is the dispatcher's own render binding (``render_callback=``
    # passed to ``director.bind_plotter``, the ONE render path) and 1 is
    # the headless one-shot GL realization for screenshot/export. The
    # other 7 are burn-down debt — subsystems that still render for
    # themselves instead of firing an event: theme re-tint, the shared
    # ``_render`` helper behind the geometry-display/prefs callbacks,
    # the dim filter, stage activation + stage toggle, clip-drag end,
    # and the escape key. Ratchets down as those subsystems join the
    # contract. Unchanged by the ADR 0084 D7 PumpSet extraction (PR 7):
    # none of the four pump bodies or the session-apply path contained
    # a ``render()`` call — the pumps deliberately leave RENDER to the
    # dispatcher's coalescer.
    "results_viewer.py": 9,
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
    # mesh_viewer.py — +1 _toggle_nodes SetVisibility (same pattern as
    # _on_mesh_filter) + +1 _on_explode_axis label remove_actor (label
    # clearing when explode is active — ADR 0056 V3 out-of-scope).
    "mesh_viewer.py": 16,
    # model_viewer.py — label-actor teardown + the _rebuild_scene
    # actor swap (its designated post-geometry-mutation reconciler).
    # +1 is the new label/scene-teardown remove_actor — a
    # V4-out-of-scope teardown call per ADR 0056.
    "model_viewer.py": 4,
    # results_viewer.py — ADR 0056 / ADR 0084 D5. Reconciler-legitimate
    # (9): the substrate materialization path the director calls as its
    # ``scene_factory`` (2 add_mesh building fill + wireframe, 2
    # SetVisibility hiding the freshly built pair), the geometry-display
    # push that the geometries subscriber runs (3 SetVisibility, ADR
    # 0058 S2b), and the geometry-removed teardown (1 remove_actor).
    # Burn-down debt (6): the point-size prefs node-cloud rebuild, the
    # node/element ID label teardowns, and the pick-highlight add/remove
    # pairs for gauss points and element cells — out-of-contract
    # subsystems that write artifacts directly. Unchanged by the ADR
    # 0084 D7 PumpSet extraction (PR 7): the moved pump bodies write
    # visibility through ``Diagram.apply_effective_visibility``, not
    # through raw actor flags, so no counted site left this file.
    # Ratchets down as those subsystems move behind the SceneLayer
    # seam (ADR 0042) and start firing events.
    # +4 (15 → 19) for ADR 0089 D1 — the feature-edge outline actor
    # is reconciler-legitimate substrate materialization, same class
    # as the fill/wireframe sites already counted: 1 add_mesh in
    # ``add_outline_actor``, 1 SetVisibility hiding the freshly built
    # outline in the scene factory, 1 SetVisibility in the
    # geometry-display push, 1 remove_actor in the geometry-removed
    # teardown (ADR 0056 reconciler carve-out).
    "results_viewer.py": 19,
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
    # results_viewer.py — ADR 0056 / ADR 0084 D5. Reconciler-legitimate
    # (6): 4 lazy ``.backends.pyvista_qt`` imports (render-surface build
    # + clip-plane application for the substrate and node cloud, the
    # clip re-apply, and the surface refresh) — this file materializes
    # the scene, so it is the designated backend caller; plus 2 lazy
    # ``pyvista`` imports for node-cloud mesh construction. Burn-down
    # debt (1): the ``pyvista`` import in the gauss-point pick
    # highlight, which belongs behind the SceneLayer seam with the rest
    # of the pick subsystem. Unchanged by the ADR 0084 D7 PumpSet
    # extraction (PR 7): the pumps import no backend at all, which is
    # why ``_pump_set.py`` enters the guard at zero.
    # +1 (7 → 8) for ADR 0089 D1 — ``add_outline_actor``'s lazy
    # ``.backends.pyvista_qt`` import (feature-edge extraction +
    # clip-plane application), the same designated-backend-caller
    # rationale as the four render-surface imports above (ADR 0056).
    "results_viewer.py": 8,
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


# ── G-ACTORS — a diagram's ``_actors`` is dead outside the diagram ──
#
# Context item 1 of ADR 0056 is this bug. Since the ADR 0042 R-B
# migration every diagram draws through backend layer handles and never
# fills ``Diagram._actors``; the field survives in ``diagrams/_base.py``
# only for the legacy teardown path. So code that walks ``d._actors``
# from OUTSIDE the diagram does nothing, and says nothing:
#
# * PR #593 (674cdceb, 2026-06-10) — ``pump_gate`` flipped
#   ``d._actors``: the composition gate was a no-op for all 11 kinds.
# * PR #620 (a345918d, 2026-06-11) — ``_sync_layer_grids``, in the SAME
#   file, still walked ``d._actors``: contour / fiber / layer / spring
#   layers stayed at the reference configuration under deformation.
#
# #593 fixed its instance and wrote the lesson down; the second
# instance sat ~130 lines away and shipped anyway. This guard greps the
# pattern instead of trusting a reader to. Scope is ALL of
# ``viewers/**`` with a hard zero and no allowlist: the pump code has
# already moved once (``_pump_set.py``, ADR 0084 D7) and will move
# again. Receivers ``self`` / ``cls`` are an owner reading its own
# field — ``Diagram`` itself, and ``ResultsPickEngine``, whose
# unrelated ``_actors`` registry is its own. Anything else is a
# foreign read: route through the diagram's reconciler-callee methods
# (``set_visible`` / ``apply_effective_visibility`` /
# ``sync_substrate_points``) instead.
#
# Mutation acceptance on the real commits (recorded in
# internal_docs/plan_agent_surface_viewers.md): 674cdceb^ -> 2 hits,
# 674cdceb -> 1 (the #620 site, a day before #620), a345918d^ -> 1,
# a345918d -> 0.

_ACTORS_ATTR = "_actors"
_OWNER_NAMES = frozenset({"self", "cls"})
_ATTR_BUILTINS = frozenset({"getattr", "hasattr", "setattr"})


def _is_owner(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in _OWNER_NAMES


def _foreign_actor_reads(tree: ast.AST) -> list[tuple[int, str]]:
    """``<x>._actors`` and ``getattr(<x>, "_actors", ...)`` (also
    ``hasattr`` / ``setattr``) where ``x`` is not ``self`` / ``cls``.

    AST-based, so a docstring or comment that NAMES the pattern (the
    ``results_viewer.py`` post-mortem note does) is not a hit.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == _ACTORS_ATTR:
            if not _is_owner(node.value):
                hits.append((node.lineno, f"{ast.unparse(node.value)}._actors"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ATTR_BUILTINS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == _ACTORS_ATTR
            and not _is_owner(node.args[0])
        ):
            hits.append(
                (node.lineno, f"{node.func.id}({ast.unparse(node.args[0])}, '_actors')")
            )
    return hits


def foreign_actor_reads_under(viewers_dir: Path) -> list[str]:
    """Every G-ACTORS hit under ``viewers_dir`` as ``rel:line: expr``.

    Takes the root as an argument so the same collector can be run
    against a ``git archive`` of a historical tree (mutation acceptance).
    """
    found: list[str] = []
    for path in sorted(viewers_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(viewers_dir).as_posix()
        found.extend(f"{rel}:{ln}: {what}" for ln, what in _foreign_actor_reads(tree))
    return found


def test_g_actors_scope_covers_the_pump_homes() -> None:
    # If the scan ever stops seeing the files the two incidents lived in
    # (and the module the pumps moved to), it would pass vacuously.
    scanned = {p.relative_to(VIEWERS_DIR).as_posix() for p in VIEWERS_DIR.rglob("*.py")}
    assert {"results_viewer.py", "_pump_set.py", "diagrams/_base.py"} <= scanned


def test_g_actors_no_foreign_diagram_actor_reads() -> None:
    hits = foreign_actor_reads_under(VIEWERS_DIR)
    assert not hits, (
        "G-ACTORS (ADR 0056 context item 1; PRs #593 / #620) — "
        "Diagram._actors is never populated since the ADR 0042 R-B "
        "migration, so walking it from outside a diagram is a silent "
        "no-op. Call the diagram's own set_visible / "
        "apply_effective_visibility / sync_substrate_points instead. If "
        "the object is not a Diagram, rename its private field rather "
        "than reaching into it:\n  " + "\n  ".join(hits)
    )


# Self-test: the collector must see the incident shapes and nothing else.
_G_ACTORS_FLAGGED = {
    # The #593 pump_gate shape.
    "pr593_gate": (
        "def pump_gate(registry, gate):\n"
        "    for d in registry.diagrams():\n"
        "        for actor in d._actors:  # noqa: SLF001\n"
        "            actor.SetVisibility(gate(d))\n"
    ),
    # The #620 _sync_layer_grids shape (a comprehension this time).
    "pr620_sync": (
        "def _sync_layer_grids(registry, pts):\n"
        "    grids = [a.GetMapper().GetInput() for d in registry.diagrams()"
        " for a in d._actors]\n"
    ),
    "through_an_owner_attribute": "def f(self):\n    return self._diagram._actors\n",
    "getattr_laundering": "def f(d):\n    return getattr(d, '_actors', [])\n",
    "hasattr_probe": "def f(d):\n    return hasattr(d, '_actors')\n",
}
_G_ACTORS_CLEAN = {
    # The base class owns the field (legacy teardown path).
    "owner_self": (
        "class Diagram:\n"
        "    def detach(self):\n"
        "        for actor in self._actors:\n"
        "            actor.remove()\n"
        "        self._actors = []\n"
    ),
    "owner_cls": "class D:\n    @classmethod\n    def f(cls):\n        return cls._actors\n",
    # The sanctioned route: the diagram's reconciler-callee methods.
    "fix_shape": (
        "def pump_gate(registry, gate, pts):\n"
        "    for d in registry.diagrams():\n"
        "        d.apply_effective_visibility(gate(d))\n"
        "        d.sync_substrate_points(pts)\n"
    ),
    # Naming the pattern in prose is not using it.
    "docstring_and_comment": (
        'def f():\n    """The old walk over ``d._actors`` was dead code."""\n'
        "    # d._actors is never populated\n    return None\n"
    ),
    # Different attributes that merely end in "actors".
    "similar_names": (
        "def f(self, r):\n"
        "    return r.dim_actors, self._explode_actors, r.dim_wire_actors\n"
    ),
}


def test_g_actors_collector_flags_the_incident_shapes() -> None:
    missed = [
        name for name, src in _G_ACTORS_FLAGGED.items()
        if not _foreign_actor_reads(ast.parse(src))
    ]
    assert not missed, f"G-ACTORS collector went blind to: {missed}"


def test_g_actors_collector_passes_the_sanctioned_shapes() -> None:
    noisy = {
        name: _foreign_actor_reads(ast.parse(src))
        for name, src in _G_ACTORS_CLEAN.items()
    }
    noisy = {k: v for k, v in noisy.items() if v}
    assert not noisy, f"G-ACTORS collector flagged sanctioned code: {noisy}"
