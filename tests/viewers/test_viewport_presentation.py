"""ADR 0089 — viewport model presentation (GUI polish phase 2c).

Covers the results substrate's designed presentation:

* the frozen module constants (``SUBSTRATE_AMBIENT``,
  ``CONTOUR_EDGE_OPACITY``, ``CONTOUR_ACTIVE_OUTLINE_PX``) and the
  criterion-10 factory width defaults (2.5 / 1.0 / 6 px);
* the static feature-edge outline: extraction, deform-follow through
  ``sync_render_surface_points`` (no new pump), and in-place
  re-extraction after a ghost change;
* the D3 "edges over fields" truth table
  (:func:`_substrate_edge_style` × ``_geometries_occluded_by_diagrams``);
* nodes-off boot default + the session snapshot that beats it;
* ``ContourStyle.cmap`` no longer defaulting to ``"jet"``;
* width defaults persisting through ``PreferencesManager``;
* qt: the boot actor state, the criterion-9 toolbar toggles (two-way
  panel ↔ toolbar sync), and live contour demotion / restore.

Headless tests use the same tiny-scene idiom as
``test_scene_instances_s2a.py``; the qt tests run per-file via
``pytest -m qt`` like every other real-window test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# =====================================================================
# Constants + factory defaults
# =====================================================================

def test_presentation_constants_match_adr_0089():
    from apeGmsh.viewers import results_viewer as rv

    assert rv.SUBSTRATE_AMBIENT == pytest.approx(0.35)
    assert rv.SUBSTRATE_DIFFUSE == pytest.approx(0.85)
    assert rv.CONTOUR_EDGE_OPACITY == pytest.approx(0.40)
    assert rv.CONTOUR_ACTIVE_OUTLINE_PX == pytest.approx(2.0)


def test_width_factory_defaults_are_adr_0089_values():
    """Criterion 10 — 2.5 / 1.0 / 6 px are the factory settings the
    Display panel's "Reset to defaults" restores."""
    from apeGmsh.viewers.ui.preferences_manager import DEFAULT_PREFERENCES

    assert DEFAULT_PREFERENCES.outline_width == pytest.approx(2.5)
    assert DEFAULT_PREFERENCES.mesh_line_width == pytest.approx(1.0)
    assert DEFAULT_PREFERENCES.node_marker_size == pytest.approx(6.0)


def test_width_prefs_persist_as_cross_session_defaults(tmp_path: Path):
    """Criterion 10 — a Display-panel width edit round-trips through
    the PreferencesManager JSON as the user's default."""
    from apeGmsh.viewers.ui.preferences_manager import PreferencesManager

    path = tmp_path / "prefs.json"
    mgr = PreferencesManager(path)
    mgr.update({
        "outline_width": 4.0,
        "mesh_line_width": 2.0,
        "node_marker_size": 9.0,
    })
    reloaded = PreferencesManager(path)
    assert reloaded.current.outline_width == pytest.approx(4.0)
    assert reloaded.current.mesh_line_width == pytest.approx(2.0)
    assert reloaded.current.node_marker_size == pytest.approx(9.0)


def test_legacy_prefs_json_without_outline_width_gets_default(
    tmp_path: Path,
):
    """A preferences.json saved before ADR 0089 has no outline_width —
    it must load with the 2.5 px factory default."""
    import json

    from apeGmsh.viewers.ui.preferences_manager import PreferencesManager

    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"point_size": 12.0}), encoding="utf-8")
    mgr = PreferencesManager(path)
    assert mgr.current.outline_width == pytest.approx(2.5)
    assert mgr.current.point_size == pytest.approx(12.0)


# =====================================================================
# D3 — "edges over fields" truth table (headless)
# =====================================================================

def test_substrate_edge_style_full_strength_without_field():
    from apeGmsh.viewers.results_viewer import _substrate_edge_style
    from apeGmsh.viewers.ui.theme import PALETTE_CATPPUCCIN_MOCHA as pal

    edge = _substrate_edge_style(pal, demoted=False, outline_width=2.5)
    assert edge["edge_color"] == pal.substrate_edge_color
    assert edge["edge_opacity"] == pytest.approx(1.0)
    assert edge["outline_width"] == pytest.approx(2.5)


def test_substrate_edge_style_demotes_over_fields():
    from apeGmsh.viewers.results_viewer import (
        CONTOUR_ACTIVE_OUTLINE_PX,
        CONTOUR_EDGE_OPACITY,
        _substrate_edge_style,
    )
    from apeGmsh.viewers.ui.theme import PALETTE_CATPPUCCIN_MOCHA as pal

    edge = _substrate_edge_style(pal, demoted=True, outline_width=2.5)
    assert edge["edge_color"] == pal.outline_color
    assert edge["edge_opacity"] == pytest.approx(CONTOUR_EDGE_OPACITY)
    assert edge["outline_width"] == pytest.approx(CONTOUR_ACTIVE_OUTLINE_PX)


def test_substrate_edge_style_honors_user_outline_width():
    """A user-sized outline restores to the USER's width, not 2.5."""
    from apeGmsh.viewers.results_viewer import _substrate_edge_style
    from apeGmsh.viewers.ui.theme import PALETTE_PAPER as pal

    edge = _substrate_edge_style(pal, demoted=False, outline_width=5.0)
    assert edge["outline_width"] == pytest.approx(5.0)


class _StubDiagram:
    def __init__(self, *, attached=True, visible=True, effective=True,
                 occludes=True):
        self.is_attached = attached
        self.is_visible = visible
        self.is_effectively_visible = effective
        self.occludes_substrate = occludes


class _StubDirector:
    """Just enough director for ``_geometries_occluded_by_diagrams``."""

    def __init__(self, diagrams, owner):
        class _Registry:
            def __init__(self, ds):
                self._ds = ds

            def diagrams(self):
                return list(self._ds)

        class _Geoms:
            def __init__(self, o):
                self._o = o

            def geometry_for_layer(self, _d):
                return self._o

        self.registry = _Registry(diagrams)
        self.geometries = _Geoms(owner)


class _StubGeom:
    def __init__(self, gid="g1"):
        self.id = gid


def test_demotion_truth_table_via_occlusion_set():
    """The demotion predicate IS the occlusion set: an effectively
    visible field diagram demotes its owner; a gate-hidden, detached,
    intent-hidden, or non-occluding one does not."""
    from apeGmsh.viewers.results_viewer import (
        _geometries_occluded_by_diagrams,
    )

    owner = _StubGeom()
    cases = [
        (_StubDiagram(), {"g1"}),
        (_StubDiagram(effective=False), set()),
        (_StubDiagram(visible=False), set()),
        (_StubDiagram(attached=False), set()),
        (_StubDiagram(occludes=False), set()),
    ]
    for diagram, expected in cases:
        director = _StubDirector([diagram], owner)
        assert _geometries_occluded_by_diagrams(director) == expected


# =====================================================================
# Feature-edge outline — extraction + deform follow (headless)
# =====================================================================

def _box_scene():
    """A 2×1×1 two-hex FEMSceneData — enough surface for real feature
    edges (12 box edges + the shared-face crease)."""
    import pyvista as pv

    from apeGmsh.viewers.scene.fem_scene import FEMSceneData

    grid = pv.ImageData(
        dimensions=(3, 2, 2), spacing=(1.0, 1.0, 1.0),
    ).cast_to_unstructured_grid()
    points = np.asarray(grid.points, dtype=np.float64).copy()
    n = grid.n_points
    node_ids = np.arange(1, n + 1, dtype=np.int64)
    grid.point_data["node_id"] = node_ids
    return FEMSceneData(
        grid=grid,
        node_ids=node_ids,
        node_id_to_idx={int(t): i for i, t in enumerate(node_ids)},
        cell_to_element_id=np.arange(1, grid.n_cells + 1, dtype=np.int64),
        element_id_to_cell={
            i + 1: i for i in range(grid.n_cells)
        },
        model_diagonal=float(np.linalg.norm([2.0, 1.0, 1.0])),
        cell_dim=np.full(grid.n_cells, 3, dtype=np.int8),
        reference_points=points,
    )


def _scene_with_outline():
    from apeGmsh.viewers.backends.pyvista_qt import (
        build_outline_edges,
        build_render_surface,
    )

    scene = _box_scene()
    rs = build_render_surface(scene.grid)
    assert rs is not None
    scene.render_surface = rs
    edges = build_outline_edges(rs, 25.0)
    return scene, rs, edges


def test_build_outline_edges_extracts_static_feature_edges():
    scene, rs, edges = _scene_with_outline()
    assert edges is not None
    assert rs.outline is edges
    assert edges.n_lines > 0
    # Every outline point maps to a grid row and sits exactly on it.
    rows = rs.outline_rows
    assert rows is not None and rows.size == edges.n_points
    np.testing.assert_allclose(
        np.asarray(edges.points),
        np.asarray(scene.grid.points)[rows],
    )
    # The extraction tag never leaks onto the render surface.
    assert "_ape_outline_row" not in rs.surface.point_data


def test_outline_follows_deform_through_render_surface_sync():
    """Checklist 4 — the DEFORM lane's existing scatter moves the
    outline; no reference-configuration outline is left behind."""
    from apeGmsh.viewers._pump_set import sync_render_surface_points

    scene, rs, edges = _scene_with_outline()
    deformed = scene.reference_points + np.array([0.0, 0.0, 0.5])
    scene.grid.points = deformed
    sync_render_surface_points(scene, deformed)
    np.testing.assert_allclose(
        np.asarray(rs.outline.points), deformed[rs.outline_rows],
    )
    # And back to reference (deformed_pts=None reads the grid).
    scene.grid.points = scene.reference_points.copy()
    sync_render_surface_points(scene, None)
    np.testing.assert_allclose(
        np.asarray(rs.outline.points),
        scene.reference_points[rs.outline_rows],
    )


def test_outline_reextracts_in_place_after_ghost_change():
    """A per-cell hide re-extracts the outline into the SAME polydata
    object (the actor's mapper keeps its bound dataset)."""
    from apeGmsh.viewers.backends.pyvista_qt import (
        refresh_outline_edges,
        refresh_render_surface,
    )

    scene, rs, edges = _scene_with_outline()
    before = np.asarray(edges.points).copy()
    ghosts = np.zeros(scene.grid.n_cells, dtype=np.uint8)
    ghosts[0] = 32  # vtkDataSetAttributes.HIDDENCELL
    scene.grid.cell_data["vtkGhostType"] = ghosts
    assert refresh_render_surface(scene.grid, rs, []) is True
    assert refresh_outline_edges(rs) is True
    assert rs.outline is edges          # identity preserved
    after = np.asarray(edges.points)
    # Hiding one of the two hexes changes the outline geometry.
    assert after.shape != before.shape or not np.allclose(after, before)


def test_scene_without_feature_edges_yields_no_outline():
    """A 1-D model (lines only) has no feature edges — the builder
    reports None and nothing downstream assumes an outline."""
    import pyvista as pv

    from apeGmsh.viewers.backends.pyvista_qt import (
        build_outline_edges,
        build_render_surface,
    )

    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    cells = np.array([2, 0, 1, 2, 1, 2], dtype=np.int64)
    celltypes = np.array([3, 3], dtype=np.uint8)   # VTK_LINE
    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    rs = build_render_surface(grid)
    if rs is None:
        pytest.skip("no render surface for line grids on this VTK")
    assert build_outline_edges(rs, 25.0) is None
    assert rs.outline is None


# =====================================================================
# D2 — nodes off at boot; session snapshot beats the default
# =====================================================================

def test_results_geometry_show_nodes_defaults_off():
    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    mgr = GeometryManager()
    assert mgr.active is not None
    assert mgr.active.show_nodes is False
    assert mgr.active.show_mesh is True


def test_session_snapshot_show_nodes_beats_boot_default():
    """A restored session's show_nodes wins over the D2 boot default —
    including legacy sessions saved before the field existed, which
    deserialize to True (what the user saw when they saved)."""
    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot,
        _deserialize_geometry,
        _serialize_geometry,
    )

    snap = GeometrySnapshot(id="g", name="Geometry 1", show_nodes=True)
    restored = _deserialize_geometry(_serialize_geometry(snap))
    assert restored.show_nodes is True
    # Legacy payload without the key → True (pre-0089 sessions showed
    # nodes; honoring the saved look is the contract).
    legacy = _serialize_geometry(snap)
    legacy.pop("show_nodes", None)
    assert _deserialize_geometry(legacy).show_nodes is True


# =====================================================================
# D3 — ContourStyle.cmap no longer defaults to "jet" (checklist 7)
# =====================================================================

def test_contour_style_cmap_defaults_from_palette_not_jet():
    from apeGmsh.viewers.core._lut_manager import is_preset
    from apeGmsh.viewers.diagrams._styles import ContourStyle
    from apeGmsh.viewers.ui.theme import THEME

    style = ContourStyle()
    assert style.cmap != "jet"
    assert is_preset(style.cmap)
    expected = THEME.current.cmap_seq
    assert style.cmap == (
        expected if is_preset(expected) else "viridis"
    )


# =====================================================================
# Toolbar glyphs (criterion 9)
# =====================================================================

def test_icon_factory_has_mesh_and_dot_glyphs():
    from apeGmsh.viewers.ui._icon_factory import glyph_names

    names = set(glyph_names())
    assert {"mesh", "dot"} <= names


# =====================================================================
# Qt — real-window boot state, toolbar toggles, contour demotion
# (local-only; run per-file: ``pytest -m qt <this file>``)
# =====================================================================

@pytest.fixture
def presentation_results(g, tmp_path: Path):
    """Tiny native Results on a solid box, with a displacement field so
    contours and deform have data."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 2, label="col")
    g.physical.add_volume("col", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size

    path = tmp_path / "presentation.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="push", kind="static", time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.vstack(
                    [np.zeros(n_nodes), np.ones(n_nodes)],
                ),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _factory_prefs(monkeypatch):
    """Pin PREFERENCES to the code defaults (the dev machine's saved
    preferences.json must not leak into assertions) and disable the
    JSON write the width callbacks perform."""
    from apeGmsh.viewers.ui import preferences_manager as pm

    monkeypatch.setattr(pm.PREFERENCES, "_current", pm.DEFAULT_PREFERENCES)
    monkeypatch.setattr(pm.PREFERENCES, "_save", lambda *_a, **_k: None)


@pytest.mark.qt
def test_boot_presentation_state(presentation_results, monkeypatch):
    """Checklist 1/2 boot half: outline actor on at the user width,
    interior edges 1 px, node cloud OFF, ambient lift applied."""
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import (
        SUBSTRATE_AMBIENT,
        ResultsViewer,
    )

    _factory_prefs(monkeypatch)
    viewer = ResultsViewer(
        presentation_results, title="0089-boot",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            gid = viewer._director.geometries.active.id
            fill, wf = viewer._scene_actors[gid]
            outline = viewer._outline_actors.get(gid)
            seen["ambient"] = float(fill.GetProperty().GetAmbient())
            seen["wf_width"] = float(wf.GetProperty().GetLineWidth())
            seen["outline_exists"] = outline is not None
            if outline is not None:
                seen["outline_visible"] = bool(outline.GetVisibility())
                seen["outline_width"] = float(
                    outline.GetProperty().GetLineWidth(),
                )
            seen["nodes_hidden"] = (
                viewer._node_cloud_actor is None
                or not bool(viewer._node_cloud_actor.GetVisibility())
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen["ambient"] == pytest.approx(SUBSTRATE_AMBIENT)
    assert seen["wf_width"] == pytest.approx(1.0)
    assert seen["outline_exists"]
    assert seen["outline_visible"]
    assert seen["outline_width"] == pytest.approx(2.5)
    assert seen["nodes_hidden"]


@pytest.mark.qt
def test_toolbar_toggles_flip_display_state_both_ways(
    presentation_results, monkeypatch,
):
    """Criterion 9 — the ``dot`` / ``mesh`` toolbar actions flip the
    active geometry's flags, and an external ``set_display`` reflects
    back into the checked state AND the panel checkboxes."""
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    _factory_prefs(monkeypatch)
    viewer = ResultsViewer(
        presentation_results, title="0089-toggles",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            geoms = viewer._director.geometries
            geom = geoms.active
            nodes_act = viewer._show_nodes_action
            mesh_act = viewer._show_mesh_action
            seen["boot_nodes_unchecked"] = not nodes_act.isChecked()
            seen["boot_mesh_checked"] = mesh_act.isChecked()
            # Toolbar → state.
            nodes_act.setChecked(True)      # fires toggled
            seen["nodes_on_after_action"] = geom.show_nodes is True
            node_actor = viewer._node_cloud_actor
            seen["node_actor_visible"] = bool(node_actor.GetVisibility())
            mesh_act.setChecked(False)
            seen["mesh_off_after_action"] = geom.show_mesh is False
            # State → toolbar (external mutator, e.g. session restore).
            geoms.set_display(geom.id, show_nodes=False, show_mesh=True)
            seen["nodes_action_synced"] = not nodes_act.isChecked()
            seen["mesh_action_synced"] = mesh_act.isChecked()
            # …and the panel checkboxes mirror too.
            panel = viewer._geometry_panel
            panel.show_geometry(geom.id)
            geoms.set_display(geom.id, show_nodes=True)
            seen["panel_checkbox_synced"] = (
                panel._cb_show_nodes.isChecked() is True
            )
            seen["panel_flip_reaches_action"] = nodes_act.isChecked()
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    for key, ok in seen.items():
        assert ok, f"{key} failed: {seen}"


@pytest.mark.qt
def test_contour_demotes_edges_and_restores_on_detach(
    presentation_results, monkeypatch,
):
    """Checklist 3 — attaching a contour drops the wireframe to 40%
    ``outline_color`` and the outline to 2.0 px; removing it restores
    the D1 widths/colors."""
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import (
        CONTOUR_ACTIVE_OUTLINE_PX,
        CONTOUR_EDGE_OPACITY,
        ResultsViewer,
    )

    _factory_prefs(monkeypatch)
    viewer = ResultsViewer(
        presentation_results, title="0089-demotion",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            from apeGmsh.viewers.diagrams._starter import (
                add_default_contour,
            )

            director = viewer._director
            gid = director.geometries.active.id
            _fill, wf = viewer._scene_actors[gid]
            outline = viewer._outline_actors.get(gid)
            diagram = add_default_contour(director)
            seen["wf_opacity_demoted"] = float(
                wf.GetProperty().GetOpacity(),
            )
            seen["outline_width_demoted"] = (
                float(outline.GetProperty().GetLineWidth())
                if outline is not None else None
            )
            geom = director.geometries.active
            geom.compositions.remove_layer(diagram)
            director.registry.remove(diagram)
            seen["wf_opacity_restored"] = float(
                wf.GetProperty().GetOpacity(),
            )
            seen["outline_width_restored"] = (
                float(outline.GetProperty().GetLineWidth())
                if outline is not None else None
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen["wf_opacity_demoted"] == pytest.approx(
        CONTOUR_EDGE_OPACITY,
    )
    assert seen["outline_width_demoted"] == pytest.approx(
        CONTOUR_ACTIVE_OUTLINE_PX,
    )
    assert seen["wf_opacity_restored"] == pytest.approx(1.0)
    assert seen["outline_width_restored"] == pytest.approx(2.5)
