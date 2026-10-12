"""ADR 0089 — viewport model presentation (GUI polish phase 2c).

Covers the results substrate's designed presentation:

* the criterion-10 factory width defaults (2.5 / 1.0 / 6 px);
* the static feature-edge outline: extraction, deform-follow through
  ``sync_render_surface_points`` (no new pump), and in-place
  re-extraction after a ghost change;
* nodes-off boot default + the session snapshot that beats it;
* ``ContourStyle.cmap`` no longer defaulting to ``"jet"``;
* width defaults persisting through ``PreferencesManager``.

The legacy ``ResultsViewer`` constants, the edges-over-fields truth
table and its real-window tests went with that viewer (X3-d, #1734).
Headless tests use the same tiny-scene idiom as
``test_scene_instances_s2a.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# =====================================================================
# Factory defaults
# =====================================================================

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
