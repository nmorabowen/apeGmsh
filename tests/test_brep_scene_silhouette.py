"""Smoke-test: build_brep_scene produces actors with theme-driven silhouette.

Runs off-screen; no Qt, just a headless PyVista Plotter. Verifies the
scene builds without error for all four dimensions after the aesthetic
Step 3 changes (silhouette + flat shading on dim=2 and dim=3).
"""
from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")
gmsh = pytest.importorskip("gmsh")


@pytest.fixture
def gmsh_box():
    """A 1×1×1 box as a minimal BRep model."""
    gmsh.initialize([])
    gmsh.model.add("test_brep_scene_silhouette")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    yield
    gmsh.finalize()


def test_scene_builds_for_all_dims(gmsh_box):
    from apeGmsh.viewers.scene.brep_scene import build_brep_scene

    p = pv.Plotter(off_screen=True)
    try:
        registry = build_brep_scene(p, dims=[0, 1, 2, 3])
        # A solid box has entities at every dim (corners, edges, faces, volume)
        assert set(registry.dim_actors) == {0, 1, 2, 3}
    finally:
        p.close()


def test_outline_matches_active_theme(gmsh_box):
    """After set_theme('paper'), d3_kwargs silhouette color is black #000000."""
    from apeGmsh.viewers.ui.theme import THEME
    from apeGmsh.viewers.scene.brep_scene import build_brep_scene

    THEME.set_theme("paper")
    try:
        p = pv.Plotter(off_screen=True)
        try:
            registry = build_brep_scene(p, dims=[3])
            # PR #199: silhouette moved off add_mesh(silhouette=) onto a
            # tracked add_silhouette actor; kwargs live here now.
            sil = registry.dim_silhouette_kwargs.get(3, {})
            assert sil.get("color") == "#000000"
            # Paper silhouette is heavier (3.0 px) than dark themes (2.5)
            assert sil.get("line_width") == pytest.approx(3.0)
        finally:
            p.close()
    finally:
        THEME.set_theme("catppuccin_mocha")
