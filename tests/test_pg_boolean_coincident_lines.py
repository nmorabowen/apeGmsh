"""Coincident-curve PG survival across a boolean cut.

Regression for the "grouping the lines one over the other" bug.  An
arch frame (columns + arc) is modelled as standalone curves that
geometrically coincide with the inner boundary of a soil region.  The
soil region is then cut out of an outer box.  The frame curves are NOT
inputs to that cut and OCC never touches them, yet the pre-fix
``remap_physical_groups`` re-discovered every non-input PG entity by
geometric matching — which cannot distinguish two coincident curves, so
each PG absorbed the other's entities:

    Frame              -> {1,2,3,4,5,6}   (absorbed the 3 soil curves)
    internal_boundary  -> {1,2,3,4,5,6,7} (absorbed the 3 frame curves)
    BC_left            -> {1,6}           (absorbed a coincident soil point)

The untouched-survivor partition in ``remap_physical_groups`` keeps the
coincident-but-distinct entities apart: an entity OCC never operated on
(stable tag + unchanged signature) is kept verbatim and excluded from
the geometric-match candidate pool.
"""
from __future__ import annotations

import gmsh

from apeGmsh import apeGmsh

L, H1, H2, BOX_OFFSET = 4.0, 3.0, 2.0, 2.0
LC = 1.0


def _pg_entities(name: str) -> set[int]:
    for dim, tag in gmsh.model.getPhysicalGroups():
        if gmsh.model.getPhysicalName(dim, tag) == name:
            return {int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)}
    return set()


def test_arch_frame_over_soil_not_conflated_by_cut() -> None:
    with apeGmsh(model_name="arch_frame", verbose=False) as g:
        geo = g.model.geometry

        # --- Arch frame: standalone columns + arc ---
        geo.add_point(-L / 2, 0.0, 0.0, mesh_size=LC, label="base_l")
        geo.add_point(L / 2, 0.0, 0.0, mesh_size=LC, label="base_r")
        geo.add_point(-L / 2, H1, 0.0, mesh_size=LC, label="sh_l")
        geo.add_point(L / 2, H1, 0.0, mesh_size=LC, label="sh_r")
        geo.add_point(0.0, H1 + H2, 0.0, mesh_size=LC, label="center")
        geo.add_line("base_l", "sh_l", label="left_col")
        geo.add_line("sh_r", "base_r", label="right_col")
        geo.add_arc("sh_l", "center", "sh_r", through_point=True, label="arch")
        g.model.select(target="base_l").to_physical(name="BC_left")
        g.model.select(target="base_r").to_physical(name="BC_right")
        g.model.select(dim=1).to_physical(name="Frame")

        # --- Soil inner boundary: coincident with the frame ---
        geo.add_point(-L / 2, 0.0, 0.0, mesh_size=LC, label="sb_l")
        geo.add_point(L / 2, 0.0, 0.0, mesh_size=LC, label="sb_r")
        geo.add_point(-L / 2, H1, 0.0, mesh_size=LC, label="ssh_l")
        geo.add_point(L / 2, H1, 0.0, mesh_size=LC, label="ssh_r")
        geo.add_point(0.0, H1 + H2, 0.0, mesh_size=LC, label="scenter")
        geo.add_line("sb_l", "ssh_l", label="left_soil")
        geo.add_line("ssh_r", "sb_r", label="right_soil")
        geo.add_arc("ssh_l", "scenter", "ssh_r", through_point=True, label="soil_arch")
        geo.add_line("sb_l", "sb_r", label="soil_bottom")
        g.model.select(
            target=["left_soil", "soil_arch", "right_soil", "soil_bottom"],
        ).to_physical(name="internal_boundary")
        geo.add_curve_loop(
            ["left_soil", "soil_arch", "right_soil", "soil_bottom"],
            label="internal_loop",
        )
        geo.add_plane_surface("internal_loop", label="internal_soil")

        # --- Outer soil box, cut by the inner region ---
        geo.add_rectangle(
            x=-L / 2 - BOX_OFFSET, y=-BOX_OFFSET, z=0.0,
            dx=2 * BOX_OFFSET + L, dy=H1 + H2 + 2 * BOX_OFFSET, label="soil_box",
        )
        g.model.boolean.cut(
            objects=["soil_box"], tools=["internal_soil"], label="soil_domain",
        )

        frame = _pg_entities("Frame")
        ib = _pg_entities("internal_boundary")
        bc_l = _pg_entities("BC_left")
        bc_r = _pg_entities("BC_right")

        # The frame (2 columns + 1 arc) is kept verbatim and never
        # absorbs the coincident soil boundary curves.
        assert len(frame) == 3, f"Frame conflated: {sorted(frame)}"
        assert frame.isdisjoint(ib), (
            f"Frame {sorted(frame)} and internal_boundary {sorted(ib)} "
            f"share coincident curves"
        )
        # The soil inner boundary keeps its 4 curves, none of them a
        # frame curve.
        assert len(ib) == 4, f"internal_boundary conflated: {sorted(ib)}"

        # Coincident support points are not merged into the BC point PGs.
        assert len(bc_l) == 1, f"BC_left absorbed a coincident point: {sorted(bc_l)}"
        assert len(bc_r) == 1, f"BC_right absorbed a coincident point: {sorted(bc_r)}"
