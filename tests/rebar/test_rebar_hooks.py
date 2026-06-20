"""P3 — hook geometry emission through g.rebar.place (ADR 0066 §3, §5).

Live gmsh: build a cage with hooks, place it, and measure the emitted
curves (tail length == resolved ACI value; true_arc produces a real arc).
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage, Hook
from apeGmsh.rebar.detailing import ACI318, BarCatalog


def _len(tag: int) -> float:
    return gmsh.model.occ.getMass(1, tag)        # curve length (occ synced)


def _metres_aci() -> ACI318:
    # model in metres: #8 -> 1.0 in * 0.0254 = 0.0254 m diameter
    return ACI318(BarCatalog(unit_length=0.0254, base="imperial"))


def test_metadata_hook_appends_tail_of_resolved_length():
    with apeGmsh(model_name="hook_meta") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.9)], db="#8",
                          material="rebar", end_hook=Hook.standard_90(),
                          name="L1")
        pl = g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")
        m = pl.members[0]
        # 1 main line + 1 metadata tail segment (no arc)
        assert len(m.line_tags) == 2
        types = [gmsh.model.getType(1, t) for t in m.line_tags]
        assert "Circle" not in types
        # 90° standard hook tail = 12·d_b = 12 · 0.0254
        assert _len(m.line_tags[-1]) == pytest.approx(12 * 0.0254, rel=1e-3)


def test_true_arc_hook_emits_a_real_arc_plus_tail():
    with apeGmsh(model_name="hook_arc") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.9)], db="#8",
                          material="rebar", end_hook=Hook.standard_90(),
                          name="L1")
        pl = g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal",
                           true_arc=True)
        m = pl.members[0]
        # main line + arc + tail
        assert len(m.line_tags) == 3
        types = [gmsh.model.getType(1, t) for t in m.line_tags]
        assert types.count("Circle") == 1            # the fillet arc
        # tail still 12·d_b
        assert _len(m.line_tags[-1]) == pytest.approx(12 * 0.0254, rel=1e-3)


def test_true_arc_180_hook_splits_into_two_arcs():
    with apeGmsh(model_name="hook_180") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.5)], db="#8",
                          material="rebar", end_hook=Hook.standard_180(),
                          name="L1")
        pl = g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal",
                           true_arc=True)
        m = pl.members[0]
        types = [gmsh.model.getType(1, t) for t in m.line_tags]
        assert types.count("Circle") == 2            # 180° → two 90° arcs


def test_explicit_numeric_hook_needs_no_standard():
    with apeGmsh(model_name="hook_explicit") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.9)], db=0.0254,
                          material="rebar",
                          end_hook=Hook(angle=90, tail=0.3, bend_radius=0.05),
                          name="L1")
        pl = g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")
        assert _len(pl.members[0].line_tags[-1]) == pytest.approx(0.3, rel=1e-3)


def test_designation_hook_without_standard_raises():
    with apeGmsh(model_name="hook_nostd") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.9)], db=0.0254,
                          material="rebar", end_hook=Hook.standard_90(),
                          name="L1")                  # tail unresolved, no std
        with pytest.raises(ValueError):
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")


def test_centroid_hook_bends_toward_core():
    with apeGmsh(model_name="hook_dir") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        # corner bar at (0.1, 0.1); a 90° hook turning toward the section
        # centroid should sweep toward +x, +y
        bar = g.rebar.bar([(0.1, 0.1, 0.1), (0.1, 0.1, 1.9)], db="#8",
                          material="rebar", end_hook=Hook.standard_90(),
                          name="L1")
        pl = g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")
        tail = pl.members[0].line_tags[-1]
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, tail)
        assert xmax > 0.1 + 1e-6 and ymax > 0.1 + 1e-6


def test_stirrup_twin_tail_closure_emitted_with_standard():
    with apeGmsh(model_name="hook_stirrup") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        tie = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db=0.0095, material="rebar",
                                   z=1.0, db_value=0.0095, name="T1")
        pl = g.rebar.place(Cage(stirrups=(tie,)), into="V", coupling="conformal")
        m = pl.members[0]
        # 4 closed-loop legs + TWO seismic-135 closure tails (twin-tail seam)
        assert len(m.line_tags) == 6


def test_stirrup_single_hook_when_twin_tail_off():
    with apeGmsh(model_name="hook_stirrup_single") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="V")
        g.rebar.use_standard(_metres_aci())
        tie = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db=0.0095, material="rebar",
                                   z=1.0, db_value=0.0095, name="T1")
        pl = g.rebar.place(Cage(stirrups=(tie,)), into="V", coupling="conformal",
                           twin_tail=False)
        m = pl.members[0]
        # 4 closed-loop legs + 1 closure tail (simplified single hook)
        assert len(m.line_tags) == 5
