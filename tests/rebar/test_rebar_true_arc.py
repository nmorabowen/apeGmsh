"""Mesh-native curved rebar geometry — `Path(curve="arc"|"spline")` and the
`circular_column(true_arc=True)` option (true circular hoops / spline spiral).

The realised FE elements stay straight 2-node chords (OpenSees has no curved
line element); `true_arc` only upgrades the *authored curve* so the mesher
seeds nodes on the true curve."""
from __future__ import annotations

import math

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Path, TieLayout


# ── L1: Path curve kinds ─────────────────────────────────────────────

def test_path_curve_validation():
    Path(((0, 0, 0), (1, 0, 0)))                       # polyline (default)
    Path(((1, 0, 0), (0, 1, 0), (-1, 0, 0)), curve="arc", arc_center=(0, 0, 0))
    Path(((0, 0, 0), (1, 1, 0), (2, 0, 0)), curve="spline")
    with pytest.raises(ValueError):
        Path(((1, 0, 0), (0, 1, 0)), curve="arc")      # arc needs a centre
    with pytest.raises(ValueError):
        Path(((0, 0, 0), (1, 0, 0)), arc_center=(0, 0, 0))   # centre w/o arc
    with pytest.raises(ValueError):
        Path(((0, 0, 0), (1, 0, 0)), curve="bezier")   # unknown kind


def test_path_curve_round_trips():
    p = Path(((1, 0, 0), (0, 1, 0), (-1, 0, 0)), curve="arc",
             arc_center=(0, 0, 0))
    q = Path.from_dict(p.to_dict())
    assert q.curve == "arc" and q.arc_center == (0.0, 0.0, 0.0)
    s = Path.from_dict(Path(((0, 0, 0), (1, 1, 0)), curve="spline").to_dict())
    assert s.curve == "spline" and s.arc_center is None


# ── circular_column true_arc ─────────────────────────────────────────

def test_circular_true_arc_hoops_are_arcs():
    with apeGmsh(model_name="ta_hoop") as g:
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.05, n_bars=8, bar_db=0.025,
            ties=TieLayout(db=0.01, spacing=0.5), true_arc=True)
    r_tie = 0.3 - 0.05 - 0.01 / 2.0
    assert cage.stirrups                               # discrete hoops
    for s in cage.stirrups:
        assert s.path.curve == "arc"
        cx, cy, cz = s.path.arc_center
        assert (cx, cy) == (0.0, 0.0)                  # ring centre on the axis
        assert s.path.points[0][2] == pytest.approx(cz)
        assert math.hypot(*s.path.points[0][:2]) == pytest.approx(r_tie)
    # longitudinal bars are straight regardless
    assert all(b.path.curve == "polyline" for b in cage.bars)


def test_circular_true_arc_spiral_is_spline():
    with apeGmsh(model_name="ta_spiral") as g:
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.05, n_bars=8, bar_db=0.025,
            ties=TieLayout(db=0.01, spacing=0.15), spiral=True, true_arc=True)
    spirals = [b for b in cage.bars if b.role == "spiral"]
    assert len(spirals) == 1 and spirals[0].path.curve == "spline"


def test_circular_polygon_is_default():
    with apeGmsh(model_name="ta_poly") as g:
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.05, n_bars=8, bar_db=0.025,
            ties=TieLayout(db=0.01, spacing=0.5))           # true_arc=False
    assert all(s.path.curve == "polyline" for s in cage.stirrups)


def test_circular_true_arc_places_embedded():
    with apeGmsh(model_name="ta_place") as g:
        cyl = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 3.0, 0.3)
        g.physical.add_volume([cyl], name="Col")
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.05, n_bars=6, bar_db=0.025,
            ties=TieLayout(db=0.01, spacing=0.6), true_arc=True)
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        assert len(g.reinforce.reinforce_defs) == len(cage.bars) + len(cage.stirrups)


def test_circular_true_arc_conformal_meshes():
    with apeGmsh(model_name="ta_conf") as g:
        g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 3.0, 0.3, label="Col")
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.06, n_bars=6, bar_db=0.02,
            ties=TieLayout(db=0.01, spacing=1.0), true_arc=True)
        g.rebar.place(cage, into="Col", coupling="conformal")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=3)               # arcs + bars embed + mesh
        fem = g.mesh.queries.get_fem_data()
        assert fem.info.n_nodes > 0


# ── hand-authored curved bars ────────────────────────────────────────

def test_hand_authored_arc_and_spline_bars():
    with apeGmsh(model_name="ta_hand") as g:
        ring = tuple((math.cos(t), math.sin(t), 0.0)
                     for t in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.0))
        arc_bar = g.rebar.bar(ring, db=0.02, material="rebar", curve="arc",
                              arc_center=(0.0, 0.0, 0.0))
        assert arc_bar.path.curve == "arc"
        assert arc_bar.path.arc_center == (0.0, 0.0, 0.0)
        spline_bar = g.rebar.bar([(0, 0, 0), (1, 1, 0), (2, 0, 0)],
                                 db=0.02, material="rebar", curve="spline")
        assert spline_bar.path.curve == "spline"
        # fluent path rejects curve= as a kwarg
        with pytest.raises(ValueError):
            g.rebar.bar(db=0.02, material="rebar", curve="spline")
