"""Live oracle for ``lysmer_base`` (#2391).

A homogeneous elastic column standing on a half-space of the same soil
has no impedance contrast, so the Lysmer base absorbs every downgoing
wave. An outcrop velocity pulse ``v(t)`` applied as ``C v(t)`` then
reaches the free surface as exactly ``v(t - H / Vs)``: the surface record
is the outcrop record delayed by the travel time.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.site_response import lysmer_base

pytestmark = pytest.mark.live

H, W, VS, RHO, NU = 20.0, 1.0, 200.0, 2.0, 0.3   # m, m, m/s, t/m^3, -
NEL = 40
DT, N_STEPS, F_RICKER, T0 = 1e-3, 500, 5.0, 0.15


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


def _column() -> Any:
    g = apeGmsh(model_name="lysmer_col", verbose=False)
    g.begin()
    try:
        G = g.model.geometry
        p = [G.add_point(0, 0, 0), G.add_point(W, 0, 0),
             G.add_point(W, H, 0), G.add_point(0, H, 0)]
        lines = [G.add_line(p[0], p[1]), G.add_line(p[1], p[2]),
                 G.add_line(p[2], p[3]), G.add_line(p[3], p[0])]
        surf = G.add_plane_surface(G.add_curve_loop(lines))
        S = g.mesh.structured
        S.set_transfinite_curve(lines[0], 2)
        S.set_transfinite_curve(lines[2], 2)
        S.set_transfinite_curve(lines[1], NEL + 1)
        S.set_transfinite_curve(lines[3], NEL + 1)
        S.set_transfinite_surface(surf)
        S.set_recombine(surf)
        g.physical.add_surface([surf], name="Soil")
        g.physical.add(1, [lines[0]], name="Base")
        g.physical.add(1, [lines[3]], name="Left")
        g.physical.add(1, [lines[1]], name="Right")
        g.physical.add(1, [lines[2]], name="Top")
        gnd = g.decouple_node_set("Base", label="base_ground")
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2), gnd
    finally:
        g.end()


def _ricker(t: np.ndarray) -> np.ndarray:
    a = (np.pi * F_RICKER * (t - T0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def test_matched_half_space_surface_is_the_delayed_outcrop(
        tmp_path: Path) -> None:
    fem, gnd = _column()
    t = np.arange(N_STEPS + 1) * DT
    v_in = 0.1 * _ricker(t)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    G_mod = RHO * VS ** 2
    soil = ops.nDMaterial.ElasticIsotropic(
        E=2.0 * G_mod * (1.0 + NU), nu=NU, rho=RHO)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=soil)
    # Tie the two sides at every elevation: a 1-D shear column.
    left = sorted(fem.nodes.select(pg="Left").ids,
                  key=lambda n: fem.nodes.coords[list(fem.nodes.ids).index(n)][1])
    right = sorted(fem.nodes.select(pg="Right").ids,
                   key=lambda n: fem.nodes.coords[list(fem.nodes.ids).index(n)][1])
    for k, (nl, nr) in enumerate(zip(left, right)):
        for d in (1, 2):
            ops.equation_constraint(constrained=(int(nr), d),
                                    retained=[(int(nl), d, -1.0)])
    base = lysmer_base(ops, "Base", gnd, rho=RHO, vs=VS)
    assert sum(base.c_s) == pytest.approx(RHO * VS * W)
    series = ops.timeSeries.Path(dt=DT, values=tuple(v_in.tolist()))
    with ops.pattern.Plain(series=series) as p:
        base.add_input(p)
    top = int(fem.nodes.select(pg="Top").ids[0])
    out = tmp_path / "top_vel.out"
    ops.recorder.Node(file=str(out), response="vel", nodes=(top,),
                      dofs=(1,), time_format="dt")
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.constraints.Lagrange()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.analysis.Transient()
    # A subprocess: on stock openseespy equationConstraint rows survive
    # wipe(), so an in-process run would poison every later live model.
    ops.py(str(tmp_path / "deck.py"), run=True, analyze_steps=N_STEPS,
           analyze_dt=DT, python=sys.executable)

    rec = np.loadtxt(out)
    t_out, v_top = rec[:, 0], rec[:, 1]
    expected = 0.1 * _ricker(t_out - H / VS)
    peak = np.abs(expected).max()
    assert np.abs(v_top).max() == pytest.approx(peak, rel=0.03)
    assert np.abs(v_top - expected).max() < 0.05 * peak
