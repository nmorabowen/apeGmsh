"""R3-S1 (#1785) live check, on stock openseespy: a stage ``Linear``
without a test runs, and gives what the same stage with a test gives.

A two-storey, one-bay elastic portal frame runs two ``Static`` stages
(gravity, then a lateral push), each with ``algorithm Linear``, once
with a convergence test and once without.  ``Linear`` never calls the
test, so the oracle is exact agreement: the same storey drifts after
the push and the same eigen periods of the loaded frame.  A closed-form
sanity check pins the first period's order of magnitude.
"""
from __future__ import annotations

import math
import runpy
from pathlib import Path
from typing import Any

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees

BAY, STOREY = 6.0, 3.0
E, A, INERTIA = 2.0e8, 0.04, 2.0e-4     # kPa, m2, m4
MASS = 20.0                         # t per storey joint
F_LATERAL = 10.0                    # kN per storey


def _frame_fem() -> Any:
    with apeGmsh(model_name="r3s1_frame", verbose=False) as g:
        geo = g.model.geometry
        for k in range(3):
            geo.add_point(0.0, k * STOREY, 0.0, label=f"L{k}")
            geo.add_point(BAY, k * STOREY, 0.0, label=f"R{k}")
        cols, beams = [], []
        for k in (1, 2):
            for side in ("L", "R"):
                geo.add_line(f"{side}{k - 1}", f"{side}{k}", label=f"C{side}{k}")
                cols.append(f"C{side}{k}")
            geo.add_line(f"L{k}", f"R{k}", label=f"B{k}")
            beams.append(f"B{k}")
        g.physical.add_curve(cols, name="columns")
        g.physical.add_curve(beams, name="beams")
        g.physical.add_point(["L0", "R0"], name="bases")
        for k in (1, 2):
            g.physical.add_point([f"L{k}", f"R{k}"], name=f"floor_{k}")
            g.physical.add_point([f"L{k}"], name=f"left_{k}")
        g.mesh.structured.set_transfinite_curve(cols + beams, n_nodes=2)
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _deck(fem: Any, path: Path, *, with_test: bool) -> None:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()
    for pg in ("columns", "beams"):
        ops.element.elasticBeamColumn(pg=pg, transf=transf, A=A, E=E, Iz=INERTIA)
    ops.fix(pg="bases", dofs=(1, 1, 1))
    for k in (1, 2):
        ops.mass(pg=f"floor_{k}", values=(MASS, MASS, 0.0))
    slots: dict[str, Any] = dict(
        algorithm=ops.algorithm.Linear(),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(),
        analysis=ops.analysis.Static(),
    )
    if with_test:
        slots["test"] = ops.test.NormDispIncr(tol=1e-10, max_iter=1)
    solver = ops.analysis.chain(**slots)
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for k in (1, 2):
                p.load(pg=f"floor_{k}", forces=(0.0, -MASS * 9.81, 0.0))
        s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=0.5))
        s.run(n_increments=2)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for k in (1, 2):
                p.load(pg=f"left_{k}", forces=(F_LATERAL * k, 0.0, 0.0))
        s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=1.0))
        s.run(n_increments=1)
    ops.py(str(path), progress=False)


def _run(osp: Any, fem: Any, deck: Path) -> tuple[list[float], list[float]]:
    runpy.run_path(str(deck), run_name="__main__")
    top = {
        k: int(fem.nodes.select(pg=f"left_{k}").ids[0]) for k in (1, 2)
    }
    ux = [osp.nodeDisp(top[1], 1), osp.nodeDisp(top[2], 1)]
    drifts = [ux[0] / STOREY, (ux[1] - ux[0]) / STOREY]
    periods = [2.0 * math.pi / math.sqrt(lam) for lam in osp.eigen(2)]
    osp.wipe()
    return drifts, periods


@pytest.mark.live
def test_linear_stage_without_test_matches_with_test(tmp_path: Path) -> None:
    osp = pytest.importorskip("openseespy.opensees")
    fem = _frame_fem()
    runs = {}
    for with_test in (True, False):
        deck = tmp_path / f"frame_{with_test}.py"
        _deck(fem, deck, with_test=with_test)
        text = deck.read_text(encoding="utf-8")
        assert ("ops.test(" in text) is with_test
        runs[with_test] = _run(osp, fem, deck)

    (d_test, t_test), (d_none, t_none) = runs[True], runs[False]
    assert d_none == pytest.approx(d_test, rel=1e-12)
    assert t_none == pytest.approx(t_test, rel=1e-12)
    # The push moved the frame, and drifts it in the push direction.
    assert d_test[0] > 0.0 and d_test[1] > 0.0
    # Sanity, closed form: a two-storey shear building with rigid beams
    # (storey stiffness 2 * 12 E I / h^3, lumped mass 2 m per floor) has
    # T1 = 2 pi / sqrt(k / M * (3 - sqrt 5) / 2).  Flexible beams soften
    # the frame, so the frame's T1 is longer, but within a factor of 2.
    k = 2 * 12 * E * INERTIA / STOREY**3
    t_shear = 2 * math.pi / math.sqrt(k / (2 * MASS) * (3 - math.sqrt(5)) / 2)
    assert t_shear < t_test[0] < 2.0 * t_shear
