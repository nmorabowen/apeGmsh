"""TEMPORARY experiment (removed before merge): is the stock equation tie exact?

Prints, for whatever OpenSees backend resolves:
  * build facts (backend, version(), equationConstraint present?);
  * single-block uniaxial patch tests per element (K / (EA/H));
  * the two-block series stack of tests/test_meshable_part_route.py for six
    plate variants (K / K_exact, top/base reaction balance, and the largest
    |u_s - sum w_i u_mi| over every emitted equationConstraint row).

The equation-tie live gate is bypassed on purpose: this measures the engine
behind the gate. Run from the repo root with PYTHONPATH=.:src
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gmsh
import numpy as np

from apeGmsh.assembly import Assembly
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import live as live_mod
from apeGmsh.opensees.emitter.live import LiveOpsEmitter, get_backend_name
from tests.test_meshable_part_route import E, H, NU, SIDE, _build_block

DELTA = 0.01
K_BLOCK = E * SIDE * SIDE / H
K_STACK = K_BLOCK / 2.0

# Measure the ENGINE: let equationConstraint through on stock.
LiveOpsEmitter._stock_build_gate = lambda self, message: None  # type: ignore[method-assign]

ops_mod = live_mod._get_ops()
print("=" * 78)
print("backend           :", get_backend_name())
print("version()         :", ops_mod.version() if hasattr(ops_mod, "version") else "?")
print("equationConstraint:", hasattr(ops_mod, "equationConstraint"))
print("criticalTimeStep  :", hasattr(ops_mod, "criticalTimeStep"))
print("gmsh              :", gmsh.__version__)
print("python            :", sys.version.split()[0])
print("=" * 78)


def _analysis(ops, handler: str) -> None:
    getattr(ops.constraints, handler)()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


def single_block(tmp: Path, mesh: str, order: int, size: float, element: str) -> None:
    fem = _build_block(tmp / "b.h5", name="b", z0=0.0, mesh=mesh, order=order,
                       size=size, vol_pg="V", bot_pg="Bot", top_pg="Top")
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    m = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    getattr(ops.element, element)(pg="V", material=m)
    ops.fix(pg="Bot", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="Top", dof=3, value=-DELTA)
    _analysis(ops, "Transformation")
    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    assert em.analyze(steps=1) == 0
    lv = em.ops
    lv.reactions()
    r = sum(lv.nodeReaction(int(t), 3) for t in fem.nodes.select(pg="Bot").ids)
    n_el = len(list(fem.elements.select(pg="V").ids))
    lv.wipe()
    print(f"PATCH {element:20s} {mesh}{order} size={size:<4} n_el={n_el:4d}  "
          f"K/(EA/H) = {abs(r) / DELTA / K_BLOCK:.9f}")


def stack(tmp: Path, mesh: str, order: int, size: float, element: str,
          method: str) -> None:
    _build_block(tmp / "c.h5", name="cover", z0=0.0, mesh="hex", order=1,
                 size=5.0, vol_pg="CoverVol", bot_pg="Base", top_pg="CoverTop")
    _build_block(tmp / "p.h5", name="plate", z0=H, mesh=mesh, order=order,
                 size=size, vol_pg="PlateVol", bot_pg="PlateBot", top_pg="PlateTop")
    kw = {} if method == "collocation" else {"method": method}
    g = (
        Assembly("two").add("cover", str(tmp / "c.h5")).add("pl", str(tmp / "p.h5"))
        .couple("cover", "pl", kind="tie", ports=("CoverTop", "PlateBot"),
                dofs=[1, 2, 3], enforce="equation", **kw)
        .materialize()
    )
    fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    m = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.stdBrick(pg="CoverVol", material=m)
    getattr(ops.element, element)(pg="pl.PlateVol", material=m)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="pl.PlateTop", dof=3, value=-DELTA)
    _analysis(ops, "Lagrange")
    em = LiveOpsEmitter(wipe=True)
    ops.build().emit(em)
    assert em.analyze(steps=1) == 0
    lv = em.ops
    lv.reactions()
    rb = sum(lv.nodeReaction(int(t), 3) for t in fem.nodes.select(pg="Base").ids)
    rt = sum(lv.nodeReaction(int(t), 3) for t in fem.nodes.select(pg="pl.PlateTop").ids)
    res, rows = 0.0, 0
    for rec in fem.elements.constraints:
        if getattr(rec, "enforce", None) != "equation":
            continue
        w = np.asarray(rec.weights, dtype=float)
        ms = [int(x) for x in rec.master_nodes]
        for d in rec.dofs:
            us = lv.nodeDisp(int(rec.slave_node), int(d))
            um = sum(wi * lv.nodeDisp(mi, int(d)) for wi, mi in zip(w, ms))
            res = max(res, abs(us - um))
            rows += 1
    n_el = len(list(fem.elements.select(pg="pl.PlateVol").ids))
    lv.wipe()
    print(f"STACK {element:20s} {mesh}{order} size={size:<4} {method:11s} "
          f"plate_el={n_el:4d} rows={rows:4d}  K/K_exact = {abs(rb) / DELTA / K_STACK:.9f}"
          f"  top/base = {abs(rt / rb):.9f}  max|row res| = {res:.2e}")


def run(fn, *args) -> None:
    with tempfile.TemporaryDirectory() as td:
        try:
            fn(Path(td), *args)
        except Exception as e:  # report and keep going: this is a survey
            print(f"FAIL {fn.__name__}{args}: {type(e).__name__}: {str(e)[:400]}")


for case in [("tet", 1, 6.0, "FourNodeTetrahedron"),
             ("tet", 2, 6.0, "TenNodeTetrahedron"),
             ("hex", 1, 3.4, "stdBrick"),
             ("hex", 1, 5.0, "stdBrick")]:
    run(single_block, *case)

if not hasattr(ops_mod, "equationConstraint"):
    print("this build has no equationConstraint: tied stacks not run")
    sys.exit(0)

for case in [("tet", 1, 6.0, "FourNodeTetrahedron", "collocation"),
             ("hex", 1, 3.4, "stdBrick", "collocation"),
             ("hex", 1, 2.5, "stdBrick", "collocation"),
             ("tet", 2, 6.0, "TenNodeTetrahedron", "collocation"),
             ("tet", 1, 6.0, "FourNodeTetrahedron", "mortar"),
             ("hex", 1, 3.4, "stdBrick", "mortar")]:
    run(stack, *case)
