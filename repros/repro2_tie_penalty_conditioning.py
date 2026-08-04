r"""Defect 2 reproduction — tie penalty default 1e18 vs N/mm/MPa units.

Two stacked 100x100x50 blocks (E = 200 000 MPa, nu = 0), independently
meshed with different sizes (non-matching interface), tied at mid-height.
Exact series stiffness: K = E*A/L = 200000*10000/100 = 2.0e7 N/mm.

Sweeps the tie route:  enforce="equation", penalty 1e10 / 1e12 / 1e18
(the default), reports K = P/u_mean and convergence per route.

Run:
  set PYTHONPATH=<worktree>\src
  C:\Users\nmb\venv\opensees_env\Scripts\python.exe repros\repro2_tie_penalty_conditioning.py
"""
import tempfile
from pathlib import Path

import numpy as np

import apeGmsh as _pkg
print(f"apeGmsh from: {_pkg.__file__}")
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

E, NU = 200_000.0, 0.0
K_EXACT = E * 100.0 * 100.0 / 100.0          # 2.0e7 N/mm
P_TOTAL = 2.0e6                              # -> u_exact = 0.1 mm

TMP = Path(tempfile.mkdtemp(prefix="apeGmsh_repro2_"))


def build_module(fname, name, z0, size):
    path = TMP / fname
    with apeGmsh(model_name=name, save_to=str(path)) as g:
        g.model.geometry.add_box(0, 0, z0, 100, 100, 50, label="blk")
        g.model.sync()
        g.model.select("blk").to_physical(f"{name}_body")
        g.model.select(dim=2).on_plane(
            (0, 0, z0 + 50), (0, 0, 1), tol=1e-3).to_physical(f"{name}_top")
        g.model.select(dim=2).on_plane(
            (0, 0, z0), (0, 0, 1), tol=1e-3).to_physical(f"{name}_bot")
        g.mesh.sizing.set_size("blk", size)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=None)
    return path


pa = build_module("mod_a.h5", "A", 0.0, 25.0)
pb = build_module("mod_b.h5", "B", 50.0, 20.0)   # conformal geometry, non-matching mesh


def run_route(enforce, stiffness):
    g = apeGmsh.from_h5(pa)
    g.compose(str(pb), label="B")
    kwargs = dict(tolerance=1.0, enforce=enforce)
    if stiffness is not None:
        kwargs["stiffness"] = stiffness
    g.constraints.tie("A_top", "B.B_bot", dofs=[1, 2, 3], **kwargs)
    fem = g._fem
    n_tie = len(list(fem.elements.constraints))
    if n_tie == 0:
        return "TIE RESOLVED 0 RECORDS", float("nan")

    top_ids = [
        int(i)
        for e in fem.nodes.physical._groups.values()
        if e.get("name") == "B.B_top"
        for i in e["node_ids"]
    ]
    f_node = P_TOTAL / len(top_ids)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.FourNodeTetrahedron(pg="A_body", material=mat)
    ops.element.FourNodeTetrahedron(pg="B.B_body", material=mat)
    ops.fix(pg="A_bot", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="B.B_top", forces=(0.0, 0.0, -f_node))

    if enforce == "equation":
        ops.constraints.Lagrange()
        ops.test.NormUnbalance(tol=1e-3, max_iter=30)
    else:
        ops.constraints.Transformation()
        ops.test.NormDispIncr(tol=1e-8, max_iter=30)
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    ret = emitter.analyze(steps=1)
    if ret != 0:
        emitter.ops.wipe()
        return f"FAILED TO CONVERGE (analyze ret={ret}, {n_tie} tie recs)", float("nan")
    u = float(np.mean([emitter.ops.nodeDisp(n, 3) for n in top_ids]))
    emitter.ops.wipe()
    k = P_TOTAL / abs(u)
    return f"K = {k:.4e} N/mm  (err {100*(k-K_EXACT)/K_EXACT:+.2f}%, {n_tie} tie recs)", k


for label, enforce, stiff in [
    ("equation           ", "equation", None),
    ("penalty K=1e10     ", "penalty", 1e10),
    ("penalty K=1e12     ", "penalty", 1e12),
    ("penalty K=1e18 (default)", "penalty", None),
]:
    try:
        msg, _ = run_route(enforce, stiff)
    except Exception as exc:  # noqa: BLE001
        msg = f"raised {type(exc).__name__}: {str(exc)[:140]}"
    print(f"{label}: {msg}")

print(f"exact closed form   : K = {K_EXACT:.4e} N/mm")
print(f"tmp dir: {TMP}")
