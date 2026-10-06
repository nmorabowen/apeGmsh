"""3D solid steel cantilever under a tip surface load, compared with Euler-Bernoulli.

Box 4.0 m (x) x 0.3 m (y) x 0.5 m (z), structured hex8 mesh, clamped at
x = 0, uniform downward traction on the free face x = L. Linear static.
Writes the model archive (model.h5) and a native results file (results.h5)
into out_02/.

Units: N, m, Pa.
"""
from pathlib import Path

import numpy as np
import openseespy.opensees as osp

from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results.capture.spec import DomainCaptureSpec

OUT = Path(__file__).parent / "out_02"
OUT.mkdir(exist_ok=True)

# ---- data ------------------------------------------------------------------
L, B, H = 4.0, 0.3, 0.5          # length (x), width (y), depth (z)
E, NU = 200e9, 0.3               # steel
P = 10e3                         # total tip load, downward [N]
MESH_SIZE = 0.05                 # hex edge length -> 80 x 6 x 10 elements

q = P / (B * H)                  # traction on the tip face [Pa]
EPS = 1e-6

# ---- geometry, groups, load, mesh -------------------------------------------
with apeGmsh(model_name="solid_cantilever") as g:
    beam = g.model.geometry.add_box(0, 0, 0, L, B, H, label="beam")
    g.physical.add_volume("beam", name="Beam")

    (g.model.select(None, dim=2)
        .in_box((-EPS, -EPS, -EPS), (EPS, B + EPS, H + EPS))
        .to_physical("Clamp"))
    (g.model.select(None, dim=2)
        .in_box((L - EPS, -EPS, -EPS), (L + EPS, B + EPS, H + EPS))
        .to_physical("Tip"))

    with g.loads.case("tip_load"):
        g.loads.surface.traction("Tip", (0.0, 0.0, -q))

    g.mesh.structured.set_transfinite_box(beam, size=MESH_SIZE)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info)

# ---- OpenSees model ---------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=7850.0)
ops.element.stdBrick(pg="Beam", material=steel)
ops.fix(pg="Clamp", dofs=(1, 1, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("tip_load")

ops.constraints.Plain()
ops.numberer.RCM()
ops.system.UmfPack()
ops.test.NormDispIncr(tol=1e-10, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()

ops.h5(OUT / "model.h5")

# ---- run + capture results --------------------------------------------------
rc = ops.analyze(steps=1)
assert rc == 0, f"analysis failed (rc={rc})"

spec = DomainCaptureSpec(opensees=ops)
spec.nodes(pg="Beam", components=["displacement"], name="disp")
results_path = OUT / "results.h5"
with ops.domain_capture(spec, path=results_path, ops=osp) as cap:
    cap.begin_stage("static", kind="static")
    cap.step(t=osp.getTime())
    cap.end_stage()

# ---- compare with Euler-Bernoulli -------------------------------------------
model = OpenSeesModel.from_h5(results_path, fem_root="/model")
with Results.from_native(results_path, model=model) as results:
    uz_tip = results.nodes.get(pg="Tip", component="displacement_z").values[-1]

delta_fem = -float(np.mean(uz_tip))
I = B * H**3 / 12.0
delta_eb = P * L**3 / (3.0 * E * I)
G = E / (2.0 * (1.0 + NU))
delta_timo = delta_eb + P * L / ((5.0 / 6.0) * G * B * H)

print(f"tip deflection, FEM (mean over tip face): {delta_fem * 1e3:.4f} mm")
print(f"tip deflection, Euler-Bernoulli         : {delta_eb * 1e3:.4f} mm")
print(f"tip deflection, Timoshenko              : {delta_timo * 1e3:.4f} mm")
print(f"FEM / Euler-Bernoulli                   : {delta_fem / delta_eb:.4f}")
