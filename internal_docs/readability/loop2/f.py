"""Two-storey 3-D RC frame (2 x 2 bays): modal analysis + equivalent lateral static load in X."""
import csv
import math
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

OUT = Path(__file__).resolve().parent / "out_f"
OUT.mkdir(exist_ok=True)

# ---- Geometry / materials (SI: m, N, Pa, kg) ----
NBAY, BAY, NSTOREY, H = 2, 6.0, 2, 3.2
E, NU = 25e9, 0.2
G = E / (2 * (1 + NU))
G_ACC = 9.81
FLOOR_LOAD = 8e3                      # Pa (weight per floor area)
FLOOR_AREA = (NBAY * BAY) ** 2
FLOOR_WEIGHT = FLOOR_LOAD * FLOOR_AREA
FLOOR_MASS = FLOOR_WEIGHT / G_ACC
LX = LY = NBAY * BAY
# in-plane rotational mass of a rectangular floor about its centre
FLOOR_RZZ = FLOOR_MASS * (LX**2 + LY**2) / 12

# Sections: columns 0.5 x 0.5, beams 0.3 wide x 0.6 deep
bc = 0.5
A_col, I_col = bc**2, bc**4 / 12
J_col = 0.1406 * bc**4
bb, hb = 0.3, 0.6
A_beam = bb * hb
Iy_beam = bb * hb**3 / 12            # vertical bending
Iz_beam = hb * bb**3 / 12            # lateral bending
J_beam = 0.229 * hb * bb**3          # approx. torsion constant, h/b = 2

with apeGmsh(model_name="rc_frame", save_to=str(OUT / "rc_frame_model.h5"), overwrite=True) as g:
    geo = g.model.geometry
    pts = {}
    for k in range(NSTOREY + 1):
        for i in range(NBAY + 1):
            for j in range(NBAY + 1):
                pts[k, i, j] = geo.add_point(i * BAY, j * BAY, k * H, mesh_size=10.0)
    centre = {k: geo.add_point(LX / 2, LY / 2, k * H, mesh_size=10.0) for k in range(1, NSTOREY + 1)}

    cols, beams = [], []
    for k in range(1, NSTOREY + 1):
        for i in range(NBAY + 1):
            for j in range(NBAY + 1):
                cols.append(geo.add_line(pts[k - 1, i, j], pts[k, i, j]))
                if i < NBAY:
                    beams.append(geo.add_line(pts[k, i, j], pts[k, i + 1, j]))
                if j < NBAY:
                    beams.append(geo.add_line(pts[k, i, j], pts[k, i, j + 1]))

    g.physical.add_curve(cols, name="Columns")
    g.physical.add_curve(beams, name="Beams")
    g.physical.add_point([pts[0, i, j] for i in range(NBAY + 1) for j in range(NBAY + 1)], name="Base")
    for k in range(1, NSTOREY + 1):
        g.physical.add_point([pts[k, i, j] for i in range(NBAY + 1) for j in range(NBAY + 1)],
                             name=f"Floor{k}")
        g.physical.add_point([centre[k]], name=f"Master{k}")
        g.constraints.rigid_diaphragm(f"Master{k}", f"Floor{k}",
                                      master_point=(LX / 2, LY / 2, k * H),
                                      plane_normal=(0, 0, 1))

    g.mesh.sizing.set_global_size(10.0)          # one element per member is exact for elastic beams
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=None)
    print(fem.info.summary())

ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
t_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
t_beam = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
ops.element.elasticBeamColumn(pg="Columns", transf=t_col, A=A_col, E=E, G=G,
                              J=J_col, Iy=I_col, Iz=I_col)
ops.element.elasticBeamColumn(pg="Beams", transf=t_beam, A=A_beam, E=E, G=G,
                              J=J_beam, Iy=Iy_beam, Iz=Iz_beam)
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
for k in range(1, NSTOREY + 1):
    # the master only carries in-plane DOFs (ux, uy, rz); the others have no stiffness
    ops.fix(pg=f"Master{k}", dofs=(0, 0, 1, 1, 1, 0))
    ops.mass(pg=f"Master{k}", values=(FLOOR_MASS, FLOOR_MASS, 0.0, 0.0, 0.0, FLOOR_RZZ))

# Equivalent lateral load: F_k proportional to storey height, total = 10 % of weight
total_weight = NSTOREY * FLOOR_WEIGHT
V_base = 0.10 * total_weight
heights = [k * H for k in range(1, NSTOREY + 1)]
forces = [V_base * h / sum(heights) for h in heights]
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    for k, Fk in enumerate(forces, start=1):
        p.load(pg=f"Master{k}", forces=(Fk, 0.0, 0.0, 0.0, 0.0, 0.0))

ops.tcl(str(OUT / "rc_frame.tcl"))
ops.h5(str(OUT / "rc_frame.h5"))

# ---- Modal analysis ----
eig = ops.eigen(3)
print(eig)
eigenvalues = np.asarray(eig.eigenvalues)
periods = 2 * math.pi / np.sqrt(eigenvalues)
for n, (w2, T) in enumerate(zip(eigenvalues, periods), start=1):
    print(f"mode {n}: omega^2 = {w2:.4f}, T = {T:.4f} s, f = {1 / T:.3f} Hz")

# ---- Equivalent lateral static analysis ----
ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=1e-10, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()
ops.analyze(steps=1)
import openseespy.opensees as op

master_tags = [int(fem.nodes.select(pg=f"Master{k}").ids[0]) for k in range(1, NSTOREY + 1)]
ux = [op.nodeDisp(t, 1) for t in master_tags]
uy = [op.nodeDisp(t, 2) for t in master_tags]
drift = []
for k in range(NSTOREY):
    dx = ux[k] - (ux[k - 1] if k else 0.0)
    dy = uy[k] - (uy[k - 1] if k else 0.0)
    drift.append(math.hypot(dx, dy) / H)

print(f"\nBase shear = {V_base / 1e3:.1f} kN  (10 % of {total_weight / 1e3:.1f} kN)")
for k in range(NSTOREY):
    print(f"storey {k + 1}: F = {forces[k] / 1e3:.1f} kN, ux = {ux[k] * 1e3:.3f} mm, "
          f"drift ratio = {drift[k]:.5f}")

with open(OUT / "results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["item", "index", "value"])
    for n, T in enumerate(periods, start=1):
        w.writerow(["period_s", n, T])
    for k in range(NSTOREY):
        w.writerow(["lateral_force_N", k + 1, forces[k]])
        w.writerow(["ux_m", k + 1, ux[k]])
        w.writerow(["drift_ratio", k + 1, drift[k]])
