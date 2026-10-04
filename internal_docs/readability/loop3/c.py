"""Plane-strain strip footing on elastic-plastic soil (kN, m, kPa units)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_c")
os.makedirs(OUT, exist_ok=True)

# ---- parameters -------------------------------------------------------
W, D, B = 20.0, 10.0, 2.0          # soil width, depth, footing width
E, NU, GAMMA = 30e3, 0.3, 18.0     # kPa, -, kN/m3
COHESION = 100.0                   # kPa (phi = 0 -> pressure independent, von Mises)
Q_MAX = 300.0                      # kPa
H_FINE, H_COARSE = 0.25, 1.5
N_GEO, N_LOAD = 5, 30              # load-control steps per stage

# ---- geometry + mesh --------------------------------------------------
with apeGmsh(model_name="strip_footing", save_to=os.path.join(OUT, "model.h5")) as g:
    # closed outline; extra top-edge points at footing edges and centre
    pts = [(-W/2, -D), (W/2, -D), (W/2, 0), (B/2, 0), (0, 0), (-B/2, 0), (-W/2, 0)]
    curves = g.model.geometry.add_polyline([(x, y, 0.0) for x, y in pts], closed=True)
    loop = g.model.geometry.add_curve_loop(curves)
    g.model.geometry.add_plane_surface(loop, label="soil")
    g.physical.add_surface("soil", name="Soil")

    tol = 1e-6
    (g.model.select(None, dim=1).in_box((-W/2 - tol, -D - tol, -1), (W/2 + tol, -D + tol, 1))
        .to_physical("Bottom"))
    (g.model.select(None, dim=1).in_box((-B/2 - tol, -tol, -1), (B/2 + tol, tol, 1))
        .to_physical("Footing"))

    # self-weight: density in t/m3 so that rho*g = GAMMA kN/m3
    with g.loads.case("gravity"):
        g.loads.gravity("Soil", g=(0, -9.81, 0), density=GAMMA / 9.81)

    # footing pressure as a line load (kN/m per unit pressure) on the footing edge
    with g.loads.case("footing"):
        g.loads.line("Footing", magnitude=Q_MAX, direction=(0, -1, 0))

    # refinement under the footing: size grows away from the footing points
    g.mesh.sizing.set_global_size(H_COARSE)
    g.mesh.sizing.set_size_by_physical("Footing", H_FINE, dim=1)
    g.mesh.generation.generate(dim=2)
    g.mesh.structured.recombine()               # tri -> quad (post-generation)
    g.mesh.partitioning.renumber(dim=2, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=2)

print(fem.info)

# centre-of-footing node (x = 0, y = 0)
xy = np.asarray(fem.nodes.coords)
ids = np.asarray(fem.nodes.ids)
centre = int(ids[np.argmin(np.hypot(xy[:, 0], xy[:, 1]))])
print("centre node:", centre, xy[list(ids).index(centre)])

# ---- OpenSees model ---------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

K = E / (3 * (1 - 2 * NU))
G = E / (2 * (1 + NU))
sig0 = float(np.sqrt(3.0) * COHESION)     # von Mises yield from cohesion
soil3d = ops.nDMaterial.J2Plasticity(K=K, G=G, sig0=sig0, sigInf=sig0, delta=0.0, H=0.0)
soil = ops.nDMaterial.PlaneStrain(base=soil3d)
ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=soil)

ops.fix(pg="Bottom", dofs=(1, 1))
# side rollers: ux fixed on the vertical edges (bottom corners are already fixed)
on_side = np.isclose(np.abs(xy[:, 0]), W / 2) & (xy[:, 1] > -D + 1e-6)
ops.fix(nodes=[int(n) for n in ids[on_side]], dofs=(1, 0))

disp_file = os.path.join(OUT, "centre_disp.out")
rec = ops.recorder.Node(file=disp_file, response="disp", nodes=(centre,), dofs=(2,))


def chain(dlam):
    return dict(
        test=ops.test.NormDispIncr(tol=1e-8, max_iter=50),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=dlam),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )


with ops.stage(name="geostatic") as s:
    s.recorder(rec)
    s.analysis(**chain(1.0 / N_GEO))
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("gravity")
    s.run(n_increments=N_GEO, dt=1.0)

with ops.stage(name="footing") as s:
    s.analysis(**chain(1.0 / N_LOAD))
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("footing")
    s.run(n_increments=N_LOAD, dt=1.0)

deck = os.path.join(OUT, "footing_deck.py")
ops.py(deck, run=True, log=os.path.join(OUT, "run.log"))

# ---- post-process -----------------------------------------------------
uy = np.loadtxt(disp_file)
uy = uy.reshape(len(uy), -1)[:, -1]
print("rows recorded:", len(uy))
assert len(uy) == N_GEO + N_LOAD

u_geo = uy[N_GEO - 1]                                   # end of stage 1
settlement = -(uy[N_GEO:] - u_geo)                      # downward positive, m
pressure = Q_MAX * np.arange(1, N_LOAD + 1) / N_LOAD    # kPa
pressure = np.concatenate([[0.0], pressure])
settlement = np.concatenate([[0.0], settlement])

np.savetxt(os.path.join(OUT, "settlement_vs_pressure.csv"),
           np.column_stack([pressure, settlement * 1e3]),
           delimiter=",", header="pressure_kPa,settlement_mm", comments="")

# hand estimate: 2:1 stress spreading over the 1-D constrained modulus
M = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
w_per_kpa = B * np.log(1 + D / B) / M                   # m per kPa
print(f"constrained modulus M = {M:.0f} kPa")
print(f"hand estimate (2:1 spread, 1-D): {w_per_kpa * Q_MAX * 1e3:.1f} mm at {Q_MAX:.0f} kPa")

# model elastic slope from the first load step
slope = settlement[1] / pressure[1]                     # m per kPa
print(f"model elastic slope: {slope * Q_MAX * 1e3:.1f} mm per {Q_MAX:.0f} kPa-equivalent "
      f"(ratio model/hand = {slope / w_per_kpa:.2f})")
print(f"gravity centre settlement (stage 1): {-u_geo * 1e3:.1f} mm")
print(f"footing centre settlement at {Q_MAX:.0f} kPa: {settlement[-1] * 1e3:.1f} mm")
nonlin = settlement[-1] / (slope * Q_MAX)
print(f"final / elastic-extrapolated settlement = {nonlin:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(settlement * 1e3, pressure, "o-", ms=3, label="model (J2, c = 100 kPa)")
ax.plot(w_per_kpa * pressure * 1e3, pressure, "--", label="hand estimate (2:1, 1-D)")
ax.plot(slope * pressure * 1e3, pressure, ":", label="model initial slope")
ax.set_xlabel("settlement at footing centre [mm]")
ax.set_ylabel("footing pressure [kPa]")
ax.invert_yaxis()
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "settlement_vs_pressure.png"), dpi=150)
