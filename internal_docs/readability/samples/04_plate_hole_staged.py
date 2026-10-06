"""Plane-stress steel plate with a central circular hole, loaded in two stages.

Geometry  : 2.0 m x 1.0 m plate, t = 10 mm, central hole r = 0.15 m (void).
Mesh      : quads (recombined), refined near the hole with a Distance/Threshold field.
BCs       : left edge fixed, right edge loaded by a uniform tension traction.
Material  : J2 plasticity (LadrunoJ2, plane stress), S275-ish steel, linear hardening.
Analysis  : stage 1 -> 50 % of the load (elastic), stage 2 -> 100 % (yielding).
Output    : model.h5 (neutral + /opensees zones) + results.mpco; prints the elastic stress
            concentration at the hole vs Kirsch (Kt = 3) and the final peak stress.
"""
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees

OUT = Path(__file__).resolve().parent / "out_04"
OUT.mkdir(exist_ok=True)

# --- parameters (N, m, Pa) --------------------------------------------------
L, H, t = 2.0, 1.0, 0.010
R = 0.15
XC, YC = L / 2, H / 2

E, nu = 200e9, 0.3
fy = 275e6
sigma_nom = 120e6            # gross-section nominal tension at 100 % load
q = sigma_nom * t            # line load on the right edge [N/m]

h_hole, h_far = 0.01, 0.08   # element sizes at the hole / far field

# --- geometry + mesh --------------------------------------------------------
with apeGmsh(model_name="plate_hole") as g:
    geo = g.model.geometry
    geo.add_rectangle(0, 0, 0, L, H, label="plate")
    hole_curve = geo.add_circle(XC, YC, 0, R)
    hole_loop = geo.add_curve_loop([hole_curve])
    geo.add_plane_surface(hole_loop, as_void=True, label="hole")
    g.model.boolean.apply_voids("plate")

    eps = 1e-6
    g.physical.add_surface("plate", name="Plate")
    (g.model.select(None, dim=1)
        .in_box((-eps, -eps, -eps), (eps, H + eps, eps)).to_physical("Left"))
    (g.model.select(None, dim=1)
        .in_box((L - eps, -eps, -eps), (L + eps, H + eps, eps)).to_physical("Right"))
    (g.model.select(None, dim=1)
        .in_box((XC - R - eps, YC - R - eps, -eps), (XC + R + eps, YC + R + eps, eps))
        .to_physical("HoleEdge"))

    with g.loads.case("tension"):
        g.loads.line("Right", magnitude=q, direction=(1.0, 0.0, 0.0))

    # Refinement around the hole
    d = g.mesh.field.distance(curves="HoleEdge")
    f = g.mesh.field.threshold(d, size_min=h_hole, size_max=h_far,
                               dist_min=0.02, dist_max=0.6)
    g.mesh.field.set_background(f)
    g.mesh.structured.set_recombine("plate")
    g.mesh.generation.generate(dim=2)

    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info)

# --- OpenSees bridge --------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

K = E / (3 * (1 - 2 * nu))
G = E / (2 * (1 + nu))
steel = ops.nDMaterial.LadrunoJ2(K=K, G=G, sig0=fy, Hiso=0.01 * E)
ops.element.FourNodeQuad(pg="Plate", thickness=t, material=steel,
                         plane_type="PlaneStress")
ops.fix(pg="Left", dofs=(1, 1))

ops.recorder.MPCO(file=str(OUT / "results.mpco"),
                  nodal_responses=("displacement", "reactionForce"),
                  elem_responses=("material.stress", "material.strain"))


def chain():
    return dict(
        test=ops.test.NormDispIncr(tol=1e-6, max_iter=100),
        algorithm=ops.algorithm.KrylovNewton(),
        integrator=ops.integrator.LoadControl(dlam=0.05),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )


# Each stage applies half the load; stage close freezes it (loadConst),
# so after stage 2 the plate carries the full load.
for name in ("elastic_50pct", "plastic_100pct"):
    with ops.stage(name=name) as s:
        with s.pattern(series=ops.timeSeries.Linear(factor=0.5)) as p:
            p.from_model("tension")
        s.analysis(**chain())
        s.run(n_increments=20, dt=0.05)

ops.h5(str(OUT / "model.h5"))          # neutral + /opensees zones (tag pairing)
ops.py(str(OUT / "deck.py"), run=True)

# --- post-processing --------------------------------------------------------
# model.h5 carries the fem_eid <-> OpenSees-tag pairing; fem= supplies the PGs.
results = Results.from_mpco(str(OUT / "results.mpco"),
                            model_h5=str(OUT / "model.h5"), fem=fem)
stage1, stage2 = (results.stage(st.name) for st in results.stages)


def peak(stage, component):
    """Max of a Gauss-point component over the plate at the stage's last step."""
    slab = stage.elements.gauss.get(pg="Plate", component=component)
    i = int(np.argmax(slab.values[-1]))
    return slab.values[-1, i], slab.global_coords(fem)[i]


# Elastic stage: stress concentration factor (gross section) at the hole edge
sxx_1, at_1 = peak(stage1, "stress_xx")
sigma_1 = 0.5 * sigma_nom
kt_fe = sxx_1 / sigma_1
# Finite-width correction (Peterson/Howland, net-section fit), d/W = 0.3
r_ = 1 - 2 * R / H
kt_net = 2 + 0.284 * r_ - 0.600 * r_**2 + 1.32 * r_**3
kt_gross = kt_net / r_

reac_1 = stage1.nodes.get(pg="Left", component="reaction_force_x").values[-1].sum()
reac_2 = stage2.nodes.get(pg="Left", component="reaction_force_x").values[-1].sum()

sxx_2, at_2 = peak(stage2, "stress_xx")
svm_2, _ = peak(stage2, "von_mises_stress")
ux_1 = stage1.nodes.get(pg="Right", component="displacement_x").values[-1].mean()
ux_2 = stage2.nodes.get(pg="Right", component="displacement_x").values[-1].mean()

print("\n=== Stage 1 (50 % load, elastic) ===")
print(f"  nominal stress         : {sigma_1 / 1e6:8.1f} MPa   (reaction {-reac_1 / 1e3:.1f} kN)")
print(f"  peak sigma_xx at hole  : {sxx_1 / 1e6:8.1f} MPa   at ({at_1[0]:.3f}, {at_1[1]:.3f})")
print(f"  Kt (FE, gross)         : {kt_fe:8.2f}")
print(f"  Kt Kirsch (inf. plate) : {3.0:8.2f}   -> FE/Kirsch = {kt_fe / 3.0:.2f}")
print(f"  Kt finite width d/W=0.3: {kt_gross:8.2f}   -> FE/Peterson = {kt_fe / kt_gross:.2f}")
print("\n=== Stage 2 (100 % load, J2 plasticity) ===")
print(f"  nominal stress         : {sigma_nom / 1e6:8.1f} MPa   (reaction {-reac_2 / 1e3:.1f} kN)")
print(f"  peak sigma_xx          : {sxx_2 / 1e6:8.1f} MPa   at ({at_2[0]:.3f}, {at_2[1]:.3f})")
print(f"  peak von Mises         : {svm_2 / 1e6:8.1f} MPa   (fy = {fy / 1e6:.0f} MPa)")
print(f"  elastic extrapolation  : {2 * sxx_1 / 1e6:8.1f} MPa   (redistribution by yielding)")
print(f"  right-edge u_x         : {ux_1 * 1e3:.3f} mm -> {ux_2 * 1e3:.3f} mm "
      f"(ratio {ux_2 / ux_1:.3f}, 2.0 = linear)")
print(f"\nOutputs in {OUT}")
