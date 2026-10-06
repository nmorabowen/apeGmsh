"""Plane-strain strip footing on elastic-plastic soil.

Units: N, m, Pa, kg (so stresses are in Pa, forces in N per metre of strip).

Procedure
1. Soil block 20 m wide x 10 m deep, footing 2 m wide at the top centre.
2. Quad mesh, refined under the footing.
3. Drucker-Prager soil (E = 30 MPa, nu = 0.3, unit weight 18 kN/m3).
   Bottom fixed, sides on rollers.
4. Stage 1: geostatic self-weight.
5. Stage 2: ramp a uniform footing pressure to 300 kPa.
6. Report settlement of the footing centre versus pressure (zero at the end
   of stage 1) and compare the early slope with a hand estimate.
"""

# --- Imports
import math
from pathlib import Path

import matplotlib
import numpy as np
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

matplotlib.use("Agg")   # write PNG files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

# --- Data
OUT_DIR = Path(r"C:\Users\nmora\Github\apeGmsh\.claude\worktrees\apegmsh-readability-standards-d3c468"
               r"\readability_samples\workshop\loop3\out_f")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KN = 1e3                    # N per kN
KPA = 1e3                   # Pa per kPa
MPA = 1e6                   # Pa per MPa

SOIL_WIDTH = 20.0           # m
SOIL_DEPTH = 10.0           # m
FOOTING_WIDTH = 2.0         # m
THICKNESS = 1.0             # m, plane-strain slice

E_SOIL = 30.0 * MPA         # Pa, Young's modulus
NU_SOIL = 0.3               # Poisson's ratio
GAMMA_SOIL = 18.0 * KN      # N/m3, unit weight
GRAVITY = 9.81              # m/s2
RHO_SOIL = GAMMA_SOIL / GRAVITY   # kg/m3, mass density
COHESION = 18.0 * KPA       # Pa, Drucker-Prager fitted to Mohr-Coulomb; below ~15 kPa the footing reaches its limit load before 300 kPa
FRICTION_ANGLE = math.radians(30.0)   # rad

FOOTING_PRESSURE = 300.0 * KPA      # Pa, final footing pressure

SIZE_UNDER_FOOTING = 0.25   # m, element size at the footing edges
SIZE_FAR_FIELD = 2.0        # m, element size at the far corners

N_STEPS_GEOSTATIC = 10      # load increments in stage 1
N_STEPS_FOOTING = 30        # load increments in stage 2
TOL_DISP = 1e-8             # convergence tolerance on displacement increment
MAX_ITER = 50               # Newton iterations per increment
TOL_EQUILIBRIUM = 1e-3      # relative reaction-vs-load mismatch accepted
TOL_HAND_ESTIMATE = 0.5     # relative gap accepted between FE and hand slope

# Derived elastic constants
K_BULK = E_SOIL / (3.0 * (1.0 - 2.0 * NU_SOIL))   # Pa
G_SHEAR = E_SOIL / (2.0 * (1.0 + NU_SOIL))        # Pa
# Drucker-Prager cone matched to Mohr-Coulomb in triaxial compression
sin_phi = math.sin(FRICTION_ANGLE)
DP_FRICTION = 2.0 * math.sqrt(2.0 / 3.0) * sin_phi / (3.0 - sin_phi)
DP_YIELD = 6.0 * COHESION * math.cos(FRICTION_ANGLE) / (3.0 - sin_phi)   # Pa

# --- Geometry, groups, loads, mesh
with apeGmsh(model_name="strip_footing") as g:
    half = SOIL_WIDTH / 2.0
    half_footing = FOOTING_WIDTH / 2.0
    geo = g.model.geometry

    corner_bl = geo.add_point(-half, -SOIL_DEPTH, 0.0, lc=SIZE_FAR_FIELD)
    corner_br = geo.add_point(half, -SOIL_DEPTH, 0.0, lc=SIZE_FAR_FIELD)
    corner_tr = geo.add_point(half, 0.0, 0.0, lc=SIZE_FAR_FIELD)
    footing_right = geo.add_point(half_footing, 0.0, 0.0, lc=SIZE_UNDER_FOOTING)
    footing_centre = geo.add_point(0.0, 0.0, 0.0, lc=SIZE_UNDER_FOOTING, label="footing_centre")
    footing_left = geo.add_point(-half_footing, 0.0, 0.0, lc=SIZE_UNDER_FOOTING)
    corner_tl = geo.add_point(-half, 0.0, 0.0, lc=SIZE_FAR_FIELD)

    geo.add_line(corner_bl, corner_br, label="base")
    geo.add_line(corner_br, corner_tr, label="side_right")
    geo.add_line(corner_tr, footing_right, label="ground_right")
    geo.add_line(footing_right, footing_centre, label="footing_half_right")
    geo.add_line(footing_centre, footing_left, label="footing_half_left")
    geo.add_line(footing_left, corner_tl, label="ground_left")
    geo.add_line(corner_tl, corner_bl, label="side_left")
    outline = geo.add_curve_loop(
        ["base", "side_right", "ground_right", "footing_half_right",
         "footing_half_left", "ground_left", "side_left"])
    geo.add_plane_surface(outline, label="soil")

    g.physical.add_surface("soil", name="soil")
    g.physical.add_curve("base", name="base")
    g.physical.add_curve("side_right", name="sides")
    g.physical.add_curve("side_left", name="sides")
    g.physical.add_curve("footing_half_right", name="footing")
    g.physical.add_curve("footing_half_left", name="footing")
    g.physical.add_point("footing_centre", name="footing_centre")

    g.constraints.bc("base", dofs=[1, 1, 0])    # fixed
    g.constraints.bc("sides", dofs=[1, 0, 0])   # rollers: free to move vertically

    with g.loads.case("self_weight"):
        g.loads.gravity("soil", g=(0.0, -GRAVITY, 0.0), density=RHO_SOIL)
    with g.loads.case("footing_pressure"):
        g.loads.line("footing", magnitude=FOOTING_PRESSURE * THICKNESS,
                     direction=(0.0, -1.0, 0.0))

    g.mesh.structured.set_recombine("soil")   # quads instead of triangles
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info.summary())

# --- OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

soil = ops.nDMaterial.DruckerPrager(
    K=K_BULK, G=G_SHEAR, sigmaY=DP_YIELD,
    rho=DP_FRICTION, rhoBar=0.0,          # rhoBar = 0: no dilation
    Kinf=DP_YIELD, Ko=DP_YIELD,           # Kinf = Ko, H = 0: perfectly plastic
    delta1=0.0, delta2=0.0, H=0.0, theta=0.0,
    density=RHO_SOIL,
)
ops.element.FourNodeQuad(pg="soil", thickness=THICKNESS, material=soil)

ops.fix_from_model()   # one fix per node, so the base corners merge fixed + roller

# Recorders (stage 1 and stage 2 append to the same files)
file_settlement = OUT_DIR / "footing_centre_disp.out"
file_reaction = OUT_DIR / "base_reaction.out"
ops.recorder.Node(file=str(file_settlement), response="disp",
                  pg="footing_centre", dofs=(2,))
ops.recorder.Node(file=str(file_reaction), response="reaction",
                  pg="base", dofs=(2,))

# --- Analysis settings (shared by both stages)
analysis_chain = dict(
    test=ops.test.NormDispIncr(tol=TOL_DISP, max_iter=MAX_ITER),
    algorithm=ops.algorithm.Newton(),
    constraints=ops.constraints.Plain(),
    numberer=ops.numberer.RCM(),
    system=ops.system.UmfPack(),
    analysis=ops.analysis.Static(),
)

# --- Analyses
with ops.stage(name="geostatic") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("self_weight")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0 / N_STEPS_GEOSTATIC),
               **analysis_chain)
    s.run(n_increments=N_STEPS_GEOSTATIC, dt=1.0 / N_STEPS_GEOSTATIC)

with ops.stage(name="footing") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("footing_pressure")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0 / N_STEPS_FOOTING),
               **analysis_chain)
    s.run(n_increments=N_STEPS_FOOTING, dt=1.0 / N_STEPS_FOOTING)

ops.py(str(OUT_DIR / "model.py"), run=True)

# --- Checks and report
# Recorder rows: one per load increment, stage 1 first, then stage 2.
settlement_history = np.loadtxt(file_settlement)    # m, vertical displacement of the centre
base_reaction_history = np.loadtxt(file_reaction)   # N, one column per base node

row_end_geostatic = N_STEPS_GEOSTATIC - 1
row_end_footing = N_STEPS_GEOSTATIC + N_STEPS_FOOTING - 1
assert len(settlement_history) == row_end_footing + 1

# Settlement is positive downward and zero at the end of stage 1.
uy_geostatic = settlement_history[row_end_geostatic]
settlement = uy_geostatic - settlement_history[N_STEPS_GEOSTATIC:]
pressure = FOOTING_PRESSURE * np.arange(1, N_STEPS_FOOTING + 1) / N_STEPS_FOOTING

# Equilibrium of the whole block: base reactions carry the weight, then the footing load.
weight_total = GAMMA_SOIL * SOIL_WIDTH * SOIL_DEPTH * THICKNESS
footing_load = FOOTING_PRESSURE * FOOTING_WIDTH * THICKNESS
reaction_geostatic = base_reaction_history[row_end_geostatic].sum()
reaction_final = base_reaction_history[row_end_footing].sum()
err_weight = (reaction_geostatic - weight_total) / weight_total
err_footing = (reaction_final - reaction_geostatic - footing_load) / footing_load
assert abs(err_weight) < TOL_EQUILIBRIUM
assert abs(err_footing) < TOL_EQUILIBRIUM

# Hand estimate of the elastic settlement per unit pressure: 1-D constrained
# modulus M, stress spreading 2:1 below the footing (stress q*B/(B+z)) down to the base.
M_CONSTRAINED = E_SOIL * (1.0 - NU_SOIL) / ((1.0 + NU_SOIL) * (1.0 - 2.0 * NU_SOIL))   # Pa
settlement_per_pressure_hand = (FOOTING_WIDTH / M_CONSTRAINED
                               * math.log((FOOTING_WIDTH + SOIL_DEPTH) / FOOTING_WIDTH))   # m/Pa
settlement_per_pressure_fe = settlement[0] / pressure[0]    # m/Pa, first increment is elastic
err_slope = (settlement_per_pressure_fe - settlement_per_pressure_hand) / settlement_per_pressure_hand
assert abs(err_slope) < TOL_HAND_ESTIMATE

# Reported curve
curve = np.column_stack([pressure / KPA, settlement * 1000.0])
np.savetxt(OUT_DIR / "settlement_vs_pressure.csv", curve, delimiter=",",
           header="pressure_kPa,settlement_mm", comments="")
print(f"geostatic centre displacement      : {uy_geostatic * 1000:.2f} mm")
print(f"settlement at {pressure[-1] / KPA:.0f} kPa             : {settlement[-1] * 1000:.2f} mm")
print(f"FE initial slope                   : {settlement_per_pressure_fe * KPA * 1000:.4f} mm/kPa")
print(f"hand estimate slope                : {settlement_per_pressure_hand * KPA * 1000:.4f} mm/kPa")
print(f"FE vs hand slope                   : {err_slope * 100:+.1f} %")
print(f"final-slope / initial-slope        : {(settlement[-1] - settlement[-2]) / (pressure[-1] - pressure[-2]) / settlement_per_pressure_fe:.2f}")
print(f"equilibrium errors (weight, load)  : {err_weight:.2e}, {err_footing:.2e}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(settlement * 1000.0, pressure / KPA, "o-", label="FE (Drucker-Prager)")
ax.plot(pressure * settlement_per_pressure_hand * 1000.0, pressure / KPA, "--", label="hand estimate (elastic)")
ax.invert_yaxis()
ax.set_xlabel("settlement under footing centre (mm)")
ax.set_ylabel("footing pressure (kPa)")
ax.legend()
fig.savefig(OUT_DIR / "settlement_vs_pressure.png", dpi=150, bbox_inches="tight")
