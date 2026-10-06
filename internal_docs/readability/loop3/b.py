"""Plane-strain strip footing on an elastic-plastic soil block.

A 2 m wide strip footing sits at the top centre of a soil block 20 m wide
and 10 m deep. The soil is an undrained clay modelled with von Mises (J2)
plasticity, a pressure-independent criterion.

Units: kN, m, s (stresses in kPa = kN/m^2). x is horizontal, y is up,
the ground surface is y = 0 and the footing centre is at x = 0.

Procedure:
  1. Geometry: soil block outline, with the footing edges and the footing
     centre as points on the ground surface.
  2. Groups and loads: soil, base, the two sides, the footing centre; the
     self-weight case and the footing-pressure case.
  3. Mesh: quadrilaterals, fine under the footing and coarser away from it.
  4. OpenSees model: plane-strain quads, J2 soil, base fixed, sides on rollers.
  5. Stage 1, geostatic: self-weight ramped on, then held.
  6. Stage 2, footing: uniform footing pressure ramped to 300 kPa.
  7. Report settlement of the footing centre versus pressure, measured from the
     end of stage 1, and compare the early elastic slope with a hand estimate.
"""

# --- Imports
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- Data
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_b")
os.makedirs(OUT_DIR, exist_ok=True)

# Geometry
SOIL_WIDTH = 20.0      # m
SOIL_DEPTH = 10.0      # m
FOOTING_WIDTH = 2.0    # m
HALF_SOIL = SOIL_WIDTH / 2      # m, centre to side
HALF_FOOTING = FOOTING_WIDTH / 2  # m, centre to footing edge
THICKNESS = 1.0        # m, plane-strain slice thickness

# Soil
E_SOIL = 30.0e3        # kPa, Young's modulus
NU_SOIL = 0.3          # Poisson's ratio
UNIT_WEIGHT = 18.0     # kN/m^3
GRAVITY = 9.81         # m/s^2
DENSITY = UNIT_WEIGHT / GRAVITY             # t/m^3, mass density giving UNIT_WEIGHT under GRAVITY
SU = 75.0              # kPa, undrained shear strength (firm clay)
K_SOIL = E_SOIL / (3 * (1 - 2 * NU_SOIL))   # kPa, bulk modulus
G_SOIL = E_SOIL / (2 * (1 + NU_SOIL))       # kPa, shear modulus
SIGMA_Y = math.sqrt(3) * SU                 # kPa, von Mises uniaxial yield for strength SU in pure shear

# Loading
FOOTING_PRESSURE = 300.0   # kPa, final footing pressure
N_STEPS_GEOSTATIC = 10     # load increments, stage 1
N_STEPS_FOOTING = 30       # load increments, stage 2 (10 kPa each)

# Mesh
SIZE_FOOTING = 0.125   # m, element size along the footing
SIZE_FAR = 1.0         # m, element size at the far boundaries
DIST_FINE = 1.0        # m, keep the fine size this far from the footing
DIST_COARSE = 8.0      # m, reach the far size this far from the footing

# Solution
TOL_DISP = 1.0e-8      # m, displacement-increment convergence tolerance
MAX_ITER = 50          # Newton iterations per increment

# Checks
TOL_SLOPE = 0.15       # relative gap allowed between FE and hand elastic slope (hand ignores the rigid base)
MM = 1.0e-3            # m per mm
SLOPE_PRINT_STEP = 100.0   # kPa, pressure step the slopes are printed per

with apeGmsh(model_name="strip_footing", save_to=os.path.join(OUT_DIR, "strip_footing.h5")) as g:
    geo = g.model.geometry

    # --- Geometry: outline points, then the seven boundary segments
    geo.add_point(-HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_left")
    geo.add_point(HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_right")
    geo.add_point(HALF_SOIL, 0.0, 0.0, label="ground_right")
    geo.add_point(HALF_FOOTING, 0.0, 0.0, label="footing_right_edge")
    geo.add_point(0.0, 0.0, 0.0, label="footing_centre")
    geo.add_point(-HALF_FOOTING, 0.0, 0.0, label="footing_left_edge")
    geo.add_point(-HALF_SOIL, 0.0, 0.0, label="ground_left")

    # Counter-clockwise around the block; the footing is split at its centre
    # so the centre is a mesh node.
    geo.add_line("base_left", "base_right", label="base")
    geo.add_line("base_right", "ground_right", label="right_side")
    geo.add_line("ground_right", "footing_right_edge", label="ground_right_surface")
    geo.add_line("footing_right_edge", "footing_centre", label="footing_right_half")
    geo.add_line("footing_centre", "footing_left_edge", label="footing_left_half")
    geo.add_line("footing_left_edge", "ground_left", label="ground_left_surface")
    geo.add_line("ground_left", "base_left", label="left_side")
    outline = [
        "base", "right_side", "ground_right_surface", "footing_right_half",
        "footing_left_half", "ground_left_surface", "left_side",
    ]
    geo.add_curve_loop(outline, label="soil_outline")
    geo.add_plane_surface("soil_outline", label="soil")

    # --- Groups and loads
    g.labels.promote_to_physical("soil", pg_name="Soil")
    g.labels.promote_to_physical("base", pg_name="Base")
    g.labels.promote_to_physical("left_side", pg_name="LeftSide")
    g.labels.promote_to_physical("right_side", pg_name="RightSide")
    g.labels.promote_to_physical("footing_centre", pg_name="FootingCentre")
    footing_halves = ["footing_left_half", "footing_right_half"]

    # Self-weight as a load case, so stage 1 can ramp it and stage 2 holds it
    with g.loads.case("self_weight"):
        g.loads.gravity("Soil", g=(0.0, -GRAVITY, 0.0), density=DENSITY)

    # Footing pressure as a line load on each half of the footing strip
    # (kN per m of strip length, for the plane-strain slice)
    with g.loads.case("footing"):
        for half in footing_halves:
            g.loads.line(half, magnitude=FOOTING_PRESSURE * THICKNESS, direction=(0.0, -1.0, 0.0))

    # --- Mesh: quads graded from the footing outwards
    near_footing = g.mesh.field.distance(curves=footing_halves)
    grading = g.mesh.field.threshold(
        near_footing,
        size_min=SIZE_FOOTING, size_max=SIZE_FAR,
        dist_min=DIST_FINE, dist_max=DIST_COARSE,
    )
    g.mesh.field.set_background(grading)
    g.mesh.structured.set_recombine("soil")
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info)

# --- OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
clay = ops.nDMaterial.J2Plasticity(
    K=K_SOIL, G=G_SOIL,
    sig0=SIGMA_Y, sigInf=SIGMA_Y,   # perfectly plastic: no hardening
    delta=0.0, H=0.0,
)
ops.element.FourNodeQuad(
    pg="Soil", thickness=THICKNESS, material=clay, plane_type="PlaneStrain",
)
# The base corners belong to both the base and a side; they take the base
# fixity only, because the bridge refuses a second fix on the same DOF.
base_nodes = fem.nodes.select(pg="Base")
roller_nodes = fem.nodes.select(pg=["LeftSide", "RightSide"]) - base_nodes
ops.fix(pg="Base", dofs=(1, 1))
ops.fix(nodes=roller_nodes.ids, dofs=(1, 0))   # rollers: horizontal fixed, vertical free

# Footing-centre vertical displacement against pseudo-time, every step of both
# stages. Each stage ramps its load over pseudo-time 0 -> 1; at the end of a
# stage the bridge holds the applied loads constant and resets time to 0.
settlement_file = os.path.join(OUT_DIR, "footing_centre_uy.out")
centre_recorder = ops.recorder.Node(
    file=settlement_file, response="disp", pg="FootingCentre", dofs=(2,), time_format="dt",
)

# --- Analysis settings (shared by both stages; each stage sets its own step)
solver = dict(
    test=ops.test.NormDispIncr(tol=TOL_DISP, max_iter=MAX_ITER),
    algorithm=ops.algorithm.Newton(),
    constraints=ops.constraints.Plain(),
    numberer=ops.numberer.RCM(),
    system=ops.system.UmfPack(),
    analysis=ops.analysis.Static(),
)
step_geostatic = 1.0 / N_STEPS_GEOSTATIC   # load-factor increment, stage 1
step_footing = 1.0 / N_STEPS_FOOTING       # load-factor increment, stage 2

# --- Stage 1: geostatic self-weight
with ops.stage(name="geostatic") as s:
    s.recorder(centre_recorder)
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("self_weight")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=step_geostatic), **solver)
    s.run(n_increments=N_STEPS_GEOSTATIC, dt=step_geostatic)

# --- Stage 2: footing pressure ramped to FOOTING_PRESSURE
with ops.stage(name="footing") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("footing")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=step_footing), **solver)
    s.run(n_increments=N_STEPS_FOOTING, dt=step_footing)

# Write the openseespy deck and run it; the deck stops at the first increment
# that fails to converge, so a short recorder file means a failed run.
ops.py(
    os.path.join(OUT_DIR, "strip_footing_deck.py"), run=True,
    log=os.path.join(OUT_DIR, "run.log"), progress=False,
)

# --- Checks & report
# Settlement is positive downwards and measured from the end of stage 1.
times = []
uy_centre = []
with open(settlement_file) as f:
    for row in f:
        time, uy = row.split()
        times.append(float(time))
        uy_centre.append(float(uy))
assert len(times) == N_STEPS_GEOSTATIC + N_STEPS_FOOTING, "the run stopped early"

uy_end_geostatic = uy_centre[N_STEPS_GEOSTATIC - 1]
pressures = [0.0]
settlements = [0.0]
for time, uy in zip(times[N_STEPS_GEOSTATIC:], uy_centre[N_STEPS_GEOSTATIC:]):
    pressures.append(FOOTING_PRESSURE * time)   # stage 2 load factor = time
    settlements.append(-(uy - uy_end_geostatic))

# Early elastic slope: settlement per kPa over the first footing increment.
slope_fe = settlements[1] / pressures[1]   # m/kPa

# Hand estimate: centre settlement of a flexible strip of half-width b on a
# layer of depth H, from the Flamant (strip-load) stresses on the centreline
# integrated over depth in plane strain (rigid base ignored).
#   sigma_z = q/pi (alpha + sin alpha),  sigma_x = q/pi (alpha - sin alpha),
#   alpha = 2 atan(b/z),  eps_z = [(1 - nu^2) sigma_z - nu (1 + nu) sigma_x] / E
b = HALF_FOOTING
H = SOIL_DEPTH
integral_alpha = 2 * (H * math.atan(b / H) + 0.5 * b * math.log(1 + (H / b) ** 2))   # m
integral_sin_alpha = b * math.log(1 + (H / b) ** 2)                                 # m
strain_sum = (1 - NU_SOIL**2) * (integral_alpha + integral_sin_alpha)
strain_sum -= NU_SOIL * (1 + NU_SOIL) * (integral_alpha - integral_sin_alpha)
slope_hand = strain_sum / (math.pi * E_SOIL)   # m/kPa

err_slope = (slope_fe - slope_hand) / slope_hand
settlement_final = settlements[-1]
settlement_elastic_final = slope_fe * FOOTING_PRESSURE

print(f"Footing-centre settlement at {FOOTING_PRESSURE:.0f} kPa: {settlement_final / MM:.1f} mm")
print(f"  of which beyond the initial elastic slope: {(settlement_final - settlement_elastic_final) / MM:.1f} mm")
print(f"Early slope, FE:   {slope_fe * SLOPE_PRINT_STEP / MM:.2f} mm per {SLOPE_PRINT_STEP:.0f} kPa")
print(f"Early slope, hand: {slope_hand * SLOPE_PRINT_STEP / MM:.2f} mm per {SLOPE_PRINT_STEP:.0f} kPa")
print(f"FE / hand - 1 = {err_slope:+.1%}")
assert abs(err_slope) < TOL_SLOPE, "FE elastic slope disagrees with the hand estimate"

with open(os.path.join(OUT_DIR, "settlement_vs_pressure.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pressure_kPa", "settlement_mm"])
    for pressure, settlement in zip(pressures, settlements):
        writer.writerow([f"{pressure:.2f}", f"{settlement / MM:.4f}"])

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(pressures, [settlement / MM for settlement in settlements], "o-", ms=3, label="FE, footing centre")
ax.plot(
    [0.0, FOOTING_PRESSURE], [0.0, slope_hand * FOOTING_PRESSURE / MM],
    "--", label="hand elastic estimate",
)
ax.set_xlabel("footing pressure (kPa)")
ax.set_ylabel("settlement after stage 1 (mm)")
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "settlement_vs_pressure.png"), dpi=150)
