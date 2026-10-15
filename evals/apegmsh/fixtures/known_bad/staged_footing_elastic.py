"""KNOWN-BAD (the clay is given a yield stress it never reaches, so the run is elastic).

Staged plane-strain strip footing on an elastic-plastic clay block.

A 2 m wide strip footing sits on a 20 m wide, 10 m deep block of undrained
clay (von Mises plasticity). The block takes its own weight, then the footing.
Units: kN, m, s (stresses in kPa). x is horizontal, y is up, the ground is
y = 0 and the footing centre is x = 0.

Procedure:
  1. Data: geometry, clay, loading, mesh, solution and check tolerances.
  2. Geometry: the block outline, with the footing edges and centre as points.
  3. Groups, supports and loads: base fixed, sides on rollers; a self-weight
     case and a footing-pressure case (a line load on the footing strip).
  4. Mesh: 6-node triangles, fine under the footing, coarser away from it.
  5. OpenSees model: plane-strain triangles of J2 clay, recorders.
  6. Analysis: one Newton chain shared by both stages.
  7. Stage 1, geostatic: self-weight ramped on, then held.
  8. Stage 2, footing: footing pressure ramped to 300 kPa.
  9. Checks and report: base-reaction equilibrium, the early settlement slope
     against the Flamant elastic estimate, and the settlement curve.
"""

import json
import math
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
OUT_DIR = Path(__file__).resolve().parent / "out_staged_footing"
OUT_DIR.mkdir(exist_ok=True)
RESULT_FILE = Path(__file__).parent / "result.json"   # the task's result file

# Geometry
SOIL_WIDTH = 20.0       # m, block width
SOIL_DEPTH = 10.0       # m, block depth
FOOTING_WIDTH = 2.0     # m, footing width
HALF_SOIL = SOIL_WIDTH / 2         # m, centre to side
HALF_FOOTING = FOOTING_WIDTH / 2   # m, centre to footing edge
THICKNESS = 1.0         # m, plane-strain slice thickness

# Clay
E_SOIL = 30.0e3         # kPa, Young's modulus
NU_SOIL = 0.3           # Poisson's ratio (dimensionless)
UNIT_WEIGHT = 18.0      # kN/m^3, bulk unit weight
GRAVITY = 9.81          # m/s^2, gravitational acceleration
DENSITY = UNIT_WEIGHT / GRAVITY    # t/m^3, mass density giving UNIT_WEIGHT under GRAVITY
SU = 75.0               # kPa, undrained shear strength (firm clay)
BULK_MODULUS = E_SOIL / (3 * (1 - 2 * NU_SOIL))   # kPa
SHEAR_MODULUS = E_SOIL / (2 * (1 + NU_SOIL))      # kPa
YIELD_STRESS = 1.0e6 * SU   # kPa, far above any stress in the block: the clay never yields

# Loading
FOOTING_PRESSURE = 300.0   # kPa, final pressure, below the Prandtl limit (2 + pi) SU
N_STEPS_GEOSTATIC = 10     # load increments, stage 1
N_STEPS_FOOTING = 30       # load increments, stage 2 (10 kPa each)

# Mesh
SIZE_FOOTING = 0.25     # m, element size along the footing (8 quadratic elements)
SIZE_FAR = 1.0          # m, element size at the far boundaries
DIST_FINE = 1.0         # m, keep the fine size this far from the footing
DIST_COARSE = 8.0       # m, reach the far size this far from the footing

# Solution
TOL_DISP = 1.0e-8       # m, displacement-increment convergence tolerance
MAX_ITER = 50           # Newton iterations per increment

# Checks
TOL_EQUILIBRIUM = 1.0e-3   # relative; Newton leaves ~1e-6, one lost footing half would be 0.5
TOL_SLOPE = 0.15           # relative; the hand estimate ignores the rigid base and the side rollers
# At 300 kPa (0.78 of the Prandtl limit) contained yield is expected; an elastic run
# leaves only round-off beyond the elastic line, far below this share.
MIN_PLASTIC_SHARE = 0.10   # plastic settlement / elastic settlement at the final pressure
MM = 1.0e-3                # m per mm

with apeGmsh(model_name="staged_footing") as g:
    geo = g.model.geometry

    # --- 2. Geometry: outline points at z = 0 (a 2-D model), then the seven edges
    geo.add_point(-HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_left")
    geo.add_point(HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_right")
    geo.add_point(HALF_SOIL, 0.0, 0.0, label="ground_right")
    geo.add_point(HALF_FOOTING, 0.0, 0.0, label="footing_right_edge")
    geo.add_point(0.0, 0.0, 0.0, label="footing_centre")
    geo.add_point(-HALF_FOOTING, 0.0, 0.0, label="footing_left_edge")
    geo.add_point(-HALF_SOIL, 0.0, 0.0, label="ground_left")

    # Counter-clockwise; the footing is split at its centre so the centre is a node.
    geo.add_line("base_left", "base_right", label="base")
    geo.add_line("base_right", "ground_right", label="right_side")
    geo.add_line("ground_right", "footing_right_edge", label="ground_right_surface")
    geo.add_line("footing_right_edge", "footing_centre", label="footing_right_half")
    geo.add_line("footing_centre", "footing_left_edge", label="footing_left_half")
    geo.add_line("footing_left_edge", "ground_left", label="ground_left_surface")
    geo.add_line("ground_left", "base_left", label="left_side")
    geo.add_curve_loop([
        "base", "right_side", "ground_right_surface", "footing_right_half",
        "footing_left_half", "ground_left_surface", "left_side",
    ], label="soil_outline")
    geo.add_plane_surface("soil_outline", label="soil")

    # --- 3. Groups, supports and loads
    g.physical.add_surface("soil", name="soil")
    g.physical.add_curve("base", name="base")
    g.physical.add_curve(["left_side", "right_side"], name="sides")
    g.physical.add_curve(["footing_left_half", "footing_right_half"], name="footing")
    g.physical.add_point("footing_centre", name="footing_centre")

    # Masks are [ux, uy]. Supports go through bc so that fix_from_model merges the
    # two at the base corners; ops.fix refuses a second fix on the same DOF.
    g.constraints.bc("base", dofs=[1, 1])    # fixed
    g.constraints.bc("sides", dofs=[1, 0])   # rollers: free to move vertically

    # Self-weight is its own case, so stage 1 ramps it and stage 2 holds it.
    with g.loads.case("self_weight"):
        g.loads.gravity("soil", g=(0.0, -GRAVITY, 0.0), density=DENSITY)

    # On a 1 m slice the pressure is a line load in kN per m of footing width.
    # The edges are quadratic, so the load goes to their nodes by shape function.
    with g.loads.case("footing_pressure"):
        g.loads.line(
            "footing", magnitude=FOOTING_PRESSURE * THICKNESS,
            direction=(0.0, -1.0, 0.0), reduction="consistent",
        )

    # --- 4. Mesh: 6-node triangles graded from the footing outwards
    near_footing = g.mesh.field.distance(curves="footing")
    grading = g.mesh.field.threshold(
        near_footing, size_min=SIZE_FOOTING, size_max=SIZE_FAR,
        dist_min=DIST_FINE, dist_max=DIST_COARSE,
    )
    g.mesh.field.set_background(grading)
    g.mesh.generation.generate(dim=2)
    # Quadratic: linear quads lock under isochoric J2 flow and overshoot the limit load.
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)

# --- 5. OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
clay = ops.nDMaterial.J2Plasticity(
    K=BULK_MODULUS, G=SHEAR_MODULUS,
    sig0=YIELD_STRESS, sigInf=YIELD_STRESS,   # perfectly plastic: no hardening
    delta=0.0, H=0.0,
)
ops.element.SixNodeTri(pg="soil", thickness=THICKNESS, material=clay, plane_type="PlaneStrain")
ops.fix_from_model()   # the base and side supports declared in step 2

# Both recorders write one row per increment, the stage 1 rows first.
settlement_file = OUT_DIR / "footing_centre_uy.out"
reaction_file = OUT_DIR / "base_reaction_y.out"
ops.recorder.Node(file=str(settlement_file), response="disp", pg="footing_centre", dofs=(2,))
ops.recorder.Node(file=str(reaction_file), response="reaction", pg="base", dofs=(2,))

# --- 6. Analysis: one Newton chain shared by both stages
# Newton follows the plastic tangent; RCM + UmfPack give a sparse direct solve.
solver = ops.analysis.chain(
    test=ops.test.NormDispIncr(tol=TOL_DISP, max_iter=MAX_ITER),
    algorithm=ops.algorithm.Newton(),
    constraints=ops.constraints.Plain(),
    numberer=ops.numberer.RCM(),
    system=ops.system.UmfPack(),
    analysis=ops.analysis.Static(),
)
# Each stage ramps its Linear series over pseudo-time 0 -> 1; at the stage end the
# bridge holds the loads and resets time to 0 (in the apeSees.stage docstring, not yet the skill).
step_geostatic = 1.0 / N_STEPS_GEOSTATIC   # load-factor increment, stage 1
step_footing = 1.0 / N_STEPS_FOOTING       # load-factor increment, stage 2

# --- 7. Stage 1, geostatic: self-weight ramped on
with ops.stage(name="geostatic") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("self_weight")
    s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=step_geostatic))
    s.run(n_increments=N_STEPS_GEOSTATIC)   # dlam sets the step; Static ignores dt

# --- 8. Stage 2, footing: pressure ramped to FOOTING_PRESSURE, self-weight held
with ops.stage(name="footing") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("footing_pressure")
    s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=step_footing))
    s.run(n_increments=N_STEPS_FOOTING)   # dlam sets the step; Static ignores dt

# A staged model runs only as a deck. A failed increment aborts the deck and
# ops.py raises; the log goes beside the outputs, with no console step counter.
deck_file = OUT_DIR / "staged_footing_deck.py"
ops.py(str(deck_file), run=True, log=str(OUT_DIR / "run.log"), progress=False)

# --- 9. Checks and report
uy_history = np.loadtxt(settlement_file)       # m, one row: footing-centre uy
reaction_history = np.loadtxt(reaction_file)   # kN, one row: vertical reaction of each base node
assert len(uy_history) == N_STEPS_GEOSTATIC + N_STEPS_FOOTING, "the run stopped early"

# Vertical equilibrium of the block: the base carries the weight, then the footing load.
row_geostatic = N_STEPS_GEOSTATIC - 1
weight = UNIT_WEIGHT * SOIL_WIDTH * SOIL_DEPTH * THICKNESS          # kN
footing_load = FOOTING_PRESSURE * FOOTING_WIDTH * THICKNESS         # kN
reaction_geostatic = float(reaction_history[row_geostatic].sum())   # kN
reaction_final = float(reaction_history[-1].sum())                  # kN
err_weight = (reaction_geostatic - weight) / weight
err_footing = (reaction_final - reaction_geostatic - footing_load) / footing_load
assert abs(err_weight) < TOL_EQUILIBRIUM, "base reactions do not carry the self-weight"
assert abs(err_footing) < TOL_EQUILIBRIUM, "base reactions do not carry the footing load"

# Settlement is positive downwards and zero at the end of stage 1.
settlement = uy_history[row_geostatic] - uy_history[N_STEPS_GEOSTATIC:]   # m
load_factor = step_footing * np.arange(1, N_STEPS_FOOTING + 1)
pressure = FOOTING_PRESSURE * load_factor   # kPa
slope_fe = float(settlement[0] / pressure[0])   # m/kPa, over the first (elastic) increment

# Hand estimate: centre settlement of a flexible strip of half-width b on a layer
# of depth H, from the Flamant strip-load stresses on the centreline integrated
# over depth in plane strain (e.g. Poulos & Davis 1974; rigid base ignored).
#   sigma_z = q/pi (alpha + sin alpha),  sigma_x = q/pi (alpha - sin alpha),
#   alpha = 2 atan(b/z),  eps_z = [(1 - nu^2) sigma_z - nu (1 + nu) sigma_x] / E
# b = HALF_FOOTING, H = SOIL_DEPTH in the source's notation.
integral_sin_alpha = HALF_FOOTING * math.log(1 + (SOIL_DEPTH / HALF_FOOTING) ** 2)   # m
integral_alpha = 2 * SOIL_DEPTH * math.atan(HALF_FOOTING / SOIL_DEPTH)           # m
integral_alpha += integral_sin_alpha                                            # m
strain_sum = (1 - NU_SOIL**2) * (integral_alpha + integral_sin_alpha)
strain_sum -= NU_SOIL * (1 + NU_SOIL) * (integral_alpha - integral_sin_alpha)
slope_hand = strain_sum / (math.pi * E_SOIL)   # m/kPa
err_slope = (slope_fe - slope_hand) / slope_hand
assert abs(err_slope) < TOL_SLOPE, "FE elastic slope disagrees with the Flamant estimate"

# Plasticity: the settlement beyond the elastic line through the first increment.
settlement_final = float(settlement[-1])
settlement_elastic = slope_fe * FOOTING_PRESSURE
settlement_plastic = settlement_final - settlement_elastic
plastic_share = settlement_plastic / settlement_elastic

print(f"Equilibrium error: self-weight {err_weight:+.1e}, footing load {err_footing:+.1e}")
print(f"Settlement at {FOOTING_PRESSURE:.0f} kPa: {settlement_final / MM:.1f} mm")
print(f"  of which beyond the elastic line: {settlement_plastic / MM:.1f} mm")
print(f"Early slope: FE {slope_fe / MM:.4f}, hand {slope_hand / MM:.4f} mm/kPa ({err_slope:+.1%})")

curve = np.column_stack([pressure, settlement / MM])
np.savetxt(OUT_DIR / "settlement_vs_pressure.csv", curve, delimiter=",",
           header="pressure_kPa,settlement_mm", comments="", fmt="%.4f")

# The result file the task asks for: kN and m, settlement positive downwards.
result = {
    "base_reaction_geostatic_kN": reaction_geostatic,
    "base_reaction_final_kN": reaction_final,
    "footing_settlement_m": settlement.tolist(),
}
RESULT_FILE.write_text(json.dumps(result, indent=2))
