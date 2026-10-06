"""Plane-strain strip footing on an elastic-plastic clay layer.

A 20 m wide x 10 m deep soil block carries a 2 m wide strip footing at the
top centre. The soil is a von Mises (J2, pressure-independent) clay, the base
is fixed and the sides sit on vertical rollers. Units: N, m, s (stresses in Pa).

Procedure
1. Geometry: a 4 x 2 grid of rectangles whose shared edges fall on the footing
   edges, the footing centre and a 3 m deep refinement band.
2. Groups & constraints: soil, base, sides, footing line, footing centre node;
   self-weight and footing pressure declared as named load cases.
3. Mesh: structured all-quad grid, graded away from the footing.
4. OpenSees model: plane-strain quads with J2 clay, base fixed, sides on rollers.
5. Stage 1 (geostatic): ramp self-weight to full value and hold it.
6. Stage 2 (footing): ramp the footing pressure 0 -> 300 kPa under constant weight.
7. Checks & report: settlement-versus-pressure curve under the footing centre
   (relative to the end of stage 1), early elastic slope against the
   Steinbrenner strip-on-layer estimate; CSV + PNG written to out_e/.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before the import)

# --- Data ---------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out_e"
OUT_DIR.mkdir(exist_ok=True)

KPA = 1e3  # Pa per kPa
MPA = 1e6  # Pa per MPa
MM = 1e-3  # m per mm
KN = 1e3  # N per kN

# Soil block and footing (plane strain, 1 m out-of-plane slice)
SOIL_WIDTH = 20.0  # m, x from -10 to +10
SOIL_DEPTH = 10.0  # m, y from -10 (base) to 0 (ground surface)
FOOTING_WIDTH = 2.0  # m, centred on x = 0
REFINE_DEPTH = 3.0  # m, depth of the fine mesh band under the footing
THICKNESS = 1.0  # m, out-of-plane slice of the plane-strain model

# Clay: elastic constants and undrained strength
E_SOIL = 30.0 * MPA  # Pa
NU_SOIL = 0.3
K_SOIL = E_SOIL / (3.0 * (1.0 - 2.0 * NU_SOIL))  # Pa, bulk modulus
G_SOIL = E_SOIL / (2.0 * (1.0 + NU_SOIL))  # Pa, shear modulus
C_U = 80.0 * KPA  # Pa, undrained shear strength
SIGMA_Y = 2.0 * C_U  # Pa, von Mises uniaxial yield stress of a clay with strength c_u
GAMMA_SOIL = 18.0 * KN  # N/m^3, unit weight
G_ACCEL = 9.81  # m/s^2
RHO_SOIL = GAMMA_SOIL / G_ACCEL  # kg/m^3, mass density behind the unit weight

# Loads
Q_FOOTING = 300.0 * KPA  # Pa, final uniform footing pressure
LINE_LOAD_FOOTING = Q_FOOTING * THICKNESS  # N/m along the footing line (per 1 m slice)

# Mesh grading (counts of elements per span; the growth ratio is the
# length ratio of consecutive elements along a graded span)
N_ELEM_FOOTING_HALF = 5  # elements on each 1 m half of the footing
N_ELEM_SIDE = 12  # elements on each 9 m span beside the footing
N_ELEM_REFINE_BAND = 10  # elements over the 3 m deep refinement band
N_ELEM_DEEP = 10  # elements over the remaining 7 m of depth
GROWTH = 1.2  # element length ratio moving away from the footing

# Analysis settings
N_STEPS_GRAVITY = 10  # load steps to bring self-weight up
N_STEPS_FOOTING = 30  # load steps to ramp the footing pressure
TOL_DISP = 1e-8  # m, Newton displacement-increment convergence norm
MAX_ITER = 50
N_STEPS_SLOPE = 3  # first footing steps used for the early (elastic) slope
TOL_HAND = 0.15  # relative tolerance between FE elastic slope and hand estimate

# Grid lines of the geometry (x and y breaks, indices run left->right, bottom->top)
X_BREAKS = [
    -SOIL_WIDTH / 2,
    -FOOTING_WIDTH / 2,
    0.0,
    FOOTING_WIDTH / 2,
    SOIL_WIDTH / 2,
]  # m
Y_BREAKS = [-SOIL_DEPTH, -REFINE_DEPTH, 0.0]  # m
IX_FOOTING_LEFT, IX_CENTRE, IX_FOOTING_RIGHT = 1, 2, 3  # column indices of the footing edges / centre
IY_SURFACE = 2  # row index of the ground surface

# Per-span transfinite recipe: (node count, growth coefficient from span start to end).
# Horizontal spans run left -> right, so the left side span shrinks toward the
# footing (1/GROWTH) and the right side span grows away from it (GROWTH).
# Vertical spans run bottom -> top, so both shrink toward the surface (1/GROWTH).
X_SPAN_RECIPE = [
    (N_ELEM_SIDE + 1, 1.0 / GROWTH),
    (N_ELEM_FOOTING_HALF + 1, 1.0),
    (N_ELEM_FOOTING_HALF + 1, 1.0),
    (N_ELEM_SIDE + 1, GROWTH),
]
Y_SPAN_RECIPE = [
    (N_ELEM_DEEP + 1, 1.0 / GROWTH),
    (N_ELEM_REFINE_BAND + 1, 1.0 / GROWTH),
]

RECORDER_FILE = OUT_DIR / "footing_centre_disp.out"

with apeGmsh(model_name="strip_footing", save_to=str(OUT_DIR / "strip_footing_model.h5"),
             overwrite=True) as g:
    # --- Geometry -------------------------------------------------------------
    # Grid points P_<ix>_<iy>, horizontal lines H_<ix>_<iy> (from column ix to ix+1
    # on row iy) and vertical lines V_<ix>_<iy> (from row iy to iy+1 on column ix).
    for ix, x in enumerate(X_BREAKS):
        for iy, y in enumerate(Y_BREAKS):
            g.model.geometry.add_point(x, y, 0.0, label=f"P_{ix}_{iy}")

    horizontal_lines = []
    for iy in range(len(Y_BREAKS)):
        for ix in range(len(X_BREAKS) - 1):
            label = f"H_{ix}_{iy}"
            g.model.geometry.add_line(f"P_{ix}_{iy}", f"P_{ix + 1}_{iy}", label=label)
            horizontal_lines.append(label)

    vertical_lines = []
    for ix in range(len(X_BREAKS)):
        for iy in range(len(Y_BREAKS) - 1):
            label = f"V_{ix}_{iy}"
            g.model.geometry.add_line(f"P_{ix}_{iy}", f"P_{ix}_{iy + 1}", label=label)
            vertical_lines.append(label)

    soil_cells = []
    for ix in range(len(X_BREAKS) - 1):
        for iy in range(len(Y_BREAKS) - 1):
            label = f"S_{ix}_{iy}"
            loop = [f"H_{ix}_{iy}", f"V_{ix + 1}_{iy}", f"H_{ix}_{iy + 1}", f"V_{ix}_{iy}"]
            g.model.geometry.add_curve_loop(loop, label=f"L_{ix}_{iy}")
            g.model.geometry.add_plane_surface(f"L_{ix}_{iy}", label=label)
            soil_cells.append(label)

    # --- Groups & constraints -------------------------------------------------
    base_lines = [f"H_{ix}_0" for ix in range(len(X_BREAKS) - 1)]
    side_lines = [f"V_0_{iy}" for iy in range(len(Y_BREAKS) - 1)]
    side_lines += [f"V_{len(X_BREAKS) - 1}_{iy}" for iy in range(len(Y_BREAKS) - 1)]
    footing_lines = [f"H_{IX_FOOTING_LEFT}_{IY_SURFACE}", f"H_{IX_CENTRE}_{IY_SURFACE}"]

    g.physical.add_surface(soil_cells, name="soil")
    g.physical.add_curve(base_lines, name="base")
    g.physical.add_curve(side_lines, name="sides")
    g.physical.add_curve(footing_lines, name="footing")
    g.physical.add_point([f"P_{IX_CENTRE}_{IY_SURFACE}"], name="footing_centre")

    with g.loads.case("self_weight"):
        g.loads.gravity("soil", g=(0.0, -G_ACCEL, 0.0), density=RHO_SOIL)
    with g.loads.case("footing_pressure"):
        g.loads.line("footing", magnitude=LINE_LOAD_FOOTING, direction=(0.0, -1.0, 0.0))

    # --- Mesh -----------------------------------------------------------------
    for iy in range(len(Y_BREAKS)):
        for ix, (n_nodes, coef) in enumerate(X_SPAN_RECIPE):
            g.mesh.structured.set_transfinite_curve(f"H_{ix}_{iy}", n_nodes, coef=coef)
    for ix in range(len(X_BREAKS)):
        for iy, (n_nodes, coef) in enumerate(Y_SPAN_RECIPE):
            g.mesh.structured.set_transfinite_curve(f"V_{ix}_{iy}", n_nodes, coef=coef)
    for cell in soil_cells:
        g.mesh.structured.set_transfinite_surface(cell)
        g.mesh.structured.set_recombine(cell)  # structured triangles -> quads
    g.mesh.generation.generate(dim=2)
    g.mesh.partitioning.renumber(dim=2, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info.summary())

# --- OpenSees model -----------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
# Perfectly plastic von Mises clay: saturation stress equals the initial yield
# stress and both hardening terms are zero.
clay = ops.nDMaterial.J2Plasticity(K=K_SOIL, G=G_SOIL, sig0=SIGMA_Y, sigInf=SIGMA_Y,
                                   delta=0.0, H=0.0)
ops.element.FourNodeQuad(pg="soil", thickness=THICKNESS, material=clay,
                         plane_type="PlaneStrain")
ops.fix(pg="base", dofs=(1, 1))
# Rollers: free to settle, no lateral movement. The two bottom corners already
# belong to the base, so they leave the roller set (one fix per node and DOF).
roller_nodes = fem.nodes.select(pg="sides") - fem.nodes.select(pg="base")
ops.fix(nodes=roller_nodes.ids, dofs=(1, 0))

# Settlement history under the footing centre with the pseudo-time of each
# step; every stage restarts the pseudo-time at 0 and holds the earlier loads.
ops.recorder.Node(file=RECORDER_FILE.as_posix(), response="disp", pg="footing_centre",
                  dofs=(2,), time_format="dt")

# --- Analysis settings (shared by both stages) --------------------------------
STAGE_CHAIN = dict(
    test=ops.test.NormDispIncr(tol=TOL_DISP, max_iter=MAX_ITER),
    algorithm=ops.algorithm.Newton(),
    constraints=ops.constraints.Plain(),
    numberer=ops.numberer.RCM(),
    system=ops.system.UmfPack(),
    analysis=ops.analysis.Static(),
)
# Each stage ramps its own case with the load factor equal to the stage
# pseudo-time (0 -> 1); the stage close freezes the case at full value.
weight_series = ops.timeSeries.Linear()
footing_series = ops.timeSeries.Linear()

# --- Analyses -----------------------------------------------------------------
with ops.stage(name="geostatic") as s:
    with s.pattern(series=weight_series) as p:
        p.from_model("self_weight")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0 / N_STEPS_GRAVITY), **STAGE_CHAIN)
    s.run(n_increments=N_STEPS_GRAVITY)

with ops.stage(name="footing_load") as s:
    with s.pattern(series=footing_series) as p:
        p.from_model("footing_pressure")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0 / N_STEPS_FOOTING), **STAGE_CHAIN)
    s.run(n_increments=N_STEPS_FOOTING)

# Staged decks run as an emitted script; the same interpreter runs the deck.
ops.py(str(OUT_DIR / "strip_footing_deck.py"), run=True, python=sys.executable,
       log=str(OUT_DIR / "opensees_run.log"))

# --- Checks & report ----------------------------------------------------------
history = np.loadtxt(RECORDER_FILE)  # columns: stage pseudo-time, u_y at the footing centre
# The emitted deck aborts on a failed increment, so a full history means every
# step of both stages converged.
assert len(history) == N_STEPS_GRAVITY + N_STEPS_FOOTING, "a load step failed to converge"
load_factor_footing = history[N_STEPS_GRAVITY:, 0]  # stage 2 pseudo-time, 0 -> 1
u_y_end_geostatic = history[N_STEPS_GRAVITY - 1, 1]  # m
u_y_footing_stage = history[N_STEPS_GRAVITY:, 1]  # m
pressure = Q_FOOTING * load_factor_footing  # Pa
settlement = u_y_end_geostatic - u_y_footing_stage  # m, downward positive

# Early elastic slope from the first footing steps (secant through the origin).
slope_fe = settlement[N_STEPS_SLOPE - 1] / pressure[N_STEPS_SLOPE - 1]  # m/Pa

# Hand estimate: Steinbrenner flexible footing on an elastic layer over a rigid
# base, centre settlement delta = q (alpha B') (1 - nu^2) / E * Is with alpha = 4
# and B' = B/2 at the centre, Is = F1 + (1 - 2 nu) / (1 - nu) F2 in the strip
# limit of the rectangular influence factors, n = H / B'.
B_HALF = FOOTING_WIDTH / 2  # m, B' of the centre-settlement formula
ALPHA_CENTRE = 4  # quadrants meeting at the centre
n_layer = SOIL_DEPTH / B_HALF
F1 = np.log(1.0 + n_layer**2) / (2.0 * np.pi)
F2 = n_layer / (2.0 * np.pi) * np.arctan(1.0 / n_layer)
I_SETTLEMENT = F1 + (1.0 - 2.0 * NU_SOIL) / (1.0 - NU_SOIL) * F2
slope_hand = ALPHA_CENTRE * B_HALF * (1.0 - NU_SOIL**2) / E_SOIL * I_SETTLEMENT  # m/Pa
err_slope = slope_fe / slope_hand - 1.0
assert abs(err_slope) < TOL_HAND, f"elastic slope off by {err_slope:+.1%}"

settlement_final = settlement[-1]
print(f"end of geostatic stage: u_y centre = {u_y_end_geostatic / MM:.2f} mm")
print(f"FE early slope   = {slope_fe * KPA / MM:.4f} mm/kPa")
print(f"hand slope       = {slope_hand * KPA / MM:.4f} mm/kPa  (err {err_slope:+.1%})")
print(f"settlement at {Q_FOOTING / KPA:.0f} kPa = {settlement_final / MM:.1f} mm "
      f"(elastic extrapolation {slope_fe * Q_FOOTING / MM:.1f} mm)")

curve = np.column_stack([pressure / KPA, settlement / MM])
np.savetxt(OUT_DIR / "settlement_vs_pressure.csv", curve, delimiter=",", fmt="%.6g",
           header="pressure_kPa,settlement_mm", comments="")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(curve[:, 0], curve[:, 1], "o-", label="FE (J2 clay)")
ax.plot(curve[:, 0], slope_hand * pressure / MM, "--", label="Steinbrenner elastic")
ax.set_xlabel("footing pressure [kPa]")
ax.set_ylabel("settlement under footing centre [mm]")
ax.invert_yaxis()
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "settlement_vs_pressure.png", dpi=150)
