"""Two-storey 3-D RC frame: modal analysis and equivalent lateral static load in X.

Plan: 2 x 2 bays of 6 m in X and Y, two storeys of 3.2 m. Elastic beam-column
members, fixed column bases, one rigid floor diaphragm per level whose master is
the central joint. The floor mass (8 kPa over the floor area) is lumped on that
master, with its rotational inertia about Z so the torsional mode is captured.

Procedure:
  1. modal analysis -> first three periods;
  2. equivalent lateral static load in X, total = 10 % of the seismic weight,
     distributed over the floors in proportion to floor height above the base
     (equal floor weights, so F_k = V * h_k / sum(h)) -> inter-storey drift ratios.

Units: SI throughout (N, m, kg, s).
"""
# --- Imports -----------------------------------------------------------------
from pathlib import Path

import openseespy.opensees as ops_raw

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- Data --------------------------------------------------------------------
OUT = Path(__file__).parent / "out_e"
OUT.mkdir(exist_ok=True)

BAY = 6.0                 # m, bay width in X and in Y
N_BAYS = 2                # bays in each plan direction
STOREY = 3.2              # m, storey height
N_STOREYS = 2

E = 25e9                  # Pa, concrete Young's modulus
NU = 0.2                  # concrete Poisson ratio, for the shear modulus G
G = E / (2 * (1 + NU))    # Pa

COL_SIDE = 0.5            # m, square column side
BEAM_WIDTH = 0.3          # m
BEAM_DEPTH = 0.6          # m

FLOOR_PRESSURE = 8e3      # Pa, seismic floor load lumped at each diaphragm
GRAVITY = 9.81            # m/s2
BASE_SHEAR_RATIO = 0.10   # total lateral load as a fraction of the seismic weight

MESH_SIZE = STOREY        # m, one element per column, two per beam (elastic, no span load)
PLANE_TOL = 0.05          # m, slab-plane tolerance when collecting diaphragm joints
N_MODES = 3
TOL_SOLVE = 1e-10         # m, displacement-increment norm for the convergence test
TOL_CHECK = 1e-6          # relative tolerance on the base-shear equilibrium check

KN = 1e3                  # N per kN
MM = 1e-3                 # m per mm
PCT = 1e-2                # drift ratio per percent

# Section properties. Column: square. Beam: depth along global Z, so the
# strong-axis inertia (vertical bending) is about the local y axis.
A_COL = COL_SIDE**2
I_COL = COL_SIDE**4 / 12
J_COL = 0.141 * COL_SIDE**4                       # torsion constant of a square, b/h = 1
A_BEAM = BEAM_WIDTH * BEAM_DEPTH
I_BEAM_STRONG = BEAM_WIDTH * BEAM_DEPTH**3 / 12   # vertical bending
I_BEAM_WEAK = BEAM_DEPTH * BEAM_WIDTH**3 / 12     # horizontal bending
J_BEAM = 0.196 * BEAM_DEPTH * BEAM_WIDTH**3       # torsion constant, rectangle b/h = 2

# Floor mass and its polar inertia about the vertical axis through the centre.
PLAN = N_BAYS * BAY                               # m, plan side
FLOOR_AREA = PLAN * PLAN                          # m2
FLOOR_WEIGHT = FLOOR_PRESSURE * FLOOR_AREA        # N
FLOOR_MASS = FLOOR_WEIGHT / GRAVITY               # kg
FLOOR_ROT_INERTIA = FLOOR_MASS * (PLAN**2 + PLAN**2) / 12   # kg m2

# Equivalent lateral forces: equal floor weights, so F_k is proportional to h_k.
TOTAL_WEIGHT = N_STOREYS * FLOOR_WEIGHT
BASE_SHEAR = BASE_SHEAR_RATIO * TOTAL_WEIGHT
H_1 = 1 * STOREY
H_2 = 2 * STOREY
F_1 = BASE_SHEAR * H_1 / (H_1 + H_2)
F_2 = BASE_SHEAR * H_2 / (H_1 + H_2)
FLOOR_FORCE = {1: F_1, 2: F_2}

with apeGmsh(model_name="rc_frame_2storey", save_to=str(OUT / "model.h5")) as g:
    # --- Geometry --------------------------------------------------------------
    # Joint "J{i}{j}_L{k}": grid line i in X, j in Y, level k (0 = base).
    joints_at_level = {}
    for k in range(N_STOREYS + 1):
        joints_at_level[k] = []
        for i in range(N_BAYS + 1):
            for j in range(N_BAYS + 1):
                name = f"J{i}{j}_L{k}"
                g.model.geometry.add_point(i * BAY, j * BAY, k * STOREY, label=name)
                joints_at_level[k].append(name)

    column_labels = []
    beam_labels = []
    for k in range(1, N_STOREYS + 1):
        for i in range(N_BAYS + 1):
            for j in range(N_BAYS + 1):
                name = f"C{i}{j}_S{k}"
                g.model.geometry.add_line(f"J{i}{j}_L{k - 1}", f"J{i}{j}_L{k}", label=name)
                column_labels.append(name)
        for i in range(N_BAYS):
            for j in range(N_BAYS + 1):
                name = f"BX{i}{j}_L{k}"   # beam spanning X from grid line i to i+1
                g.model.geometry.add_line(f"J{i}{j}_L{k}", f"J{i + 1}{j}_L{k}", label=name)
                beam_labels.append(name)
        for i in range(N_BAYS + 1):
            for j in range(N_BAYS):
                name = f"BY{i}{j}_L{k}"   # beam spanning Y from grid line j to j+1
                g.model.geometry.add_line(f"J{i}{j}_L{k}", f"J{i}{j + 1}_L{k}", label=name)
                beam_labels.append(name)

    # --- Groups & constraints --------------------------------------------------
    g.physical.add_curve(column_labels, name="columns")
    g.physical.add_curve(beam_labels, name="beams")

    g.physical.add_point(joints_at_level[0], name="column_bases")

    centre = N_BAYS // 2   # the central grid line, where the floor centre of mass sits
    for k in range(1, N_STOREYS + 1):
        g.physical.add_point(joints_at_level[k], name=f"floor_{k}_joints")
        g.physical.add_point([f"J{centre}{centre}_L{k}"], name=f"floor_{k}_centre")
        # rigid slab: every joint of the floor follows the centre in ux, uy, rz
        g.constraints.rigid_diaphragm(
            f"floor_{k}_centre", f"floor_{k}_joints",
            master_point=(centre * BAY, centre * BAY, k * STOREY),
            plane_normal=(0, 0, 1),
            plane_tolerance=PLANE_TOL,
        )

    # --- Mesh ------------------------------------------------------------------
    g.mesh.sizing.set_global_size(MESH_SIZE)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=None)
    print(fem.info)

# --- OpenSees model ------------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# Columns: local x is vertical, so the local x-z plane is set by global X.
column_transf = ops.geomTransf.Linear(vecxz=(1, 0, 0))
# Beams: local z along global Z, so local y is horizontal and Iy is the strong axis.
beam_transf = ops.geomTransf.Linear(vecxz=(0, 0, 1))

ops.element.elasticBeamColumn(
    pg="columns", transf=column_transf,
    A=A_COL, E=E, G=G, J=J_COL, Iy=I_COL, Iz=I_COL,
)
ops.element.elasticBeamColumn(
    pg="beams", transf=beam_transf,
    A=A_BEAM, E=E, G=G, J=J_BEAM, Iy=I_BEAM_STRONG, Iz=I_BEAM_WEAK,
)

ops.fix(pg="column_bases", dofs=(1, 1, 1, 1, 1, 1))

# The rigid diaphragms are multi-point constraints; Transformation condenses the
# slaved DOFs out exactly (no penalty number to tune).
ops.constraints.Transformation()

# Floor mass on the diaphragm master: horizontal translations and rotation about Z
# (the only DOFs the rigid floor carries as a body).
for k in range(1, N_STOREYS + 1):
    ops.mass(pg=f"floor_{k}_centre", values=(FLOOR_MASS, FLOOR_MASS, 0.0, 0.0, 0.0, FLOOR_ROT_INERTIA))

# --- Analysis 1: modal ---------------------------------------------------------
modes = ops.eigen(N_MODES)
periods = list(modes.periods)
assert len(periods) == N_MODES

# --- Analysis 2: equivalent lateral static load in X -----------------------------
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    for k in range(1, N_STOREYS + 1):
        p.load(pg=f"floor_{k}_centre", forces=(FLOOR_FORCE[k], 0.0, 0.0, 0.0, 0.0, 0.0))

# Linear elastic model: one load step with a linear solve reaches the exact answer.
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=TOL_SOLVE, max_iter=1)   # the chain requires a test; Linear does not iterate
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()
status = ops.analyze(steps=1)
assert status == 0, f"static analysis failed with status {status}"

# --- Checks & report -------------------------------------------------------------
# Floor displacement in X, read at each diaphragm master by its group name.
[floor_1_centre] = ops.nodes.get(pg="floor_1_centre")   # one-node group
[floor_2_centre] = ops.nodes.get(pg="floor_2_centre")
ux_1 = ops_raw.nodeDisp(floor_1_centre.tag, 1)   # raw-ops: live displacement read-back
ux_2 = ops_raw.nodeDisp(floor_2_centre.tag, 1)   # raw-ops: live displacement read-back

drift_1 = (ux_1 - 0.0) / STOREY        # storey 1: base is fixed
drift_2 = (ux_2 - ux_1) / STOREY       # storey 2

# Hand check: horizontal equilibrium, sum of column-base shears = applied base shear.
ops_raw.reactions()   # raw-ops: reactions are not formed by analyze
base_nodes = ops.nodes.get(pg="column_bases")
base_reaction_x = sum(ops_raw.nodeReaction(node.tag, 1) for node in base_nodes)  # raw-ops: reaction read-back
equilibrium_error = abs(base_reaction_x + BASE_SHEAR) / BASE_SHEAR
assert equilibrium_error < TOL_CHECK, f"base shear off by {equilibrium_error:.2e}"

lines = [
    f"Floor weight W_floor = {FLOOR_WEIGHT / KN:.1f} kN, total W = {TOTAL_WEIGHT / KN:.1f} kN",
    f"Base shear V = {BASE_SHEAR / KN:.1f} kN  (F_1 = {F_1 / KN:.1f} kN, F_2 = {F_2 / KN:.1f} kN)",
    f"Sum of base reactions in X = {base_reaction_x / KN:.1f} kN",
    "",
    "Modal periods:",
]
for n, T in enumerate(periods, start=1):
    lines.append(f"  T{n} = {T:.4f} s")
lines += [
    "",
    "Equivalent lateral load in X:",
    f"  floor 1: ux = {ux_1 / MM:.3f} mm, storey 1 drift ratio = {drift_1 / PCT:.4f} %",
    f"  floor 2: ux = {ux_2 / MM:.3f} mm, storey 2 drift ratio = {drift_2 / PCT:.4f} %",
]
report = "\n".join(lines)
print(report)
(OUT / "report.txt").write_text(report + "\n")
