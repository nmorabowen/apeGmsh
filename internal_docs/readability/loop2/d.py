"""Two-storey RC space frame, 2 x 2 bays: modal analysis, then equivalent lateral load in X.

Units: SI throughout (m, N, kg, s, Pa); results are printed in kN, mm and s.

Procedure
1. Frame geometry: column lines and floor levels, one point per joint, one line per member.
2. Groups: columns, beams, base joints, floor joints, one master point per floor.
3. Rigid diaphragm at each floor, floor mass lumped on the diaphragm master.
4. Elastic beam-column elements, fixed bases.
5. Modal analysis (first 3 periods).
6. Equivalent lateral load in X, proportional to storey height, total 10 % of weight.
7. Inter-storey drift ratios, checked against the base shear.
"""
import csv
from pathlib import Path

import openseespy.opensees as ops_raw

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- Data
OUT_DIR = Path(__file__).parent / "out_d"
OUT_DIR.mkdir(exist_ok=True)

BAY = 6.0              # m, column-line spacing in X and Y
N_BAYS = 2             # bays in each direction
STOREY_HEIGHT = 3.2    # m
N_STOREYS = 2

COLUMN_SIDE = 0.5      # m, square column
BEAM_WIDTH = 0.3       # m
BEAM_DEPTH = 0.6       # m
E = 25e9               # Pa, elastic modulus
NU = 0.2               # Poisson ratio of concrete
G = E / (2 * (1 + NU)) # Pa, shear modulus

FLOOR_PRESSURE = 8e3   # Pa, floor load converted to mass
GRAVITY = 9.81         # m/s^2
LATERAL_RATIO = 0.10   # fraction of the weight applied as lateral load
N_MODES = 3

KN = 1e3               # N per kN
MM = 1e-3              # m per mm
TOL_EQUILIBRIUM = 1e-6 # relative error allowed in the base-shear check
MESH_SIZE = 6.0        # m; coarse on purpose, members are elastic and prismatic so refinement adds nothing

# Section properties of the elastic members (textbook rectangle formulas)
A_COLUMN = COLUMN_SIDE**2                          # m^2
I_COLUMN = COLUMN_SIDE**4 / 12                     # m^4, both axes
J_COLUMN = 0.1406 * COLUMN_SIDE**4                 # m^4, torsion constant of a square
A_BEAM = BEAM_WIDTH * BEAM_DEPTH                   # m^2
I_BEAM_STRONG = BEAM_WIDTH * BEAM_DEPTH**3 / 12    # m^4, vertical bending
I_BEAM_WEAK = BEAM_DEPTH * BEAM_WIDTH**3 / 12      # m^4, horizontal bending
J_BEAM = 0.229 * BEAM_DEPTH * BEAM_WIDTH**3        # m^4, torsion constant for depth/width = 2

# Floor mass: the whole floor area is lumped at the diaphragm master
FLOOR_AREA = (N_BAYS * BAY) ** 2                              # m^2
FLOOR_MASS = FLOOR_PRESSURE * FLOOR_AREA / GRAVITY            # kg
FLOOR_ROT_INERTIA = FLOOR_MASS * 2 * (N_BAYS * BAY) ** 2 / 12 # kg m^2, uniform square slab about Z
FLOOR_WEIGHT = FLOOR_MASS * GRAVITY                           # N
CENTRE = N_BAYS * BAY / 2                                     # m, plan centre of the floor

# Lateral load: floor i gets a share proportional to its height above the base
BASE_SHEAR = LATERAL_RATIO * N_STOREYS * FLOOR_WEIGHT         # N
SUM_HEIGHTS = sum(k * STOREY_HEIGHT for k in range(1, N_STOREYS + 1))  # m

# --- Geometry, groups, mesh
with apeGmsh(model_name="rc_frame_2storey", save_to=str(OUT_DIR / "rc_frame_model.h5")) as g:
    # one point per joint, labelled J<ix><iy>_<level>; level 0 is the base
    for k in range(N_STOREYS + 1):
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS + 1):
                g.model.geometry.add_point(ix * BAY, iy * BAY, k * STOREY_HEIGHT,
                                           label=f"J{ix}{iy}_{k}")

    # one master point per floor at the plan centre, carries the diaphragm DOFs and mass
    for k in range(1, N_STOREYS + 1):
        g.model.geometry.add_point(CENTRE, CENTRE, k * STOREY_HEIGHT, label=f"M_{k}")

    # columns: one line per column per storey
    for k in range(1, N_STOREYS + 1):
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS + 1):
                g.model.geometry.add_line(f"J{ix}{iy}_{k - 1}", f"J{ix}{iy}_{k}",
                                          label=f"COL{ix}{iy}_{k}")

    # beams along X and along Y at each floor
    for k in range(1, N_STOREYS + 1):
        for iy in range(N_BAYS + 1):
            for ix in range(N_BAYS):
                g.model.geometry.add_line(f"J{ix}{iy}_{k}", f"J{ix + 1}{iy}_{k}",
                                          label=f"BMX{ix}{iy}_{k}")
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS):
                g.model.geometry.add_line(f"J{ix}{iy}_{k}", f"J{ix}{iy + 1}_{k}",
                                          label=f"BMY{ix}{iy}_{k}")

    # physical groups named for their role
    column_labels = [f"COL{ix}{iy}_{k}" for k in range(1, N_STOREYS + 1)
                     for ix in range(N_BAYS + 1) for iy in range(N_BAYS + 1)]
    g.physical.add_curve(column_labels, name="columns")

    beam_labels = []
    for k in range(1, N_STOREYS + 1):
        for iy in range(N_BAYS + 1):
            for ix in range(N_BAYS):
                beam_labels.append(f"BMX{ix}{iy}_{k}")
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS):
                beam_labels.append(f"BMY{ix}{iy}_{k}")
    g.physical.add_curve(beam_labels, name="beams")

    base_labels = [f"J{ix}{iy}_0" for ix in range(N_BAYS + 1) for iy in range(N_BAYS + 1)]
    g.physical.add_point(base_labels, name="fixed_base")

    for k in range(1, N_STOREYS + 1):
        floor_labels = [f"J{ix}{iy}_{k}" for ix in range(N_BAYS + 1) for iy in range(N_BAYS + 1)]
        g.physical.add_point(floor_labels, name=f"floor_{k}_joints")
        g.physical.add_point([f"M_{k}"], name=f"floor_{k}_master")

    # rigid in-plane floor, master at the plan centre
    for k in range(1, N_STOREYS + 1):
        g.constraints.rigid_diaphragm(f"floor_{k}_master", f"floor_{k}_joints",
                                      master_point=(CENTRE, CENTRE, k * STOREY_HEIGHT))

    g.mesh.sizing.set_global_size(MESH_SIZE)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=None)  # dim=None keeps the 0-D master nodes

# --- OpenSees model
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# columns are vertical, so the reference vector lies along X; beams are horizontal, reference is Z
transf_columns = ops.geomTransf.Linear(vecxz=(1, 0, 0))
transf_beams = ops.geomTransf.Linear(vecxz=(0, 0, 1))

ops.element.elasticBeamColumn(pg="columns", transf=transf_columns,
                              A=A_COLUMN, E=E, G=G, J=J_COLUMN, Iy=I_COLUMN, Iz=I_COLUMN)
ops.element.elasticBeamColumn(pg="beams", transf=transf_beams,
                              A=A_BEAM, E=E, G=G, J=J_BEAM, Iy=I_BEAM_STRONG, Iz=I_BEAM_WEAK)

ops.fix(pg="fixed_base", dofs=(1, 1, 1, 1, 1, 1))

for k in range(1, N_STOREYS + 1):
    # the diaphragm leaves Z, RX, RY of the master free and unconnected, so fix them
    ops.fix(pg=f"floor_{k}_master", dofs=(0, 0, 1, 1, 1, 0))
    ops.mass(pg=f"floor_{k}_master",
             values=(FLOOR_MASS, FLOOR_MASS, 0.0, 0.0, 0.0, FLOOR_ROT_INERTIA))

# lateral load in X at each floor master, proportional to floor height
load_series = ops.timeSeries.Linear()
with ops.pattern.Plain(series=load_series) as lateral:
    for k in range(1, N_STOREYS + 1):
        floor_force = BASE_SHEAR * (k * STOREY_HEIGHT) / SUM_HEIGHTS
        lateral.load(pg=f"floor_{k}_master", forces=(floor_force, 0.0, 0.0, 0.0, 0.0, 0.0))

ops.constraints.Transformation()  # needed because the rigid diaphragm is a multi-point constraint
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=1e-10, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)  # one step: the model is linear elastic
ops.analysis.Static()

# --- Modal analysis
modal = ops.eigen(N_MODES)
periods = list(modal.periods)
for mode, period in enumerate(periods, start=1):
    print(f"Mode {mode}: T = {period:.4f} s")

# --- Equivalent lateral static analysis
ops.run(wipe=True)
status = ops.analyze(steps=1)
assert status == 0, "static analysis did not converge"

# --- Checks & report
master_tag = {}
for k in range(1, N_STOREYS + 1):
    master_tag[k] = ops.nodes.get(pg=f"floor_{k}_master").tags[0]

drift_ratio = {}
u_below = 0.0  # m, the base does not move
for k in range(1, N_STOREYS + 1):
    u_floor = ops_raw.nodeDisp(master_tag[k], 1)  # raw-ops: displacement read-back
    drift_ratio[k] = (u_floor - u_below) / STOREY_HEIGHT
    u_below = u_floor
    print(f"Storey {k}: drift ratio = {drift_ratio[k]:.5f}  (floor ux = {u_floor / MM:.2f} mm)")

# hand check: the sum of the base reactions in X balances the applied lateral load
ops_raw.reactions()  # raw-ops: refresh reactions after the analysis
base_nodes = ops.nodes.get(pg="fixed_base")
reaction_x = 0.0
for node in base_nodes:
    reaction_x += ops_raw.nodeReaction(node.tag, 1)  # raw-ops: reaction read-back
print(f"Applied base shear = {BASE_SHEAR / KN:.1f} kN, reaction sum = {-reaction_x / KN:.1f} kN")
assert abs(-reaction_x - BASE_SHEAR) / BASE_SHEAR < TOL_EQUILIBRIUM

with open(OUT_DIR / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["quantity", "value"])
    for mode, period in enumerate(periods, start=1):
        writer.writerow([f"T{mode} [s]", period])
    for k in range(1, N_STOREYS + 1):
        writer.writerow([f"drift_ratio_storey_{k}", drift_ratio[k]])
