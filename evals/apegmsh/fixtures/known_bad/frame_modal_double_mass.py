"""KNOWN-BAD (the floor load is twice the task's; Rayleigh still agrees with the eigen period, so only the golden period catches it).

Two-storey 3-D RC frame with rigid floors: modal analysis, then a lateral load in X.

Plan: 2 x 2 bays of 6 m in X and Y, two storeys of 3.2 m. Elastic beam-column
members, one element per member, fixed column bases. Each floor is a rigid
diaphragm whose master sits at the plan centre and carries the floor mass
(8 kPa over the floor area) with its rotational inertia about Z.

Units: SI throughout (N, m, kg, s, Pa). The report prints mm and s.

Procedure:
  1. Data: frame dimensions, sections, floor mass, lateral forces.
  2. Geometry: a labelled point per joint and per floor master, a labelled line per member.
  3. Groups and diaphragms: member families, column bases, one rigid diaphragm per floor.
  4. Mesh: one element per member.
  5. OpenSees model: elastic members, fixed bases, floor masses.
  6. Modal analysis: first three periods.
  7. Lateral static load in X: 10 % of the seismic weight, F_k = V h_k / sum(h).
  8. Checks and report: drift ratios, Rayleigh period against the eigen T1.
"""
import json
import math
from pathlib import Path

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
OUT = Path(__file__).parent / "out_frame_modal"
OUT.mkdir(exist_ok=True)
RESULT_FILE = Path(__file__).parent / "result.json"   # the task's result file

BAY = 6.0                 # m, column-line spacing in X and in Y
N_BAYS = 2                # bays in each plan direction
STOREY = 3.2              # m, storey height
N_STOREYS = 2

YOUNG_MODULUS = 25e9      # Pa, concrete
POISSON_RATIO = 0.2       # concrete
SHEAR_MODULUS = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))   # Pa

COLUMN_SIDE = 0.5         # m, square column
BEAM_WIDTH = 0.3          # m
BEAM_DEPTH = 0.6          # m

FLOOR_PRESSURE = 16e3     # Pa, seismic floor load, lumped at each floor master
GRAVITY = 9.81            # m/s2
BASE_SHEAR_RATIO = 0.10   # lateral load as a fraction of the seismic weight
N_MODES = 3

# m, slab-plane band that collects the diaphragm joints; the 1 m default is a third of a storey.
PLANE_TOL = 0.05
# Rayleigh vs eigen T1. Rayleigh's error is second order in the trial-shape error, and the
# static shape under the h_k pattern is within about 1 % of the first mode of this frame;
# 0.5 % leaves margin. Both periods come from the same stiffness, so this checks the
# mass the eigen solve used and the solve itself (a singular K fails), not the members.
TOL_RAYLEIGH = 0.005

MM = 1e-3                 # m per mm
PCT = 1e-2                # drift ratio per percent

# Sections: rectangle formulas. Torsion constant J = beta h b^3, with beta by h/b
# from Timoshenko and Goodier's table.
COLUMN_AREA = COLUMN_SIDE**2                              # m2
COLUMN_INERTIA = COLUMN_SIDE**4 / 12                      # m4, both axes of the square
COLUMN_TORSION = 0.141 * COLUMN_SIDE**4                   # m4, beta = 0.141 at h/b = 1
BEAM_AREA = BEAM_WIDTH * BEAM_DEPTH                       # m2
BEAM_INERTIA_STRONG = BEAM_WIDTH * BEAM_DEPTH**3 / 12     # m4, vertical bending
BEAM_INERTIA_WEAK = BEAM_DEPTH * BEAM_WIDTH**3 / 12       # m4, horizontal bending
BEAM_TORSION = 0.229 * BEAM_DEPTH * BEAM_WIDTH**3         # m4, beta = 0.229 at h/b = 2

# Floor mass, and its polar inertia about Z for a uniform square slab: m (a^2 + b^2) / 12.
PLAN_SIDE = N_BAYS * BAY                                  # m
PLAN_CENTRE = PLAN_SIDE / 2                               # m, centre of mass in X and in Y
FLOOR_WEIGHT = FLOOR_PRESSURE * PLAN_SIDE**2              # N
FLOOR_MASS = FLOOR_WEIGHT / GRAVITY                       # kg
FLOOR_ROT_INERTIA = FLOOR_MASS * 2 * PLAN_SIDE**2 / 12    # kg m2

# Equivalent lateral forces. Equal floor weights, so F_k = V h_k / sum(h).
FLOORS = range(1, N_STOREYS + 1)                          # floor levels above the base
BASE_SHEAR = BASE_SHEAR_RATIO * N_STOREYS * FLOOR_WEIGHT  # N
SUM_OF_HEIGHTS = sum(k * STOREY for k in FLOORS)          # m
floor_force = {}
for k in FLOORS:
    floor_force[k] = BASE_SHEAR * k * STOREY / SUM_OF_HEIGHTS   # N

with apeGmsh(model_name="rc_frame_2storey") as g:
    # --- 2. Geometry
    # Joint "J{i}{j}_L{k}": column line i in X, j in Y, level k (0 = base).
    joints_at_level = {}
    for k in range(N_STOREYS + 1):
        joints_at_level[k] = []
        for i in range(N_BAYS + 1):
            for j in range(N_BAYS + 1):
                name = f"J{i}{j}_L{k}"
                g.model.geometry.add_point(i * BAY, j * BAY, k * STOREY, label=name)
                joints_at_level[k].append(name)
    # One master per floor at the centre of mass, so the floor mass sits where it acts.
    for k in FLOORS:
        g.model.geometry.add_point(PLAN_CENTRE, PLAN_CENTRE, k * STOREY, label=f"floor_{k}_master")

    column_labels = []
    for k in FLOORS:
        below = k - 1
        for i in range(N_BAYS + 1):
            for j in range(N_BAYS + 1):
                name = f"COL{i}{j}_S{k}"
                g.model.geometry.add_line(f"J{i}{j}_L{below}", f"J{i}{j}_L{k}", label=name)
                column_labels.append(name)

    beam_labels = []
    for k in FLOORS:
        for i in range(N_BAYS):            # beams spanning X, from line i to i + 1
            i_next = i + 1
            for j in range(N_BAYS + 1):
                name = f"BX{i}{j}_L{k}"
                g.model.geometry.add_line(f"J{i}{j}_L{k}", f"J{i_next}{j}_L{k}", label=name)
                beam_labels.append(name)
        for i in range(N_BAYS + 1):        # beams spanning Y, from line j to j + 1
            for j in range(N_BAYS):
                j_next = j + 1
                name = f"BY{i}{j}_L{k}"
                g.model.geometry.add_line(f"J{i}{j}_L{k}", f"J{i}{j_next}_L{k}", label=name)
                beam_labels.append(name)

    # --- 3. Groups and diaphragms
    g.physical.add_curve(column_labels, name="columns")
    g.physical.add_curve(beam_labels, name="beams")
    g.physical.add_point(joints_at_level[0], name="column_bases")
    for k in FLOORS:
        g.physical.add_point(joints_at_level[k], name=f"floor_{k}_joints")
        g.physical.add_point([f"floor_{k}_master"], name=f"floor_{k}_master")
        # Rigid slab: every floor joint follows the master in ux, uy and rz. The master
        # group holds one node, so that node is the master and the slab plane passes
        # through it (master_point is omitted).
        g.constraints.rigid_diaphragm(
            f"floor_{k}_master", f"floor_{k}_joints",
            plane_normal=(0, 0, 1),
            plane_tolerance=PLANE_TOL,
        )

    # --- 4. Mesh
    # Two nodes per member line: elastic prismatic members with nodal loads need no interior nodes.
    g.mesh.structured.one_element_per_curve("columns", "beams")
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

# --- 5. OpenSees model
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# Columns: local x is vertical, vecxz along global X. The section is square, so
# any horizontal vector gives the same stiffness; X is the plain choice.
column_transf = ops.geomTransf.Linear(vecxz=(1, 0, 0))
# Beams (X and Y spans alike): vecxz along global Z puts local z vertical and
# local y horizontal, so Iy carries vertical bending and takes the strong inertia.
beam_transf = ops.geomTransf.Linear(vecxz=(0, 0, 1))

ops.element.elasticBeamColumn(pg="columns", transf=column_transf,
                              A=COLUMN_AREA, E=YOUNG_MODULUS, G=SHEAR_MODULUS, J=COLUMN_TORSION,
                              Iy=COLUMN_INERTIA, Iz=COLUMN_INERTIA)
ops.element.elasticBeamColumn(pg="beams", transf=beam_transf,
                              A=BEAM_AREA, E=YOUNG_MODULUS, G=SHEAR_MODULUS, J=BEAM_TORSION,
                              Iy=BEAM_INERTIA_STRONG, Iz=BEAM_INERTIA_WEAK)

ops.fix(pg="column_bases", dofs=(1, 1, 1, 1, 1, 1))
for k in FLOORS:
    # The master is attached only to the diaphragm, which carries ux, uy and rz. Left free,
    # its uz, rx, ry make the stiffness singular: periods of hours, yet analyze still returns 0
    # (#1333). The Rayleigh check in step 8 is what catches it.
    ops.fix(pg=f"floor_{k}_master", dofs=(0, 0, 1, 1, 1, 0))
    ops.mass(pg=f"floor_{k}_master",
             values=(FLOOR_MASS, FLOOR_MASS, 0.0, 0.0, 0.0, FLOOR_ROT_INERTIA))

# Analysis chain. Transformation condenses the diaphragm slaves out exactly (no penalty
# to tune); it is declared ahead of eigen, which would otherwise auto-pick it with a warning.
ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.algorithm.Linear()                 # linear elastic model: one solve is exact
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()

# --- 6. Modal analysis
modes = ops.eigen(N_MODES)
periods = list(modes.periods)

# --- 7. Lateral static load in X
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as lateral:
    for k in FLOORS:
        lateral.load(pg=f"floor_{k}_master", forces=(floor_force[k], 0.0, 0.0, 0.0, 0.0, 0.0))
status = ops.analyze(steps=1)
assert status == 0, f"static analysis failed with status {status}"

# --- 8. Checks and report
# Floor displacement in X at each master, and the storey drift ratio below it.
# Live read at each floor master, by its label.
floor_ux = {}
drift_ratio = {}
ux_below = 0.0   # m, the base is fixed
for k in FLOORS:
    floor_ux[k] = ops.nodes.one(label=f"floor_{k}_master").read("displacement_x")
    drift_ratio[k] = (floor_ux[k] - ux_below) / STOREY
    ux_below = floor_ux[k]

# Hand check, Rayleigh's method (Chopra, Dynamics of Structures, ch. 8):
# T = 2 pi sqrt(sum m_k u_k^2 / sum F_k u_k), u_k the static floor displacements under F_k.
# The plan is symmetric, so the X and Y translation modes share one period: T1 = T2.
kinetic_term = sum(FLOOR_MASS * floor_ux[k]**2 for k in FLOORS)   # kg m2
work_term = sum(floor_force[k] * floor_ux[k] for k in FLOORS)     # N m
rayleigh_period = 2 * math.pi * math.sqrt(kinetic_term / work_term)   # s
rayleigh_error = abs(rayleigh_period - periods[0]) / periods[0]
assert rayleigh_error < TOL_RAYLEIGH, f"Rayleigh period off by {rayleigh_error:.2%}"

lines = []
for mode, period in enumerate(periods, start=1):
    lines.append(f"T{mode} = {period:.4f} s")
lines.append(f"Rayleigh period = {rayleigh_period:.4f} s ({rayleigh_error:.3%} from T1)")
for k in FLOORS:
    ux_mm = floor_ux[k] / MM
    drift_pct = drift_ratio[k] / PCT
    lines.append(f"Floor {k}: ux = {ux_mm:.3f} mm, storey {k} drift ratio = {drift_pct:.4f} %")
report = "\n".join(lines)
print(report)
(OUT / "report.txt").write_text(report + "\n")

# The result file the task asks for: SI units.
floor_displacements = []
for k in FLOORS:
    floor_displacements.append(float(floor_ux[k]))
result = {
    "periods_s": periods,
    "floor_ux_m": floor_displacements,
}
RESULT_FILE.write_text(json.dumps(result, indent=2))
