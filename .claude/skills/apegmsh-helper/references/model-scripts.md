# Model scripts: Script English

Read this before writing a **user model script**: a flat procedure that builds, solves and
checks one structure. It does not cover changing apeGmsh itself. Read the rules, then only
the one reference script nearest your task; keep its shape and change the data and the structure.

## The rules

A model script reads as an engineering procedure that a structural engineer can open cold.
It never costs modelling power: if a rule would force a weaker model, the rule gives way.
These rules cover user model scripts, not library code.

1. **Shape.** The docstring gives the problem, the unit system and a numbered procedure. Then
   come the imports, the data, the geometry, the groups/supports/loads, the mesh, the OpenSees
   model, the analysis, and the checks and report. Each step opens with `# --- N. <step>`,
   matching the docstring.
2. **Flat and local.** No `main()`, no helper functions, nothing defined and left unused,
   nothing the task did not ask for. Outputs go to a folder beside the script, never to an
   absolute path.
3. **Fidelity.** Build the model the task describes with the verb that names each part
   (material law, stages, supports, loads as given). Never use an elastic, unstaged,
   point-load or hand-rebuilt stand-in.
4. **Labels, not tags.** Every object the procedure names gets an engineering label when it is
   created. Find edges, supports and results by that label, never by a tag, an index, a Python
   handle, coordinates or a tolerance box.
5. **One object, one name.** A physical group keeps the name of the label it groups and is made
   in one call, from a list if needed.
6. **Named numbers.** Each engineering number sits in the data section, one per line, with its
   unit and meaning (`SU = 75.0  # kPa, undrained shear strength`). A tuned value says why it
   was tuned. State the unit system once, and convert for printing by dividing by a named unit
   (`settlement / MM`). Spell names out; a textbook symbol appears only inside its formula.
7. **One action per statement.** No nested comprehensions, no long one-line formulas, no
   arithmetic inside an f-string. Dividing by a named unit (`{force / KN:.1f}`) is the one
   exception.
8. **One OpenSees handle.** Drive OpenSees through the `apeSees` bridge. Read results from its
   recorders with `np.loadtxt`, under a comment that says what one row holds. For live
   read-back, import raw openseespy at the top as `ops_raw` and mark each use
   `# raw-ops: <why>`. Name a beam `geomTransf` after its member family, and say which way its
   `vecxz` points and why.
9. **Check, assert, report.** Write the hand check as a few named steps, with a comment giving
   the formula or source. Assert that the run succeeded, and that each check agrees within a
   named tolerance whose comment says why it has that size. Report only numbers the run
   produced.
10. **Comments say why, and stay true.** Every non-default setting, and any step a reader would
    not expect, carries its reason. A reason longer than the line goes above it.

**Check agent-written scripts.** Run `python scripts/lint_model_script.py <script.py>` (the lint
lives beside this skill; it needs only the standard library) on every model script an agent
writes, and fix each finding before handing the script over. It covers rules 1 to 2 (S1, S2, S3),
4 (V1, V3), 6 (V4, V5), 7 (T1), 8 (T2) and 9 (T4), and prints `path:line: RULE message`;
`--rules S1,V5` narrows it. It is advisory, never a gate: fidelity (rule 3), role naming (rule 5),
symbol choice, the quality of a hand check and whether a comment is true still need a reviewer.

## Reference scripts

Each script runs on stock openseespy and asserts its own hand check.
`tests/test_skill_model_scripts.py` runs all three in the `live` lane. Pick the family nearest
your task:

| Script | Shows |
|---|---|
| `pratt_truss.py` | 2-D truss: one element per member (`set_transfinite_curve(list_of_groups, n_nodes=2)`), member families as groups, recorders read with `np.loadtxt`, method-of-sections check |
| `frame_modal.py` | 3-D frame: `geomTransf` per member family with its `vecxz` reason, rigid floor diaphragms with a fixed master, eigen, then a lateral load and a Rayleigh check |
| `staged_footing.py` | 2-D staged soil: `g.constraints.bc` + `ops.fix_from_model()`, a self-weight case and a pressure case, two `ops.stage` blocks, deck run, equilibrium and Flamant checks |

Lines marked `API gap` name a workaround that will go away when the gap is closed. Do not
copy the workaround into a script where the gap does not apply.

### 2-D Pratt truss: linear static

<!-- reference-script: pratt_truss.py -->
```python
"""2-D Pratt truss bridge under bottom-chord joint loads: linear static analysis.

Six panels of 4 m (span 24 m), depth 4 m, steel members. Pin at the left end,
roller at the right. A 100 kN downward load acts at each of the five interior
bottom-chord joints. Every member is one pin-ended truss element.

Joints: bottom chord B0..B6 (left to right), top chord T1..T5 above B1..B5.
The end posts run B0-T1 and T5-B6; the Pratt diagonals slope down towards
midspan (T1-B2, T2-B3, T4-B3, T5-B4), so under gravity they pull.
Units: N, m, Pa throughout (forces printed in kN, deflection in mm).

Procedure:
    1. Data
    2. Geometry: the joints, and each bar labelled after its two joints
    3. Groups and loads: one group per member family, supports, loaded joints
    4. Mesh: one truss element per member
    5. OpenSees model: material, members, supports, load pattern
    6. Analysis: one linear static step, with recorders on every family
    7. Checks and report: bottom chord by sections, end posts by joints,
       then the axial-force range per family and the midspan deflection
"""
import math
from pathlib import Path

import numpy as np
import openseespy.opensees as ops_raw

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
OUT_DIR = Path(__file__).parent / "out_pratt_truss"
OUT_DIR.mkdir(exist_ok=True)

N_PANELS = 6              # number of panels along the span
PANEL = 4.0               # m, panel length
DEPTH = 4.0               # m, chord centreline to chord centreline

E_STEEL = 200e9           # Pa, Young's modulus of structural steel
A_CHORD = 6.0e-3          # m2, top and bottom chords
A_END_POST = 6.0e-3       # m2, inclined end posts (each carries a full support reaction)
A_VERTICAL = 3.0e-3       # m2, verticals
A_DIAGONAL = 4.0e-3       # m2, interior diagonals

JOINT_LOAD = 100e3        # N, downward load at each interior bottom-chord joint

# Newton, not Linear, so the test really checks equilibrium; on a linear model
# it converges in one iteration.
TOL_UNBALANCE = 1e-3      # N, out-of-balance force norm (1e-8 of one joint load)
MAX_ITER = 10             # iterations allowed per step (a linear model needs one)

# The truss is statically determinate (21 bars + 3 reactions = 2 x 12 joints), so
# its bar forces follow from statics alone; what is left is the recorder file's
# rounding to 6 significant digits (API gap: the bridge exposes no -precision).
TOL_STATICS = 1e-5        # relative, FE vs hand bar force

KN = 1e3                  # N per kN, for printing
MM = 1e-3                 # m per mm, for printing

# Members by family, each bar named "start-end" after its two joints.
MEMBERS = {
    "bottom_chord": ["B0-B1", "B1-B2", "B2-B3", "B3-B4", "B4-B5", "B5-B6"],
    "top_chord": ["T1-T2", "T2-T3", "T3-T4", "T4-T5"],
    "end_posts": ["B0-T1", "T5-B6"],
    "verticals": ["B1-T1", "B2-T2", "B3-T3", "B4-T4", "B5-T5"],
    "diagonals": ["T1-B2", "T2-B3", "T4-B3", "T5-B4"],
}
LOADED_JOINTS = ["B1", "B2", "B3", "B4", "B5"]
MIDSPAN_JOINT = "B3"

with apeGmsh(model_name="pratt_truss", verbose=False) as g:
    # --- 2. Geometry
    for i in range(N_PANELS + 1):
        g.model.geometry.add_point(i * PANEL, 0.0, 0.0, label=f"B{i}")
    for i in range(1, N_PANELS):
        g.model.geometry.add_point(i * PANEL, DEPTH, 0.0, label=f"T{i}")
    for bars in MEMBERS.values():
        for bar in bars:
            start, end = bar.split("-")
            g.model.geometry.add_line(start, end, label=bar)

    # --- 3. Groups and loads
    for family, bars in MEMBERS.items():
        g.physical.from_labels(bars, name=family)
    g.labels.promote_to_physical("B0")
    g.labels.promote_to_physical("B6")
    g.labels.promote_to_physical(MIDSPAN_JOINT)
    g.physical.from_labels(LOADED_JOINTS, name="loaded_joints")

    with g.loads.case("gravity"):
        g.loads.point.force("loaded_joints", force=(0.0, -JOINT_LOAD, 0.0))

    # --- 4. Mesh
    # API gap: no one-element-per-member switch; two nodes per curve gives one element.
    g.mesh.structured.set_transfinite_curve(list(MEMBERS), n_nodes=2)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

# --- 5. OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
steel = ops.uniaxialMaterial.ElasticMaterial(E=E_STEEL)
ops.element.Truss(pg="bottom_chord", A=A_CHORD, material=steel)
ops.element.Truss(pg="top_chord", A=A_CHORD, material=steel)
ops.element.Truss(pg="end_posts", A=A_END_POST, material=steel)
ops.element.Truss(pg="verticals", A=A_VERTICAL, material=steel)
ops.element.Truss(pg="diagonals", A=A_DIAGONAL, material=steel)

ops.fix(pg="B0", dofs=(1, 1))   # pin
ops.fix(pg="B6", dofs=(0, 1))   # roller: free to slide, so the chords carry no support thrust

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("gravity")

# --- 6. Analysis
for family in MEMBERS:
    ops.recorder.Element(file=str(OUT_DIR / f"{family}.out"), pg=family, response=("axialForce",))
ops.recorder.Node(file=str(OUT_DIR / "midspan.out"), pg=MIDSPAN_JOINT, dofs=(2,), response="disp")

# Plain: the model has no multi-point constraints. RCM + BandGeneral: a small banded system.
ops.constraints.Plain()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormUnbalance(tol=TOL_UNBALANCE, max_iter=MAX_ITER)
ops.algorithm.Newton()
ops.integrator.LoadControl(dlam=1.0)   # the full load in one step: the model is linear
ops.analysis.Static()
status = ops.analyze(steps=1)
assert status == 0, f"static analysis failed (code {status})"

# API gap: no bridge verb closes live recorders, and OpenSees buffers their rows until then.
ops_raw.wipe()   # raw-ops: clearing the domain closes the recorder files, flushing every row

# --- 7. Checks and report
# One row per step; each column is the axial force of one bar of the family (N, tension +).
axial_force = {}
for family in MEMBERS:
    axial_force[family] = np.loadtxt(OUT_DIR / f"{family}.out")
# One row per step: the vertical displacement of the midspan joint (m, up +).
midspan_deflection = float(np.loadtxt(OUT_DIR / "midspan.out"))

reaction = len(LOADED_JOINTS) * JOINT_LOAD / 2   # N, at each support, by symmetry

# Method of sections: the bottom chord peaks in the two midspan panels. Cut
# panel B2-B3 and take moments about T2; the top chord and the diagonal T2-B3
# both pass through T2, so only the bottom chord resists the moment of the
# left free body (the reaction at B0 and the load at B1):
#     N = (R * x_T2 - P * (x_T2 - x_B1)) / DEPTH
x_T2 = 2 * PANEL          # m, from the pin
x_B1 = PANEL              # m, from the pin
moment_about_T2 = reaction * x_T2 - JOINT_LOAD * (x_T2 - x_B1)
bottom_chord_hand = moment_about_T2 / DEPTH
bottom_chord_fe = axial_force["bottom_chord"].max()
bottom_chord_error = bottom_chord_fe / bottom_chord_hand - 1
assert abs(bottom_chord_error) < TOL_STATICS, f"bottom chord off by {bottom_chord_error:.1e}"

# Method of joints at B0: the end post alone balances the vertical reaction,
#     N = -R / sin(theta), sin(theta) = DEPTH / end-post length (compression -)
end_post_length = math.hypot(PANEL, DEPTH)
end_post_hand = -reaction * end_post_length / DEPTH
end_post_fe = axial_force["end_posts"].min()
end_post_error = end_post_fe / end_post_hand - 1
assert abs(end_post_error) < TOL_STATICS, f"end post off by {end_post_error:.1e}"

print(f"Bottom chord peak: FE {bottom_chord_fe / KN:.1f} kN, "
      f"method of sections {bottom_chord_hand / KN:.1f} kN ({bottom_chord_error:+.1e})")
print(f"End posts:         FE {end_post_fe / KN:.1f} kN, "
      f"method of joints {end_post_hand / KN:.1f} kN ({end_post_error:+.1e})")
print("Axial force range per family (kN, tension +):")
for family, forces in axial_force.items():
    print(f"  {family:13s} {forces.min() / KN:7.1f} to {forces.max() / KN:7.1f}")
print(f"Midspan deflection at {MIDSPAN_JOINT}: {midspan_deflection / MM:.2f} mm")
```

### 3-D frame with rigid floors: modal, then lateral load

<!-- reference-script: frame_modal.py -->
```python
"""Two-storey 3-D RC frame with rigid floors: modal analysis, then a lateral load in X.

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
import math
from pathlib import Path

import openseespy.opensees as ops_raw

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
OUT = Path(__file__).parent / "out_frame_modal"
OUT.mkdir(exist_ok=True)

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

FLOOR_PRESSURE = 8e3      # Pa, seismic floor load, lumped at each floor master
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
        # Rigid slab: every floor joint follows the master in ux, uy and rz. master_point
        # sets the slab plane and picks the nearest node as master, so it repeats the
        # master's position although the master group holds one node (API gap).
        g.constraints.rigid_diaphragm(
            f"floor_{k}_master", f"floor_{k}_joints",
            master_point=(PLAN_CENTRE, PLAN_CENTRE, k * STOREY),
            plane_normal=(0, 0, 1),
            plane_tolerance=PLANE_TOL,
        )

    # --- 4. Mesh
    # Two nodes per member line: elastic prismatic members with nodal loads need no interior nodes.
    g.mesh.structured.set_transfinite_curve("columns", n_nodes=2)
    g.mesh.structured.set_transfinite_curve("beams", n_nodes=2)
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
# Linear does not iterate, so the tolerance is inert; the bridge still requires a test (API gap).
ops.test.NormDispIncr(tol=1e-10, max_iter=1)
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
# A one-node group still returns a set, and there is no read-back by label (API gap).
floor_ux = {}
drift_ratio = {}
ux_below = 0.0   # m, the base is fixed
for k in FLOORS:
    [master] = ops.nodes.get(pg=f"floor_{k}_master")
    floor_ux[k] = ops_raw.nodeDisp(master.tag, 1)   # raw-ops: live displacement read-back
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
```

### Staged strip footing on J2 clay

<!-- reference-script: staged_footing.py -->
```python
"""Staged plane-strain strip footing on an elastic-plastic clay block.

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

import math
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
OUT_DIR = Path(__file__).resolve().parent / "out_staged_footing"
OUT_DIR.mkdir(exist_ok=True)

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
YIELD_STRESS = math.sqrt(3) * SU   # kPa, von Mises uniaxial yield for strength SU in pure shear

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
# API gap: no stage-level shared chain, so the chain is a dict passed to each stage.
solver = dict(
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
    s.analysis(integrator=ops.integrator.LoadControl(dlam=step_geostatic), **solver)
    s.run(n_increments=N_STEPS_GEOSTATIC, dt=step_geostatic)   # API gap: the step is stated twice

# --- 8. Stage 2, footing: pressure ramped to FOOTING_PRESSURE, self-weight held
with ops.stage(name="footing") as s:
    with s.pattern(series=ops.timeSeries.Linear()) as p:
        p.from_model("footing_pressure")
    s.analysis(integrator=ops.integrator.LoadControl(dlam=step_footing), **solver)
    s.run(n_increments=N_STEPS_FOOTING, dt=step_footing)   # API gap: the step is stated twice

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
assert plastic_share > MIN_PLASTIC_SHARE, "the clay never yielded"

print(f"Equilibrium error: self-weight {err_weight:+.1e}, footing load {err_footing:+.1e}")
print(f"Settlement at {FOOTING_PRESSURE:.0f} kPa: {settlement_final / MM:.1f} mm")
print(f"  of which beyond the elastic line: {settlement_plastic / MM:.1f} mm")
print(f"Early slope: FE {slope_fe / MM:.4f}, hand {slope_hand / MM:.4f} mm/kPa ({err_slope:+.1%})")

curve = np.column_stack([pressure, settlement / MM])
np.savetxt(OUT_DIR / "settlement_vs_pressure.csv", curve, delimiter=",",
           header="pressure_kPa,settlement_mm", comments="", fmt="%.4f")
```
