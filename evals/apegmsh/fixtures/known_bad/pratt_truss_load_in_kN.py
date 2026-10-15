"""KNOWN-BAD (the joint load written in kN in an N-m-Pa model, so every force is 1000 times too small).

2-D Pratt truss bridge under bottom-chord joint loads: linear static analysis.

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
    6. Analysis: one linear static step
    7. Checks and report: bottom chord by sections, end posts by joints,
       then the axial-force range per family and the midspan deflection
"""
import json
import math
from pathlib import Path

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- 1. Data
RESULT_FILE = Path(__file__).parent / "result.json"

N_PANELS = 6              # number of panels along the span
PANEL = 4.0               # m, panel length
DEPTH = 4.0               # m, chord centreline to chord centreline

E_STEEL = 200e9           # Pa, Young's modulus of structural steel
A_CHORD = 6.0e-3          # m2, top and bottom chords
A_END_POST = 6.0e-3       # m2, inclined end posts (each carries a full support reaction)
A_VERTICAL = 3.0e-3       # m2, verticals
A_DIAGONAL = 4.0e-3       # m2, interior diagonals

JOINT_LOAD = 100.0        # kN, downward load at each interior bottom-chord joint

# Newton, not Linear, so the test really checks equilibrium; on a linear model
# it converges in one iteration.
TOL_UNBALANCE = 1e-3      # N, out-of-balance force norm (1e-8 of one joint load)
MAX_ITER = 10             # iterations allowed per step (a linear model needs one)

# The truss is statically determinate (21 bars + 3 reactions = 2 x 12 joints), so
# its bar forces follow from statics alone. The live read is full precision, so this
# bound only has to cover the solver's roundoff; it is kept loose on purpose.
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
    # One element per member: each bar has only its two end joints as nodes.
    g.mesh.structured.one_element_per_curve(*MEMBERS)
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

# --- 7. Checks and report
# Live read at the last committed state: one axial force per bar of the family, in the
# group's element order (N, tension +).
axial_force = {}
for family in MEMBERS:
    axial_force[family] = ops.elements.get(pg=family).read("axial_force")
# The vertical displacement of the midspan joint, by its geometry label (m, up +).
midspan_deflection = ops.nodes.one(label=MIDSPAN_JOINT).read("displacement_y")

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

# The result file the task asks for: SI units, tension +, up +.
result = {
    "bottom_chord_max_axial_N": float(bottom_chord_fe),
    "end_post_axial_N": float(end_post_fe),
    "midspan_uy_m": float(midspan_deflection),
}
RESULT_FILE.write_text(json.dumps(result, indent=2))
