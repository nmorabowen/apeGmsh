"""2-D Pratt truss bridge under bottom-chord gravity joint loads -- linear static.

Six panels of 4 m (span 24 m), depth 4 m. Pin at the left support, roller at
the right. A 100 kN downward point load acts at each of the five interior
bottom-chord joints. Every member is one pin-ended truss element.

Joint names: bottom chord B0..B6 (left to right), top chord T1..T5 (above
B1..B5). Inclined end posts run B0-T1 and T5-B6; the Pratt diagonals slope
down towards midspan (T1-B2, T2-B3, T4-B3, T5-B4), so they work in tension.

Reports the peak chord and diagonal axial forces, the midspan deflection, and
checks the bottom-chord force next to midspan against the method of sections.

Units: N, m, Pa (forces printed in kN, deflection in mm).
"""
from pathlib import Path

from apeGmsh import Results, apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results.capture.spec import DomainCaptureSpec

# --- Data -------------------------------------------------------------------
OUT = Path(__file__).parent / "out_c"
OUT.mkdir(exist_ok=True)

N_PANELS = 6            # -
PANEL = 4.0             # m, panel length
H = 4.0                 # m, truss depth (chord centreline to centreline)
SPAN = N_PANELS * PANEL # m

E = 200e9               # Pa, structural steel
A_CHORD = 6.0e-3        # m2, top and bottom chords
A_END_POST = 6.0e-3     # m2, inclined end posts (carry the full reaction)
A_VERTICAL = 3.0e-3     # m2, verticals
A_DIAGONAL = 4.0e-3     # m2, interior diagonals

P = 100e3               # N, gravity load at each interior bottom joint
KN = 1e3                # N per kN, for reporting
MM = 1e-3               # m per mm, for reporting

# Solver: the Linear algorithm solves once; the test still has to be declared.
TOL = 1e-10             # m, displacement-increment tolerance
MAX_ITER = 10           # -

# Two mesh nodes per member: one pin-ended truss element between joints.
NODES_PER_MEMBER = 2

# Member connectivity by family, as (start joint, end joint).
MEMBERS = {
    "bottom_chord": [("B0", "B1"), ("B1", "B2"), ("B2", "B3"),
                     ("B3", "B4"), ("B4", "B5"), ("B5", "B6")],
    "top_chord":    [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5")],
    "end_posts":    [("B0", "T1"), ("T5", "B6")],
    "verticals":    [("B1", "T1"), ("B2", "T2"), ("B3", "T3"),
                     ("B4", "T4"), ("B5", "T5")],
    "diagonals":    [("T1", "B2"), ("T2", "B3"), ("T4", "B3"), ("T5", "B4")],
}
LOADED_JOINTS = ["B1", "B2", "B3", "B4", "B5"]
MIDSPAN_JOINT = "B3"
# The bottom-chord bar just left of midspan; its force is the hand-check target.
CHECKED_BAR = "B2-B3"

# --- Geometry -----------------------------------------------------------------
with apeGmsh(model_name="pratt_truss", verbose=False) as g:
    for i in range(N_PANELS + 1):
        g.model.geometry.add_point(i * PANEL, 0, 0, label=f"B{i}")
    for i in range(1, N_PANELS):
        g.model.geometry.add_point(i * PANEL, H, 0, label=f"T{i}")

    # Each bar gets its own label "start-end" so a member can be read back by name.
    for bars in MEMBERS.values():
        for start, end in bars:
            g.model.geometry.add_line(start, end, label=f"{start}-{end}")

    # --- Groups & loads -------------------------------------------------------
    for family, bars in MEMBERS.items():
        bar_labels = [f"{start}-{end}" for start, end in bars]
        g.physical.from_labels(bar_labels, name=family)
    g.labels.promote_to_physical("B0")
    g.labels.promote_to_physical("B6")
    g.physical.from_labels(LOADED_JOINTS, name="loaded_joints")

    with g.loads.case("gravity"):
        g.loads.point.force("loaded_joints", force=(0, -P, 0))

    # --- Mesh -----------------------------------------------------------------
    for bars in MEMBERS.values():
        for start, end in bars:
            g.mesh.structured.set_transfinite_curve(f"{start}-{end}", n_nodes=NODES_PER_MEMBER)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)
    print(fem.info)

# --- OpenSees model -------------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
ops.element.Truss(pg="bottom_chord", A=A_CHORD, material=steel)
ops.element.Truss(pg="top_chord", A=A_CHORD, material=steel)
ops.element.Truss(pg="end_posts", A=A_END_POST, material=steel)
ops.element.Truss(pg="verticals", A=A_VERTICAL, material=steel)
ops.element.Truss(pg="diagonals", A=A_DIAGONAL, material=steel)

ops.fix(pg="B0", dofs=(1, 1))   # pin
ops.fix(pg="B6", dofs=(0, 1))   # roller: free to slide, so no spurious chord force

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("gravity")

# --- Analysis -------------------------------------------------------------------
# Linear elastic and small-displacement: one full load step is exact.
ops.constraints.Plain()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=TOL, max_iter=MAX_ITER)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1)
ops.analysis.Static()

model_h5 = OUT / "pratt_model.h5"
run_h5 = OUT / "pratt_run.h5"
ops.h5(model_h5)

spec = DomainCaptureSpec(opensees=ops)
spec.nodes(components="displacement", label=MIDSPAN_JOINT)
spec.gauss(components="axial_force", pg=list(MEMBERS))
with ops.domain_capture(spec, path=run_h5) as cap:
    cap.begin_stage("gravity", kind="static")
    status = ops.analyze(steps=1)
    cap.step(t=1.0)
    cap.end_stage()
assert status == 0, f"static analysis failed (code {status})"

# --- Checks & report --------------------------------------------------------------
results = Results.from_native(run_h5, model=OpenSeesModel.from_h5(model_h5))

# Axial force per bar, tension positive, read back by bar label.
rows = ["family,bar,axial_force_kN"]
peak = {}
for family, bars in MEMBERS.items():
    peak[family] = 0.0
    for start, end in bars:
        bar = f"{start}-{end}"
        N = results.elements.gauss.get(label=bar, component="axial_force").values[-1, 0]
        rows.append(f"{family},{bar},{N / KN:.3f}")
        if abs(N) > abs(peak[family]):
            peak[family] = N
(OUT / "member_forces.csv").write_text("\n".join(rows) + "\n")

# Chords = top + bottom; the end posts are reported on their own.
if abs(peak["top_chord"]) > abs(peak["bottom_chord"]):
    N_chord_max = peak["top_chord"]
else:
    N_chord_max = peak["bottom_chord"]
N_diag_max = peak["diagonals"]
N_end_post_max = peak["end_posts"]

uy_mid = results.nodes.get(label=MIDSPAN_JOINT, component="displacement_y").values[-1, 0]
N_checked = results.elements.gauss.get(label=CHECKED_BAR, component="axial_force").values[-1, 0]

# Method of sections: cut panel 3 (x = 8..12 m) and take moments about T2
# (x = 8 m, top chord). The top chord and the diagonal T2-B3 both pass
# through T2, so only the bottom chord B2-B3 resists that moment.
R_support = len(LOADED_JOINTS) * P / 2
x_T2 = 2 * PANEL
M_T2 = R_support * x_T2 - P * (x_T2 - PANEL)
N_hand = M_T2 / H
error_pct = 100 * (N_checked - N_hand) / N_hand

print(f"Span {SPAN:.0f} m, {N_PANELS} panels, depth {H:.0f} m, P = {P / KN:.0f} kN per joint")
print(f"Max chord axial force      : {N_chord_max / KN:9.1f} kN  (+ tension)")
print(f"Max diagonal axial force   : {N_diag_max / KN:9.1f} kN")
print(f"Max end-post axial force   : {N_end_post_max / KN:9.1f} kN")
print(f"Midspan deflection ({MIDSPAN_JOINT})    : {uy_mid / MM:9.2f} mm")
print(f"Bar {CHECKED_BAR}: FE {N_checked / KN:.1f} kN vs method of sections {N_hand / KN:.1f} kN"
      f" (error {error_pct:.2e} %)")
print(f"Member forces written to {OUT / 'member_forces.csv'}")
