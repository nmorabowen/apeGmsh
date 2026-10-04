"""2-D Pratt truss bridge, 6 panels x 4 m (span 24 m), height 4 m.

Linear static analysis under 100 kN gravity point loads at the interior
bottom-chord joints. Reports peak chord / diagonal axial forces and midspan
deflection, and checks the midspan bottom-chord force by the method of sections.
Units: kN, m (so stresses and E are in kPa).
"""
import csv
import os
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
import openseespy.opensees as ops_raw

# ---------------------------------------------------------------- data
n_panels = 6
panel = 4.0                 # m, panel length
H = 4.0                     # m, truss height
span = n_panels * panel     # m

E = 200e6                   # kPa, structural steel
A_chord = 0.012             # m2, top and bottom chords
A_vertical = 0.006          # m2, verticals (compression posts)
A_diagonal = 0.008          # m2, diagonals (tension)
P = 100.0                   # kN, gravity load per interior bottom joint

nodes_per_member = 2        # one truss element per member (interior nodes would be mechanisms)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_a")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- geometry
with apeGmsh(model_name="pratt_truss", save_to=os.path.join(out_dir, "pratt_truss.h5")) as g:
    geo = g.model.geometry

    # Joints are labelled bottom_i / top_i, i = 0..n_panels, left to right.
    for i in range(n_panels + 1):
        x = i * panel
        geo.add_point(x, 0.0, 0.0, label=f"bottom_{i}")
        geo.add_point(x, H, 0.0, label=f"top_{i}")

    # Members, one label per member.
    member_labels = []
    for i in range(n_panels):
        geo.add_line(f"bottom_{i}", f"bottom_{i+1}", label=f"bc_{i}")
        geo.add_line(f"top_{i}", f"top_{i+1}", label=f"tc_{i}")
        member_labels += [f"bc_{i}", f"tc_{i}"]
    for i in range(n_panels + 1):
        geo.add_line(f"bottom_{i}", f"top_{i}", label=f"vert_{i}")
        member_labels.append(f"vert_{i}")

    # Pratt pattern: diagonals fall toward midspan (tension), verticals carry compression.
    half = n_panels // 2
    for i in range(half):
        geo.add_line(f"top_{i}", f"bottom_{i+1}", label=f"diag_{i}")
        member_labels.append(f"diag_{i}")
    for i in range(half, n_panels):
        geo.add_line(f"top_{i+1}", f"bottom_{i}", label=f"diag_{i}")
        member_labels.append(f"diag_{i}")

    # ------------------------------------------------------------ groups & loads
    for i in range(n_panels):
        g.physical.add_curve(f"bc_{i}", name="bottom_chord")
        g.physical.add_curve(f"tc_{i}", name="top_chord")
        g.physical.add_curve(f"diag_{i}", name="diagonal")
    for i in range(n_panels + 1):
        g.physical.add_curve(f"vert_{i}", name="vertical")

    g.physical.add_point("bottom_0", name="pin_support")
    g.physical.add_point(f"bottom_{n_panels}", name="roller_support")
    for i in range(1, n_panels):
        g.physical.add_point(f"bottom_{i}", name="loaded_joints")
    g.physical.add_point(f"bottom_{half}", name="midspan_joint")

    # ------------------------------------------------------------ mesh
    for member in member_labels:
        g.mesh.structured.set_transfinite_curve(member, nodes_per_member)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=None)

# ---------------------------------------------------------------- OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
bottom_chords = ops.element.Truss(pg="bottom_chord", A=A_chord, material=steel)
ops.element.Truss(pg="top_chord", A=A_chord, material=steel)
ops.element.Truss(pg="vertical", A=A_vertical, material=steel)
ops.element.Truss(pg="diagonal", A=A_diagonal, material=steel)

ops.fix(pg="pin_support", dofs=(1, 1))
ops.fix(pg="roller_support", dofs=(0, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="loaded_joints", forces=(0.0, -P))

# ---------------------------------------------------------------- analysis
ops.constraints.Plain()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=1e-8, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()
ops.run()
ops.analyze(steps=1)

# ---------------------------------------------------------------- checks & report
midspan_node = ops.nodes.get(pg="midspan_joint").tags[0]
# raw-ops: the bridge has no post-run query for nodal displacement or element force
midspan_deflection = ops_raw.nodeDisp(midspan_node, 2)  # m, negative = downward

kinds = ("bottom chord", "top chord", "vertical", "diagonal")
forces = {kind: [] for kind in kinds}       # kind -> list of axial forces, kN (+ tension)
force_at = {}                               # ((xi, yi), (xj, yj)) -> axial force, kN
rows = []
for ele in ops_raw.getEleTags():            # raw-ops: element tags are not exposed per group
    ni, nj = ops_raw.eleNodes(ele)
    xi, yi = ops_raw.nodeCoord(ni)
    xj, yj = ops_raw.nodeCoord(nj)
    N = ops_raw.eleResponse(ele, "axialForce")[0]   # raw-ops: element force read-back
    if xi == xj:
        kind = "vertical"
    elif yi == yj and yi == 0.0:
        kind = "bottom chord"
    elif yi == yj:
        kind = "top chord"
    else:
        kind = "diagonal"
    forces[kind].append(N)
    force_at[((round(xi, 6), round(yi, 6)), (round(xj, 6), round(yj, 6)))] = N
    rows.append((ele, kind, xi, yi, xj, yj, N))

chord_forces = forces["bottom chord"] + forces["top chord"]
max_chord_tension = max(chord_forces)
max_chord_compression = min(chord_forces)
max_diag_tension = max(forces["diagonal"])
min_diag_force = min(forces["diagonal"])   # stays positive: every diagonal is in tension

# Method of sections: cut the panel just left of midspan, take moments about the
# top joint at the cut's left end, where the top chord and the diagonal meet.
x_midspan = span / 2.0
x_cut_left = x_midspan - panel
n_loads_left = round(x_cut_left / panel)   # loaded joints at x = panel .. x_cut_left
reaction = P * (n_panels - 1) / 2.0                          # kN, symmetric loading
moment_reaction = reaction * x_cut_left                      # kN m about top joint at x_cut_left
moment_loads = 0.0
for k in range(1, n_loads_left + 1):
    lever = x_cut_left - k * panel
    moment_loads = moment_loads + P * lever
bottom_chord_hand = (moment_reaction - moment_loads) / H     # kN, tension

bottom_chord_fe = force_at[((x_cut_left, 0.0), (x_midspan, 0.0))]
chord_error_pct = 100.0 * (bottom_chord_fe - bottom_chord_hand) / bottom_chord_hand

csv_path = os.path.join(out_dir, "member_forces.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["element", "kind", "xi", "yi", "xj", "yj", "axial_force_kN"])
    writer.writerows(rows)

print(f"Max chord tension          : {max_chord_tension:9.2f} kN")
print(f"Max chord compression      : {max_chord_compression:9.2f} kN")
print(f"Max diagonal tension       : {max_diag_tension:9.2f} kN")
print(f"Min diagonal force (tension): {min_diag_force:9.2f} kN")
print(f"Midspan deflection         : {midspan_deflection * 1000.0:9.3f} mm")
print(f"Bottom chord at midspan, FE   : {bottom_chord_fe:9.2f} kN")
print(f"Bottom chord at midspan, hand : {bottom_chord_hand:9.2f} kN")
print(f"Difference                    : {chord_error_pct:9.4f} %")
