"""2-D Pratt truss bridge, 6 panels x 4 m, h = 4 m. Linear static."""
import os
import csv
import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_b")
os.makedirs(OUT, exist_ok=True)

# ---- geometry parameters (m, N, Pa)
N_PANEL, L_PANEL, H = 6, 4.0, 4.0
SPAN = N_PANEL * L_PANEL
P_JOINT = 100e3            # N, downward at each interior bottom joint
E_STEEL = 200e9
A_CHORD, A_VERT, A_DIAG = 8.0e-3, 3.0e-3, 4.0e-3   # m^2

with apeGmsh(model_name="pratt_truss", save_to=os.path.join(OUT, "pratt.h5")) as g:
    geo = g.model.geometry
    bot = [geo.add_point(i * L_PANEL, 0.0, 0.0) for i in range(N_PANEL + 1)]
    top = {i: geo.add_point(i * L_PANEL, H, 0.0) for i in range(1, N_PANEL)}

    members = {"BottomChord": [], "TopChord": [], "Verticals": [], "Diagonals": [], "EndPosts": []}
    for i in range(N_PANEL):
        members["BottomChord"].append(geo.add_line(bot[i], bot[i + 1]))
    for i in range(1, N_PANEL - 1):
        members["TopChord"].append(geo.add_line(top[i], top[i + 1]))
    for i in range(1, N_PANEL):
        members["Verticals"].append(geo.add_line(bot[i], top[i]))
    # end posts
    members["EndPosts"].append(geo.add_line(bot[0], top[1]))
    members["EndPosts"].append(geo.add_line(bot[N_PANEL], top[N_PANEL - 1]))
    # Pratt diagonals slope down toward midspan (tension under gravity)
    half = N_PANEL // 2
    for i in range(1, half):
        members["Diagonals"].append(geo.add_line(top[i], bot[i + 1]))
    for i in range(half + 1, N_PANEL):
        members["Diagonals"].append(geo.add_line(top[i], bot[i - 1]))

    for name, tags in members.items():
        g.physical.add_curve(tags, name=name)
        for tag in tags:
            g.mesh.structured.set_transfinite_curve(tag, 2)   # one truss element per member

    # supports and loaded joints, selected by position
    eps = 1e-6
    (g.model.select(None, dim=0).in_box((-eps, -eps, -eps), (eps, eps, eps)).to_physical("Pin"))
    (g.model.select(None, dim=0).in_box((SPAN - eps, -eps, -eps), (SPAN + eps, eps, eps)).to_physical("Roller"))
    (g.model.select(None, dim=0).in_box((L_PANEL - eps, -eps, -eps), (SPAN - L_PANEL + eps, eps, eps))
        .to_physical("LoadedJoints"))
    (g.model.select(None, dim=0).in_box((SPAN / 2 - eps, -eps, -eps), (SPAN / 2 + eps, eps, eps))
        .to_physical("Midspan"))

    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)
    print(fem.info.summary())

# ---- OpenSees model
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
steel = ops.uniaxialMaterial.ElasticMaterial(E=E_STEEL)
ops.element.Truss(pg="BottomChord", A=A_CHORD, material=steel)
ops.element.Truss(pg="TopChord", A=A_CHORD, material=steel)
ops.element.Truss(pg="EndPosts", A=A_CHORD, material=steel)
ops.element.Truss(pg="Verticals", A=A_VERT, material=steel)
ops.element.Truss(pg="Diagonals", A=A_DIAG, material=steel)
ops.fix(pg="Pin", dofs=(1, 1))
ops.fix(pg="Roller", dofs=(0, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="LoadedJoints", forces=(0.0, -P_JOINT))

ops.test.NormDispIncr(tol=1e-8, max_iter=10)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.constraints.Plain()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.analysis.Static()

ops.py(os.path.join(OUT, "pratt_deck.py"))
ops.run()
ops.analyze(steps=1)

# ---- results straight from the live domain
import openseespy.opensees as osp

# OpenSees element tags are not the mesh ids, so match members by end nodes
domain_ele = {frozenset(osp.eleNodes(t)): t for t in osp.getEleTags()}
rows = []
for pg in members:
    sel = fem.elements.select(pg=pg)
    for e, (n1, n2) in zip(sel.ids, sel.connectivity):
        tag = domain_ele[frozenset((int(n1), int(n2)))]
        force = osp.eleResponse(tag, "axialForce")[0]
        rows.append((pg, int(e), int(n1), int(n2), force))

with open(os.path.join(OUT, "member_forces.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "element", "node_i", "node_j", "axial_N"])
    w.writerows(rows)

def extreme(groups):
    sel = [r for r in rows if r[0] in groups]
    return max(sel, key=lambda r: abs(r[4]))

max_bot = extreme(["BottomChord"])
max_top = extreme(["TopChord"])
max_diag = extreme(["Diagonals"])
mid_node = int(fem.nodes.select(pg="Midspan").ids[0])
mid_defl = osp.nodeDisp(mid_node, 2)

# bottom-chord member touching midspan joint
mid_bot = [r for r in rows if r[0] == "BottomChord" and mid_node in (r[2], r[3])]

# ---- hand check: method of sections, cut through panel 3-4 (x = 8..12 m)
# Moment about the top joint at x = 8 m, where top chord and diagonal meet.
R = P_JOINT * (N_PANEL - 1) / 2
x_cut = (half - 1) * L_PANEL
M = R * x_cut - sum(P_JOINT * (x_cut - k * L_PANEL) for k in range(1, half - 1))
F_hand = M / H

print(f"max bottom chord : {max_bot[4]/1e3:10.2f} kN (element {max_bot[1]})")
print(f"max top chord    : {max_top[4]/1e3:10.2f} kN (element {max_top[1]})")
print(f"max diagonal     : {max_diag[4]/1e3:10.2f} kN (element {max_diag[1]})")
print(f"midspan deflection: {mid_defl*1e3:.3f} mm")
for r in mid_bot:
    print(f"midspan bottom chord el {r[1]}: FE {r[4]/1e3:.2f} kN | hand {F_hand/1e3:.2f} kN | "
          f"diff {100*(r[4]-F_hand)/F_hand:.4f} %")
