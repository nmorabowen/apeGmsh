import os
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

out_dir = r"C:\Users\nmora\Github\apeGmsh\.claude\worktrees\apegmsh-readability-standards-d3c468\readability_samples\workshop\loop3\out_a"
os.makedirs(out_dir, exist_ok=True)

soil_width, soil_depth = 20.0, 10.0
footing_width = 2.0
footing_x0 = (soil_width - footing_width) / 2
E, nu = 30e6, 0.3
density = 18e3 / 9.81

with apeGmsh(model_name="strip_footing", save_to=os.path.join(out_dir, "strip_footing.h5")) as g:
    # Geometry
    pts = [
        g.model.geometry.add_point(0, 0, 0),
        g.model.geometry.add_point(soil_width, 0, 0),
        g.model.geometry.add_point(soil_width, soil_depth, 0),
        g.model.geometry.add_point(0, soil_depth, 0)
    ]

    lines = [
        g.model.geometry.add_line(pts[0], pts[1], label="bottom"),
        g.model.geometry.add_line(pts[1], pts[2], label="right"),
        g.model.geometry.add_line(pts[2], pts[3], label="top"),
        g.model.geometry.add_line(pts[3], pts[0], label="left")
    ]

    loop = g.model.geometry.add_curve_loop(lines)
    surf = g.model.geometry.add_plane_surface([loop], label="soil")

    # Physical groups
    g.physical.add_surface("soil", name="Soil")
    g.physical.add_curve(lines[0], name="Bottom")
    g.physical.add_curve(lines[3], name="Left")
    g.physical.add_curve(lines[1], name="Right")

    # Mesh
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(f"\n{fem.info}")

ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

material = ops.nDMaterial.ElasticIsotropic(E=E, nu=nu, rho=density)
ops.element.Tri31(pg="Soil", material=material, thickness=1.0)

# Simple boundary: fix bottom, leave sides free
ops.fix(pg="Bottom", dofs=(1, 1))

# Find monitor node (topmost center)
footing_center = footing_x0 + footing_width / 2
max_y = -1e10
mon_x, mon_y = 0, 0
for coord in fem.nodes.coords:
    if abs(coord[0] - footing_center) < 0.5 and coord[1] > max_y:
        max_y = coord[1]
        mon_x, mon_y = coord[0], coord[1]

# Find node ID
node_id = 1
for i, coord in enumerate(fem.nodes.coords):
    if abs(coord[0] - mon_x) < 1e-6 and abs(coord[1] - mon_y) < 1e-6:
        node_id = i + 1
        break

print(f"Monitor node: {node_id} at ({mon_x:.2f}, {mon_y:.2f})")

# Analysis setup
ops.system.BandGeneral()
ops.numberer.RCM()
ops.constraints.Plain()
ops.test.NormDispIncr(tol=1e-6, max_iter=25)
ops.algorithm.Newton()
ops.integrator.LoadControl(dlam=0.1)
ops.analysis.Static()

# Load and analyze
footing_force = 300.0 * 1e3 * footing_width
print("\n=== Footing Load ===")

with ops.pattern.Plain(series=ops.timeSeries.Linear(factor=1.0)) as p:
    p.load(node=node_id, forces=(0, -footing_force))

import opensees

settlements, pressures = [], []
for step in range(1, 11):
    ret = ops.analyze(steps=1)
    if ret != 0:
        print(f"Diverged at step {step}")
        break
    u = opensees.nodeDisp(node_id, 2)
    p = (step / 10) * 300.0
    settlements.append(u)
    pressures.append(p)
    print(f"  Step {step:2d}: P={p:6.1f} kPa, u={u:.8e} m")

# Save
import csv
with open(os.path.join(out_dir, "settlement_vs_pressure.csv"), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Pressure (kPa)", "Settlement (m)"])
    for p, s in zip(pressures, settlements):
        w.writerow([f"{p:.1f}", f"{s:.8e}"])

elastic = (300.0 * 1e3 * footing_width / E) * 0.5
print(f"\n=== Results ===")
print(f"Hand elastic: {elastic:.8e} m")
print(f"Model settlement: {settlements[-1]:.8e} m")
print(f"Ratio: {settlements[-1] / elastic:.2f}" if elastic > 0 else "")
print(f"Output: {out_dir}")
