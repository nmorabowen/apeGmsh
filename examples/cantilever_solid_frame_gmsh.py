"""
Cantilever Beam: Solid-Frame Coupling with Gmsh Visualization
=============================================================

Same model as cantilever_solid_frame_coupling.py, but:
  - The solid mesh is built using pyGmsh (structured quad → shell mesh)
  - After OpenSees analysis, results are mapped back to Gmsh
  - Deformed shape + displacement magnitude shown as Gmsh views
  - Exported as .msh for interactive visualization in Gmsh GUI

Workflow:
  1. Build rectangular solid mesh in pyGmsh
  2. Extract numbered mesh → create OpenSees model
  3. Add frame element + dup node coupling manually
  4. Run analysis
  5. Map displacements back → g.view.add_node_vector / add_node_scalar
  6. Export .msh with embedded views
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import openseespy.opensees as ops
from apeGmsh import apeGmsh

# ================================================================
#  1. PARAMETERS
# ================================================================
L      = 10.0        # Total length [m]
h      = 1.0         # Depth [m]
b      = 0.5         # Width / thickness [m]
E      = 30.0e6      # Young's modulus [kPa]
nu     = 0.2         # Poisson's ratio
G      = E / (2 * (1 + nu))
P      = -100.0      # Tip load [kN]

L_solid = L / 2.0
L_frame = L / 2.0

nx_s = 20            # elements along x (solid)
ny_s = 8             # elements through depth

A  = b * h
Iz = b * h**3 / 12.0
Iy = h * b**3 / 12.0
J  = 0.5 * (Iy + Iz)

# ================================================================
#  2. BUILD MESH WITH pyGmsh
# ================================================================
g = apeGmsh(model_name="cantilever", verbose=True)
g.begin()

# Geometry: rectangle [0, L_solid] x [-h/2, h/2]  +  frame line
dx = L_solid / nx_s
dy = h / ny_s

p1 = g.model.geometry.add_point(0.0,     -h/2, 0.0)
p2 = g.model.geometry.add_point(L_solid, -h/2, 0.0)
p3 = g.model.geometry.add_point(L_solid,  h/2, 0.0)
p4 = g.model.geometry.add_point(0.0,      h/2, 0.0)

# Frame endpoints (centroid at interface → tip)
p_master = g.model.geometry.add_point(L_solid, 0.0, 0.0)
p_tip    = g.model.geometry.add_point(L,       0.0, 0.0)

l1 = g.model.geometry.add_line(p1, p2, label="bottom")
l2 = g.model.geometry.add_line(p2, p3, label="right")
l3 = g.model.geometry.add_line(p3, p4, label="top")
l4 = g.model.geometry.add_line(p4, p1, label="left")
l_frame = g.model.geometry.add_line(p_master, p_tip, label="frame")

loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
surf = g.model.geometry.add_plane_surface(loop, label="solid_region")

# Physical groups
g.physical.add(2, [surf],    name="SolidRegion")
g.physical.add(1, [l4],      name="FixedSupport")
g.physical.add(1, [l2],      name="Interface")
g.physical.add(1, [l_frame], name="FrameElement")

# Transfinite meshing → structured quads + frame line
import gmsh
gmsh.model.mesh.setTransfiniteCurve(l1, nx_s + 1)
gmsh.model.mesh.setTransfiniteCurve(l3, nx_s + 1)
gmsh.model.mesh.setTransfiniteCurve(l2, ny_s + 1)
gmsh.model.mesh.setTransfiniteCurve(l4, ny_s + 1)
gmsh.model.mesh.setTransfiniteSurface(surf)
gmsh.model.mesh.setRecombine(2, surf)  # quad mesh
gmsh.model.mesh.setTransfiniteCurve(l_frame, 2)  # 2 nodes → 1 line element

g.mesh.set_order(1)
g.mesh.generate(2)

print(f"\nMesh generated: structured {nx_s}x{ny_s} quads + 1 frame line element")

# ================================================================
#  3. EXTRACT MESH DATA
# ================================================================
mesh = g.mesh.get_numbered_mesh(dim=2, method="simple")

print(f"Nodes:    {mesh.n_nodes}")
print(f"Elements: {mesh.n_elems}")

# Get interface and fixed-support nodes from physical groups
# Physical group tags: FixedSupport=2, Interface=3 (from add order above)
fixed_nodes_gmsh     = g.physical.get_nodes(1, 2)['tags']   # FixedSupport
interface_nodes_gmsh = g.physical.get_nodes(1, 3)['tags']   # Interface

# Map to solver IDs
fixed_node_ids = sorted([mesh.gmsh_to_solver_node[int(gt)]
                         for gt in fixed_nodes_gmsh
                         if int(gt) in mesh.gmsh_to_solver_node])

interface_node_ids = sorted([mesh.gmsh_to_solver_node[int(gt)]
                             for gt in interface_nodes_gmsh
                             if int(gt) in mesh.gmsh_to_solver_node])

print(f"Fixed nodes:     {len(fixed_node_ids)}")
print(f"Interface nodes: {len(interface_node_ids)}")

# Get interface node y-coordinates (for sorting and identification)
interface_data = []
for sid in interface_node_ids:
    idx = np.where(mesh.node_ids == sid)[0][0]
    y = mesh.node_coords[idx, 1]
    interface_data.append((sid, y))
interface_data.sort(key=lambda x: x[1])  # sort by y

# ================================================================
#  4. BUILD OPENSEES MODEL
# ================================================================
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# --- Solid nodes (from Gmsh mesh) ---
for i in range(mesh.n_nodes):
    nid = int(mesh.node_ids[i])
    x, y, z = mesh.node_coords[i]
    ops.node(nid, float(x), float(y), 0.0)

# Constrain to 2D: fix uz(3), rx(4), ry(5)
interface_id_set = set(interface_node_ids)
for i in range(mesh.n_nodes):
    nid = int(mesh.node_ids[i])
    if nid in fixed_node_ids:
        ops.fix(nid, 1, 1, 1, 1, 1, 1)  # full fixity
    elif nid in interface_id_set:
        # Interface: fix out-of-plane, leave rz free for equalDOF coupling
        ops.fix(nid, 0, 0, 1, 1, 1, 0)
    else:
        ops.fix(nid, 0, 0, 1, 1, 1, 0)

# --- Shell section & elements ---
ops.section('ElasticMembranePlateSection', 1, E, nu, b, 0.0)

for i in range(mesh.n_elems):
    eid = int(mesh.element_ids[i])
    conn = [int(n) for n in mesh.connectivity[i]]
    ops.element('ShellMITC4', eid, *conn, 1)

max_solver_nid = int(mesh.node_ids.max())
max_solver_eid = int(mesh.element_ids.max())

# --- Frame nodes ---
master_tag = max_solver_nid + 1
tip_tag    = max_solver_nid + 2
ops.node(master_tag, L_solid, 0.0, 0.0)
ops.node(tip_tag,    L,       0.0, 0.0)
ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
ops.fix(tip_tag,    0, 0, 1, 1, 1, 0)

# --- Duplicated interface nodes ---
dup_tags = []
for k, (sid, y) in enumerate(interface_data):
    dn = tip_tag + 1 + k
    ops.node(dn, L_solid, float(y), 0.0)
    dup_tags.append(dn)

# --- Frame element ---
ops.geomTransf('Linear', 1, 0.0, 0.0, 1.0)
frame_eid = max_solver_eid + 1
ops.element('elasticBeamColumn', frame_eid, master_tag, tip_tag,
            A, E, G, J, Iy, Iz, 1)

# --- Coupling: rigidLink + equalDOF ---
for dn in dup_tags:
    ops.rigidLink('beam', master_tag, dn)

for dn, (sid, y) in zip(dup_tags, interface_data):
    ops.equalDOF(dn, sid, 1, 2)

print(f"\nOpenSees model:")
print(f"  Shell elements: {mesh.n_elems}")
print(f"  Frame element:  {frame_eid}")
print(f"  Master: {master_tag}, Tip: {tip_tag}")
print(f"  Dup nodes: {dup_tags[0]}..{dup_tags[-1]}")

# ================================================================
#  5. ANALYSIS
# ================================================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(tip_tag, 0.0, P, 0.0, 0.0, 0.0, 0.0)

ops.constraints('Penalty', 1e14, 1e14)
ops.numberer('RCM')
ops.system('UmfPack')
ops.test('NormDispIncr', 1e-6, 50)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')

print("\nRunning analysis...")
ok = ops.analyze(10)
if ok != 0:
    print("*** FAILED ***")
else:
    print("Analysis OK!\n")

# ================================================================
#  6. EXTRACT RESULTS
# ================================================================
tip_d    = ops.nodeDisp(tip_tag)
master_d = ops.nodeDisp(master_tag)
delta_EB = P * L**3 / (3 * E * Iz)

print(f"Tip deflection:  {tip_d[1]:+.6e} m")
print(f"Euler-Bernoulli: {delta_EB:+.6e} m")
print(f"Ratio:           {tip_d[1]/delta_EB:.4f}")

# Build displacement array for ALL Gmsh nodes (solid + frame)
# Solid nodes: mapped via NumberedMesh
gmsh_tags_solid = np.array(
    [mesh.solver_to_gmsh_node[int(sid)] for sid in mesh.node_ids],
    dtype=int,
)
disp_solid = np.zeros((mesh.n_nodes, 3))
for i in range(mesh.n_nodes):
    nid = int(mesh.node_ids[i])
    d = ops.nodeDisp(nid)
    disp_solid[i, 0] = d[0]
    disp_solid[i, 1] = d[1]

# Frame nodes: get their Gmsh tags from the FrameElement physical group
frame_pg = g.physical.get_nodes(1, 4)  # tag=4, "FrameElement"
frame_gmsh_tags = frame_pg['tags']
frame_coords    = frame_pg['coords'].reshape(-1, 3)
print(f"Frame Gmsh nodes: {frame_gmsh_tags}")

# Map frame Gmsh nodes to OpenSees displacements (master, tip)
disp_frame = np.zeros((len(frame_gmsh_tags), 3))
for i, gt in enumerate(frame_gmsh_tags):
    cx, cy = frame_coords[i, 0], frame_coords[i, 1]
    # Identify: master is at (L_solid, 0), tip is at (L, 0)
    if abs(cx - L_solid) < 1e-6 and abs(cy) < 1e-6:
        d = ops.nodeDisp(master_tag)
    elif abs(cx - L) < 1e-6 and abs(cy) < 1e-6:
        d = ops.nodeDisp(tip_tag)
    else:
        d = [0.0] * 6
    disp_frame[i, 0] = d[0]
    disp_frame[i, 1] = d[1]

# Concatenate solid + frame data
gmsh_tags_ordered = np.concatenate([gmsh_tags_solid, frame_gmsh_tags.astype(int)])
disp_array = np.vstack([disp_solid, disp_frame])
disp_mag = np.linalg.norm(disp_array, axis=1)

print(f"Total view nodes: {len(gmsh_tags_ordered)} (solid={mesh.n_nodes}, frame={len(frame_gmsh_tags)})")

# ================================================================
#  7. GMSH POST-PROCESSING VIEWS
# ================================================================
# Displacement vector (deformed shape, vector_type=5)
v1 = g.view.add_node_vector(
    "Displacement [m]",
    node_tags=list(gmsh_tags_ordered.astype(int)),
    vectors=disp_array,
    vector_type=5,   # 5 = displacement (deformed shape overlay)
)

# Displacement magnitude scalar (contour plot)
v2 = g.view.add_node_scalar(
    "|Displacement| [m]",
    node_tags=list(gmsh_tags_ordered.astype(int)),
    values=disp_mag.tolist(),
)

# uy component (vertical deflection)
v3 = g.view.add_node_scalar(
    "uy [m]",
    node_tags=list(gmsh_tags_ordered.astype(int)),
    values=disp_array[:, 1].tolist(),
)

# ux component (axial)
v4 = g.view.add_node_scalar(
    "ux [m]",
    node_tags=list(gmsh_tags_ordered.astype(int)),
    values=disp_array[:, 0].tolist(),
)

print(f"\nGmsh views created: {g.view.count()}")
for tag, name in g.view.list_views().items():
    print(f"  View {tag}: {name}")

# ================================================================
#  8. CONFIGURE VIEW OPTIONS FOR NICE OUTPUT
# ================================================================
# IMPORTANT: only show ONE view at a time to avoid overlapping chaos.
# Default: show the |Displacement| contour on the deformed shape.

# ── View 0 (Displacement vector): deformed shape overlay ────────────
gmsh.view.option.setNumber(v1, "Visible", 0)          # hide (used only as deformation source)

# ── View 1 (|Displacement|): main contour on deformed shape ─────────
gmsh.view.option.setNumber(v2, "Visible", 1)
gmsh.view.option.setNumber(v2, "VectorType", 5)       # not used for scalar, but harmless
gmsh.view.option.setNumber(v2, "DisplacementFactor", 200.0)  # deform the contour plot
gmsh.view.option.setNumber(v2, "ShowElement", 1)       # show element edges
gmsh.view.option.setNumber(v2, "ColormapNumber", 2)    # "Jet" colormap
gmsh.view.option.setNumber(v2, "IntervalsType", 2)     # continuous colormap (not stepped)
gmsh.view.option.setNumber(v2, "NbIso", 20)            # number of iso levels

# ── View 2 (uy): hidden by default, toggle in GUI ───────────────────
gmsh.view.option.setNumber(v3, "Visible", 0)
gmsh.view.option.setNumber(v3, "DisplacementFactor", 200.0)
gmsh.view.option.setNumber(v3, "ShowElement", 1)
gmsh.view.option.setNumber(v3, "ColormapNumber", 2)
gmsh.view.option.setNumber(v3, "IntervalsType", 2)
gmsh.view.option.setNumber(v3, "NbIso", 20)

# ── View 3 (ux): hidden by default, toggle in GUI ───────────────────
gmsh.view.option.setNumber(v4, "Visible", 0)
gmsh.view.option.setNumber(v4, "DisplacementFactor", 200.0)
gmsh.view.option.setNumber(v4, "ShowElement", 1)
gmsh.view.option.setNumber(v4, "ColormapNumber", 2)
gmsh.view.option.setNumber(v4, "IntervalsType", 2)
gmsh.view.option.setNumber(v4, "NbIso", 20)

# ── Hide the underlying mesh so it doesn't clash with the view ──────
gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
gmsh.option.setNumber("Mesh.SurfaceFaces", 0)
gmsh.option.setNumber("Mesh.VolumeEdges", 0)

# ── Set 2D XY viewing angle (top-down for a 2D model) ──────────────
gmsh.option.setNumber("General.Trackball", 0)
gmsh.option.setNumber("General.RotationX", 0)
gmsh.option.setNumber("General.RotationY", 0)
gmsh.option.setNumber("General.RotationZ", 0)

# ================================================================
#  9. EXPORT .msh WITH VIEWS
# ================================================================
out_msh = "cantilever_solid_frame_deformed.msh"
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.write(out_msh)
print(f"\nExported: {out_msh}")
print("Open in Gmsh GUI to explore deformed shape interactively:")
print(f"  gmsh {out_msh}")

# Also export a screenshot-ready .pos file for the displacement view
out_pos = "cantilever_displacement.pos"
gmsh.view.write(v1, out_pos)
print(f"Exported displacement view: {out_pos}")

# ================================================================
#  10. LAUNCH GUI (interactive — comment out for batch runs)
# ================================================================
ops.wipe()
print("\nLaunching Gmsh GUI...")
g.launch_gui()

# ================================================================
#  11. CLEANUP
# ================================================================
g.end()
print("\nDone!")
