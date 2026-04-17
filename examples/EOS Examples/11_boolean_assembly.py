# %% [markdown]
# # 11 — Boolean Operations in an Assembly
#
# **Curriculum slot:** Tier 3, slot 11.
# **Prerequisite:** 10 — Parts Basics.
#
# ## Purpose
#
# When two geometric regions meet at a common edge, Gmsh's mesher
# will **not** automatically produce a conformal mesh across the
# interface — each region is meshed independently and the edge
# nodes won't match. A plane-stress analysis on the two-piece
# geometry would behave like two completely disconnected plates
# with a slit between them.
#
# The fix is a **boolean fragment operation**: carving the two
# shapes against each other so Gmsh sees a single CAD entity whose
# internal surfaces share edges. After fragment the mesh is
# conformal automatically.
#
# apeGmsh surfaces two ways to do this:
#
# | Call | Works on | Outcome |
# |---|---|---|
# | ``g.model.boolean.fragment(objects, tools, dim=...)`` | raw DimTags | carves and returns the new DimTag list; caller must re-tag |
# | ``g.parts.fragment_all(dim=...)`` | parts registered with ``g.parts.register`` | carves all registered parts against each other and **updates the parts registry** in place — a re-query like ``g.parts.get("plateA")`` returns the post-fragment DimTags |
#
# This notebook uses the part-aware version so the ``g.parts``
# registry stays accurate across the geometry edit.
#
# ## Problem — two plates meeting at $x = L_A$
#
# Two 2-D rectangles meeting at $x = L_A$, pulled in uniaxial
# tension along $+x$. Without fragment, each plate would mesh
# independently and the interface would be a slit. After fragment,
# the two plates become a single conformal assembly whose total
# displacement under uniform traction $\sigma$ obeys
#
# $$
# u_{x,\text{right}} \;=\; \dfrac{\sigma\,(L_A + L_B)}{E}.
# $$
#
# That's the verification.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import gmsh
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Geometry
L_A = 1.0                      # length of plate A
L_B = 1.5                      # length of plate B
H   = 0.5                      # common height
L_total = L_A + L_B

# Loading
sigma = 1.0e6                  # uniform traction on right edge [Pa]
E  = 2.1e11
nu = 0.3
thk = 1.0                      # plane-stress thickness [m]

LC = min(L_A, L_B, H) / 10.0


# %% [markdown]
# ## 2. Geometry — two adjacent rectangles, built independently
#
# Plate A occupies $[0, L_A] \times [0, H]$. Plate B occupies
# $[L_A, L_A + L_B] \times [0, H]$. They share the $x = L_A$ edge
# at the CAD level after synchronization, but Gmsh will still
# mesh each surface independently unless we tell it otherwise.

# %%
g_ctx = apeGmsh(model_name="11_boolean_assembly", verbose=False)
g = g_ctx.__enter__()

# Plate A
pA0 = g.model.geometry.add_point(0.0,   0.0, 0.0, lc=LC)
pA1 = g.model.geometry.add_point(L_A,   0.0, 0.0, lc=LC)
pA2 = g.model.geometry.add_point(L_A,   H,   0.0, lc=LC)
pA3 = g.model.geometry.add_point(0.0,   H,   0.0, lc=LC)
lA0 = g.model.geometry.add_line(pA0, pA1)
lA1 = g.model.geometry.add_line(pA1, pA2)
lA2 = g.model.geometry.add_line(pA2, pA3)
lA3 = g.model.geometry.add_line(pA3, pA0)
loopA = g.model.geometry.add_curve_loop([lA0, lA1, lA2, lA3])
sA    = g.model.geometry.add_plane_surface([loopA])

# Plate B
pB0 = g.model.geometry.add_point(L_A,     0.0, 0.0, lc=LC)
pB1 = g.model.geometry.add_point(L_total, 0.0, 0.0, lc=LC)
pB2 = g.model.geometry.add_point(L_total, H,   0.0, lc=LC)
pB3 = g.model.geometry.add_point(L_A,     H,   0.0, lc=LC)
lB0 = g.model.geometry.add_line(pB0, pB1)
lB1 = g.model.geometry.add_line(pB1, pB2)
lB2 = g.model.geometry.add_line(pB2, pB3)
lB3 = g.model.geometry.add_line(pB3, pB0)
loopB = g.model.geometry.add_curve_loop([lB0, lB1, lB2, lB3])
sB    = g.model.geometry.add_plane_surface([loopB])

g.model.sync()

# Register both surfaces as parts (before fragment — so the parts
# registry tracks the original identities).
g.parts.register("plateA", [(2, sA)])
g.parts.register("plateB", [(2, sB)])

print("pre-fragment DimTags:")
print(f"  plateA : {g.parts.get('plateA').entities}")
print(f"  plateB : {g.parts.get('plateB').entities}")


# %% [markdown]
# ## 3. Fragment all parts against each other
#
# ``g.parts.fragment_all(dim=2)`` runs a boolean fragment between
# every pair of parts at dim=2. Gmsh carves the shared edge so the
# two plates become CAD-conformal — subsequent meshing will place
# coincident nodes on the interface.
#
# The parts registry is updated in place. After the call, plateA's
# entities DimTag(s) may have changed (same footprint, possibly new
# tag numbers), but the registry still knows which entities belong
# to which part.

# %%
new_tags = g.parts.fragment_all(dim=2)
print(f"fragment_all returned {len(new_tags)} top-level DimTags")
print("post-fragment DimTags:")
print(f"  plateA : {g.parts.get('plateA').entities}")
print(f"  plateB : {g.parts.get('plateB').entities}")


# %% [markdown]
# ## 4. Physical groups for BC + load targeting

# %%
# Far-left edge (x = 0): symmetry / fix
# Far-right edge (x = L_total): applied traction
# After fragment, the ORIGINAL point tags pA0, pA3, pB1, pB2 still
# reference the same geometric points at the corners, so we can
# use them directly.
left_corner_bottom  = pA0
left_corner_top     = pA3
right_corner_bottom = pB1
right_corner_top    = pB2

g.physical.add(0, [left_corner_bottom], name="anchor")   # y-anchor

# Resolve the "left" and "right" curves from the fragmented geometry
# by querying gmsh for curves at x ~= 0 and x ~= L_total.
left_curves: list[int] = []
right_curves: list[int] = []
TOL = 1e-6
for (dim, tag) in gmsh.model.getEntities(dim=1):
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, xmax = bb[0], bb[3]
    xmid = 0.5 * (xmin + xmax)
    dx = abs(xmax - xmin)
    if dx < TOL and abs(xmid - 0.0) < TOL:
        left_curves.append(tag)
    elif dx < TOL and abs(xmid - L_total) < TOL:
        right_curves.append(tag)

print(f"left curves  : {left_curves}")
print(f"right curves : {right_curves}")

g.physical.add(1, left_curves,  name="left")
g.physical.add(1, right_curves, name="right")


# %% [markdown]
# ## 5. Mesh
#
# After fragment, Gmsh sees the two plates as CAD-conformal so the
# 2-D mesh respects the interface automatically — you should see a
# row of nodes along $x = L_A$ that belongs to both plates.

# %%
g.mesh.generation.generate(2)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# Sanity — the part node map after fragmentation
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
shared = node_map["plateA"] & node_map["plateB"]
print(f"plateA nodes : {len(node_map['plateA'])}")
print(f"plateB nodes : {len(node_map['plateB'])}")
print(f"shared nodes : {len(shared)}  "
      f"(= the number of mesh nodes along the x={L_A} interface)")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Plane-stress tri31 across both plates. Since the mesh is now
# conformal the interface nodes are shared and no tie/MPC is
# needed.

# %%
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]))

ops.nDMaterial("ElasticIsotropic", 1, E, nu)

# Iterate elements of the fragmented surfaces (all of them, not
# per-part — the plane-stress material is the same everywhere).
for group in fem.elements.get(element_type="tri3"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("tri31", int(eid),
                    int(nodes[0]), int(nodes[1]), int(nodes[2]),
                    thk, "PlaneStress", 1)

# Left edge: symmetry in x (ux = 0)
for n in fem.nodes.get(target="left").ids:
    ops.fix(int(n), 1, 0)
# Bottom-left corner: y-anchor
for n in fem.nodes.get(target="anchor").ids:
    ops.fix(int(n), 0, 1)

# Right edge: horizontal traction distributed by tributary length
right_ids = list(fem.nodes.get(target="right").ids)
# sort by y
right_coords = np.array([
    (int(n), float(fem.nodes.coords[i, 1]))
    for i, n in enumerate(fem.nodes.ids)
    if int(n) in set(int(x) for x in right_ids)
])
order = np.argsort(right_coords[:, 1])
rs = right_coords[order]
ys = rs[:, 1]
trib = np.zeros(len(ys))
trib[0]  = (ys[1]  - ys[0])  / 2.0
trib[-1] = (ys[-1] - ys[-2]) / 2.0
for i in range(1, len(ys) - 1):
    trib[i] = (ys[i+1] - ys[i-1]) / 2.0
trib *= H / trib.sum()  # normalise

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for (nid_y, tr) in zip(rs, trib):
    nid = int(nid_y[0])
    Fx = sigma * float(tr) * thk
    ops.load(nid, Fx, 0.0)

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification
#
# Mean right-edge $u_x$ compared to $u = \sigma\,(L_A + L_B)/E$.

# %%
mean_ux = float(np.mean([ops.nodeDisp(int(r[0]), 1) for r in rs]))
analytical = sigma * L_total / E
err = abs(mean_ux - analytical) / abs(analytical) * 100.0

print(f"FEM mean ux :  {mean_ux:.6e}  m")
print(f"Analytical  :  {analytical:.6e}  m   (sigma*(LA+LB)/E)")
print(f"Error       :  {err:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **``g.parts.fragment_all(dim=...)``** as the one-liner that turns
#   a multi-part assembly into a conformal CAD body while keeping
#   the parts registry accurate.
# * **Shared-node interface meshes.** Once fragment is done, the
#   two plates share mesh nodes at the interface automatically —
#   no ``g.constraints.tie`` is needed here. Slot 12 covers the
#   case where conformal meshing is not possible (two parts with
#   different element sizes) and a tie constraint is required.
# * **Pattern for locating edges after a boolean op.** Using
#   ``gmsh.model.getEntities(dim=1)`` + bounding-box filtering is
#   the canonical way to find curves by geometric location when
#   their tag numbers have been reshuffled by fragmentation.

# %%
g_ctx.__exit__(None, None, None)
