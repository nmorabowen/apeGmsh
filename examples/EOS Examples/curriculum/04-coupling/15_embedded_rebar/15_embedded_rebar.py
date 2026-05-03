# %% [markdown]
# # 15 — Reinforced Concrete via ASDEmbeddedNodeElement
#
# **Curriculum slot:** Tier 4, slot 15.
# **Prerequisite:** 11 — Boolean Assembly, 01 — Hello Plate.
#
# ## Purpose
#
# There are two patterns for modelling a rebar inside a concrete
# member:
#
# | Pattern | Mechanism | When to use |
# |---|---|---|
# | **Conformal mesh** | ``g.model.boolean.fragment`` carves the rebar into the concrete so they share nodes. | Rebar geometry drives the mesh; OK when rebar layout is simple. |
# | **ASDEmbeddedNodeElement** (this slot) | ``g.constraints.embedded(host, embedded)`` — rebar nodes float anywhere inside host elements and are kinematically coupled via shape functions. | Rebar geometry is independent of the concrete mesh; keep a coarse solid mesh under any rebar pattern. |
#
# This slot uses the second path. The rebar line is **not** fragmented
# into the concrete surface. Instead, apeGmsh's ``resolve_embedded``
# routine locates each rebar mesh node inside the containing tri3
# concrete element and emits an ``ASDEmbeddedNodeElement`` that ties
# the rebar node to that triangle's corners through linear
# shape-function weights.
#
# ## Problem — 2D plane-stress concrete strip with a single rebar
#
# A rectangular concrete strip of length $L$ and depth $H$ (plane
# stress, unit thickness) with a single longitudinal rebar running
# along the centreline at $y = H/2$. Left edge fixed in $x$; right
# edge pulled in tension with total force $F$.
#
# For an axial load applied to a parallel composition (concrete
# cross section area $A_c$ with Young's modulus $E_c$; steel area
# $A_s$ with $E_s$) the composite Young's modulus gives the right-
# edge displacement
#
# $$
# u_{x,\text{right}}
#   \;=\; \dfrac{F\,L}{E_c\,A_c + E_s\,A_s}.
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
L    = 2.0        # concrete strip length [m]
H    = 0.30       # concrete strip depth  [m]
t    = 1.0        # plane-stress thickness [m]

# Concrete
E_c  = 30.0e9     # Young's modulus of concrete [Pa]
nu_c = 0.2
A_c  = H * t      # gross concrete section (A_s << A_c, no subtraction)

# Rebar
A_s = np.pi * (0.012 / 2.0) ** 2   # one #12 mm rebar [m^2]
E_s = 200.0e9                       # Young's modulus of steel [Pa]

# Loading
F = 50_000.0      # total right-edge pull  [N]

# Mesh
LC = 0.08         # concrete characteristic size [m]


# %% [markdown]
# ## 2. Geometry — concrete strip + free-floating rebar line
#
# Crucially we do **not** fragment the rebar into the concrete. The
# two entities keep their own meshes; the embedded constraint will
# stitch them together at solve time.

# %%
g_ctx = apeGmsh(model_name="15_embedded_rebar", verbose=False)
g = g_ctx.__enter__()

# Concrete rectangle
p_BL = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_BR = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
p_TR = g.model.geometry.add_point(L,   H,   0.0, lc=LC)
p_TL = g.model.geometry.add_point(0.0, H,   0.0, lc=LC)
lB = g.model.geometry.add_line(p_BL, p_BR)
lR = g.model.geometry.add_line(p_BR, p_TR)
lT = g.model.geometry.add_line(p_TR, p_TL)
lL = g.model.geometry.add_line(p_TL, p_BL)
loop = g.model.geometry.add_curve_loop([lB, lR, lT, lL])
concrete_surf = g.model.geometry.add_plane_surface([loop])

# Rebar line at y = H/2 — independent of the concrete surface
p_rb0 = g.model.geometry.add_point(0.0, H / 2, 0.0, lc=LC)
p_rb1 = g.model.geometry.add_point(L,   H / 2, 0.0, lc=LC)
rebar_line = g.model.geometry.add_line(p_rb0, p_rb1)

g.model.sync()


# %% [markdown]
# ## 3. Physical groups

# %%
g.physical.add(2, [concrete_surf],  name="concrete")
g.physical.add(1, [rebar_line],     name="rebar")
g.physical.add(1, [lL],             name="left")
g.physical.add(1, [lR],             name="right")


# %% [markdown]
# ## 4. Embedded constraint
#
# Declares the kinematic coupling *before* meshing. Each mesh node
# the mesher places on the rebar line will be located inside the
# containing concrete tri3 at resolve time and kinematically coupled
# to that triangle's corners.

# %%
g.constraints.embedded(host_label="concrete", embedded_label="rebar")


# %% [markdown]
# ## 5. Mesh

# %%
g.mesh.generation.generate(2)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

embedded_recs = list(fem.elements.constraints.interpolations())
print(f"embedded records: {len(embedded_recs)} rebar node(s) tied to tri3 hosts")

rebar_line_elems: list[tuple[int, tuple[int, int]]] = []
for group in fem.elements.get(target="rebar"):
    for eid, nodes in zip(group.ids, group.connectivity):
        rebar_line_elems.append((int(eid), (int(nodes[0]), int(nodes[1]))))
print(f"rebar truss elements: {len(rebar_line_elems)}")


# %% [markdown]
# ## 6. OpenSees ingest + analysis

# %%
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]))

# Concrete: nDMaterial ElasticIsotropic + tri31 plane-stress
ops.nDMaterial("ElasticIsotropic", 1, E_c, nu_c)
for group in fem.elements.get(target="concrete"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("tri31", int(eid),
                    int(nodes[0]), int(nodes[1]), int(nodes[2]),
                    t, "PlaneStress", 1)

# Rebar: uniaxialMaterial Elastic + truss.
MAX_CON_EID = 0
for group in fem.elements.get(target="concrete"):
    if len(group.ids) > 0:
        MAX_CON_EID = max(MAX_CON_EID, int(max(group.ids)))

ops.uniaxialMaterial("Elastic", 100, E_s)
next_eid = MAX_CON_EID + 1
for _, (ni, nj) in rebar_line_elems:
    ops.element("truss", next_eid, ni, nj, A_s, 100)
    next_eid += 1

# Embedded couplings: ASDEmbeddedNodeElement per rebar node.
next_eid = max(next_eid, 1_000_000)
for rec in embedded_recs:
    ops.element(
        "ASDEmbeddedNodeElement",
        next_eid,
        int(rec.slave_node),
        *(int(m) for m in rec.master_nodes),
    )
    next_eid += 1

# BCs: fix ux on the left edge, anchor uy at one corner
for n in fem.nodes.get(target="left").ids:
    ops.fix(int(n), 1, 0)
ops.fix(int(p_BL), 0, 1)

# Load: uniform traction on the right concrete edge, distributed
# by tributary length (same pattern as slot 01). Only the concrete
# boundary nodes carry the pull; embedded rebar nodes do not take
# direct nodal loads.
right_ids = set(int(r) for r in fem.nodes.get(target="right").ids)
concrete_corners = set()
for group in fem.elements.get(target="concrete"):
    concrete_corners.update(int(n) for row in group.connectivity for n in row)
right_concrete_ids = sorted(right_ids & concrete_corners)

right_coords = np.array([
    (int(n), float(fem.nodes.coords[i, 1]))
    for i, n in enumerate(fem.nodes.ids)
    if int(n) in set(right_concrete_ids)
])
order = np.argsort(right_coords[:, 1])
rs = right_coords[order]
ys = rs[:, 1]
trib = np.zeros(len(ys))
trib[0]  = (ys[1]  - ys[0])  / 2.0
trib[-1] = (ys[-1] - ys[-2]) / 2.0
for i in range(1, len(ys) - 1):
    trib[i] = (ys[i + 1] - ys[i - 1]) / 2.0
trib *= H / trib.sum()

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for (nid_y, tr) in zip(rs, trib):
    nid = int(nid_y[0])
    Fx = F * float(tr) / H
    ops.load(nid, Fx, 0.0)

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification
#
# Mean right-edge $u_x$ on the concrete boundary vs the composite-
# section answer $F L / (E_c A_c + E_s A_s)$.

# %%
mean_ux = float(np.mean([ops.nodeDisp(int(r[0]), 1) for r in rs]))
analytical = F * L / (E_c * A_c + E_s * A_s)
err = abs(mean_ux - analytical) / abs(analytical) * 100.0

unreinforced = F * L / (E_c * A_c)

print(f"FEM mean ux       :  {mean_ux:.6e}  m")
print(f"Composite (c + s) :  {analytical:.6e}  m   (FL / (Ec*Ac + Es*As))")
print(f"Unreinforced      :  {unreinforced:.6e}  m   (concrete only)")
print(f"Stiffening ratio  :  {unreinforced / analytical:.4f}  (how much stiffer)")
print(f"Error             :  {err:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Mesh-independent rebar.** The rebar line is meshed on its own
#   resolution, then stitched into whatever triangles the mesher
#   placed in the concrete. No fragment, no shared-edge constraint.
# * **Kinematic coupling via shape functions.** Each rebar node
#   rides on $u_{\text{rebar}} = \sum N_i(\xi,\eta)\,u_{\text{host},i}$
#   through an ``ASDEmbeddedNodeElement``, which is exactly the
#   embedded-element formulation used by Abaqus and STKO.
# * **Path to arbitrary layouts.** The same API scales to multiple
#   rebars, stirrups, and 3D tet hosts (where the host is a tet4
#   and the coupling uses four shape functions instead of three).

# %%
g_ctx.__exit__(None, None, None)
