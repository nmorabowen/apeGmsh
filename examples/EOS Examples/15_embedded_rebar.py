# %% [markdown]
# # 15 — Reinforced Concrete via Conformal Meshing
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
# | **Conformal mesh** (this slot) | ``g.model.boolean.fragment`` carves the rebar into the concrete so they share nodes. Rebar elements live on those shared nodes as ``truss`` / ``beamColumn``; concrete lives as ``tri31`` / ``quad`` / tet. | Rebar geometry is known before meshing and you can tolerate mesh density driven by the rebar layout. |
# | **ASDEmbeddedNodeElement** (not yet in apeGmsh) | ``g.constraints.embedded(host, embedded)`` — rebar nodes float arbitrarily inside host elements and are kinematically coupled via shape functions. | Rebar geometry is independent of the concrete mesh; lets you keep a coarse solid mesh with any rebar pattern. |
#
# The second path depends on ``resolve_embedded``, which isn't yet
# implemented in apeGmsh's constraint resolver (flagged
# ``_opensees_constraints.py`` as "deferred"). This slot uses the
# first, fragment-based, path — which is the pattern the existing
# ``examples/01_embedded_rebars.ipynb`` uses too.
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
A_c_gross = H * t
A_c       = A_c_gross   # simplification: don't subtract the steel area (OK for A_s << A_c)

# Rebar
A_s = np.pi * (0.012 / 2.0)**2    # one #12 mm rebar [m^2]
E_s = 200.0e9                      # Young's modulus of steel [Pa]

# Loading
F = 50_000.0      # total right-edge pull  [N]

# Mesh
LC = 0.08         # concrete characteristic size [m]


# %% [markdown]
# ## 2. Geometry — concrete strip + embedded rebar line

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

# Rebar line at y = H/2
p_rb0 = g.model.geometry.add_point(0.0, H/2, 0.0, lc=LC)
p_rb1 = g.model.geometry.add_point(L,   H/2, 0.0, lc=LC)
rebar_line = g.model.geometry.add_line(p_rb0, p_rb1)

g.model.sync()


# %% [markdown]
# ## 3. Fragment: carve the rebar into the concrete
#
# After fragment the concrete surface is split into two triangles
# sharing the rebar line as a common edge — so mesh nodes along
# that line are **shared** between concrete elements and rebar
# elements. This is the key that makes "conformal rebar" work.

# %%
g.model.boolean.fragment(
    objects=[(2, concrete_surf)],
    tools=[(1, rebar_line)],
    dim=2,                      # 2D fragment (the default dim=3 would filter out surfaces)
)

# After fragment the surface + line tags may have changed. Look up
# the carved surfaces and the rebar-line entity afresh.
concrete_surfs_after = [t for d, t in gmsh.model.getEntities(dim=2)]
print(f"concrete surfaces after fragment: {concrete_surfs_after}")

# Find the rebar line by bounding-box lookup (it should still span
# x in [0, L] at y == H/2).
TOL = 1e-6
rebar_line_after = None
for (dim, tag) in gmsh.model.getEntities(dim=1):
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, xmax, ymax = bb[0], bb[1], bb[3], bb[4]
    if (abs(xmin) < TOL and abs(xmax - L) < TOL
            and abs(ymin - H/2) < TOL and abs(ymax - H/2) < TOL):
        rebar_line_after = tag
        break
assert rebar_line_after is not None, "could not locate the rebar line after fragment"
print(f"rebar line after fragment: {rebar_line_after}")


# %% [markdown]
# ## 4. Physical groups

# %%
# Left + right edges for BC + load
# Bounding-box lookup is robust against fragment renumbering.
left_curves: list[int] = []
right_curves: list[int] = []
for (dim, tag) in gmsh.model.getEntities(dim=1):
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, xmax = bb[0], bb[3]
    if abs(xmax - xmin) < TOL:    # vertical
        if abs(xmin) < TOL:
            left_curves.append(tag)
        elif abs(xmin - L) < TOL:
            right_curves.append(tag)

g.physical.add(2, concrete_surfs_after, name="concrete")
g.physical.add(1, [rebar_line_after],   name="rebar")
g.physical.add(1, left_curves,          name="left")
g.physical.add(1, right_curves,         name="right")


# %% [markdown]
# ## 5. Mesh

# %%
g.mesh.generation.generate(2)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# How many mesh nodes lie on the rebar line? They'll be referenced
# by BOTH the rebar truss elements and the surrounding concrete
# triangles (because of fragment).
rebar_node_ids = list(fem.nodes.get(target="rebar").ids)
print(f"rebar line nodes: {len(rebar_node_ids)}")

# And how many truss-candidate line elements are on the rebar line?
# Each will become an OpenSees truss between two consecutive rebar
# nodes.
rebar_line_elems: list[tuple[int, tuple[int, int]]] = []
for group in fem.elements.get(target="rebar"):
    for eid, nodes in zip(group.ids, group.connectivity):
        rebar_line_elems.append((int(eid), (int(nodes[0]), int(nodes[1]))))
print(f"rebar line elements: {len(rebar_line_elems)}")


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

# Rebar: uniaxialMaterial Elastic + truss on shared nodes.
# Emit with fresh element tags above any tri31 tag.
MAX_CON_EID = 0
for group in fem.elements.get(target="concrete"):
    if len(group.ids) > 0:
        MAX_CON_EID = max(MAX_CON_EID, int(max(group.ids)))

ops.uniaxialMaterial("Elastic", 100, E_s)
next_eid = MAX_CON_EID + 1
for _, (ni, nj) in rebar_line_elems:
    ops.element("truss", next_eid, ni, nj, A_s, 100)
    next_eid += 1

# BCs: fix ux on the left edge, anchor uy at one corner
for n in fem.nodes.get(target="left").ids:
    ops.fix(int(n), 1, 0)
# Pin the bottom-left corner in y
ops.fix(int(p_BL), 0, 1)

# Load: uniform traction on the right edge, distributed by
# tributary length (same pattern as slot 01)
right_node_ids = list(fem.nodes.get(target="right").ids)
right_coords = np.array([
    (int(n), float(fem.nodes.coords[i, 1]))
    for i, n in enumerate(fem.nodes.ids)
    if int(n) in set(int(r) for r in right_node_ids)
])
order = np.argsort(right_coords[:, 1])
rs = right_coords[order]
ys = rs[:, 1]
trib = np.zeros(len(ys))
trib[0]  = (ys[1]  - ys[0])  / 2.0
trib[-1] = (ys[-1] - ys[-2]) / 2.0
for i in range(1, len(ys) - 1):
    trib[i] = (ys[i+1] - ys[i-1]) / 2.0
trib *= H / trib.sum()

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for (nid_y, tr) in zip(rs, trib):
    nid = int(nid_y[0])
    Fx = F * float(tr) / H     # share of F by tributary length
    ops.load(nid, Fx, 0.0)

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification
#
# Mean right-edge $u_x$ vs the composite-section answer
# $F L / (E_c A_c + E_s A_s)$.

# %%
mean_ux = float(np.mean([ops.nodeDisp(int(r[0]), 1) for r in rs]))
analytical = F * L / (E_c * A_c + E_s * A_s)
err = abs(mean_ux - analytical) / abs(analytical) * 100.0

# Sanity: also compute "concrete only" to show the stiffening effect
unreinforced = F * L / (E_c * A_c)

print(f"FEM mean ux       :  {mean_ux:.6e}  m")
print(f"Composite (c + s) :  {analytical:.6e}  m   (FL / (Ec*Ac + Es*As))")
print(f"Unreinforced      :  {unreinforced:.6e}  m   (concrete only)")
print(f"Stiffening ratio  :  {unreinforced / analytical:.4f}  (how much stiffer)")
print(f"Error             :  {err:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Reinforced-concrete modelling via fragment.** For plane-stress
#   or plane-strain RC this is the standard apeGmsh workflow:
#   carve the rebar lines into the concrete surface with
#   ``g.model.boolean.fragment``, so rebar nodes are shared with
#   concrete corners. Rebar truss / beam elements then live on
#   those shared nodes.
# * **Parallel-stiffness verification.** For linear-elastic
#   materials the composite axial stiffness is $E_c A_c + E_s A_s$.
#   Any RC axial test should reproduce this before trusting any
#   nonlinear (cracked, yielding) follow-up.
# * **Pathway to the ``ASDEmbeddedNodeElement`` future.** When
#   ``resolve_embedded`` lands in apeGmsh's resolver, the same
#   geometry will work without ``fragment`` — rebars can float
#   anywhere inside the concrete and get interpolated onto
#   arbitrary tet / quad / tri hosts.

# %%
g_ctx.__exit__(None, None, None)
