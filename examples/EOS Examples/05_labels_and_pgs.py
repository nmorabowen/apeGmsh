# %% [markdown]
# # 05 — Labels and Physical Groups
#
# **Curriculum slot:** Tier 2, slot 05.
# **Prerequisite:** 02 — 2D Cantilever Beam.
#
# ## Purpose
#
# apeGmsh gives you **two** naming namespaces for geometry:
#
# | Namespace | Call | Lives as | Survives |
# |---|---|---|---|
# | Label (Tier 1) | ``g.labels.add(dim, tags, name)`` | a Gmsh physical group whose name is prefixed with ``_label:`` | most geometry edits — label-tagged entities are re-identified after fragment / cut / fuse |
# | Physical group (Tier 2) | ``g.physical.add(dim, tags, name=...)`` | a plain Gmsh physical group | its member entity tags. If the mesh is regenerated after geometry edits, membership may shift. |
#
# Both wind up as OpenSees-visible entities — the difference is how
# apeGmsh keeps the name attached across geometry operations.
#
# **When to use which?**
#
# * **Labels** when the identity has to survive boolean / fragment
#   / cut / fuse. Example: "top_flange" as the upper surface of a
#   beam that you'll later fragment against other parts.
# * **Physical groups** when the identity is stable and you want a
#   named export target (OpenSees recorders read PG names).
#
# ## This notebook does two things
#
# 1. Builds a 3-point cantilever, tags two different points in the
#    two namespaces, and shows that **the same set of four FEMData
#    accessor calls resolves both kinds of names consistently**:
#    ``target=``, ``pg=``, ``label=``, ``tag=``.
# 2. Demonstrates the **auto-resolution precedence** that
#    ``target="..."`` uses when a name could match more than one
#    source.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Tiny cantilever, same mechanical setup as slot 02.
L = 3.0
P = 10_000.0
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 6.0


# %% [markdown]
# ## 2. Geometry

# %%
g_ctx = apeGmsh(model_name="05_labels_and_pgs", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0,   0.0, 0.0, lc=LC)
p_mid  = g.model.geometry.add_point(L/2,   0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,     0.0, 0.0, lc=LC)

ln_L = g.model.geometry.add_line(p_base, p_mid)
ln_R = g.model.geometry.add_line(p_mid,  p_tip)

g.model.sync()


# %% [markdown]
# ## 3. Tag entities TWO different ways
#
# * ``p_tip``  -> **label** ``"tip"``     (Tier 1)
# * ``p_mid``  -> **physical group** ``"mid_pg"`` (Tier 2)
# * ``p_base`` -> both (label ``"base_lbl"`` and PG ``"base_pg"``)
# * ``[ln_L, ln_R]`` -> PG ``"beam"`` (needed to assign elements)
#
# The last tag on ``p_base`` is what lets us demonstrate precedence:
# when two different namespaces both claim a name, auto-resolution
# (``target="..."``) uses the **label** first.

# %%
# --- label (Tier 1) ---
g.labels.add(0, [p_tip],  "tip")
g.labels.add(0, [p_base], "base_lbl")

# --- physical group (Tier 2) ---
g.physical.add(0, [p_mid],  name="mid_pg")
g.physical.add(0, [p_base], name="base_pg")
g.physical.add(1, [ln_L, ln_R], name="beam")

print("labels registered :", g.labels.get_all())


# %% [markdown]
# ## 4. Mesh

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 5. The three accessor dialects
#
# For any named entity, ``fem.nodes.get`` and ``fem.elements.get``
# accept three equivalent forms, plus raw DimTag lists:
#
# | Call | What it does |
# |---|---|
# | ``fem.nodes.get(target="foo")`` | auto-resolve — **label → PG → part** (matches ``LoadsComposite``) |
# | ``fem.nodes.get(label="foo")``  | force the label namespace |
# | ``fem.nodes.get(pg="foo")``     | force the PG namespace |
# | ``fem.nodes.get(target=[(dim, tag)])`` | raw geometry DimTag(s) resolved through Gmsh |
#
# **Precedence.** When the same name ``"foo"`` is registered as both
# a label and a PG, ``target="foo"`` returns the **label** entity —
# same tie-breaker as ``g.loads.*`` and ``g.constraints.*``. Use the
# explicit ``label=`` / ``pg=`` form when you want to force one
# namespace or the other.

# %%
print("--- tip (label only) ---")
print(f"  target='tip'          -> {sorted(int(n) for n in fem.nodes.get(target='tip').ids)}")
print(f"  label='tip'           -> {sorted(int(n) for n in fem.nodes.get(label='tip').ids)}")

print()
print("--- mid_pg (PG only) ---")
print(f"  target='mid_pg'       -> {sorted(int(n) for n in fem.nodes.get(target='mid_pg').ids)}")
print(f"  pg='mid_pg'           -> {sorted(int(n) for n in fem.nodes.get(pg='mid_pg').ids)}")

print()
print("--- base: has label 'base_lbl' AND PG 'base_pg' ---")
print(f"  target='base_lbl'          -> {sorted(int(n) for n in fem.nodes.get(target='base_lbl').ids)}")
print(f"  label='base_lbl'           -> {sorted(int(n) for n in fem.nodes.get(label='base_lbl').ids)}")
print(f"  target='base_pg'           -> {sorted(int(n) for n in fem.nodes.get(target='base_pg').ids)}")
print(f"  pg='base_pg'               -> {sorted(int(n) for n in fem.nodes.get(pg='base_pg').ids)}")


# %% [markdown]
# ## 6. Consistency verification
#
# Run ONE OpenSees analysis and compare tip displacement obtained
# via three different accessor styles. If the namespaces are wired
# correctly, all three should match to machine precision.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 0, 1, 0)
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E, G, J, Iy, Iz, 1)

# Fix the base node — access it via LABEL
for n in fem.nodes.get(label="base_lbl").ids:
    ops.fix(int(n), 1, 1, 1, 1, 1, 1)

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
# Load the tip — access via LABEL
for n in fem.nodes.get(target="tip").ids:
    ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)


# %% [markdown]
# ## 7. Result extraction and verification
#
# Pull the tip $u_z$ three different ways and compare to analytical.

# %%
disp_via_target  = ops.nodeDisp(int(next(iter(fem.nodes.get(target="tip").ids))), 3)
disp_via_label   = ops.nodeDisp(int(next(iter(fem.nodes.get(label="tip").ids))),  3)

analytical = -P * L**3 / (3.0 * E * Iz)
err_target = abs(disp_via_target - analytical) / abs(analytical) * 100.0
err_label  = abs(disp_via_label  - analytical) / abs(analytical) * 100.0

print(f"disp via target='tip'  :  {disp_via_target:.6e}  m   err {err_target:.4f} %")
print(f"disp via label='tip'   :  {disp_via_label:.6e}  m   err {err_label:.4f} %")
print(f"analytical             :  {analytical:.6e}  m")
print()
same = abs(disp_via_target - disp_via_label) < 1e-15
print(f"both accessor forms give identical tip disp? {same}")


# %% [markdown]
# ## What this unlocks
#
# * **Knowing where names live.** Labels are survivors of geometry
#   edits; PGs are the canonical OpenSees-visible names. Both work
#   with ``fem.nodes.get(...)`` / ``fem.elements.get(...)`` and with
#   ``g.loads.*`` / ``g.constraints.*``.
# * **Four accessor dialects** for the same target: ``target=``
#   (auto), ``label=``, ``pg=``, ``tag=`` (raw DimTag list). Use the
#   most specific form in production code — it fails loudly if the
#   entity is misnamed, whereas ``target=`` would silently try the
#   other namespaces.
# * **Auto-resolution order** (label → PG → part label) is the
#   default everywhere in apeGmsh. Any future notebook that says
#   ``target="..."`` is using this same lookup chain.

# %%
g_ctx.__exit__(None, None, None)
