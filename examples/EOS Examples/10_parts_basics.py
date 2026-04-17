# %% [markdown]
# # 10 — Parts Basics
#
# **Curriculum slot:** Tier 3, slot 10.
# **Prerequisite:** 04 — 2D Portal Frame.
#
# ## Purpose
#
# apeGmsh's ``Parts`` composite is a third namespace, on top of
# Labels and Physical Groups. It groups a set of DimTags under a
# single **part label**. The intent:
#
# * Give a set of entities an identity that survives boolean
#   operations (``fragment``, ``cut``, ``fuse``). After fragmentation
#   the registry still knows which post-fragment DimTags belong to
#   the original part, so downstream code can re-query them without
#   re-tagging.
# * Provide a stable mesh-node view through
#   ``g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)``.
# * Represent assembly semantics — each part can have its own
#   translate / rotate at ``add`` time and participate in
#   ``fragment_all()`` / ``fuse_group()``.
#
# ## Problem — two independent cantilevers as two parts
#
# Two identical cantilever beams, each running along $+x$ at
# different $y$ levels. Both have a fixed base and a tip load $P$.
# We register each as a ``Part``, verify the per-part node-map
# resolution, and confirm that both cantilevers independently
# satisfy $\delta = P L^{3} / (3 E I)$.
#
# > ### Note
# >
# > ``fem.nodes.get(target="my_part")`` and
# > ``fem.elements.get(target="my_part")`` both resolve part labels
# > directly — they search **label → PG → part** in that order,
# > matching ``g.loads.*`` and ``g.constraints.*``. The explicit
# > workaround below (``g.parts.build_node_map`` + direct
# > ``Instance.entities`` iteration) still works and is shown here
# > as a reference for when you need the raw mapping; prefer
# > ``fem.nodes.get(target=...)`` in day-to-day code.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import gmsh
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L = 3.0
P = 10_000.0
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 10.0


# %% [markdown]
# ## 2. Geometry — two parallel beams

# %%
g_ctx = apeGmsh(model_name="10_parts_basics", verbose=False)
g = g_ctx.__enter__()

# Beam A along y = 0
p0a = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p1a = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
lnA = g.model.geometry.add_line(p0a, p1a)

# Beam B along y = 1
p0b = g.model.geometry.add_point(0.0, 1.0, 0.0, lc=LC)
p1b = g.model.geometry.add_point(L,   1.0, 0.0, lc=LC)
lnB = g.model.geometry.add_line(p0b, p1b)

g.model.sync()


# %% [markdown]
# ## 3. Register the two parts
#
# A part is a named bag of DimTags. Each part's bag here contains
# its line and both endpoint points — all three DimTags get tracked
# under the part label.

# %%
instA = g.parts.register("beamA", [(1, lnA), (0, p0a), (0, p1a)])
instB = g.parts.register("beamB", [(1, lnB), (0, p0b), (0, p1b)])

print(f"parts registered: {g.parts.labels()}")
print(f"beamA entities  : {dict(instA.entities)}")
print(f"beamB entities  : {dict(instB.entities)}")


# %% [markdown]
# ## 4. Physical groups for BC / load targeting
#
# Since FEMData's ``get(target=...)`` does not resolve part labels,
# we create PGs for the four nodes we'll address directly (the two
# bases and the two tips). The line elements for each beam are
# identified through the part's ``entities`` below.

# %%
g.physical.add(0, [p0a], name="baseA")
g.physical.add(0, [p1a], name="tipA")
g.physical.add(0, [p0b], name="baseB")
g.physical.add(0, [p1b], name="tipB")


# %% [markdown]
# ## 5. Mesh + node-map

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()

# The canonical way to ask "which mesh nodes belong to part X?".
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)

print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")
for lbl, nset in node_map.items():
    print(f"part '{lbl}': {len(nset)} mesh nodes")

shared = node_map["beamA"] & node_map["beamB"]
print(f"shared nodes   : {len(shared)}  (should be 0 — disjoint beams)")


# %% [markdown]
# ## 6. Per-part element identification
#
# Each part's line entities live in ``Instance.entities[1]``. We
# query gmsh directly for the mesh elements on those curves and use
# those for element emission.

# %%
def line_elements_of(inst) -> list[tuple[int, tuple[int, int]]]:
    """Return [(elem_tag, (node_i, node_j)), ...] for every line
    element on the part's dim=1 entities."""
    out: list[tuple[int, tuple[int, int]]] = []
    for tag in inst.entities.get(1, []):
        etypes, elem_tags, elem_nodes = gmsh.model.mesh.getElements(1, int(tag))
        for etype, etags, enodes in zip(etypes, elem_tags, elem_nodes):
            # gmsh element type 1 = 2-node line
            if int(etype) != 1:
                continue
            arr = np.asarray(enodes, dtype=np.int64).reshape(-1, 2)
            for eid, row in zip(etags, arr):
                out.append((int(eid), (int(row[0]), int(row[1]))))
    return out


elems_A = line_elements_of(instA)
elems_B = line_elements_of(instB)
print(f"part 'beamA' line elements: {len(elems_A)}")
print(f"part 'beamB' line elements: {len(elems_B)}")


# %% [markdown]
# ## 7. OpenSees ingest + analysis

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)

# Part-scoped element emission
for elems in (elems_A, elems_B):
    for eid, (ni, nj) in elems:
        ops.element("elasticBeamColumn", eid, ni, nj,
                    A, E, G, J, Iy, Iz, 1)

# Base fixes — PG lookup (FEMData does handle PGs)
for pg_name in ("baseA", "baseB"):
    for n in fem.nodes.get(target=pg_name).ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)

# Tip loads
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for pg_name in ("tipA", "tipB"):
    for n in fem.nodes.get(target=pg_name).ids:
        ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 8. Verification — both parts independently satisfy δ = PL³/3EI

# %%
tip_A = int(next(iter(fem.nodes.get(target="tipA").ids)))
tip_B = int(next(iter(fem.nodes.get(target="tipB").ids)))
d_A = ops.nodeDisp(tip_A, 3)
d_B = ops.nodeDisp(tip_B, 3)
analytical = -P * L**3 / (3.0 * E * Iz)
err_A = abs(d_A - analytical) / abs(analytical) * 100.0
err_B = abs(d_B - analytical) / abs(analytical) * 100.0

print(f"Part A tip u_z :  {d_A:.6e}  m   err {err_A:.4f} %")
print(f"Part B tip u_z :  {d_B:.6e}  m   err {err_B:.4f} %")
print(f"Analytical     :  {analytical:.6e}  m")


# %% [markdown]
# ## What this unlocks
#
# * **A third namespace for assembly modelling.** Parts persist
#   across boolean operations (slot 11) and can be placed with
#   translate/rotate at ``g.parts.add(part, translate=..., rotate=...)``
#   time — the basis for every repeated-instance model (truss bays,
#   frame stories, reinforcement arrays).
# * **``build_node_map`` as the canonical part-to-nodes bridge.**
#   Every later curriculum notebook that composes multiple parts
#   uses this call once after meshing to produce a
#   ``dict[label → set[node_id]]`` for addressing.
# * **Direct element iteration through ``Instance.entities``.** The
#   pattern ``gmsh.model.mesh.getElements(dim, tag)`` for each DimTag
#   in the part is the portable fallback whenever higher-level
#   accessors don't know about parts.

# %%
g_ctx.__exit__(None, None, None)
