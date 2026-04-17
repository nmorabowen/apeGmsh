# %% [markdown]
# # 12 — Interface Coupling via ``equal_dof``
#
# **Curriculum slot:** Tier 3, slot 12.
# **Prerequisite:** 10 — Parts Basics, 11 — Boolean Assembly.
#
# ## Purpose
#
# Slot 11 showed how a **boolean fragment** produces a conformal
# mesh at the interface of two parts — the CAD itself becomes one
# body, so Gmsh places a single shared node along the common edge.
#
# This slot covers the alternative: **the two parts stay
# geometrically separate.** Each part meshes independently, the
# junction has two coincident-but-distinct mesh nodes, and a
# constraint glues the DOFs of the second node to the first.
#
# The simplest form of that constraint is ``equal_dof`` —
# coordinate-matched nodes share all requested DOFs. For
# non-matching meshes with no coincident nodes, a **tie** with
# shape-function interpolation is needed instead (slot 16).
#
# ## Problem — two beams joined end-to-end by ``equal_dof``
#
# Two linear-elastic cantilever beams meeting at $x = L_A$. Beam A
# spans $[0, L_A]$ with its fixed base at $x=0$. Beam B spans
# $[L_A, L_A + L_B]$ with a downward tip load $P$ at $x = L_A+L_B$.
# Each beam is a **separate part**, so the junction is represented
# by two coincident points ($p_{A1}$ and $p_{B0}$) and after
# meshing by two coincident nodes.
#
# With the junction tied (``equal_dof`` on all 6 DOFs) the
# structure behaves as a single cantilever of total length
# $L = L_A + L_B$, so the analytical tip deflection is the
# classical
#
# $$
# \delta_{\text{tip}} \;=\; \dfrac{P\,L^{3}}{3\,E\,I},
# \qquad L = L_A + L_B.
# $$
#
# Without the tie, the two beams float independently and the
# analysis either fails (beam B has no boundary condition) or
# produces unbounded displacement. That "before vs after" contrast
# is the teaching point.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L_A, L_B = 1.5, 1.5
L = L_A + L_B
P = 10_000.0
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 20.0


# %% [markdown]
# ## 2. Geometry — two independent beams

# %%
g_ctx = apeGmsh(model_name="12_interface_tie", verbose=False)
g = g_ctx.__enter__()

# Beam A
pA0 = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
pA1 = g.model.geometry.add_point(L_A, 0.0, 0.0, lc=LC)     # junction, side A
lnA = g.model.geometry.add_line(pA0, pA1)

# Beam B — starts at x = L_A, DISTINCT point from pA1
pB0 = g.model.geometry.add_point(L_A, 0.0, 0.0, lc=LC)      # junction, side B (coincident with pA1)
pB1 = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
lnB = g.model.geometry.add_line(pB0, pB1)

g.model.sync()

# Register parts (do NOT fragment — keep the two bodies distinct)
instA = g.parts.register("beamA", [(1, lnA), (0, pA0), (0, pA1)])
instB = g.parts.register("beamB", [(1, lnB), (0, pB0), (0, pB1)])


# %% [markdown]
# ## 3. Physical groups + the tie

# %%
g.physical.add(0, [pA0], name="base")
g.physical.add(0, [pB1], name="tip")


# %% [markdown]
# ## 4. Declare the interface tie
#
# ``g.constraints.equal_dof`` matches coincident nodes across the
# two parts' interface entities and emits ``NodePairRecord`` per
# match. The search is bounded to ``master_entities`` /
# ``slave_entities`` so we don't accidentally pick up nodes far
# from the junction.

# %%
g.constraints.equal_dof(
    master_label="beamA",
    slave_label="beamB",
    master_entities=[(0, pA1)],       # the junction point on side A
    slave_entities=[(0, pB0)],        # the junction point on side B
    tolerance=1e-9,                    # they're exactly coincident
    dofs=[1, 2, 3, 4, 5, 6],           # couple all 6 DOFs
)


# %% [markdown]
# ## 5. Mesh + resolve

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# Inspect the constraint records that came out.
equal_dof_pairs = list(fem.nodes.constraints.equal_dofs())
print(f"equal_dof pairs resolved : {len(equal_dof_pairs)}")
for rec in equal_dof_pairs:
    print(f"  master {rec.master_node} <-> slave {rec.slave_node}  "
          f"dofs={rec.dofs}")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# The standard ingest (nodes, elements, fixes, loads) plus one new
# step: iterate ``fem.nodes.constraints.equal_dofs()`` and emit
# ``ops.equalDOF`` per pair.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)

# Elements per part (pattern reused from slot 10)
import gmsh
def line_elements_of(inst):
    out = []
    for tag in inst.entities.get(1, []):
        etypes, etags, enodes = gmsh.model.mesh.getElements(1, int(tag))
        for etype, elist, nlist in zip(etypes, etags, enodes):
            if int(etype) != 1:
                continue
            arr = np.asarray(nlist, dtype=np.int64).reshape(-1, 2)
            for eid, row in zip(elist, arr):
                out.append((int(eid), (int(row[0]), int(row[1]))))
    return out

for inst in (instA, instB):
    for eid, (ni, nj) in line_elements_of(inst):
        ops.element("elasticBeamColumn", eid, ni, nj,
                    A, E, G, J, Iy, Iz, 1)

# Interface tie — one ops.equalDOF per pair record
for rec in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(int(rec.master_node), int(rec.slave_node), *rec.dofs)

# BCs + loads (unchanged)
for n in fem.nodes.get(target="base").ids:
    ops.fix(int(n), 1, 1, 1, 1, 1, 1)

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for n in fem.nodes.get(target="tip").ids:
    ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)

# For equalDOF constraints, "Plain" handler is OK when all linked
# DOFs are free (no conflicting SPs) — which is our case here
# because the junction nodes are not themselves fixed.
ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification — the tied assembly behaves as a single beam

# %%
tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
d_tip = ops.nodeDisp(tip_node, 3)
analytical = -P * L**3 / (3.0 * E * Iz)
err = abs(d_tip - analytical) / abs(analytical) * 100.0

print(f"FEM tip u_z  :  {d_tip:.6e}  m")
print(f"Analytical   :  {analytical:.6e}  m   (PL^3 / 3EI, L = LA + LB)")
print(f"Error        :  {err:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Interface coupling mechanism.** ``g.constraints.equal_dof``
#   finds coincident mesh nodes across two parts and produces one
#   ``NodePairRecord`` per pair. Emission to OpenSees is a
#   one-liner: ``ops.equalDOF(master, slave, *dofs)``.
# * **Unfragmented-assembly workflow.** Keep parts geometrically
#   separate, mesh each independently, tie them through
#   constraints. This is the preferred path whenever the two
#   parts have different element types or different material
#   laws — fragmenting would force a single mesh across the
#   interface.
# * **Scaffolding for slot 16.** Slot 16 replaces ``equal_dof``
#   with ``g.constraints.tie`` to handle non-matching meshes, where
#   each slave node projects onto a master element face and gets
#   shape-function weights. The resolve / record / ingest chain
#   established here is the same — only the emission step swaps
#   ``ops.equalDOF`` for ``ASDEmbeddedNodeElement`` (the canonical
#   OpenSees vehicle for weighted MP constraints).

# %%
g_ctx.__exit__(None, None, None)
