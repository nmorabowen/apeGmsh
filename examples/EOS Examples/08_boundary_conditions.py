# %% [markdown]
# # 08 — Boundary Conditions Walkthrough
#
# **Curriculum slot:** Tier 2, slot 08.
# **Prerequisite:** 03 — Simply-Supported Beam.
#
# ## Purpose
#
# Three ways to state a boundary condition in apeGmsh / OpenSees and
# when each is correct:
#
# | Mechanism | OpenSees call | apeGmsh composite | Use for |
# |---|---|---|---|
# | **Homogeneous DOF fix** | ``ops.fix(node, mask…)`` | ``g.loads.face_sp(disp_xyz=None, rot_xyz=None)`` (auto-flagged homogeneous) | pinned / fixed / rolled supports with zero prescribed value |
# | **Non-homogeneous SP** | ``ops.sp(node, dof, value)`` + a ``ops.pattern "Plain"`` block to own it | ``g.loads.face_sp(disp_xyz=(dx, dy, dz))`` | prescribed displacement / settlement cases |
# | **Distributed face load** (not a BC) | ``ops.eleLoad`` or per-node ``ops.load`` | ``g.loads.face_load(force_xyz=(...))`` / ``g.loads.surface(...)`` | applied tractions |
#
# ``face_sp`` records land in ``fem.nodes.sp`` (an ``SPSet``) after
# the mesh resolves. Records with ``is_homogeneous=True`` should be
# emitted outside any pattern (they are time-invariant); records
# with ``is_homogeneous=False`` have to live inside a ``pattern``
# block so their prescribed value is scaled by the pattern's
# ``timeSeries``.
#
# ## Problem
#
# Cantilever beam, fixed at the base, with a **prescribed vertical
# displacement** $\Delta = 10 \text{ mm}$ at the free tip. From
# elementary beam theory, that constraint requires the tip to pull
# on the beam with force
#
# $$
# V_{\text{tip}} \;=\; \dfrac{3\,E\,I}{L^{3}}\,\Delta,
# $$
#
# and produces a **base moment reaction**
#
# $$
# M_{\text{base}} \;=\; \dfrac{3\,E\,I}{L^{2}}\,\Delta.
# $$
#
# The tip rotation is free, so the beam deforms like half of a
# doubly-supported beam with a central support release.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L = 3.0
Delta = 10.0e-3        # prescribed tip displacement [m]

E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 10.0


# %% [markdown]
# ## 2. Geometry

# %%
g_ctx = apeGmsh(model_name="08_boundary_conditions", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()

g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [ln],     name="beam")


# %% [markdown]
# ## 3. Declare BCs through the composite
#
# Both endpoints use ``g.loads.face_sp``. At the base we pass no
# ``disp_xyz`` so every target DOF is constrained to 0 — that's a
# homogeneous fix. At the tip we pass ``disp_xyz=(0, 0, -Delta)``
# so DOF 3 gets a non-homogeneous prescribed value.

# %%
# Base: homogeneous fix on [ux, uy, uz] (the translational DOFs
# that face_sp exposes). Rotational DOFs are NOT covered by face_sp;
# we still need ops.fix on DOFs 4-6 for a "truly fixed" end, so for
# this example the base uses ops.fix directly — documented below.
g.loads.face_sp(target="base", dofs=[1, 1, 1])   # homogeneous on ux, uy, uz

# Tip: prescribe u_z = -Delta. The other two translational DOFs
# stay free (dofs=[0, 0, 1]).
g.loads.face_sp(target="tip", dofs=[0, 0, 1], disp_xyz=(0.0, 0.0, -Delta))


# %% [markdown]
# ## 4. Mesh

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")
print(f"SP records: {len(list(fem.nodes.sp))}")


# %% [markdown]
# ## 5. Inspect the resolved SPRecords
#
# apeGmsh resolves each ``face_sp`` into one ``SPRecord`` per
# (node, DOF) pair. ``is_homogeneous`` tells the solver whether the
# record can live outside any pattern (True — bare constraint) or
# needs a scaled ``timeSeries`` (False — prescribed disp).

# %%
print(f"{'node':>6} {'dof':>4} {'value':>14}  is_hom")
for rec in fem.nodes.sp:
    print(f"{rec.node_id:>6} {rec.dof:>4} {rec.value:>14.6e}  {rec.is_homogeneous}")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Two distinct emission paths:
#
# * **Homogeneous** SPs: emit as ``ops.fix(node, …mask…)`` before any
#   pattern. These are time-invariant.
# * **Non-homogeneous** SPs: emit as ``ops.sp(node, dof, value)``
#   inside a ``ops.pattern "Plain"`` with a scaling ``timeSeries``.
#
# The base's **rotational** DOFs are still free after ``face_sp``
# (which only covers translations). We add them explicitly via
# ``ops.fix`` below so the base is truly clamped.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E, G, J, Iy, Iz, 1)

# --- homogeneous SPs from face_sp (ux, uy, uz = 0 at base) ---
# Group by node so we emit a single ops.fix call per node.
hom_by_node: dict[int, list[int]] = {}
for rec in fem.nodes.sp.homogeneous():
    hom_by_node.setdefault(int(rec.node_id), []).append(int(rec.dof))

for nid, dofs in hom_by_node.items():
    mask = [1 if d in dofs else 0 for d in (1, 2, 3, 4, 5, 6)]
    ops.fix(nid, *mask)

# Still need to fix rotational DOFs at the base (face_sp only covers
# translations). Additional fix: skip DOFs already constrained.
base_nodes = set(int(n) for n in fem.nodes.get(target="base").ids)
for nid in base_nodes:
    existing = hom_by_node.get(nid, [])
    extra = [d for d in (4, 5, 6) if d not in existing]
    if extra:
        mask = [1 if d in extra else 0 for d in (1, 2, 3, 4, 5, 6)]
        ops.fix(nid, *mask)

# --- non-homogeneous SPs inside a scaled pattern ---
nonhom_records = list(fem.nodes.sp.prescribed())
if nonhom_records:
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for rec in nonhom_records:
        ops.sp(int(rec.node_id), int(rec.dof), float(rec.value))

# Constraint handler note: OpenSees's "Plain" handler IGNORES
# non-zero SPs (prescribed displacements). For displacement control
# we need "Transformation" (or "Penalty" with a large factor, or
# "Lagrange"). "Transformation" is the cleanest — it partitions the
# stiffness matrix into free/constrained DOFs exactly.
ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Transformation")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
status = ops.analyze(1)
assert status == 0
ops.reactions()
print("analysis converged")


# %% [markdown]
# ## 7. Verification
#
# Three checks:
#
# * tip $u_z$ should equal the prescribed $-\Delta$ to machine
#   precision (the constraint enforces it),
# * base moment reaction magnitude should equal $3EI\Delta/L^{2}$,
# * base vertical-force reaction magnitude should equal
#   $3EI\Delta/L^{3}$.

# %%
tip_node  = int(next(iter(fem.nodes.get(target="tip").ids)))
base_node = int(next(iter(fem.nodes.get(target="base").ids)))

tip_uz        = ops.nodeDisp(tip_node, 3)
base_Fz       = ops.nodeReaction(base_node, 3)
base_My       = ops.nodeReaction(base_node, 5)   # in-plane moment

analytical_V  = 3.0 * E * Iz * Delta / L**3
analytical_M  = 3.0 * E * Iz * Delta / L**2

err_uz = abs(tip_uz - (-Delta)) / Delta * 100.0
err_V  = abs(abs(base_Fz) - analytical_V) / analytical_V * 100.0
err_M  = abs(abs(base_My) - analytical_M) / analytical_M * 100.0

print("Prescribed tip displacement")
print(f"  FEM u_z     :  {tip_uz:.6e}  m")
print(f"  Prescribed  :  {-Delta:.6e}  m")
print(f"  Error       :  {err_uz:.4f} %")
print()
print("Base vertical-force reaction")
print(f"  FEM         :  {base_Fz:.6e}  N")
print(f"  Analytical  :  {-analytical_V:.6e}  N   (3EI*Delta/L^3)")
print(f"  Error       :  {err_V:.4f} %")
print()
print("Base moment reaction")
print(f"  FEM         :  {base_My:.6e}  N*m")
print(f"  Analytical  :  {-analytical_M:.6e}  N*m   (3EI*Delta/L^2)")
print(f"  Error       :  {err_M:.4f} %")


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment to see the deformed cantilever under the prescribed tip
# displacement.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * **Three BC mechanisms** — homogeneous ``fix``, non-homogeneous
#   ``sp``, and distributed face loads via ``face_load`` — and the
#   explicit convention for which OpenSees call each one becomes.
# * **The ``SPSet`` composite** — ``fem.nodes.sp.homogeneous()`` and
#   ``fem.nodes.sp.prescribed()`` separate the two emission paths so
#   the ingest code doesn't have to inspect ``.value`` or
#   ``.is_homogeneous`` manually.
# * **Reaction-based verification pattern** for prescribed-
#   displacement problems: the reaction at the fixed end equals the
#   analytical stiffness-times-displacement term. Any later notebook
#   that uses displacement control (pushover, monotonic load path)
#   reuses this reaction-extraction template.

# %%
g_ctx.__exit__(None, None, None)
