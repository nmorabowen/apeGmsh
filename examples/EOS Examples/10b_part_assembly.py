# %% [markdown]
# # 10b — Part Assembly with Transforms
#
# **Curriculum slot:** Tier 3, slot 10.5.
# **Prerequisite:** 10 — Parts Basics.
#
# ## Purpose
#
# Slot 10 showed how to *tag* existing geometry with ``g.parts.register``.
# That's the simplest part workflow but doesn't demonstrate the most
# useful application of the Parts composite: **instancing a template
# geometry multiple times with transforms.**
#
# The canonical apeGmsh idiom:
#
# ```python
# col = Part(name="column").begin()
# # ... build the column geometry inside its own session ...
# col.end()
#
# with apeGmsh(model_name="assembly") as g:
#     g.parts.add(col, label="col_0", translate=(0.0, 0, 0))
#     g.parts.add(col, label="col_1", translate=(4.0, 0, 0))
#     g.parts.add(col, label="col_2", translate=(8.0, 0, 0))
# ```
#
# Under the hood ``Part.end()`` auto-persists the Part's geometry
# to a STEP file (OS tempfile, garbage-collected with the Part),
# and each ``g.parts.add(col, translate=...)`` imports that STEP
# into the main session with the transform applied. This is the
# same mechanism ``g.parts.import_step(file_path, translate=...)``
# uses directly when the part comes from a pre-built CAD file —
# ideal for dropping a library of CAD parts into a scene.
#
# ## Problem — three identical columns in a row
#
# A 3 m vertical cantilever beam template, replicated at $x = 0$,
# $x = 4$, $x = 8$. Each column gets its own fixed base and tip
# point load $P$. Each column's tip deflection must match the
# classical $P L^{3} / (3\,E\,I)$ — the transform (pure
# translation) doesn't alter stiffness.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh
from apeGmsh.core.Part import Part

# --- Geometry ---
L = 3.0                     # column height
N_COLS = 3                  # how many instances to place
DX = 4.0                    # spacing between columns in +x
COLUMN_X_OFFSETS = [i * DX for i in range(N_COLS)]

# --- Loading ---
P = 10_000.0                # tip horizontal load (pushed in +x) [N]

# --- Material ---
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5

LC = L / 10.0


# %% [markdown]
# ## 2. Build the template Part in isolation
#
# ``Part(name=...).begin()`` opens an isolated Gmsh session for the
# part. We build exactly the geometry we want (line + two
# endpoints). ``end()`` auto-persists the Part to a tempfile.

# %%
col_template = Part(name="cantilever_column").begin(verbose=False)
col_template.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
col_template.model.geometry.add_point(0.0, 0.0, L,   lc=LC)
# Note: the line comes along automatically when add_line is called.
# Using the geometry add_line keeps the part's internal tag space clean.
p0 = 1   # first point added inside the part; its OCC tag is 1
p1 = 2   # second point; OCC tag 2
col_template.model.geometry.add_line(p0, p1)
col_template.model.sync()
col_template.end()


# %% [markdown]
# ## 3. Assemble three instances in the main session
#
# Each call to ``g.parts.add(part, label=..., translate=...)``
# imports the template's STEP into the assembly session at the
# given translate offset and registers it as a new ``Instance``.

# %%
g_ctx = apeGmsh(model_name="10b_part_assembly", verbose=False)
g = g_ctx.__enter__()

for i, x_off in enumerate(COLUMN_X_OFFSETS):
    g.parts.add(
        col_template,
        label=f"col_{i}",
        translate=(x_off, 0.0, 0.0),
    )

print(f"parts registered: {g.parts.labels()}")
for lbl in g.parts.labels():
    print(f"  {lbl} entities: {dict(g.parts.get(lbl).entities)}")


# %% [markdown]
# ## 4. Mesh

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

for lbl in sorted(g.parts.labels()):
    n = len(list(fem.nodes.get(target=lbl).ids))
    e = sum(len(gr.ids) for gr in fem.elements.get(target=lbl))
    print(f"  {lbl}: {n} nodes, {e} elements")


# %% [markdown]
# ## 5. Per-column base + tip node identification
#
# Each column's base is the mesh node at $z = 0$; the tip is the
# node at $z = L$. We classify the part's nodes by coordinate.

# %%
tag_to_idx = {int(t): i for i, t in enumerate(fem.nodes.ids)}

def base_and_tip(label: str) -> tuple[int, int]:
    """Return (base_node_id, tip_node_id) for part ``label``."""
    base = tip = None
    for nid in fem.nodes.get(target=label).ids:
        z = float(fem.nodes.coords[tag_to_idx[int(nid)], 2])
        if abs(z - 0.0) < 1e-9:
            base = int(nid)
        elif abs(z - L) < 1e-9:
            tip = int(nid)
    assert base is not None and tip is not None, \
        f"could not find base/tip nodes for part '{label}'"
    return base, tip

column_nodes = {lbl: base_and_tip(lbl) for lbl in sorted(g.parts.labels())}
for lbl, (b, t) in column_nodes.items():
    print(f"  {lbl}: base={b}, tip={t}")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Standard pattern. Element emission iterates per part label via
# ``fem.elements.get(target=lbl)`` — now that slot 10's FEMData gap
# is fixed, the part label is a first-class target.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)      # vecxz = +x; columns run along +z

for lbl in sorted(g.parts.labels()):
    for group in fem.elements.get(target=lbl):
        for eid, nodes in zip(group.ids, group.connectivity):
            ops.element("elasticBeamColumn", int(eid),
                        int(nodes[0]), int(nodes[1]),
                        A, E, G, J, Iy, Iz, 1)

# Fix each base; apply horizontal load at each tip
for lbl, (base, tip) in column_nodes.items():
    ops.fix(int(base), 1, 1, 1, 1, 1, 1)

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for lbl, (base, tip) in column_nodes.items():
    ops.load(int(tip), P, 0.0, 0.0, 0.0, 0.0, 0.0)     # +x tip load

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification — all three columns must give identical δ_tip
#
# Each vertical cantilever loaded horizontally at its tip has the
# classical $\delta_{\text{tip}} = P L^{3} / (3 E I)$. With
# vecxz = ($+$x, 0, 0) and the column along $+z$, in-plane bending
# is about local $+y$ — controlled by $I_{y}$.

# %%
analytical = P * L**3 / (3.0 * E * Iy)
print(f"{'part':>6s}  {'tip dx':>14s}  {'error %':>10s}")
worst = 0.0
for lbl, (base, tip) in column_nodes.items():
    d_tip = ops.nodeDisp(int(tip), 1)        # u_x (horizontal)
    err = abs(d_tip - analytical) / abs(analytical) * 100.0
    worst = max(worst, err)
    print(f"{lbl:>6s}  {d_tip:>14.6e}  {err:>9.4f} %")

print(f"\nanalytical reference : {analytical:.6e}  m")
print(f"worst-case error     : {worst:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Template-plus-transforms assembly workflow** via the canonical
#   ``g.parts.add(part, translate=..., rotate=...)``. Build the
#   geometry once as a ``Part``, then place it many times with
#   translate/rotate. Works for CAD imports via
#   ``g.parts.import_step(file_path, translate=..., rotate=...)``.
# * **Per-part node + element queries** via
#   ``fem.nodes.get(target=part_label)`` and
#   ``fem.elements.get(target=part_label)``. The label resolves
#   through the same chain used everywhere else in apeGmsh:
#   label → PG → part.
# * **Coordinate-based role extraction** for post-mesh part
#   navigation. ``base_and_tip(label)`` classifies the part's
#   mesh nodes by coordinate — the pattern scales to any
#   assembly geometry where specific nodes need to be addressed.

# %%
g_ctx.__exit__(None, None, None)
