# %% [markdown]
# # 07 — Load Patterns and Linear Superposition
#
# **Curriculum slot:** Tier 2, slot 07.
# **Prerequisite:** 02 — 2D Cantilever Beam.
#
# ## Purpose
#
# The ``g.loads.pattern()`` context manager groups load definitions
# under a **named pattern**. apeGmsh attaches the pattern name to
# every ``LoadDef`` you register inside the ``with`` block, and the
# resolved records (available as ``fem.nodes.loads.by_kind('nodal')``
# / ``fem.elements.loads.by_kind('element')``) carry the pattern
# name forward. At the OpenSees emission stage we group records by
# pattern and emit one ``ops.timeSeries`` + ``ops.pattern "Plain"``
# block per group.
#
# This notebook shows:
#
# 1. How to register two independent patterns.
# 2. How to filter resolved records by pattern name.
# 3. How to emit one OpenSees pattern per group.
# 4. **Linear superposition** — the combined-pattern tip deflection
#    equals the sum of the dead-only and live-only tip deflections.
#
# ## Problem
#
# The same cantilever from slot 02 (length $L = 3$ m, the usual
# elastic beam section), loaded with:
#
# * **Pattern "dead"** — uniform distributed $w$ in $-z$ (self-weight
#   or any permanent action), expressed via ``g.loads.line``. This
#   gives a cantilever under uniform load with
#
#   $$\delta_{w} \;=\; \dfrac{w\,L^{4}}{8\,E\,I}.$$
#
# * **Pattern "live"** — tip point load $P$ in $-z$, via
#   ``g.loads.point``, giving the classical
#
#   $$\delta_{P} \;=\; \dfrac{P\,L^{3}}{3\,E\,I}.$$
#
# We run **three** analyses:
#
# | Run | Active patterns | Expected tip disp |
# |---|---|---|
# | dead only | ``"dead"`` | $-\,w L^{4} / (8\,E\,I)$ |
# | live only | ``"live"`` | $-\,P L^{3} / (3\,E\,I)$ |
# | combined  | both       | sum of the above |
#
# Verification: the combined result must equal the sum of the two
# individual results to machine precision (linear problem).

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L = 3.0
w = 2_000.0            # distributed "dead" load [N/m]
P = 10_000.0           # tip "live" point load   [N]

E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 10.0


# %% [markdown]
# ## 2. Geometry + patterns

# %%
g_ctx = apeGmsh(model_name="07_load_patterns", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()

g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [ln],     name="beam")


# %% [markdown]
# ## 3. Declare loads **inside named patterns**
#
# Every ``LoadDef`` created inside ``with g.loads.pattern(name):``
# carries ``pattern = name``. Outside any ``pattern()`` context the
# default pattern name is ``"default"``.

# %%
with g.loads.pattern("dead"):
    g.loads.line(target="beam", magnitude=w, direction=(0.0, 0.0, -1.0))

with g.loads.pattern("live"):
    g.loads.point(target="tip", force_xyz=(0.0, 0.0, -P))

print(f"patterns registered: {g.loads.patterns()}")


# %% [markdown]
# ## 4. Mesh + resolve

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")
print(f"nodal-load records after resolve: {len(list(fem.nodes.loads.by_kind('nodal')))}")


# %% [markdown]
# ## 5. Analysis driver
#
# The ``run`` function below builds a fresh OpenSees model on each
# call, applies **only the patterns named in** ``active_patterns``,
# analyses, and returns the tip deflection.

# %%
def run(active_patterns: list[str]) -> float:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # -- nodes + elements --
    for nid, xyz in fem.nodes.get():
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    for group in fem.elements.get(target="beam"):
        for eid, nodes in zip(group.ids, group.connectivity):
            ops.element("elasticBeamColumn", int(eid),
                        int(nodes[0]), int(nodes[1]),
                        A, E, G, J, Iy, Iz, 1)

    # -- fix the base --
    for n in fem.nodes.get(target="base").ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)

    # -- one OpenSees pattern per apeGmsh pattern --
    all_records = list(fem.nodes.loads.by_kind("nodal"))
    for pat_tag, pat_name in enumerate(active_patterns, start=1):
        records = [r for r in all_records if r.pattern == pat_name]
        if not records:
            continue
        ts_tag = pat_tag
        ops.timeSeries("Constant", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)
        for rec in records:
            fx, fy, fz = rec.force_xyz if rec.force_xyz else (0, 0, 0)
            mx, my, mz = rec.moment_xyz if rec.moment_xyz else (0, 0, 0)
            ops.load(int(rec.node_id), fx, fy, fz, mx, my, mz)

    ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0); ops.analysis("Static")
    status = ops.analyze(1)
    assert status == 0

    tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
    return ops.nodeDisp(tip_node, 3)


# %% [markdown]
# ## 6. Three runs: dead-only, live-only, combined

# %%
d_dead = run(["dead"])
d_live = run(["live"])
d_both = run(["dead", "live"])

print(f"dead-only tip disp  : {d_dead:.6e}  m")
print(f"live-only tip disp  : {d_live:.6e}  m")
print(f"combined  tip disp  : {d_both:.6e}  m")


# %% [markdown]
# ## 7. Verification — analytical values + linearity check

# %%
analytical_dead = -w * L**4 / (8.0 * E * Iz)
analytical_live = -P * L**3 / (3.0 * E * Iz)
analytical_both = analytical_dead + analytical_live

err_dead = abs(d_dead - analytical_dead) / abs(analytical_dead) * 100.0
err_live = abs(d_live - analytical_live) / abs(analytical_live) * 100.0
err_both = abs(d_both - analytical_both) / abs(analytical_both) * 100.0

# Linearity: d_both should equal d_dead + d_live
super_sum = d_dead + d_live
err_linearity = abs(d_both - super_sum) / abs(d_both) * 100.0

print("Dead-load case (wL^4 / 8EI)")
print(f"  FEM        :  {d_dead:.6e}  m")
print(f"  Analytical :  {analytical_dead:.6e}  m")
print(f"  Error      :  {err_dead:.4f} %")
print()
print("Live-load case (PL^3 / 3EI)")
print(f"  FEM        :  {d_live:.6e}  m")
print(f"  Analytical :  {analytical_live:.6e}  m")
print(f"  Error      :  {err_live:.4f} %")
print()
print("Combined case (analytical superposition)")
print(f"  FEM        :  {d_both:.6e}  m")
print(f"  Analytical :  {analytical_both:.6e}  m")
print(f"  Error      :  {err_both:.4f} %")
print()
print("Numerical linearity: d_both vs (d_dead + d_live)")
print(f"  d_dead + d_live :  {super_sum:.6e}  m")
print(f"  d_both          :  {d_both:.6e}  m")
print(f"  Error           :  {err_linearity:.4f} %")


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment in Jupyter to open the results viewer on the combined
# run.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * **Named patterns** — use ``g.loads.pattern("foo")`` to group
#   any mix of ``point``, ``line``, ``surface``, ``gravity``, etc.
#   Every later notebook that separates dead / live / seismic
#   reuses this structure.
# * **Pattern-filtered emission** — filter
#   ``fem.nodes.loads.by_kind("nodal")`` by ``rec.pattern`` to
#   decide what enters each ``ops.pattern "Plain"`` block. One
#   OpenSees pattern per apeGmsh pattern is the standard mapping.
# * **Distributed loading via ``g.loads.line``** — this replaces
#   the manual ``ops.eleLoad`` loop from slot 03 with
#   ``g.loads.line(target="beam", magnitude=w, direction=(...))``.
#   The ``reduction="tributary"`` default (not shown because it's
#   the default) means the line load is converted to per-node
#   point forces by tributary length — no ``eleLoad`` at all on
#   the OpenSees side.

# %%
g_ctx.__exit__(None, None, None)
