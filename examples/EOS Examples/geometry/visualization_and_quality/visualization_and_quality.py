# %% [markdown]
# # Visualization and mesh quality
#
# **Companion to:** the `g.plot.*` API (`src/apeGmsh/viz/Plot.py`).
#
# ## Purpose
#
# `g.plot` is apeGmsh's built-in matplotlib 3D viewer.  It's a thin,
# *method-chainable* wrapper around `mpl_toolkits.mplot3d` that
# understands the apeGmsh data model directly — no need to extract
# nodes, assemble polygons, hunt down entity tags.  Every method
# returns `self`, so you can layer plots with a fluent chain.
#
# This notebook walks through **every public `g.plot.*` method** on a
# single shared model — a bracket plate with a circular hole plus a
# tilted stiffener — and then takes advantage of the same viewer to
# explore **mesh quality**: visual heatmaps, numerical stats, and a
# before/after comparison with one pass of `.refine()`.
#
# ## What the Plot API exposes
#
# | Method | Draws |
# |---|---|
# | `g.plot.geometry(...)` | BRep: points + curves + (optional) filled surfaces |
# | `g.plot.mesh(...)` | Finite-element mesh — nodes, edges, faces |
# | `g.plot.quality(...)` | Surface elements coloured by a Gmsh quality metric |
# | `g.plot.label_entities(...)` | BRep entity tags as floating text |
# | `g.plot.label_nodes(...)` | Mesh node tags |
# | `g.plot.label_elements(...)` | Mesh element tags |
# | `g.plot.physical_groups(...)` | BRep entities coloured by physical group |
# | `g.plot.physical_groups_mesh(...)` | Mesh elements coloured by physical group |
# | `g.plot.figsize((w, h))` | Set figure size before/during drawing |
# | `g.plot.show()` | Render and reset handles |
# | `g.plot.clear()` | Discard the current figure without rendering |
#
# Every drawing method accepts `show=False` so you can chain layers;
# the last call in the chain leaves `show=True` (the default) to
# flush the figure to the notebook.

# %% [markdown]
# ## 1. Imports
#
# The plotting API is behind the `[plot]` optional-dependency
# group (`pip install apeGmsh[plot]` pulls `matplotlib` + `scipy`).

# %%
import numpy as np
import pandas as pd

from apeGmsh import apeGmsh

# %% [markdown]
# ## 2. Build a bracket: plate with a hole + tilted stiffener
#
# A square plate with a circular hole (the hole is the canary for
# correct surface triangulation — a naive XY-projection approach
# would fill it).  A second plate is tilted 30° about Y and offset
# in +X to act as a stiffener; it exercises the arbitrary-orientation
# triangulation path.

# %%
g = apeGmsh(model_name="viz_and_quality", verbose=False)
g.begin()
geom = g.model.geometry

LC_PLATE = 0.09

# --- Plate with a hole in the XY plane -------------------------------
p1 = geom.add_point(0.0, 0.0, 0.0, lc=LC_PLATE)
p2 = geom.add_point(1.0, 0.0, 0.0, lc=LC_PLATE)
p3 = geom.add_point(1.0, 1.0, 0.0, lc=LC_PLATE)
p4 = geom.add_point(0.0, 1.0, 0.0, lc=LC_PLATE)
l_outer = [
    geom.add_line(p1, p2),
    geom.add_line(p2, p3),
    geom.add_line(p3, p4),
    geom.add_line(p4, p1),
]
loop_outer = geom.add_curve_loop(l_outer)

hole = geom.add_circle(0.5, 0.5, 0.0, radius=0.18)
loop_hole = geom.add_curve_loop([hole])

plate = geom.add_plane_surface([loop_outer, loop_hole])

# --- Tilted stiffener (30° about Y, offset in +X) --------------------
c, s = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
def rot_y(x, y, z, dx=1.5):
    return c * x + s * z + dx, y, -s * x + c * z + 0.4

p5 = geom.add_point(*rot_y(0.0, 0.0, 0.0), lc=LC_PLATE)
p6 = geom.add_point(*rot_y(0.8, 0.0, 0.0), lc=LC_PLATE)
p7 = geom.add_point(*rot_y(0.8, 1.0, 0.0), lc=LC_PLATE)
p8 = geom.add_point(*rot_y(0.0, 1.0, 0.0), lc=LC_PLATE)
l_tilt = [
    geom.add_line(p5, p6),
    geom.add_line(p6, p7),
    geom.add_line(p7, p8),
    geom.add_line(p8, p5),
]
loop_tilt = geom.add_curve_loop(l_tilt)
stiff = geom.add_plane_surface([loop_tilt])

g.model.sync()

# Physical groups — shows up in `physical_groups()` / `_mesh()` later
g.physical.add(1, l_outer,  name="plate_edge")
g.physical.add(1, [hole],   name="hole_edge")
g.physical.add(2, [plate],  name="plate")
g.physical.add(2, [stiff],  name="stiffener")

g.mesh.generation.generate(2)

fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# %% [markdown]
# ## 3. `geometry()` — BRep view
#
# `geometry()` draws the CAD itself: points (orange), curves (navy)
# and — if `show_surfaces=True` (the default) — filled surfaces
# projected onto their best-fit plane with hole cut-outs honoured.
# `label_tags=True` adds entity-tag annotations (`P3`, `C4`, `S1`).

# %%
g.plot.figsize((9, 7)).geometry(label_tags=True)

# %% [markdown]
# ## 4. `mesh()` — the triangular mesh
#
# Same viewer, now drawing the 2D elements.  The refined strip
# around the hole is visible as a denser ring of triangles.

# %%
g.plot.mesh()

# %% [markdown]
# ## 5. `physical_groups()` — BRep overlay coloured by PG
#
# Each physical group gets its own colour + legend entry.  `[S]`,
# `[C]`, `[P]` prefixes in the legend mark the group dimension
# (surface / curve / point).

# %%
g.plot.physical_groups()

# %% [markdown]
# ## 6. `physical_groups_mesh()` — same, but coloured by element
#
# Use this when you want to *see* which elements belong to which
# physical group after meshing — the mesh colouring is the unambiguous
# answer, independent of where the BRep entity happens to sit.

# %%
g.plot.physical_groups_mesh()

# %% [markdown]
# ## 7. `label_entities()` — tag overlay
#
# Drop entity tags on any combination of dims.  Useful when you're
# debugging a chain of boolean operations and need to know which
# `(dim, tag)` pair corresponds to which piece of the model.

# %%
g.plot.geometry(show=False).label_entities(dims=[0, 1, 2])

# %% [markdown]
# ## 8. `label_nodes()` and `label_elements()` — mesh annotations
#
# Both accept `stride=N` so you can thin out the labels when the mesh
# is dense.  Here we also demo **chaining**: mesh + element labels
# on a single axes.

# %%
(g.plot
   .figsize((10, 8))
   .mesh(show=False)
   .label_elements(dim=2, stride=4))

# %% [markdown]
# ## 9. Mesh quality — visual heatmap
#
# `quality()` colours each 2D element by a Gmsh metric.  The default
# `"minSICN"` (minimum Scaled Inverse Condition Number) ranges 0 → 1:
#
# * `1.0` — an equilateral triangle (the ideal)
# * `0.0` — a degenerate (zero-area) element
#
# In practice anything above ~0.3 is fine for a linear solve; below
# ~0.1 is a warning.  The colour bar makes bad elements jump out.

# %%
g.plot.figsize((10, 7)).quality(quality_name="minSICN")

# %% [markdown]
# ### 9b. Other quality metrics
#
# Gmsh exposes several metrics — swap `quality_name` to get a
# different view of the same mesh.  `"gamma"` is the
# inscribed-to-circumscribed-radius ratio (≈ triangle fatness);
# `"minSIGE"` is the minimum scaled inverse gradient error and
# penalises high-aspect-ratio elements more aggressively than
# `minSICN`.

# %%
g.plot.figsize((10, 7)).quality(quality_name="gamma")

# %%
g.plot.figsize((10, 7)).quality(quality_name="minSIGE")

# %% [markdown]
# ## 10. Mesh quality — numerical stats
#
# The visual heatmap is great for spotting problems; for an
# automated check you want numbers.  We pull the same metric
# straight from the Gmsh API and summarise it.

# %%
import gmsh

def quality_stats(metric: str = "minSICN") -> pd.Series:
    """Aggregate stats for a surface-element quality metric."""
    all_tags: list[int] = []
    for _, ent_tag in gmsh.model.getEntities(dim=2):
        etypes, etags_list, _ = gmsh.model.mesh.getElements(dim=2, tag=ent_tag)
        for etype, tags in zip(etypes, etags_list):
            # Triangles / quads only — skip anything lower-dim
            props = gmsh.model.mesh.getElementProperties(etype)
            if props[1] != 2:
                continue
            all_tags.extend(int(t) for t in tags)

    q = np.asarray(gmsh.model.mesh.getElementQualities(all_tags, qualityName=metric))
    return pd.Series({
        "metric":   metric,
        "n_elems":  len(q),
        "min":      float(q.min()),
        "p05":      float(np.percentile(q, 5)),
        "median":   float(np.median(q)),
        "mean":     float(q.mean()),
        "max":      float(q.max()),
        "< 0.3":    int((q < 0.3).sum()),
        "< 0.1":    int((q < 0.1).sum()),
    })

pd.concat(
    [quality_stats("minSICN"), quality_stats("gamma")],
    axis=1,
).T

# %% [markdown]
# ## 11. Before / after: one pass of `.refine()`
#
# `g.mesh.generation.refine()` uniformly splits every element once.
# That roughly quadruples element count and usually *improves*
# quality because the new elements are produced by bisection of
# existing edges — but the comparison below lets us verify rather
# than assume.

# %%
stats_before = quality_stats("minSICN")

g.mesh.generation.refine()

stats_after = quality_stats("minSICN")

# Side-by-side comparison
pd.concat({"before": stats_before, "after": stats_after}, axis=1)

# %% [markdown]
# ### Visual confirmation of the refined mesh

# %%
g.plot.figsize((10, 7)).quality(quality_name="minSICN")

# %% [markdown]
# ## What this unlocks
#
# * **One viewer, nine methods, fluent chaining.**  `g.plot` is the
#   full BRep + mesh + quality surface of apeGmsh in one namespace.
# * **Correct triangulation on any geometry.**  The plate-with-hole
#   renders with the hole visible; the tilted stiffener comes out
#   with correct topology regardless of its orientation — both rely
#   on the best-fit-plane + point-in-polygon triangulation shipped
#   in `viz/Plot.py`.
# * **Quality is both visual and numerical.**  Use `quality()` for
#   eyeballing; pull raw arrays via `gmsh.model.mesh.getElementQualities`
#   for CI thresholds and automated reports.
# * **Optional-dep boundary.**  The plotting machinery imports
#   lazily — `from apeGmsh import apeGmsh` works on a lean install;
#   `g.plot.*` only demands matplotlib/scipy when you call a drawing
#   method.

# %%
g.end()
