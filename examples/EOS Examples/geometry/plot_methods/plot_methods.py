# %% [markdown]
# # Plot smoke test — exercise every `g.plot.*` method
#
# Smoke-test notebook for the matplotlib `Plot` composite
# (`src/apeGmsh/viz/Plot.py`). Builds a small mixed-dimension
# model — specifically including a **tilted surface** — so that
# each drawing method has non-trivial input.
#
# Runs headless: `matplotlib.use("Agg")` + `savefig` into
# `tests_out/plot_methods/` rather than `plt.show()`.  Open the
# PNGs to eyeball the output.
#
# After a refactor of `Plot.py`, re-run this file and diff the
# PNGs (or just look at them) to catch regressions.

# %%
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

from apeGmsh import apeGmsh

OUT = Path(__file__).parent / "tests_out" / "plot_methods"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

print(f"output dir: {OUT}")

# %% [markdown]
# ## 1. Build a small mixed-dim model with three surfaces
#
# Three surfaces exercise three orientations:
#
# * **horizontal** — baseline, XY projection is the surface itself
# * **tilted 30° about Y** — XY projection is foreshortened but
#   still a valid quadrilateral, so `Delaunay(polygon[:, :2])`
#   returns correct *topology* and the 3D triangles come out fine
# * **vertical (YZ plane)** — XY projection collapses to a **line
#   segment**: this is where `Delaunay(polygon[:, :2])` genuinely
#   fails (degenerate input → QHull error or empty triangulation),
#   and the surface renders as nothing.  **This is the real
#   Delaunay-bug canary** that Phase 1 must fix.

# %%
g = apeGmsh(model_name="plot_smoke", verbose=False)
g.begin()

geom = g.model.geometry

# --- Surface 1: horizontal square in the XY plane (baseline) ---
p1 = geom.add_point(0.0, 0.0, 0.0, lc=0.25)
p2 = geom.add_point(1.0, 0.0, 0.0, lc=0.25)
p3 = geom.add_point(1.0, 1.0, 0.0, lc=0.25)
p4 = geom.add_point(0.0, 1.0, 0.0, lc=0.25)
l12 = geom.add_line(p1, p2)
l23 = geom.add_line(p2, p3)
l34 = geom.add_line(p3, p4)
l41 = geom.add_line(p4, p1)
loop_flat = geom.add_curve_loop([l12, l23, l34, l41])
surf_flat = geom.add_plane_surface([loop_flat])

# --- Surface 2: same square, tilted 30° about the Y axis ---
c, s = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
def rot_y(x, y, z, dx=1.5):
    """Rotate about Y, shift in +X so the two plates don't overlap."""
    return c * x + s * z + dx, y, -s * x + c * z + 0.5

p5 = geom.add_point(*rot_y(0.0, 0.0, 0.0), lc=0.25)
p6 = geom.add_point(*rot_y(1.0, 0.0, 0.0), lc=0.25)
p7 = geom.add_point(*rot_y(1.0, 1.0, 0.0), lc=0.25)
p8 = geom.add_point(*rot_y(0.0, 1.0, 0.0), lc=0.25)
l56 = geom.add_line(p5, p6)
l67 = geom.add_line(p6, p7)
l78 = geom.add_line(p7, p8)
l85 = geom.add_line(p8, p5)
loop_tilt = geom.add_curve_loop([l56, l67, l78, l85])
surf_tilt = geom.add_plane_surface([loop_tilt])

# --- Surface 3: vertical plate parallel to YZ (XY projection = line) ---
# Delaunay(polygon[:, :2]) gets colinear points → raises → the
# `except Exception: pass` fallback kicks in a centroid-fan
# triangulation, which renders correctly for *convex* shapes.
p11 = geom.add_point(-0.5, 0.0, 0.0, lc=0.25)
p12 = geom.add_point(-0.5, 1.0, 0.0, lc=0.25)
p13 = geom.add_point(-0.5, 1.0, 1.0, lc=0.25)
p14 = geom.add_point(-0.5, 0.0, 1.0, lc=0.25)
l_11_12 = geom.add_line(p11, p12)
l_12_13 = geom.add_line(p12, p13)
l_13_14 = geom.add_line(p13, p14)
l_14_11 = geom.add_line(p14, p11)
loop_vert = geom.add_curve_loop([l_11_12, l_12_13, l_13_14, l_14_11])
surf_vert = geom.add_plane_surface([loop_vert])

# --- Surface 4: plate WITH A HOLE (the real canary) ---
# The outer loop is a 1 × 1 square at z=0; the inner loop is a
# circle of radius 0.15 centred on (3.0, 0.5).  The current
# boundary-sampler chains *all* boundary curves (outer + inner)
# into one polygon before triangulating, so Delaunay fills the
# hole with triangles.  Pre-fix: hole appears solid.  Post-fix
# (honouring the inner loop): hole appears as a gap.
p15 = geom.add_point(2.5, 0.0, 0.0, lc=0.15)
p16 = geom.add_point(3.5, 0.0, 0.0, lc=0.15)
p17 = geom.add_point(3.5, 1.0, 0.0, lc=0.15)
p18 = geom.add_point(2.5, 1.0, 0.0, lc=0.15)
l_outer = [geom.add_line(a, b) for a, b in
           [(p15, p16), (p16, p17), (p17, p18), (p18, p15)]]
loop_outer_hole = geom.add_curve_loop(l_outer)

# Circular hole — a single `add_circle` returns one curve tag
circle = geom.add_circle(3.0, 0.5, 0.0, radius=0.15)
loop_hole = geom.add_curve_loop([circle])

surf_holed = geom.add_plane_surface([loop_outer_hole, loop_hole])

# --- A standalone line (not on any surface) to exercise dim=1 mesh path ---
p9  = geom.add_point(0.0, -0.5, 0.0, lc=0.25)
p10 = geom.add_point(1.0, -0.5, 0.0, lc=0.25)
l_standalone = geom.add_line(p9, p10)

g.model.sync()

# --- Physical groups (exercise all 4 dims where possible) ---
g.physical.add(0, [p1],                    name="corner_origin")
g.physical.add(1, [l_standalone],          name="beam_line")
g.physical.add(1, [l23, l67],              name="right_edges")
g.physical.add(2, [surf_flat],             name="plate_flat")
g.physical.add(2, [surf_tilt],             name="plate_tilted")
g.physical.add(2, [surf_vert],             name="plate_vertical")
g.physical.add(2, [surf_holed],            name="plate_holed")

g.mesh.generation.generate(2)
print(f"mesh: {g.mesh.queries.get_fem_data().info.n_nodes} nodes")

# %% [markdown]
# ## 2. Exercise each `Plot` method, save each figure as a PNG
#
# Every drawing method below is invoked with `show=False` so we
# can grab the figure handle (`g.plot._fig`) and `savefig` it.
# After Phase 2 of the refactor (`show=False` as default), the
# explicit kwargs become redundant — the test still works.

# %%
def _save(name: str) -> None:
    """Save the current Plot figure as a PNG, then clear handles."""
    g.plot.savefig(OUT / f"{name}.png", dpi=110)
    g.plot.clear()
    print(f"  wrote {name}.png")

# %%
# --- geometry ---
g.plot.figsize((9, 7)).geometry(label_tags=True)
_save("01_geometry")

# --- mesh ---
g.plot.mesh()
_save("02_mesh")

# --- quality heatmap (dim=2 only in current impl) ---
g.plot.quality()
_save("03_quality")

# --- label_entities ---
g.plot.geometry().label_entities(dims=[0, 1, 2])
_save("04_label_entities")

# --- label_nodes (strided so the plot stays readable) ---
g.plot.mesh().label_nodes(stride=10)
_save("05_label_nodes")

# --- label_elements ---
g.plot.mesh().label_elements(dim=2, stride=5)
_save("06_label_elements")

# --- physical_groups (BRep overlay) ---
g.plot.physical_groups()
_save("07_physical_groups")

# --- physical_groups_mesh (mesh overlay) ---
g.plot.physical_groups_mesh()
_save("08_physical_groups_mesh")

# --- Chained layered plot ---
(g.plot
   .figsize((10, 8))
   .geometry(show_surfaces=False)
   .mesh(alpha=0.4)
   .label_entities(dims=[2]))
_save("09_chained_geom_mesh_labels")

# %% [markdown]
# ## 3. Regression markers (visual checks to do by eye)
#
# Open each PNG in `tests_out/plot_methods/` and check:
#
# * **`01_geometry`**: **all four plates** render as filled
#   coloured patches.  The **plate with a hole** (far right) is
#   the real Delaunay-bug canary: pre-fix, the hole appears
#   filled in because the current implementation chains outer +
#   inner boundary curves into a single polygon and
#   Delaunay-triangulates the whole thing.  Post-fix, the hole
#   should be visible as a gap.  The vertical plate and tilted
#   plate both render correctly pre-fix (convex shapes trigger
#   either Delaunay or the centroid-fan fallback, both of which
#   give valid triangles on convex input).
# * **`02_mesh`**: triangular mesh visible on both plates +
#   line segment below.
# * **`03_quality`**: both plates coloured by `minSICN` with a
#   colorbar.
# * **`04_label_entities`** … **`06_label_elements`**: tags show
#   up near the expected locations; fonts legible.
# * **`07_physical_groups`**: five coloured groups, legend lists
#   `corner_origin / beam_line / right_edges / plate_flat /
#   plate_tilted`.
# * **`08_physical_groups_mesh`**: same five groups, but now
#   coloured *by mesh element* (not BRep).
# * **`09_chained_geom_mesh_labels`**: all three layers visible
#   in one plot (curves + mesh + surface tags).

# %%
g.end()
print("DONE — open PNGs in:", OUT)
