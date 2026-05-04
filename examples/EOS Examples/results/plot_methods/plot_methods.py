# %% [markdown]
# # Results plot smoke test — exercise every `results.plot.*` method
#
# Smoke-test notebook for the matplotlib `ResultsPlot` composite
# (`src/apeGmsh/results/plot/_plot.py`). Loads a pre-baked
# Results capture.h5 from the curriculum example and exercises every
# public plot method against it. Mirrors the shape of
# `examples/EOS Examples/geometry/plot_methods/plot_methods.py`.
#
# Runs headless: `matplotlib.use("Agg")` + `savefig` into
# `tests_out/plot_methods/`. Open the PNGs to eyeball the output.
#
# After a refactor of the plot module, re-run this file and diff the
# PNGs (or just look at them) to catch regressions.

# %%
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

from apeGmsh import Results

# Inputs: pre-baked Results captures from the curriculum / case-studies
# trees. We walk up the path until we find a checkout whose example
# outputs are populated — that way the smoke test runs both from the
# main repo and from a worktree (`.claude/worktrees/...`) where only
# script files are checked out.
HERE = Path(__file__).resolve().parent


def _find_repo() -> Path:
    """Return the apeGmsh root that has materialized example outputs."""
    candidates: list[Path] = []
    for parent in HERE.parents:
        candidates.append(parent)
    # Also fall back to the user-level main checkout.
    candidates.append(Path.home() / "Github" / "apeGmsh")
    for cand in candidates:
        probe = (
            cand / "examples" / "EOS Examples" / "curriculum"
            / "01-fundamentals" / "01_hello_plate" / "outputs" / "capture.h5"
        )
        if probe.exists():
            return cand
    raise FileNotFoundError(
        "Couldn't locate apeGmsh example outputs. Run the example "
        "notebooks at least once, or set REPO manually."
    )


REPO = _find_repo()
PLATE_H5 = (
    REPO / "examples" / "EOS Examples" / "curriculum"
    / "01-fundamentals" / "01_hello_plate" / "outputs" / "capture.h5"
)
FOOTING_H5 = (
    REPO / "examples" / "EOS Examples" / "case_studies"
    / "footing_flexible_slab_on_springs" / "outputs" / "capture.h5"
)
CHEVRON_H5 = (
    REPO / "examples" / "EOS Examples" / "case_studies"
    / "chevron_brace_retrofit_sine" / "outputs" / "capture.h5"
)

OUT = HERE / "tests_out" / "plot_methods"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

print(f"output dir: {OUT}")

# %% [markdown]
# ## 1. Open the Results and probe metadata

# %%
plate = Results.from_native(PLATE_H5)
plate_sid = plate.stages[0].id

print("plate stages :", [s.name for s in plate.stages])
print("plate n_nodes:", len(plate.fem.nodes.ids))
print("plate types  :", [t.name for t in plate.fem.elements.types])
print("plate nodal  :", plate.nodes.available_components(stage=plate_sid))

# %% [markdown]
# ## 2. mesh — undeformed surface
#
# Sanity check that the triangulation comes out clean.

# %%
ax = plate.plot.mesh()
ax.figure.savefig(OUT / "01_plate_mesh.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 3. contour — nodal displacement field
#
# Default styling, then a custom cmap + clim + no edges.

# %%
ax = plate.plot.contour("displacement_y", stage=plate_sid)
ax.figure.savefig(OUT / "02_plate_contour_uy.png", dpi=120)
plt.close(ax.figure)

ax = plate.plot.contour(
    "displacement_x", stage=plate_sid,
    cmap="coolwarm",
    clim=(-1.0e-3, 1.0e-3),
    edge_color=None,
)
ax.figure.savefig(OUT / "03_plate_contour_ux.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 4. deformed — warped shape, optional scalar overlay

# %%
ax = plate.plot.deformed(scale=500, stage=plate_sid)
ax.figure.savefig(OUT / "04_plate_deformed_plain.png", dpi=120)
plt.close(ax.figure)

ax = plate.plot.deformed(
    scale=500, stage=plate_sid,
    component="displacement_y", cmap="viridis",
)
ax.figure.savefig(OUT / "05_plate_deformed_contour.png", dpi=120)
plt.close(ax.figure)

ax = plate.plot.deformed(scale=500, stage=plate_sid, ghost=False)
ax.figure.savefig(OUT / "06_plate_deformed_noghost.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 5. history — node component vs time
#
# The plate is single-step (static), so the trace is degenerate
# (one sample). Exercises the read + 2-D plotting path nonetheless.

# %%
nid = int(plate.fem.nodes.ids[5])
ax = plate.plot.history(node=nid, component="displacement_x", stage=plate_sid)
ax.figure.savefig(OUT / "07_plate_history_node.png", dpi=120)
plt.close(ax.figure)

ax = plate.plot.history(
    point=tuple(plate.fem.nodes.coords[5]),
    component="displacement_y", stage=plate_sid,
    color="C1", linewidth=1.5,
)
ax.figure.savefig(OUT / "08_plate_history_point.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 6. Custom layout — pass your own axes
#
# `results.plot.*` accepts `ax=` so figures can be embedded in
# larger matplotlib layouts (side-by-side comparisons, paper figs).

# %%
fig = plt.figure(figsize=(13, 5))
ax_undef = fig.add_subplot(121, projection="3d")
ax_def = fig.add_subplot(122, projection="3d")

plate.plot.mesh(ax=ax_undef)
ax_undef.set_title("undeformed")

plate.plot.deformed(
    ax=ax_def, scale=500, stage=plate_sid,
    component="displacement_y",
)
ax_def.set_title("deformed × 500")

fig.tight_layout()
fig.savefig(OUT / "09_plate_side_by_side.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## 7. 3-D solid — boundary face extraction
#
# Re-runs the contour + deformed paths against a 3-D tet4 mesh to
# verify the boundary-face logic (faces shared by >1 element are
# dropped; only the outer hull is rendered).

# %%
if FOOTING_H5.exists():
    footing = Results.from_native(FOOTING_H5)
    sid = footing.stages[0].id
    print("footing types:", [t.name for t in footing.fem.elements.types])

    ax = footing.plot.mesh()
    ax.figure.savefig(OUT / "10_footing_mesh.png", dpi=120)
    plt.close(ax.figure)

    ax = footing.plot.contour("displacement_z", stage=sid)
    ax.figure.savefig(OUT / "11_footing_contour.png", dpi=120)
    plt.close(ax.figure)

    ax = footing.plot.deformed(
        scale=20, stage=sid, component="displacement_z",
    )
    ax.figure.savefig(OUT / "12_footing_deformed.png", dpi=120)
    plt.close(ax.figure)
else:
    print(f"footing capture not found at {FOOTING_H5} — skipping 3-D pass")

# %% [markdown]
# ## 8. reactions — force arrows at supports (Phase 2)
#
# Reaction moments are not rendered (no curved-arrow primitive in
# matplotlib); use the interactive viewer for moment glyphs.

# %%
ax = plate.plot.reactions(stage=plate_sid)
ax.figure.savefig(OUT / "13_plate_reactions.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 9. vector_glyph — generic vector field
#
# Arrows from any ``<prefix>_{x,y,z}`` component triple. Useful for
# displacement fields, velocity, applied forces — anything vector.

# %%
ax = plate.plot.vector_glyph("displacement", stage=plate_sid)
ax.figure.savefig(OUT / "14_plate_vector_disp.png", dpi=120)
plt.close(ax.figure)

# Anchor at the deformed shape so arrows track the warp.
ax = plate.plot.vector_glyph(
    "displacement", stage=plate_sid,
    deformed=True, deform_scale=500,
    color="C2",
)
ax.figure.savefig(OUT / "15_plate_vector_disp_def.png", dpi=120)
plt.close(ax.figure)

# %% [markdown]
# ## 10. loads — applied nodal force arrows (broker, no time)

# %%
try:
    ax = plate.plot.loads()
    ax.figure.savefig(OUT / "16_plate_loads.png", dpi=120)
    plt.close(ax.figure)
except Exception as exc:
    print(f"loads skipped: {exc}")

# %% [markdown]
# ## 11. line_force — beam internal-force diagrams (Phase 2)
#
# Renders the classic envelope-plus-fill diagram in the beam's local
# frame. Component-to-axis mapping is automatic
# (``shear_y`` / ``bending_moment_z`` → local y;
# ``axial_force`` → local z); override with ``axis="y"`` or
# ``axis="z"``.

# %%
if CHEVRON_H5.exists():
    chevron = Results.from_native(CHEVRON_H5)
    csid = chevron.stages[-1].id
    print("chevron line_stations:",
          chevron.elements.line_stations.available_components(stage=csid))

    for comp in ("axial_force", "shear_y", "bending_moment_z"):
        ax = chevron.plot.line_force(comp, stage=csid)
        ax.figure.savefig(OUT / f"17_chevron_lf_{comp}.png", dpi=120)
        plt.close(ax.figure)

    # reactions on a 3-D-frame model
    ax = chevron.plot.reactions(stage=csid, target_frac=0.10)
    ax.figure.savefig(OUT / "18_chevron_reactions.png", dpi=120)
    plt.close(ax.figure)
else:
    print(f"chevron capture not found at {CHEVRON_H5} — skipping line_force")

print("done — see PNGs in:", OUT)
