# Region Tools — Section Cuts & Scope Boxes vs ParaView

Not a plan — analysis. After [`plotting-comparison.md`](plotting-comparison.md)
(rendering layer) and [`diagram-catalog-comparison.md`](diagram-catalog-comparison.md)
(what fields get plotted), this is the third axis: tools that *restrict what's visible
or what's integrated* to a region of the model.

## TL;DR

| | apeGmsh | ParaView |
|---|---|---|
| **Section cuts as engineering-meaningful integration** (force/moment) | **Ahead** — `apeGmsh.cuts` is an STKO spec producer | Not a built-in; user composes filters |
| **Section cuts as pure visualization** | Static two-tone quad; ignores deformation | Interactive Slice filter; drag-the-plane; contour painted on the cut |
| **Clip planes** | BRep only; one axis-aligned plane; visual only | Any implicit function on any dataset; interactive widget |
| **Scope boxes** | **None** | Built-in via Clip filter + box implicit function + 3D box widget |
| **Cylinder / sphere clip** | None | Yes |
| **Multiple simultaneous clip regions** | No | Yes (chained filters) |

We're ahead where engineering meaning matters (the STKO integration story). We're behind
on visualization interactivity and on the basic "show me only what's in this region"
toolbox.

---

## What we have

### 1. `apeGmsh.cuts` — section cuts as a spec producer

The [`apeGmsh.cuts`](../../src/apeGmsh/cuts) subpackage builds `SectionCutDef` objects
from Gmsh physical groups, then converts them to STKO_to_python's `SectionCutSpec` for
downstream beam/shell/solid integration (force, moment, time history).

Per [`cuts/ARCHITECTURE.md`](../../src/apeGmsh/cuts/ARCHITECTURE.md):

> **apeGmsh's job is to produce `SectionCutSpec` objects from physical groups;
> STKO_to_python's job is to consume them.**

Capabilities:
- Plane builders ([`_planes.py`](../../src/apeGmsh/cuts/_planes.py)): horizontal,
  vertical, 3-point, SVD-fit, from-physical-group-surface.
- Bounding polygon from physical surface ([`_polygons.py`](../../src/apeGmsh/cuts/_polygons.py))
  — uses Cyrus–Beck + Sutherland–Hodgman in STKO.
- Sweeps ([`_sweeps.py`](../../src/apeGmsh/cuts/_sweeps.py)): one filter, N planes —
  useful for shear envelopes along a beam.
- FEM eid → OpenSees tag map ([`_tag_map.py`](../../src/apeGmsh/cuts/_tag_map.py))
  reading `model.h5`.
- Outputs `SectionCut(F, M, time, …)` — a full force/moment time history.

**This is the engineering-meaningful side.** Nothing in ParaView does this directly. A
ParaView user would have to compose Slice + Calculator + Integrate Variables manually,
and they'd lose the beam/shell/solid-aware integration kernel.

### 2. `SectionCutDiagram` — section cuts as visualization

[`viewers/diagrams/_section_cut.py`](../../src/apeGmsh/viewers/diagrams/_section_cut.py)
renders a `SectionCutDef` as a static quad in the viewer:

- Two-tone quad (kept side vs discarded side, via front/back face property).
- Small fixed-fraction normal arrow at the centroid → orbit-edge-on fallback.
- Quad extent: `cut.bounding_polygon` if set, else AABB of filter elements.
- **Static** — ignores deformation (decision D6); `update_to_step` is a no-op.
- **Tag-map-driven** — requires `FemToOpsTagMap` to translate OpenSees tags → FEM eids.

It's a faithful *visualization of the spec*. It is **not** a live interactive cutting
plane.

### 3. `ClipPlaneOverlay` — visual clip on BRep only

[`viewers/overlays/clip_plane_overlay.py`](../../src/apeGmsh/viewers/overlays/clip_plane_overlay.py)
is a separate feature, model.viewer only:

- One `vtkPlane` attached to the mappers of every BRep dim actor in `EntityRegistry`.
- VTK clipping discards fragments on the positive side; flipping the normal reverses
  which half is hidden.
- Axis-aligned only (X / Y / Z) per
  [`clip_plane_overlay.py:35-39`](../../src/apeGmsh/viewers/overlays/clip_plane_overlay.py).
- Geometry untouched — Gmsh state stays clean.
- UI in [`_clip_plane_panel.py`](../../src/apeGmsh/viewers/ui/_clip_plane_panel.py) and
  [`clipping_tab.py`](../../src/apeGmsh/viewers/ui/clipping_tab.py).

**Notably absent from `mesh.viewer` and `results.viewer`.** If you want to clip a
results scene you can't.

### 4. What we have *no* version of

- **Scope box** — a 3D rectangular region used to filter visible cells.
- **Field-driven cutting plane** — drag a plane through a results scene, see the
  contour painted on the cut.
- **Multiple simultaneous clip planes** (intersection of half-spaces).
- **Cylinder, sphere, or arbitrary-implicit clip.**
- **Box, cylinder, sphere widgets for interactive region edit.**

---

## What ParaView has

ParaView's region toolbox is built on three composable primitives:

### Clip filter (`vtkClipDataSet`)

- Takes an **implicit function**: plane, box, sphere, cylinder, scalar threshold, or any
  user-defined function.
- Output: the inside (or outside) of the implicit region, as a 3D subset of the input
  dataset.
- Interactive: comes with `vtkBoxWidget` / `vtkSphereWidget` / `vtkPlaneWidget` for
  drag-the-handles editing.
- **This is the scope-box equivalent** when you pass a box implicit function.

### Slice filter (`vtkCutter`)

- Same implicit-function input.
- Output: a 2D surface where the function value = 0 (the slice).
- This is the *field-driven cutting plane*: drag the plane, see the contour rendered on
  the cut surface.

### Extract Subset / Threshold

- `vtkExtractGeometry`: extract cells whose centroid (or bounds) is inside an implicit
  region. Equivalent to Clip but preserves cell integrity (no cells split at the
  boundary).
- `vtkThreshold`: extract cells where a scalar array is in a range. A *field-driven*
  scope, not a geometric one.

### Widget machinery

- `vtkBoxWidget`, `vtkSphereWidget`, `vtkPlaneWidget` — VTK widgets with grab-handle
  interaction, rendered in the view.
- Bound to the Clip / Slice filter's implicit function via property links.

The pattern: **filters define operations, widgets define editing UI, both share an
implicit function as the contract.**

---

## Side-by-side per concept

### Section cuts (with force/moment integration)

| | apeGmsh | ParaView |
|---|---|---|
| Mechanism | `SectionCutDef` → STKO integration kernel | Slice + Calculator + Integrate Variables (manual composition) |
| Engineering correctness | Beam / shell / solid aware; shared-edge resolution; three validators | Generic — user supplies the math |
| Workflow | One method call per cut | Multi-step pipeline |
| Lock-in | Coupled to STKO_to_python | None |

**Verdict:** We win. Don't break this.

### Section cuts (visualization only)

| | apeGmsh | ParaView |
|---|---|---|
| Display | Two-tone quad + normal arrow | Slice output (a 2D mesh) painted with whatever array the user picks |
| Interactivity | Edit via dialog ([`registry.replace`](../../src/apeGmsh/viewers/diagrams/_registry.py)) | Drag the plane widget in 3D |
| Field-driven? | No — geometry only | Yes — contour painted on the cut |
| Deformation-aware | No (D6) | Yes (slices the deformed mesh) |

**Verdict:** ParaView's slice is more interactive and immediately useful for "show me
σ_vm on this cut plane." Our `SectionCutDiagram` is the right *visualization of the
engineering object*, but it doesn't give you the "what does the stress field look like
*on* the cut" view.

### Clip planes

| | apeGmsh | ParaView |
|---|---|---|
| Where | `model.viewer` only | All views, all datasets |
| Function | Axis-aligned plane (X/Y/Z) | Any implicit function |
| Interactivity | Slider + axis dropdown | Drag handles in 3D |
| Multiple planes? | No | Yes (chain filters) |

**Verdict:** Real gap. `results.viewer` and `mesh.viewer` have no clip story at all.

### Scope boxes

| | apeGmsh | ParaView |
|---|---|---|
| Built-in? | No | Yes — Clip with box implicit function + `vtkBoxWidget` |
| Use cases we'd want | "Show only the column at grid intersection B-2", "Inspect a 5 m³ region near the connection" | Same |

**Verdict:** Pure gap. Scope boxes are a standard CAD/BIM/FEM tool (Revit calls them
"Scope Boxes" explicitly).

---

## What's worth doing

### Tier 1 — high value, scoped

1. **Scope box for `results.viewer` and `mesh.viewer`**
   - 3D box widget (PyVista has `add_box_widget`); hide cells whose centroid is outside.
   - Cheap implementation: cell-mask on the FEMScene grid.
   - Single box for v1; multiple boxes (intersection / union) can wait.
   - Estimated cost: **~3 days**.

2. **Field-driven cutting plane for `results.viewer`**
   - PyVista `plotter.add_mesh_clip_plane(mesh, ...)` does much of this for free; or
     wrap `vtkCutter` + `vtkPlaneWidget` directly.
   - Reuses the existing contour color/LUT — just paints on the cut surface.
   - **Don't conflate with `SectionCutDiagram`** — different concept (visualization-only
     vs engineering-integration).
   - Estimated cost: **~2 days**.

3. **Extend `ClipPlaneOverlay` to `mesh.viewer` and `results.viewer`**
   - Same machinery, different actor source. Generalize the `EntityRegistry` dependency.
   - Add support for non-axis-aligned plane orientation (user clicks "set normal" on a
     face or types a vector).
   - Estimated cost: **~2 days**.

Total Tier 1 cost: **~1 week** for three high-value visualization tools we're flat-out
missing.

### Tier 2 — power-user

4. **Multiple simultaneous clip planes** (intersection of half-spaces). Three clip
   planes are enough to clip to any convex region. Cost: ~1 day on top of #3.
5. **Cylinder clip** — useful for cylindrical structures (e.g. inspect a column).
   PyVista has cylinder widget. Cost: ~1 day.
6. **Interactive editing of `SectionCutDiagram` plane** — drag handles in 3D to move
   the cut plane, with live re-emission of the STKO `SectionCutSpec` and re-execution
   of the integration. This is the natural marriage of our engineering-meaningful cut
   with ParaView-style interactivity. Cost: **~1 week** — non-trivial because it
   requires re-running STKO on every drag.

### Tier 3 — defer

7. Sphere clip.
8. Arbitrary implicit function clip.
9. Box-widget multiple-region selection (e.g. "show me everything inside box A AND box B").

---

## Naming discipline

We should be careful not to conflate these:

- **Section cut** = an engineering operation that integrates forces/moments over a
  plane (current `SectionCutDiagram` + `apeGmsh.cuts`). Has a name in structural
  engineering. Keep this name for the engineering thing.
- **Slice plane** / **cutting plane** = pure visualization, painted contour on the cut.
  Don't reuse the name "section cut."
- **Scope box** = a 3D box that hides everything outside. Borrow Revit's term.
- **Clip plane** = a half-space that hides everything on one side. Already in use for
  `ClipPlaneOverlay`.

ParaView itself muddles "Clip" (3D subset) and "Slice" (2D cut surface) but the
distinction matters for users.

---

## Recommendations as `future/` docs

Three candidates:

- **`future/scope-box.md`** — 3D box widget filtering cell visibility. Probably the
  highest-leverage single addition; standard tool from CAD/BIM. **Tier 1.**
- **`future/slice-plane.md`** — field-driven cutting plane for results. Cheap, immediate
  user value, doesn't disturb `SectionCutDiagram`. **Tier 1.**
- **`future/clip-plane-everywhere.md`** — generalize `ClipPlaneOverlay` to all three
  viewers, with non-axis-aligned planes and multiple simultaneous clips. **Tier 1.**
- **`future/interactive-section-cut.md`** — drag-handles editing of `SectionCutDef` with
  live STKO re-integration. **Tier 2** — large because of the integration cost.

The first three are roughly a week total and would significantly close the gap on
visualization interactivity. The fourth is a power-user upgrade to our existing
engineering-cut feature.

---

## Honest summary

We are not behind ParaView on the *engineering* side of section cuts — `apeGmsh.cuts` +
STKO is sophisticated. We are behind on *interactivity* and on the basic "filter to a
region" toolbox:

- No scope box.
- No interactive cutting plane (with field painted on the cut).
- Clip plane only exists in `model.viewer`, not the other two.

Closing those gaps is ~1 week of work spread over three small features, each shippable
independently. Best to file them as separate `future/` docs and pick them up after the
must-ship list (08, 04, 01, 05, 06) lands.
