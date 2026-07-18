# Future: Field-Driven Slice Plane

**Status:** future · **Estimated cost:** ~2d · **Depends on:** plan 04
**Discussed in:** [`../region-tools-comparison.md`](../region-tools-comparison.md)

## Goal

A drag-the-plane cutting tool that produces a 2D surface through the model with the
contour field painted on it. **Distinct from `SectionCutDiagram`**, which is
engineering-meaningful (STKO integration). This one is visualization-only.

## Why

- Today: `SectionCutDiagram` is static and OpenSees-tag-driven. No way to interactively
  explore "what does σ_vm look like on a plane through this region?"
- ParaView's `Slice` filter is exactly this — drag a plane, see the contour painted on
  the cut surface.

## Naming discipline

Two distinct concepts, easily confused:

- **Section cut** (`SectionCutDiagram`): engineering integration. Produces F/M time
  histories via STKO. Keep this name.
- **Slice plane** (this doc): visualization. Produces a 2D contour on a cut. **Don't
  reuse "section cut" for this.**

## Mechanism

- PyVista's `plotter.add_mesh_clip_plane(mesh, ...)` or direct `vtkCutter` + `vtkPlane`
  + `vtkPlaneWidget`.
- Output is fed to a mapper colored by whichever scalar the parent Contour exposes (or
  via the LUT manager from plan 06).
- Plane widget for interactive editing.

## Files

- New: `src/apeGmsh/viewers/diagrams/_slice_plane.py` — a new diagram kind that lives
  alongside Contour.
- Modify: `_styles.py` — add `SlicePlaneStyle`.
- Modify: `_kind_catalog.py` — add `slice_plane` kind.

## Risks

- **Naming confusion with section_cut.** Use explicit UI labels: "Slice Plane (visual)"
  vs "Section Cut (forces & moments)." Possibly add tooltip text disambiguating.
- **Widget vs camera controls** — plane widget interaction with PyVista's camera may
  fight. Disable camera during widget drag.

## Done criteria

- New `slice_plane` kind available; drag handles work in 3D.
- Contour color follows whichever scalar is on the substrate (or user-selectable).
- Survives stage changes; updates the cut surface live.

## Out of scope

- Multiple simultaneous slice planes (trivial to add later).
- Sphere / cylinder slice surfaces.
- Live integration of slice (that's `interactive-section-cut.md`).
