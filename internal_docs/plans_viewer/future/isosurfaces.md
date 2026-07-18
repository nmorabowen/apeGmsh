# Future: Isosurfaces & Isolines

**Status:** future · **Estimated cost:** ~2–3d · **Depends on:** plan 04
**Discussed in:** [`../diagram-catalog-comparison.md`](../diagram-catalog-comparison.md)

## Goal

Add an `isosurface` diagram kind that extracts equi-value surfaces from a 3D scalar
field, and (optionally) an `isolines` kind for banded contour lines on 2D surfaces.

## Why

- Today's "Contour" is surface coloring (paint the boundary with scalar values). True
  isosurfaces (equi-value surfaces *through* a 3D solid) are missing.
- For solid-element models, isosurfaces of stress / damage / temperature are a standard
  visualization.

## Naming caveat

ParaView calls isosurfaces "Contour" — confusing. **We must not reuse the name.** Stick
with `isosurface` for the 3D extraction, `contour` for our existing surface coloring,
`isolines` for banded 2D contour lines.

## Mechanism

- **Isosurfaces:** `vtkContourFilter` (despite the name; in 3D mode it produces 2D
  iso-surfaces from a 3D scalar field). Multiple iso-values produce a stack of nested
  surfaces.
- **Isolines:** `vtkBandedPolyDataContourFilter` for stepped/banded contour lines on 2D
  surfaces.

## Files

- New: `src/apeGmsh/viewers/diagrams/_isosurface.py`
- Modify: `_styles.py` — `IsosurfaceStyle` with `iso_values: list[float]`, colormap.
- Modify: `_kind_catalog.py` — expose the kind when 3D solid data exists.

## Risks

- Only meaningful for 3D solid models. Validate the mesh has 3D cells before exposing
  the kind in the catalog.
- Performance: extracting isosurfaces re-runs on every stage change. For very large
  meshes, warn or throttle.
- UI: editing a list of iso-values needs a small list editor (add / remove / drag).

## Done criteria

- New `isosurface` kind available when 3D solid data exists.
- Multiple iso-values render as separate translucent surfaces.
- Settings panel: editable iso-value list + colormap + opacity.
- Re-extracts correctly on stage change.

## Out of scope

- Adaptive iso-value selection (e.g., quartile-driven).
- Volume rendering (ray-marching through 3D scalar field) — different problem entirely.
- Auto-band-by-N (specify "10 evenly spaced bands" rather than explicit values).
