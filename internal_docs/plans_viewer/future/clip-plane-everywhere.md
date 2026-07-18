# Future: Clip Planes Everywhere

**Status:** future · **Estimated cost:** ~2d · **Depends on:** none
**Discussed in:** [`../region-tools-comparison.md`](../region-tools-comparison.md)

## Goal

Generalize `ClipPlaneOverlay` (today: `model.viewer` only, single axis-aligned plane)
to all three viewers, and allow non-axis-aligned plane normals + up to 3 simultaneous
clips.

## Why

- `mesh.viewer` and `results.viewer` have no clip story today.
- Inspecting the interior of a 3D solid model in `results.viewer` is impossible without
  clipping.
- Axis-aligned is limiting — real workflows want arbitrary normals (e.g., "clip
  perpendicular to this brace").

## Mechanism

- Generalize `EntityRegistry` dependency in `clip_plane_overlay.py`; let the overlay
  accept any iterable of actors (works for FEMScene grid, BRep actors, diagram actors).
- VTK mapper supports multiple clip planes natively via `vtkMapper::AddClippingPlane`.
- Non-axis-aligned normals via:
  - explicit XYZ spinners, OR
  - "set from active face" button (use selection-system face pick), OR
  - 3D plane widget for drag-handles editing.

## Files

- Modify: `src/apeGmsh/viewers/overlays/clip_plane_overlay.py` — accept generic actor
  source; support multiple planes.
- Modify: `src/apeGmsh/viewers/core/clipping_controller.py` — multi-plane state.
- Modify: `src/apeGmsh/viewers/mesh_viewer.py`, `results_viewer.py` — wire the overlay.
- Modify: `src/apeGmsh/viewers/ui/_clip_plane_panel.py` — arbitrary-normal UI, planes list.

## Risks

- Multi-plane intersection: verify VTK mapper behaves correctly with 2+ planes (it
  intersects half-spaces, which is what we want).
- Persistence: clip-plane state per viewer in session JSON. Schema bump.

## Done criteria

- Clip plane works in all three viewers.
- Up to 3 simultaneous planes; their intersection is what's visible.
- Non-axis-aligned normals editable.
- State persists across launches.

## Out of scope

- Cylinder / sphere clip (separate widgets, different doc).
- Field-driven clip (clip by scalar threshold — that's `vtkThreshold`, different
  concept).
- Clip in `model.viewer` going beyond what we already have.
