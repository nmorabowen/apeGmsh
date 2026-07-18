# Future: Render Quality & Lifecycle

**Status:** future · **Estimated cost:** ~2d · **Depends on:** none
**Discussed in:** [`../plotting-comparison.md`](../plotting-comparison.md)

## Goal

Enable depth peeling for transparency in all three viewers and add a
still-vs-interactive render lifecycle so camera drag uses cheaper rendering.

## Why

- Transparent overlapping diagrams (contour over deformed mesh, stacked section cuts,
  multiple isosurfaces) z-fight today. The Gauss-marker world-space-sphere workaround
  is partly compensating for missing depth peeling.
- With multiple heavy diagrams on, camera rotation in `results.viewer` stutters because
  every frame is full quality.

## Mechanism

- **Depth peeling:** one-line config in `viewer_window.py` after the plotter is
  created — `renderer.SetUseDepthPeeling(1); renderer.SetMaximumNumberOfPeels(4);
  renderer.SetOcclusionRatio(0.0)`.
- **Still vs interactive:** hook the VTK interactor's `LeftButtonPressEvent` /
  `LeftButtonReleaseEvent`. On press: disable anti-aliasing, optionally set a lower
  `DesiredUpdateRate`. On release: restore.

## Files

- `src/apeGmsh/viewers/ui/viewer_window.py` — depth peeling config in init.
- New: `src/apeGmsh/viewers/core/_render_lifecycle.py` — interactor hooks.

## Risks

- Depth peeling has a perf cost (~10-20%); expose as a Preferences toggle.
- Still/interactive can flicker on release; debounce ~100ms before quality restore.

## Done criteria

- Two transparent diagrams stacked render in correct depth order.
- Camera drag with 3 diagrams on remains responsive; quality restores on release.
- User-toggleable in Preferences.

## Out of scope

- LOD actor swap (full geometric decimation).
- FXAA / SSAO / shadows.
- OSPRay / ANARI ray tracing.
