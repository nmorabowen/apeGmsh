# Future: Selection as Render Overlay

**Status:** future · **Estimated cost:** ~2d · **Depends on:** plan 04
**Discussed in:** [`../plotting-comparison.md`](../plotting-comparison.md)

## Goal

Render the active selection as a separate highlighted overlay actor, drawn on top of
all geometry without depth-test occlusion — instead of re-coloring the original
geometry in place.

## Why

- Today: `ColorManager` mutates cell colors on the original actor. Works for opaque
  models. **Breaks when picking through transparency** — the highlight is invisible
  behind a translucent contour layer.
- The Gauss-point z-fighting class of bug has the same root cause: trying to make
  selection visible by direct mutation of the underlying geometry.

## Mechanism

- On selection change: extract selected cells/nodes via `vtkExtractSelection` (or a
  cell-mask + `pv.UnstructuredGrid.extract_cells`).
- Render the extract as a bright-outlined, slightly-offset overlay actor.
- Add to a **non-composited renderer** layer (rendered after compositing, no depth
  test) so it's always visible.

## Files

- New: `src/apeGmsh/viewers/core/_selection_overlay.py`
- Modify: `src/apeGmsh/viewers/core/color_manager.py` — delegate highlight to the
  overlay; keep idle/hover state on the base actor.

## Risks

- Performance: selection-change rebuilds the overlay actor. For large selections
  (1000s of elements), throttle on a 50ms debounce.
- PyVista plotter doesn't expose a second renderer layer by default; reach through to
  `plotter.renderer.GetRenderWindow().AddRenderer(layer1)`.

## Done criteria

- Picking an element through a transparent contour highlights it visibly.
- `ColorManager` still handles idle/hover states on the base actor.
- Large selections (1000+ cells) don't stutter the UI.

## Out of scope

- Hardware-accelerated ID-pick render (`vtkHardwareSelector`). Different problem —
  picking accuracy vs picking visibility.
- Selection animation / pulse effect.
