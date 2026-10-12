### CHANGED — `export_animation` re-steps a `ResultsSession`, no Qt window (X2-b, #1663)

`Results.export_animation` and the studio `animate` verb (`kind=history`,
`kind=yield`) no longer build a Qt `ResultsViewer` or a `ResultsDirector`.
They realize one session mesh view into an offscreen plotter and re-step it
for each captured step, so the verb runs headless with no window flash.
The frame schedule, `deform=`, `fps=`, `step_stride=` and `window_size=`
are unchanged, and `stage=None` resolves as before (the scoped stage, else
the only stage, else an error naming `stage=`). A failed frame now leaves
no file at the path.

Changed (behaviour), per the maintainer ruling of 2026-10-11 (#1663):

- `setup=` is now `setup(plotter, view)`, where `view` is the session
  `MeshView` (for example `view.contour = Contour("displacement_x")`). A
  callback written against the old `director` argument raises `TypeError`
  before any file is written.
- `stage="__all__"` (the old viewer's combined stage) now raises
  `ValueError`. Animate one stage at a time with `stage=<id or name>`.

Also: `camera=None` now follows the still rule (`xy` for a planar model,
`iso` otherwise). The frames take the session look: no axes triad, the
session's mesh shading, and exactly `window_size` pixels.
