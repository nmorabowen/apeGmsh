### CHANGED — `export_animation` re-steps a `ResultsSession`, no Qt window (X2-b, #1663)

`Results.export_animation` and the studio `animate` verb (`kind=history`,
`kind=yield`) no longer build a Qt `ResultsViewer` or a `ResultsDirector`.
They realize one session mesh view into an offscreen plotter and re-step it
for each captured step, so the verb runs headless with no window flash.
The frame schedule, `deform=`, `stage=`, `fps=`, `step_stride=` and
`window_size=` are unchanged. A failed frame now leaves no file at the path.

Breaking for `setup=`: the callback is now `setup(plotter, view)`, where
`view` is the session `MeshView` (for example
`view.contour = Contour("displacement_x")`). A callback written against the
old `director` argument raises `TypeError`. `camera=None` now follows the
still rule (`xy` for a planar model, `iso` otherwise), and `stage=None`
animates the last stage. The frames take the session look: no axes triad,
the session's mesh shading, and exactly `window_size` pixels.
