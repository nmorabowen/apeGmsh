### ADDED — X2 numeric oracle: the data the old results path draws (X2-e, #1684)

`tests/viewers/golden_numeric/` (`oracle.npz` + `manifest.json`) pins what
the director / Qt `ResultsViewer` path draws for `Results.render` (mesh,
contour, deformed, reactions), `render_pack`, `assess(figures=True)` and
`export_animation` (default, `deform=`, `setup=` contour) on the X2-a plate:
the drawn coordinates, contour scalars and colour range, reaction arrows
(anchor and vector), the history line, and the frame schedule `[0, 3, 4]`.
The capture is path-agnostic: it reads the scene at
`pyvista` `BasePlotter.screenshot` and matplotlib `Figure.savefig`, so X2-b
and X2-d can run the same tests through the session path after
re-baselining the image goldens. Tests only; no `src/` change.
