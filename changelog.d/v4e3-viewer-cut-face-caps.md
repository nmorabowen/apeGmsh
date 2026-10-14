### ADDED — apeGmshViewer: caps and the contour on the cut face (V4e3, #1904; ADR 0083 slice 3)

A section plane through a solid in the app no longer opens onto an empty
box. Each active plane now fills the cut with a cap, as the Qt results
viewer does (ADR 0083 S3). The cap is the slice of the drawn volumetric
cells (tet, hex, wedge, pyramid), with the deformation of the step shown
baked in. Cells the legend hides or the stage has not activated are left
out, and each cap is trimmed by the other active planes. With a contour on,
the cap carries the contour's field, interpolated to the cut, with the
contour's colormap, range and no-data grey. No data stays no data, never 0.
An element field drawn per cell colours the cap per cell, and the
node-located modes carry their extrapolated corner values. With no contour,
the cap is a flat fill. The caps are sliced again when a drag ends (a gizmo
or the panel slider) and when the step changes, never per drag tick, and
they are hidden mid-drag. The panel's **Contour on cut face** footer button
is live (on by default), and a mesh with no volumetric cell (shells, beams,
2-D) gets no cap and says so. The slice matches PyVista's slice of the same
tet, hex and wedge meshes through the Qt backend's `slice_layer` to 1e-9
(`apeGmshViewer/fixtures/slice.oracle.json`, written by
`scripts/make_viewer_results_fixtures.py --slice-oracle`). The deformed
contour scrub with six capped planes is measured in
`apeGmshViewer/MEASUREMENTS.md`.
