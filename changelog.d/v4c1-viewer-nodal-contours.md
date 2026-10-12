### ADDED — apeGmshViewer: nodal contours (V4c1, #1756)

On a results stage, the legend's **contour** mode colours the mesh by a
nodal field at the step shown: a field picker (the components the file
names, the magnitude of a translational vector, `[k]` for an unnamed one),
jet by default with viridis and coolwarm, and a colour range that defaults
to the field's whole history on the stage (read in the background in main;
until it is done the legend says "range: this step (history scan n%)"),
with "this step" and a user-fixed range as the other modes. A node the
field does not record is drawn in a named no-data grey, never as zero and
never blended from a neighbour. A nodal component whose value columns
disagree with `_ids` (a partial write) is refused on its own, with a
banner, instead of refusing the file. `apeGmshViewer/MEASUREMENTS.md` has
the contour scrub rows (worst 246 steps/s, peak 398 MB on the V4a files),
and `scripts/make_viewer_results_fixtures.py --contour` writes the new
contour fixtures.
