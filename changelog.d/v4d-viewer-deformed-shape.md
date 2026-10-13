### ADDED — apeGmshViewer: deformed shape (V4d, #1765)

On a results stage, the legend's **deformed** toggle moves every drawn node
to `x + scale · u`, with `u` the file's nodal `displacement_x/y/z` at the
step shown (rotations are not used), through the node id of each drawn
vertex, never a match by position. The scale defaults to 1 (true scale) and
the legend states any other; **reference ghost** draws the undeformed mesh
behind it at 0.3 opacity, as the Qt viewer does. Beams and trusses stay
straight chords, and the outline edges follow their own nodes. The deformed
mesh takes contour colours, stage activation
and the time scrubber, and a click picks what is drawn where it lands. A
drawn node with no displacement (not recorded, or NaN) is drawn undeformed
and marked, and the banner states how many (`partial`); it is never moved by
a displacement of 0. The deformed shape is refused, with the reason, when a
displacement axis was refused as a partial write, when the components name
no axis, or when `displacement_z` is missing from a model that is not
declared 2-D (`/meta/ndm` 2): it is never drawn from the surviving axes.
`apeGmshViewer/MEASUREMENTS.md` has the deformed contour scrub rows (worst
209 steps/s, the reviewer's run at 47 % CPU, peak 454 MB on the V4a files), and
`scripts/make_viewer_results_fixtures.py --deform` writes a 2-D fixture.
