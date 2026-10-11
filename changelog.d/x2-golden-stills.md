### ADDED — golden stills and frame counts for the director path (X2-a, #1662)

`tests/viewers/golden_stills/` pins what the `ResultsDirector` path renders
today, as the oracle the X2 move onto `ResultsSession` must match:
`results.render` for each view (mesh, contour, deformed, reactions),
`results.render_pack`'s file set and stills, the `export_animation` frame
schedule, count and first and last frames, and the figure names of
`results.assess(figures=True)`. The fixture is a planar quad plate built in
the test. Stills are compared after 4x4 block averaging, within a per-channel
tolerance kept in `manifest.json`; the same test proves that a wrong step,
a wrong component or a missing deformation fails it. It runs in `suite`,
where Mesa makes a missing still a failure, not a skip.
