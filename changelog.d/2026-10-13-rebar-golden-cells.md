### ADDED — a rebar cage in the golden emit corpus (X-f4, #1847)

`g.rebar` now has a committed deck oracle. The golden corpus gains a
`rebar_column_cage` cell family beside the stub grid: an `ACI318_seismic`
column cage (3x3 #6 bars, #3 ties @ 0.25 in a 0.4 x 0.4 x 1.5 box), placed
once `conformal` with `emit_elements=True` (262 `CorotTruss`) and once
`embedded` with cross-ties (366 `CorotTruss`, 400 `LadrunoEmbeddedRebar`).
Each placement pins a Tcl and a py deck. A change to bar placement, hooks,
`emit_elements` or the embedded coupling now fails `test_golden_corpus`
with a diff. Regenerate with `python -m tests.opensees.golden.regen`.
