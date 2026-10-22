### FIXED — replaying a legacy archive no longer re-emits a partly fixed `rigidDiaphragm` silently (#2435)

#2427 refuses, at emit, a `rigidDiaphragm` whose master is fixed on some
in-plane DOFs only under Transformation. Under that handler OpenSees
accumulates the master's total in-plane displacement at every step, so
the master's response is silently wrong.

H5 replay writes the archived rows directly and skipped that check. An
archive written before #2427 therefore replayed the wrong deck with no
error, through `OpenSeesModel.from_h5(...).build(...)`.

`OpenSeesModel.build` now runs the same check, before any emitter
exists, so a live build does not wipe the domain first. It reads the
archive's rows as the emit reads the model: fix masks, stage fixes and
removals, pattern `sp` rows (counted as varying), and the recorded
`constraints` handlers. An archive the bridge wrote and accepted
therefore always replays.

An `equalDOF` into a `rigidDiaphragm` node now counts as a chain for the
#2396 rewrite. It keeps its declared form and warns
`PartlyFixedTieWarning`, rather than fixing the diaphragm master, which
would leave that master partly fixed under the diaphragm. Archives
that are exact still replay unchanged: a master fixed only out of plane,
or a deck under Lagrange.
