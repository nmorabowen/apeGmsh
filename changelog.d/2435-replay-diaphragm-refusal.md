### FIXED — replaying a legacy archive no longer re-emits a partly fixed `rigidDiaphragm` silently (#2435)

#2427 refuses, at emit, a `rigidDiaphragm` whose master is fixed on some
in-plane DOFs only under Transformation. Under that handler OpenSees
accumulates the master's total in-plane displacement at every step, so
the master's response is silently wrong.

H5 replay writes the archived rows directly and skipped that check. An
archive written before #2427 therefore replayed the wrong deck with no
error, through `OpenSeesModel.from_h5(...).build(...)`.

Replay now runs the same check, flat and staged, on the archived fix
masks, stage fixes and removals, and the recorded `constraints`
handlers. It raises `BridgeError` before anything is emitted. Archives
that are exact still replay unchanged: a master fixed only out of plane,
or a deck under Lagrange.
