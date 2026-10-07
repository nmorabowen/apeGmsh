### FIXED — a nan or inf argument is refused at emit, not written into the deck (#1356)

The py and Tcl deck formatters rendered `float("nan")` and `inf` as bare
`nan` / `inf` tokens: the openseespy deck died with `NameError` and Tcl's
`Tcl_GetDouble` rejected the token. `ops.py(...)` and `ops.tcl(...)` now raise
`BridgeError` naming the command and the argument (for example
`uniaxialMaterial Elastic: argument 2 is nan`). Plain floats, numpy scalars
and floats inside a list argument are all checked; no deck is left on disk.
