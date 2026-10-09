### FIXED — numpy scalars reach openseespy as plain numbers on the live route (program slice R0-c, #1574; fixes #1352)

`ops.analyze(...)` handed primitive arguments to openseespy unchanged, and
openseespy parses only Python `int` / `float` / `str` / `bool`: an
`np.float32` area, an `np.int64` iteration cap or an `np.bool_` flag died
with `OpenSeesError` (`np.float64` passed only because it subclasses
`float`). #1351 fixed the py and Tcl decks; the same script failed live.
`LiveOpsEmitter` now binds the openseespy module behind `_CoercingOps`,
one proxy that passes every positional argument of every verb through
`emitter.base.plain_scalar` before the real call (numpy integers to
`int`, floats to `float`, `np.bool_` to `bool`, 0-d arrays to their
scalar). Calls whose arguments are already plain are forwarded without
rebuilding the tuple; `str`, `None`, lists, tuples and any other object
pass through unchanged, and containers are not walked. A numpy array with
elements, passed as one argument, raises `TypeError` before openseespy
registers anything. Capability probes (`getattr(self._ops, verb, None)`,
`hasattr`) read the module as before, and the backend verdict stays keyed
on the module itself. `LiveOpsEmitter.ops` now returns the proxy; the raw
module is `get_ops()`.
