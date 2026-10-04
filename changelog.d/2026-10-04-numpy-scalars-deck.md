### FIXED — numpy scalars reach the py and Tcl decks as plain numbers (#1336)

A value computed with numpy (`E=np.sqrt(4.0) * 50.0`, an `np.int64`
iteration cap, an `np.float32` area) used to reach the emitted decks
through `repr`. Under numpy 2 that writes `np.float64(100.0)`, so the
openseespy deck died with `NameError: name 'np' is not defined` and the
Tcl deck carried a token OpenSees cannot parse. The py and Tcl value
formatters, and the py strategy-ladder rungs, now write any numpy scalar
or 0-d array as the plain Python number (`100.0`, `10`, `1`/`0` for
`np.bool_`). Subclasses of `int` and `float` are written as their base
value. A numpy array with more than one element, passed as a single
value, now raises `TypeError` instead of being written as `array([...])`.
