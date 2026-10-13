### CHANGED — R1b round-trip oracle: class B now requires declare-before-use (K2c, #1816)

The round-trip oracle (`tests/opensees/golden/test_roundtrip_oracle.py`)
called a replayed deck class B ("same lines, another order") on multiset
equality alone, which cannot see a fatal reorder: a replay that writes
`uniaxialMaterial Parallel 3 1 2` above `Steel01 1` aborts in OpenSees
(#1814). A row is now B only if the replay also declares every tag before a
line uses it. The check is static and reads the archive's own reference
metadata (`/opensees/decls` tags and the `decl_params` references), not the
Tcl grammar; node and element tag references are outside it. A new `live`
test runs the bridge and replay openseespy decks of the B rows in a
subprocess as the second net. `test_h5_decl_params::_params_frame` moves
from B to C, citing #1814. The ledger count does not change.
