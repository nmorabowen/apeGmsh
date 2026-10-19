### FIXED — a replayed deck keeps each argument's int/float type (#2194)

The bridge archives positional arguments as `float64`, and the replay read
every integral float back as an `int`. A deck rebuilt by
`OpenSeesModel.from_h5(...).build(...)` therefore wrote `element quad ...
1 PlaneStrain 1` where the bridge wrote `1.0`, and `200000000000` for
`E=200e9`. This affected materials, sections, beam integrations, time
series, dampings, element arguments, and the `-rayleigh` tail of a
stage-bound region.

The replay now rebuilds each declaration from `/opensees/decl_params`, runs
its emit on a recording emitter, and takes the int/float type of each slot.
The archived values are kept as they are. A stage region's `-rayleigh`
coefficients are read as floats, as a flat region's already were, and a
`-rayleigh` without four coefficients raises `MalformedH5Error`.

An archive without `/opensees/decl_params` (below opensees 2.26.0, or a
rewrite of one) replays as before, with integral floats read as ints. So
does a row whose argument count differs from its declaration's emit. When a
declaration's class is unknown to this build, or its rebuild fails, the
replay warns `ReplaySlotTypeWarning` and keeps the by-value types for that
declaration. A fanned-out element row whose FEM element is missing from the
archive's neutral zone raises `MalformedH5Error`.

In the R1b round-trip oracle, all 31 class-A ledger rows clear and
`N_WAIVED_ORACLE` goes from 32 to 1. The one row left is the class-S
`_broad_partitioned`, which is bridge-only by design. The live net now runs
every replayable corpus model.
