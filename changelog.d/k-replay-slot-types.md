### FIXED — a replayed deck keeps the int/float type of material, section, element and stage-region arguments (#2194)

The bridge archives positional arguments as `float64`, and the replay read
every integral float back as an `int`. A deck rebuilt by
`OpenSeesModel.from_h5(...).build(...)` therefore wrote `element quad ...
1 PlaneStrain 1` where the bridge wrote `1.0`, and `200000000000` for
`E=200e9`.

Three read sites now keep the bridge's types:

- the tagged stores: materials, sections, beam integrations, time series
  and dampings;
- the element arguments;
- the `-rayleigh` tail of a stage-bound region.

For the first two, the replay rebuilds each declaration from
`/opensees/decl_params`, runs its emit on a recording emitter, and takes
the int/float type of each slot. The archived values are kept as they are.
The stage-region tail is read as four floats, as a flat region's already
was. The bridge always writes four coefficients, so a `-rayleigh` with
fewer raises `MalformedH5Error`.

Patterns and recorders are not covered yet. Their integral floats, such as
an `H5DRM` argument or a recorder's `-dT 1.0`, still replay as ints.

An archive without `/opensees/decl_params` (below opensees 2.26.0, or a
rewrite of one) still reads integral floats as ints at the first two sites.
Its stage-region `-rayleigh` tail now replays as floats, the bridge's text.
A row whose argument count differs from its declaration's emit is left as
read. When a declaration's class is unknown to this build, or its rebuild
fails, the replay warns `ReplaySlotTypeWarning` and keeps the by-value
types for that declaration. A fanned-out element row whose FEM element is
missing from the archive's neutral zone raises `MalformedH5Error`.

In the R1b round-trip oracle, all 31 class-A ledger rows clear and
`N_WAIVED_ORACLE` goes from 32 to 1. The one row left is the class-S
`_broad_partitioned`, which is bridge-only by design. The live net now runs
every replayable corpus model.
