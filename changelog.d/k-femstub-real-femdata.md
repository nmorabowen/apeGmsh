### FIXED — bridge-only refusal covers composed files; R1b oracle ledger 35 → 33 (#1880, #1797)

`OpenSeesModel.from_h5(path, fem_root="/model")` on a composed results file
built from a bridge-only archive (a stamped `/model/meta`, no `/model/nodes`,
the `/opensees` zone at the file root) now raises `BridgeOnlyArchiveError`
up front instead of h5py's raw `KeyError: 'nodes'`. The guard looked for the
bridge zone only under `fem_root`; it now also looks at `opensees_root`,
where the reader reads it. Other files without a neutral zone keep the
error they raised before.

Tests: the two `test_h5_stages_writer` bridges now build on a real two-quad
`FEMData`, which makes them the same models as the `test_h5_stages_reader`
replay bridges. The R1b round-trip oracle replays each one once, and its two
class-S rows are gone (`N_WAIVED_ORACLE` 35 → 33). The golden emit corpus
stays on `FEMStub` by design, as its README now records.
