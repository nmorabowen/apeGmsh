### ADDED — golden emit corpus and canonical H5 dump (C2.1, #1255)

`tests/opensees/golden/` pins the bridge's emitted decks as the oracle for
pure moves. The grid is the 7 `FEMStub` factories x 5 emit modes (flat,
partitioned, staged, staged + partitioned, per-rank) x 3 outputs (Tcl, py,
Tcl with recorders): 105 cells in `MANIFEST.json`, 86 golden and 19 n/a,
each with its reason. Each golden cell also pins a canonical dump of the
same model's `ops.h5()` archive, written by the new
`scripts/h5_canonical_dump.py`: one line per group, dataset and attribute,
with dtype, shape and a byte-order-independent sha1, which hashes floats
at 12 significant digits. Deck floats are compared within a 1e-12 relative
tolerance, which absorbs last-ulp libm differences between platforms.
Integers and text stay exact. Only the wall-clock
`/meta@created_iso`, the release-bound `/meta@apeGmsh_version` and the
derived digest `/meta/lineage@model_hash` are masked. `test_golden_corpus.py` runs in `suite`. It never rewrites a golden,
and it fails if any workflow calls the regen entry point,
`python -m tests.opensees.golden.regen`.
