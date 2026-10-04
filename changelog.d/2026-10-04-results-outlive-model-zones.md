### CHANGED — a results file outlives its embedded model zones (ADR 0113 D9, #1303 PR-4)

`NativeReader` no longer refuses a `results.h5` whose embedded `/model`
(neutral) or `/opensees` zone is below its compatibility floor. The file
opens with `/stages` read-only: each such zone is listed in the new
`NativeReader.unavailable_zones` (zone id to the refusal text), one
`UserWarning` per flagged zone says so at open, and `fem()` /
`opensees_model()` refuse with the same text instead of reading the
zone. Nothing is rewritten. An embedded zone *newer* than the reader
still refuses the whole file (ADR 0023 INV-4), as does the results zone
itself in either direction.

`Results.from_native` opens such a file too. With `fem=` supplied the
embedded `/model` is never read (`_resolve_fem` returns the supplied
snapshot before asking the reader, on every constructor); without it the
file still opens and `Results.fem` raises the reader's refusal with a
`fem=` hint until `results.bind(fem)` supplies a snapshot. The embedded
`/opensees` is never read by `from_native`, whose `model=` is required;
`OpenSeesModel.from_h5` on the flagged file refuses as before, so the
model comes from a sidecar archive.

Two tests join the floor: `tests/test_app_floor_drift.py` parses the
app's `ZONE_FLOOR` table in `apeGmshViewer/src/reader/read.ts` as text
and holds it equal to Python's `reader_floor` constants without importing
across the ADR 0112 D4 wall (INV-4 of ADR 0113), and the #1300
results-twin test is now the named INV-9 (no laundering) test: the `ndm`
`NativeWriter.write_opensees_from` forwards under the restamped
`/model/meta` is the value the source's own reader resolves through its
shim, never the raw attribute. `tests/results/test_results_outlive_model_zones.py`
holds INV-11 at the facade: both zones, with and without `fem=`.
