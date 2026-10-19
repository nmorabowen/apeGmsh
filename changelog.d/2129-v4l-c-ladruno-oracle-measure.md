### ADDED — apeGmshViewer: a cross-language pairing oracle and a `.ladruno` read row in `npm run measure` (#2129)

A new test, `tests/test_app_ladruno_pairing_oracle.py`, runs both halves of
the ADR 0122 D3 pairing on the same files. Python writes a model with the
bridge and stamps copies of a fork-written `.ladruno` with
`stamp_provenance`. Then `node` runs the app's own reader and `pairResults`
on them, through `apeGmshViewer/scripts/pair_oracle.ts`. A file stamped
against the model pairs by `model_hash`. An unstamped file is unpairable. A
stamp from another model on the same mesh is a mismatch, never a `fem_hash`
pairing. Locally, the test skips with the reason when node 22.18 or later,
or the app's `npm ci`, is missing. The `app-tests` CI job runs it with
`APEGMSH_APP_ORACLE_REQUIRED=1`, so there a missing node or h5wasm fails the
job instead of skipping.

`npm run measure -- <stem>.h5` with a `<stem>.ladruno` beside the model now
prints the `.ladruno` read row. The row gives the index time, the time to
read one displacement step cold and warm, and the dataset's chunks and
filters. It gives the same for `<stem>.results.h5` of the same run when that
file exists. `MEASUREMENTS.md` records three runs on a real fork run. One step
costs the chunked, deflated `.ladruno` 3.5 to 5.8 times the contiguous native
read. The file is half the size.
