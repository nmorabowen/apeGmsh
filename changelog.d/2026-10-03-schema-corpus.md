### ADDED — the schema corpus: one real file per schema minor (ADR 0113 D8, #1303 PR-3)

`scripts/build_schema_corpus.py` checks each schema era of the neutral and
opensees zones out of git into a temporary worktree and runs an era-stable
generator against that era's frozen writer: the deterministic box through
`FEMData.to_h5` for the neutral zone, a one-bay elastic frame through
`apeSees(fem).h5()` for the opensees zone. Each era's own reader records a
semantic dump beside its file, and each opensees era stores its own
`build("tcl")` deck. The files, dumps, decks and `MANIFEST.json` are
committed under `tests/fixtures/schema_corpus/`.
`tests/opensees/h5/test_schema_corpus.py` holds ADR 0113 INV 5, 6, 7, 10
and 12: every file opens through today's `FEMData.from_h5` and
`OpenSeesModel.from_h5` and reproduces its era's dump outside the shim
ledger, opening never changes a file's bytes, a tampered copy fails the
`snapshot_id` check, every minor from floor to current has a file or a
recorded gap, and a floor set to the current minor turns the check red.
The corpus does not prove the opensees floor 2.11.0: every opensees-2.11
writer stamped neutral 2.7.0, below the neutral floor, so no
opensees-2.11 file opens. A strict xfail records this until the
maintainer raises the floor.
