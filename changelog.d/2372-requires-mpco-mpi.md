### CHANGED — `/opensees@requires` names `"mpco"` and `"mpi"`; `build('live')` and the live net act on them (opensees 2.33.0, #2372, #2220)

Three corpus models carried a solve stamp claiming any backend could run
them (`requires=()`), but their bridge decks aborted on stock openseespy.
Two of them, and the golden cell's partitioned twin, declared a
`recorder mpco` (pip openseespy ships none) or a Mumps solver, or were
partitioned (a rank-gated deck that needs a multi-rank run). The stamp
could not say so, because these needs depend on argument values, which
the `VERBS` rows do not carry.

- The stamp writer (`H5Emitter.solve_stamp`) now derives two value
  tokens from its stores, global and staged: `"mpco"` for any
  `recorder mpco`, and `"mpi"` for a Mumps-family `system` or a
  partitioned `@solve_mode`. Opensees schema 2.33.0 (additive). Readers
  accept any token. A file below 2.33.0 keeps the `@requires` it was
  written with, and a `to_h5` rewrite writes today's tokens.
- `build('live')` refuses per token: `"fork"` and `"mpco"` need a Ladruno
  build, `"mpi"` always refuses in process, and so does an unknown token.
  A partitioned archive therefore no longer builds into a live domain;
  use `build('tcl')` / `build('py')` and run the deck under `mpiexec`.
- **`model_hash` changes once** (maintainer ruling on #2372): `@requires`
  stays hashed, so re-emitting a model with an MPCO recorder, a Mumps
  system or partitions stores a different `model_hash` than 2.32 did. A
  results file paired with the older archive shows a plain hash mismatch.
  Every other model hashes as before.
- The live net (`test_roundtrip_oracle_live.py`) reads the stamp the way
  `build('live')` does. A deck whose stamp claims nothing the backend
  lacks and which aborts now fails the net. A deck whose stamp names
  `"mpco"` must abort at its MPCO recorder on stock. A deck whose stamp
  names `"mpi"` skips by name. The `#2220` rows and the golden twin's
  stand-in are gone.
- `_param_stage_model` declares its Fill quad, so it solves; its nodes 5
  and 6 were orphans. The kitchen-sink fixture is singular on every build
  (it removes the Rock quad and the base x-fix). It stays a known failure
  under that reason until #2376 fixes it. The net now also checks that its
  replay fails at the same stage and increment.
