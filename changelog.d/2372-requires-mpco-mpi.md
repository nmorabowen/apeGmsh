### CHANGED — `/opensees@requires` names `"mpco"` and `"mpi"`; `build('live')` and the live net act on them (opensees 2.33.0, #2372, #2220)

Three corpus models carried a solve stamp claiming any backend could run
them (`requires=()`), but their bridge decks aborted on stock openseespy.
Two of them, and the golden cell's partitioned twin, declared a
`recorder mpco` (pip openseespy ships none) or a Mumps solver, or were
partitioned (a rank-gated deck that needs a multi-rank run). The stamp
could not say so, because these needs depend on argument values, which
the `VERBS` rows do not carry.

- The stamp writer (`H5Emitter.solve_stamp`) now derives two value
  tokens from its stores, global and staged. `"mpco"` covers any
  `recorder mpco`. `"mpi"` covers an MPI `system` (`Mumps` or
  `ParallelProfileSPD`, unless it carries the ADR 0027 runtime fallback)
  or a partitioned `@solve_mode`. `MPIDiagonal` needs nothing: both stock
  and the fork fall back to the serial solver. Opensees schema 2.33.0
  (additive). Readers accept any token. A file below 2.33.0 keeps the
  `@requires` it was written with, and a `to_h5` rewrite writes today's
  tokens.
- `build('live')` refuses per token of its **flat** replay
  (`OpenSeesModel.replay_requires`). `"fork"` and `"mpco"` need a Ladruno
  build. An unknown token refuses. `"mpi"` refuses only when the archived
  chain declares an MPI system. The message names that system and points
  to a serial one. A partitioned mode alone does not bind a flat replay,
  so a partitioned archive without an MPI system builds into a live domain
  exactly as before.
- **`model_hash` changes once** (maintainer ruling on #2372): `@requires`
  stays hashed, so re-emitting a model with an MPCO recorder, an MPI
  system or partitions stores a different `model_hash` than 2.32 did. A
  results file paired with the older archive shows a plain hash mismatch.
  Every other model hashes as before.
- The live net (`test_roundtrip_oracle_live.py`) reads the stamp the way
  `build('live')` does:
  - A deck that aborts, while its stamp claims nothing the backend lacks,
    now fails the net.
  - A deck whose stamp names `"mpco"` must abort at its MPCO recorder on
    stock.
  - A rank-gated bridge deck is not judged in one process. Its flat replay
    still runs whenever the partitioned mode was the only reason for
    `"mpi"`.
  - The `#2220` rows and the golden twin's stand-in are gone.
- `_param_stage_model` declares its Fill quad, so it solves; its nodes 5
  and 6 were orphans. The kitchen-sink fixture is singular on every build
  (it removes the Rock quad and the base x-fix). It stays a known failure
  under that reason until #2376 fixes it. The net now also checks that its
  replay fails at the same stage and increment.
