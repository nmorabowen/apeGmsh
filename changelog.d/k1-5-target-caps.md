### ADDED — `TargetCaps`, the `/opensees` solve stamp (opensees 2.24.0) and the `emitter-sniff` quirk rule (ADR 0114 D6, K1-5 #1462)

`src/apeGmsh/opensees/emitter/caps.py` is the typed home of what an emit target
can do. `TargetCaps` is a frozen dataclass with six fields (`archival`,
`supports_partitions`, `per_rank_fragments`, `suppress_analysis_chain_auto_emit`,
`model_reissue_purges`, `emit_stage_markers`); the `Emitter` Protocol declares
`caps: TargetCaps` as its one non-method name, and each of the five emitters
carries a value equal to the flags it declared before (`TclEmitter` purges on a
`model` re-issue, `LiveOpsEmitter` cannot consume partition brackets,
`H5Emitter` is archival). The old class attributes stay until the bridge reads
`emitter.caps` instead of probing them (K1-5 phase B); the verbs lock
(`tests/opensees/contract/test_verbs_lock.py`) now requires `caps` on every
emitter and keeps the Protocol's method count at 75.

The opensees schema moves to **2.24.0** (additive): `H5Emitter.set_solve_stamp`
takes one `SolveStamp` (`will_solve`, `solve_refusals`, `requires`) and `write`
stamps it as `/opensees@will_solve` (int8), `@solve_refusals` and `@requires`
(vlen str), together; `H5Model.solve_stamp()` reads them back and returns
`None` for a file without them (every file below 2.24.0). A partial or
malformed stamp raises `MalformedH5Error`. The attributes fold into
`model_hash`. The bridge does not call the hook yet; phase B stamps it from
`BuiltModel.emit` and echoes it on the H5 → H5 rewrite. The corpus gains
`tests/fixtures/schema_corpus/opensees_2.24.*`.

`scripts/check_quirks.py` gains `emitter-sniff`: a `type(e).__name__ ==
"XEmitter"` / `e.__class__.__name__` comparison or an `isinstance` /
`issubclass` against a class named `*Emitter`, anywhere under `src/apeGmsh/`
outside `opensees/emitter/`, is a finding. It flags the two sniffs in
`apesees.py` and the live guard in `_internal/compose.py` until phase B
replaces them with `caps.archival`.
