### CHANGED — `TargetCaps` replaces the emitter sniffs; the archive stamps its solve (opensees 2.24.0); `emitter-sniff` quirk rule (ADR 0114 D6/R4, K1-5 #1462)

`src/apeGmsh/opensees/emitter/caps.py` is the typed home of what an emit target
can do. `TargetCaps` is a frozen dataclass (`archival`, `supports_partitions`,
`per_rank_fragments`, `suppress_analysis_chain_auto_emit`,
`model_reissue_purges`, `emit_stage_markers`, `supports_stages`); the `Emitter`
Protocol declares `caps: TargetCaps` as its one non-method name, and each of
the five emitters carries a value equal to the flags it declared before
(`TclEmitter` purges on a `model` re-issue, `LiveOpsEmitter` consumes neither
partition brackets nor stages, `H5Emitter` is archival). The bridge reads
`emitter.caps.<field>` everywhere it used to `getattr`-probe an attribute,
set one behind `type: ignore[attr-defined]` (now `dataclasses.replace` on the
instance), or ask for the emitter's class: `BuiltModel.emit`'s
`type(emitter).__name__ == "H5Emitter"`, `_guard_mass_from_model`'s
`isinstance` and the staged replay's `isinstance(emitter, LiveOpsEmitter)` are
gone, and so are the class attributes `TclEmitter.model_reissue_purges`,
`LiveOpsEmitter.supports_partitions` and the deck emitters'
`_emit_stage_markers`. The verbs lock requires `caps` on every emitter and
keeps the Protocol's method count at 75.

The opensees schema moves to **2.24.0** (additive): every `apeSees.h5` archive
now carries `/opensees@will_solve` (`staged or any(Analysis)`), `@solve_mode`
(the archive's own partition mode), `@solve_refusals`, `@solve_refusals_flat`
and `@requires`. An archival emit no longer skips the solve-time gates
silently: it probes `validate_ladruno_up_solver`, `validate_serial_mumps` and
`validate_up_pressure_datum` as a solve would, once under the archive's own
partition mode and once flat (the maintainer's ruling on #1462, 2026-10-10),
and records the ids of those that refuse per mode, so a LadrunoUP model with a
solve and no system archives where `ops.tcl` refuses, and a partitioned one
(legal, it rides the auto-emitted general solver) records that its flat replay
is not. `@requires` is the union of the `VERBS.requires` tokens over the verbs
the program holds, derived by the writer (a fork element type rides the generic
`element` verb and does not reach it until K4). `OpenSeesModel.build('tcl' |
'py' | 'live')` emits flat and fails closed on the stored flat verdict,
`build('live')` refuses a `fork` requirement on a stock openseespy before
touching the domain, `to_h5` echoes the stamp hash-stable, and
`H5Model.solve_stamp()` / `OpenSeesModel.solve_stamp` read it (`None` below
2.24.0). The attributes fold into `model_hash`, so the 86 golden h5dumps gain
five attribute lines each; no deck byte changes. The corpus gains
`tests/fixtures/schema_corpus/opensees_2.24.*`.

`scripts/check_quirks.py` gains `emitter-sniff`: a `type(e).__name__ ==
"XEmitter"` / `e.__class__.__name__` comparison or an `isinstance` /
`issubclass` against a class named `*Emitter`, anywhere under `src/apeGmsh/`
outside `opensees/emitter/`, is a finding; the self-test proves it on the
commit that had the three sniffs, and the checkout is clean with no waiver.
