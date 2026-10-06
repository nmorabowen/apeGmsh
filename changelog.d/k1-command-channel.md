### ADDED — `Emitter.command()`, the last Protocol method, and the H5 `_refuse` / `_ledger` helpers (ADR 0114 D2/D3, slice K1-2, #1360)

`Emitter.command(verb, *args)` is Protocol method 75 and the last
(`verbs.EMITTER_METHOD_COUNT = 75`; the lock fails a 76th). Tcl writes
`verb a1 a2`, py writes `ops.verb(a1, a2)`, live calls `ops.verb(*args)` and
raises naming the row's `requires` when the binding lacks it, recording
appends `("command", (verb, *args), {})`, and H5 refuses until K1-4 adds
`/opensees/commands`. The channel is fail-closed: a token needs a
`via="command"` row in `VERBS`, or every emitter raises `ValueError`; no such
row ships yet, and the lock scans every `x.command(...)` call under
`src/apeGmsh/opensees/` for a literal verb with a row, called from a
primitive's `_emit` (or K2's replay). There is no user-facing `ops.command()`.

`H5Emitter` gains `_refuse(verb, detail)`, which raises `H5RefusedVerb` (a
`NotImplementedError`) naming the verb and its `VERBS` row, and
`_ledger(verb)`, which counts a dropped call in `_ledger_counts` and stays
silent; every `refuse` and `ledger` body now goes through them, and the lock
reads that off the source. A stage-only verb (`set_time`, `remove_sp`,
`update_material_stage`, ...) emitted outside a `stage_open` / `stage_close`
bracket now raises instead of returning silently (ADR 0114 D2). No emitted
deck byte and no schema changes.
