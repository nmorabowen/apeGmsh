### ADDED — the `VERBS` table and the Emitter Protocol freeze (ADR 0114, slice K1-1, #1359)

`src/apeGmsh/opensees/emitter/verbs.py` lists every Emitter verb with its
family, scope, archive store, required capabilities, return shape and what
`H5Emitter` does with it today: 56 verbs archive, 4 refuse
(`equalDOF_mixed`, staged `update_parameter`, `set_node_vel`,
`set_node_accel`) and 14 are dropped silently (the ledger: global `rayleigh`,
`modal_damping`, the `eigen` and modal-response family, `profiler`, the
contact, rebar and embed passes, and `equationConstraint`). A new lock in the
`lock-tests` job, `tests/opensees/contract/test_verbs_lock.py`, freezes the
Protocol at 74 methods, ties each row to a Protocol method and its `H5Emitter`
source, and keeps the ledger shrink-only. ADR 0114 records the ratified K0
design and supersedes ADR 0019 INV-5: tags are archive facts, and replay
never allocates one. No emitted deck or schema changes.
