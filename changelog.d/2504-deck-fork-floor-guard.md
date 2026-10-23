### ADDED — a Tcl or openseespy deck that switches `materialState` on a Ladruno solid checks the running build first, and stops by name below `FORK_PIN` (#2504)

LadrunoQuad, LadrunoCST, LadrunoLST, LadrunoBrick and LadrunoBrick20 hand
`materialState` to their materials only on a fork build at or after
`FORK_PIN` 100eb9ae (fork commit `64a0341a6`). On an older build, such as
79e062367, the per-carrier sweep of a plastic ManzariDafalias flip
initialises the first element only, and the run gives a wrong answer
without a warning (#2445).

A deck whose sweep, or whose `s.update_parameter("materialState", ...)`,
reaches one of these elements now checks for the fork's `ladrunoSchema`
command before its first command: `if {[info commands ladrunoSchema] eq ""}
{ error ... }` in Tcl, and `if not hasattr(ops, 'ladrunoSchema'): raise
SystemExit(...)` after the openseespy import. Every fork build that
registers `ladrunoSchema` descends from the fix, and stock OpenSees lacks
it, so the deck stops with a message naming the floor on stock and on an
older fork. A streamed Tcl deck (`ops.tcl(stream=True)`) has its start on
disk already, so it checks right before the flip.

The floor comes from one table, `stage_emit.MATERIAL_STATE_FORK_FLOOR`,
which the stage close also reads for its fork-floor classes. The check is
derived from the emit calls: a deck without such an update is unchanged,
the deck replayed from `model.h5` carries the same lines, and `model_hash`
does not change. On `main` the stage close still refuses these classes;
the check takes effect when #2457 admits them.
