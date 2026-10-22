### CHANGED — every ManzariDafalias deck states `updateMaterialStage -stage 0`; the class-wide stage check runs at emit (#2440, #2437)

**Behaviour change: stage 0 is explicit.** Upstream OpenSees starts
`ManzariDafalias` elastoplastic and the Ladruno fork starts it elastic, so
one deck gave two answers. One `SSPbrick` under a pressure of 100 settled
2.08e-3 on stock 3.8.0 and 1.31e-3 on the fork (#2437). The Tcl and py
decks and the in-process run (`apeSees.analyze`) now issue
`updateMaterialStage -material <tag> -stage 0` for every
`ManzariDafalias` and `LadrunoSANISAND` material. With it, stock gives the
fork's answer to round-off.

- **Where it goes.** The line acts only through elements already in the
  domain, so a line beside the `nDMaterial` declaration would do nothing.
  The deck writes it at the first `analyze` or `eigen` after the elements.
  It writes it again at the next one if a later stage adds elements, so a
  material whose elements a stage activates is reached as well.
- **A user flip still wins.** Once an `s.update_material_stage` names one
  of these materials, or an `s.update_parameter("materialState" |
  "updateMaterialStage", ..., material=m)` sets its flag, the deck writes
  no more stage-0 lines. A user stage 1
  therefore still switches plasticity on.
- **The archive does not change.** The line belongs to the deck writer,
  not to the model, so `model.h5` stores none of it and `model_hash` is
  unchanged. A deck replayed from an archive writes the line too, older
  archives included.
- **`ops.run()`.** It builds the live model and leaves the solve to its
  caller, so it issues the line at its end. The caller's own `analyze`
  then starts elastic on stock as on the fork.
- **Decks without a solve still diverge.** A deck file that drives no
  `analyze` or `eigen` (`ops.tcl(path)` without `analyze_steps`) writes no
  line, since the writer cannot tell where its elements end. A script
  that sources such a deck and then solves on stock starts elastoplastic
  unless it states `updateMaterialStage ... -stage 0` itself.

**The class-wide stage check moves to the emit.** This supersedes the
"when the stage closes" wording of the #2439 entry. The refusal of a flip
that names only some of a group's materials, by either route, now runs
when the model is emitted, beside validator V7. No check is left at stage
close: a later stage can still activate or remove the elements that decide
which materials are live, so a check there could refuse a correct model.
The refusal and its message are the same as #2439's, but it now sees two
more things:

- It sees the final model, so a group material registered after the stage
  block counts. The stage-close check missed that material, and the flip
  switched it silently.
- It models `s.remove_element`. A material whose elements are all removed
  by the flip line need not be named. The stage-close check refused that
  case. A flip whose named materials have all been removed is refused,
  because it would switch nothing.

**In-process `analyze` on stock refuses a fork-only chain before the wipe.**
A fork-only `system`, `integrator`, `numberer`, `test` or constraint
handler, declared or auto-emitted, is now refused before the live emitter
wipes the domain. Before, it was refused at its own command, after the wipe
and the model emit, so a refused run had already cleared the user's domain.
The check reads the declared chain and costs about 0.5 ms; the fork skips
it.
