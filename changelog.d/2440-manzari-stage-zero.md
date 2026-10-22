### CHANGED — every ManzariDafalias deck states `updateMaterialStage -stage 0`; the class-wide stage check runs at emit (#2440, #2437)

**Behaviour change: stage 0 is explicit.** Upstream OpenSees starts
`ManzariDafalias` elastoplastic and the Ladruno fork starts it elastic, so
one deck gave two answers. One `SSPbrick` under a pressure of 100 settled
2.08e-3 on stock 3.8.0 and 1.31e-3 on the fork (#2437). The Tcl and py decks
now write `updateMaterialStage -material <tag> -stage 0` for every
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
- **Decks without a solve.** A deck that drives no `analyze` or `eigen`
  (`ops.tcl(path)` without `analyze_steps`) writes no line, because nothing
  reads the flag there. A script that sources such a deck and then solves
  must set the stage itself.

**The class-wide stage check moves to the emit.** The #2439 refusal of a
flip that names only some of a group's materials now runs when the model is
emitted, beside validator V7, instead of when the stage closes. It is the
same refusal and the same message, but it now sees two more things:

- It sees the final model, so a group material registered after the stage
  block counts. The stage-close check missed that material, and the flip
  switched it silently.
- It models `s.remove_element`. A material whose elements are all removed
  by the flip line need not be named. The stage-close check refused that
  case. A flip whose named materials have all been removed is refused,
  because it would switch nothing.
