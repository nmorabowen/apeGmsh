### FIXED — an element registered after a stage flip no longer slips past the stage-flag route checks (#2464)

The stage-flag route checks (#2443, #2445, #2458, #2460) ran only at stage
close, so they saw only the elements registered so far. The deck emits
every element spec, a later one included, before the stage's flip lines.
So a model with a carrier registered after the stage passed `.tcl()`,
`.py()` and `.h5()` with a silent wrong answer. For example, a FourNodeQuad
on a sand flipped plastic got a bare `updateMaterialStage` with no sweep,
the #2445 wrong answer. A DruckerPrager neighbour, a stress-density class,
a FourNodeQuad host or an activated element registered late slipped through
the same way.

The emit now runs the same checks again on the final model, beside
validator V7 (`stage_validate.validate_stage_routes`). Each stage is
checked in program order, with the stage close's own message, so the
bridge refuses what the `model.h5` replay refuses (#2465). The stage close
keeps its early refusal. The fork-reset check knows the run is on the fork
when the bridge's `OpenSeesTarget` pins `mode="fork"` (a new
`BuiltModel.fork_pinned`), or when a stage's analysis chain declares a
fork-only type such as `system Pardiso`. The replay reads that from the
same chains, so an unpinned model on a fork-only chain is refused at emit
too.
