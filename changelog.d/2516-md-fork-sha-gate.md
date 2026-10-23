### FIXED — a deck that a stale fork's ManzariDafalias constructor reset would change checks the running build by exact `ladrunoBuild` sha; the stage-close refusal moves into the deck (#2516)

On a Ladruno fork build without commit `43b235bbb` (fork PR #1005), such as
79e062367, the ManzariDafalias and LadrunoSANISAND constructors set their
class-wide stage flag back to elastic. Every new element copies its material
through that constructor, so a stage that activates elements after an earlier
stage flipped the group ran every material of the group elastic, with no
warning (#2444). The stage-close refusal of that pattern ran only for
`OpenSeesTarget(mode="fork")`. A `mode="auto"` deck run on such a build got no
refusal. A fork-pinned model was refused even at `FORK_PIN`, which carries the
fix.

Now a Tcl or openseespy deck that builds an element or material after a flip
of the group, and solves before flipping it again, checks the running build
before its first command. If the build has no `ladrunoBuild` command it runs
on: stock OpenSees keeps the flag (measured on `live-stock`). If it has the
command, its answer must be one of the shas that
`stage_emit.FORK_BUILD_HAS_MD_STATIC_FIX` measured to carry the fix, which
today is `FORK_PIN` bdb69cea5. Otherwise the deck stops with a message naming
the flag, the fix and the ways out. An unmeasured or unanswering fork build is
refused too: a sha cannot be ordered, so a newer build joins the table once it
is measured. The check applies to every target mode. It sits before every rank
block, or right before the solve in a streamed deck. The archive replay
carries it, and `model_hash` does not change.

The stage close no longer refuses the pattern on any target. A staged model
runs only as a deck (`ops.analyze`, `ops.run` and `OpenSeesModel.build("live")`
refuse stages), so the deck is the only place that sees the build. The deck's
check replaces the stage-close refusal: it lifts the refusal at `FORK_PIN` and
keeps it, by name, everywhere else. When the fork registers a capability
marker for the fix (#2518), `stage_emit.MD_STATIC_RESET_GUARD` becomes a
command probe in one place.
