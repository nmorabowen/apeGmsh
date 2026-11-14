### ADDED — `ops.mpm`: the fork's MPM engine, emitted with its particle set beside the deck (ADR 0134, #2999)

The Ladruno fork's Material Point Method engine is now a primitive family,
`ops.mpm.*`, with one member per `mpm` sub-command: `grid`, `body`, `fix`,
`wall`, `gravity`, `timeSeries`, `damping`, `flip`, `massCutoff`, `safety`,
`units`, `output` and `analyze`. A body names a physical group, a material by
label, a density and the particles per cell. Users never type a tag: the
body's material tag comes from the bridge's allocator, and the grid's tag (a
fork load-pattern tag) is written above every `pattern` tag of the deck.

`ops.tcl(path)` and `ops.py(path)` lattice-seed the bodies on the bridge's
FEMData and write `<deck stem>.particles.h5` beside the deck. The deck reads
it with an `mpm particles -file` line anchored to its own folder. The two are
checked before either is written and written together, and an existing
particle set is replaced only when an apeGmsh emit wrote it. Every
declaration is a `/opensees/commands` row of `model.h5`, so
`OpenSeesModel.build("tcl" | "py", out=path)` re-seeds the same particles
beside the replayed deck, under a new `SET_ID`.

MPM is deck-only. The live route refuses a model with an `mpm` member on
every build, before the domain is touched, and so do a partitioned or
per-rank emit and a staged model. The family needs a fork release that
carries `mpm` (`ladruno-v20261114` or later).
