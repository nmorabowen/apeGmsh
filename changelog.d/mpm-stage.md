### ADDED — `ops.mpm.stage` stages the particles of one material in an MPM deck (ADR 0134 Amendment 4)

A new member of the `mpm` family, `ops.mpm.stage(material, stage)`
(`mpm.Stage`), emits the fork's `mpm stage <matTag> <s>`. The fork stages
every particle whose material is `material`, through the material's own
`materialState` parameter, and passes `stage` through unchanged. The row is
written where it was declared among the `ops.mpm.analyze` rows, so a deck
such as `stage(dp, 0)`, `analyze`, `stage(dp, 2)`, `analyze` runs an elastic
stage and then an elastoplastic one. `material` is a handle or a registered
name, and the deck writes its allocated tag.

Two classes are admitted: `DruckerPrager` with stages 0, 1 and 2, and
`LadrunoSANISAND` with stages 0 and 1. SANISAND's stage is class-wide, so
staging one SANISAND material stages every SANISAND particle. It needs fork
`ladruno-v20261123b` or later.

apeGmsh refuses each of these by name before the deck is written:

- a material of another class, a subclass included;
- a stage that is a bool, a float or a string, or that its class does not
  read;
- a stage whose material no body uses (the fork would refuse the deck);
- a stage with no `ops.mpm.analyze` after it;
- two stages of one material with no analyze between them;
- two SANISAND materials staged to different values with no analyze between
  them.

It also refuses a `LadrunoSANISAND` body. The fork runs SANISAND particles
in its explicit mode F only on a B-spline grid, and `ops.mpm.grid` admits
the gimp and linear bases only. A model with `ops.stage` blocks is still
refused.

The live-fork lane runs a weak Drucker-Prager column on the pinned fork,
staged 0 and then 2. At the end of stage 0 the deep particles hold
`σxx/σyy = ν/(1−ν)`, and after stage 2 every particle is on or inside the
yield surface.
