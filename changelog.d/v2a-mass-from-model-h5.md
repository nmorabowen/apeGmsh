### FIXED — `apeSees.h5()` archives a `mass_from_model()` model instead of raising (ADR 0112 amendment 5, #1304)

`apeSees.h5()` raised `BridgeError` ("deck/live-only") whenever the bridge
declared `mass_from_model()`, so under ADR 0112 D1's unconditional write every
such model would fail at the end of its run. The H5 emit now skips the
redundant mass stream, since the masses already persist in the neutral zone's
`/masses`, and marks the archive with `/opensees/bcs@mass_from_model = 1`.
`OpenSeesModel.build` streams the neutral masses back onto each node's
effective ndf when the marker is set, so a replayed deck carries the same
masses as the original, node by node and dof by dof, and `to_h5` keeps the
archive a fixed point. The double-count guard against an explicit `ops.mass`
on the same node still raises on every emitter. OpenSees zone 2.21.0 → 2.22.0
(additive): opensees-zone 2.20.x files no longer open until the per-zone compatibility floor (#1303) lands.
