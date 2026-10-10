### REMOVED (breaking): `g.compose` and the v1 `Assembly` verbs (ADR 0117 D7)

`g.compose`, `apeGmsh.compose`, `FEMData.compose`, `Assembly.add`,
`Assembly.couple(part_a, part_b, kind=, ports=)` and `Assembly.materialize`
are gone, with no deprecation period. Composition is `Assembly`
(`from apeGmsh.assembly import Assembly`): `instance`, the tie and
coupling verbs, and `bridge`. The compose readers `g.compose_inspect`,
`g.compose_list` and `g.compose_tree` stay, and read an assembly archive.
`AssemblyError` is still `apeGmsh.assembly.AssemblyError`. To migrate:

- **Save the host, then instance it.** There is no live host session: write
  the host with `apeSees(fem).model(...)` and `.h5()`, then
  `instance("host", "host.h5")`. Its groups become `host.<name>`.
- `add(label, source, ...)` becomes `instance(label, source, ...)`; a
  rotation `(x, y, z, theta)` becomes `((x, y, z), theta)`.
- `couple(kind="tie" | "equal_dof", ports=...)` becomes `tie` /
  `equal_dof` on dotted ports `"{instance}.{group}"`; the other couplings
  are `rigid_link`, `rigid_diaphragm`, `rigid_body`, `embedded` and
  `couple(kind="kinematic" | "distributing")`.
- `materialize()` becomes `bridge(ndm=, ndf=)`, which returns the `apeSees`
  bridge; `asm.h5(path)` writes the archive.
- **Supports, masses and loads are restated on the bridge.** A composed
  v1 session carried the module's `g.constraints.bc`, `g.masses` and load
  cases into its deck; the bridge emits them only when asked:
  `ops.fix_from_model()`, `ops.mass_from_model()` and, inside a pattern,
  `p.from_model(case)` (or `ops.fix` / `ops.mass` / `p.load` on dotted
  groups). The build warns (`UnconsumedModelDefinitionWarning`) when the
  model defines supports or masses the deck leaves out, so nothing is
  dropped silently.
- `max_compose_depth`, `properties` and `compose_size_per_module` are
  gone; the nested-compose depth is fixed at 3.

Also: the `/interfaces` refusal of `bridge()` now names a node-pair
`zeroLength` declared on an interface pair instead of a stage claim (#1590
review).
