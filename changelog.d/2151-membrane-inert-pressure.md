### FIXED — `ops.build()` refuses a `LadrunoMembrane` `pressure=` that no pattern activates (#2151)

The fork's `LadrunoMembrane` reads `-pressure` only inside
`LadrunoMembrane::addLoad`, as the reference value that a pattern's
`eleLoad -type -surfaceLoad` scales. `-selfWeight` falls back to it only
when the membrane has no `rho` x thickness, and `-beamUniform` never
reads it. A membrane that set `pressure=` with no such pattern carried
no load: `analyze` returned 0 and every support reaction read exactly
0.0, with no error (#2143).

`ops.build()`, and so `tcl()`, `py()`, `h5()` and the live run, now raises
`InertMembranePressureError` (a `BridgeError`) when a membrane sets a
nonzero `pressure=` and no `Plain` pattern, top level or a stage's
`s.pattern(...)`, reads it: a `surface_load`, or a nonzero `self_weight`
on a membrane without `rho` x thickness (`rho=`, the section's `rho` or
the `mat=` density), targeting it by its handle (`ele=slab`) or `pg=`.
An integer `ele=` tag is not matched, since a membrane's tag is planned
at emit. The message names the membrane and the fix,
`p.surface_load(ele=<membrane>)`, or dropping `pressure=` for a modal or
stiffness-only model. A
`label=` target does not count: it resolves to mesh cells, and a membrane
has none. A membrane without `pressure=` still builds, and so does one
whose `rho=` gives it mass for a modal or `UniformExcitation` run.
