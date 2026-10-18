### FIXED — `ops.build()` refuses a `LadrunoMembrane` `pressure=` that no pattern activates (#2151)

The fork's `LadrunoMembrane` reads `-pressure` only inside
`LadrunoMembrane::addLoad`, as the reference value that a pattern's
`eleLoad -type -surfaceLoad` (or `-selfWeight`) scales. A membrane that
set `pressure=` with no such pattern carried no load: `analyze` returned
0 and every support reaction read exactly 0.0, with no error (#2143).

`ops.build()`, and so `tcl()`, `py()`, `h5()` and the live run, now raises
`InertMembranePressureError` (a `BridgeError`) when a membrane sets a
nonzero `pressure=` and no `Plain` pattern, top level or a stage's
`s.pattern(...)`, targets it with `surface_load`, `self_weight` or
`beam_uniform` by its handle (`ele=slab`) or `pg=`. An integer `ele=` tag
is not matched, since a membrane's tag is planned at emit. The message
names the membrane and the fix, `p.surface_load(ele=<membrane>)`. A
`label=` target does not count: it resolves to mesh cells, and a membrane
has none. A membrane without `pressure=` still builds, and so does one
whose `rho=` gives it mass for a modal or `UniformExcitation` run.
