### The `.out` route records fibres and per-Gauss-point damage

On a build without MPCO (pip `openseespy`), `spec.emit_recorders(...)` and the
Tcl/Python exports now record a `fibers` record as `section fiberData2` (every
section) with an `integrationPoints` pair, and a `damage` gauss record as one
`material <gp> damage` recorder per Gauss point. `Results.from_recorders(...)`
reads them back under the MPCO route's names (`fiber_stress`, `fiber_strain`,
`damage_tension`, `damage_compression`). A fibres record must name one
catalogued element class and hold one fibre section; a damage record must
name its element class. Both need `model=`, as every element record does.
A `layers` record is now refused by name on every `.out` route (it was skipped
with a warning).
