### FIXED — a raw deck's line stations over a FEMData archive no longer mis-slice mixed point counts (#2328)

`spec.emit_default` runs a raw openseespy deck with a FEMData archive
(`fem.to_h5`). Before this fix, a `line_stations` record could read one
element's station as another's, with no warning. That happened when the
record held elements with different point counts and its columns divided
evenly. Take two displacement beams of lengths 1 and 13, with Legendre 2
and Legendre 4: an equal slice gives each 3 stations, and both blocks pass
the order check. Element 1 then reads element 2's base moment as its own
third station.

- **The counts come from the live domain.** At `__enter__`, before the
  run, each element of a `line_stations` record read from `section force`
  has its section locations read with `sectionLocation`. The read slices
  the `.out` pair by those counts, under the rule the bridge's
  `beamIntegration` counts follow. Unequal counts are refused by count,
  naming each element.
- **An element with no sections is refused by FEM id before the run.**
- Elastic beams read from `localForce`, and records that name no class,
  are unchanged.

The bridge-archive route (`from_recorders(model=)` and `emit_default`
over `ops.h5`) already refused this record by its counts, and it is
unchanged.
