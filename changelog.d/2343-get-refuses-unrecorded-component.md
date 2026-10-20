### FIXED — a `Results` read of a component the stage does not record refuses by name instead of returning empty (#2343)

`results.nodes.get(component="displacement")` returned an empty slab on a
file that recorded `displacement_x`, `_y` and `_z`, with no error and no
warning. The read looked like "no data" when the data was there. A typo
(`"displacment_x"`) or a name at the wrong level did the same, on every
reader (native, MPCO, `.ladruno`) and every composite.

Each composite `get()` now raises `ComponentNotRecorded` (a `ValueError`) when
the read comes back empty and the name is not in the listing for the stage
being read. The composites are `results.nodes`, `results.elements` and its
`gauss`, `line_stations`, `fibers`, `layers` and `springs`. The message names
the level and the stage, and lists what is recorded. For a family name it also
lists the members to read one at a time.
The spatial queries (`nearest_to`, `in_box`, `in_sphere`, `on_plane`) and
`results.plot.history` read through `get()` and refuse the same way.

A recorded name on an empty selection still returns an empty slab. A read
that returns data never consults the listing, so its cost is unchanged.
Derived and custom scalars resolve before the reader, as before.
