### FIXED — a `Results` read of a component family refuses by name instead of returning empty, and an unrecorded name warns (#2343)

`results.nodes.get(component="displacement")` returned an empty slab on a
file that recorded `displacement_x`, `_y` and `_z`, with no error and no
warning. The read looked like "no data" when the data was there. This
happened on every reader (native, MPCO, `.ladruno`) and every composite.

Each composite `get()` now checks an empty read against the components that
the stage being read lists at that level. The composites are
`results.nodes`, `results.elements`, and the `gauss`, `line_stations`,
`fibers`, `layers` and `springs` levels.

- **A family name** raises `ComponentNotRecorded` (a `ValueError`), naming the
  members to read one at a time. A family name is one that prefixes listed
  names, such as `"displacement"`, `"stress"` or `"bending_moment"`.
- **A name outside apeGmsh's vocabulary**, such as a typo, still returns the
  empty slab. It now also emits one `ComponentNotRecordedWarning` per call,
  listing what is recorded.
- **Stays silent and empty:** a canonical name the stage does not record,
  such as `displacement_z` on a 2-D model, which the library's own x/y/z
  loops ask for. Also silent: a recorded name on an empty selection, and a
  read that leaves a recorded bucket unread. That read already names the
  bucket.

A read that returns data never consults the listing, so its cost is
unchanged. The spatial queries (`nearest_to`, `in_box`, `in_sphere`,
`on_plane`) and `results.plot` read through `get()` and behave the same way.
