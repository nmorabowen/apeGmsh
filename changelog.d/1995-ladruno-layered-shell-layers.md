### FIXED — `.ladruno` reads layered-shell layer state, as MPCO does (#1995)

A layered shell (`LayeredShell` on `ASDShellQ4`, `ASDShellT3`, `ShellMITC4`
or `ShellDKGQ`) recorded with `-E section.fiber.stress section.fiber.strain`
to a `.ladruno` read back as nothing. `available_components(FIBERS)` listed
`fiber_stress` and `fiber_strain`, then `elements.fibers.get(...)` returned
an empty slab: the fiber read kept only one-component fiber blocks, and a
shell writes five components per layer. `elements.layers` was always empty.

`Results.from_ladruno(...).elements.layers` now returns the `LayerSlab` the
MPCO reader returns: components `fiber_stress_<c>` / `fiber_strain_<c>`,
one row per (element, surface Gauss point, layer) at sub-GP 0, the layer
thickness, and the shell frame. A `.ladruno` stores no shell local axes, so
the frame is computed from the file's node coordinates, as on MPCO (#1996).
A row with no frame reads NaN and warns `ShellFrameUnavailableWarning`,
naming the elements. The fiber level no longer lists a component that only
a layered shell carries.

A bucket of several components per fiber that the layer level is unable to
read (a class with no `LAYER_CATALOG` entry, or an irregular block layout)
warns `LayerBucketUnreadWarning`, naming the bucket, at listing and at the
fiber read. A layer row whose section the file does not assign raises
`LayerSectionUnresolved`, naming the bucket and the pairs.

The fork-lane file `test_fiber_layer_readback_live.py` checks every layer
component, the thickness and the frame against `eleResponse(e, "material",
gp, "fiber", k, ...)` at rtol 1e-10, on both formats and for each of the
four bridge shells. Its `[Ladruno]` strict xfails are removed.
