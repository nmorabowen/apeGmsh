### FIXED — a `.ladruno` read of `nodal_resisting_*` returns the per-node forces, or refuses by name (#2071)

A `.ladruno` file recorded with the element response `globalForce` answered
`res.elements.get(component="nodal_resisting_force_x")` with an empty slab
and no warning. The `.ladruno` element level reads by token, so only the raw
`globalForce` name found the data.

The reader now decodes the `globalForce` columns into the canonical
`nodal_resisting_force_{x,y,z}` and `nodal_resisting_moment_{x,y,z}`, as a
`(T, E, n_nodes)` slab in element node order: the shape and order the MPCO
reader returns for the same names. It reads all three spellings OpenSees
writes: `Px_1`… for beams, `P<dof>_<node>` for quads and bricks, and
`P<node>_<dof>` for trusses and `zeroLength`, telling the last two apart by
their column order. An indexed column above the spatial dimension (a 2-D
`P3_1`) can hold a rotation, a pore pressure or a PML field, so it stays
undecoded, and a moment request that meets it warns
(`NodalForceColumnsUndecodedWarning`). `available_components` lists the
canonical names a file can answer, and the raw `globalForce` read is as before.

A request the file is unable to answer raises `NodalForceNotRecorded`, naming
the element responses the file records: a file without `globalForce`, a
component its buckets lack, or a local-frame name. A partition set raises
only when every part is unable to answer.
