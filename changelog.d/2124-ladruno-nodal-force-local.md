### FIXED — a `.ladruno` read of `nodal_resisting_*_local_*` returns the `localForce` end forces, as MPCO does (#2124)

A `.ladruno` file answered the local-frame names
`res.elements.get(component="nodal_resisting_force_local_x")` (and the other
`_local_` names) with `NodalForceNotRecorded`, though it recorded
`localForce`. The MPCO reader read the same names from the same response.

The reader now decodes a `localForce` bucket into
`nodal_resisting_{force,moment}_local_{x,y,z}`, as a `(T, E, n_nodes)` slab
in element node order. The layout is the one MPCO uses:
`NODAL_FORCE_CATALOG`, keyed by the class in the bucket key. The bucket's
width and every column name (`N_1`, `Vy_1`…, the 2-D `V_1`/`M_1`, or `N1`
without the underscore) must spell that layout. A class the catalog does not
list, or a bucket whose names disagree with it (2-D names on a 3-D class),
is left out with a `NodalForceColumnsUndecodedWarning` naming the bucket and
the reason. When no bucket answers, the read raises `NodalForceNotRecorded`,
naming the buckets it left out. The local names read even when the file's
spatial dimension is unknown, and `available_components` lists them. On a
fork-recorded L frame of each catalogued beam class, the `.ladruno` read
equals the MPCO read bit for bit.
