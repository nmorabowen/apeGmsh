### FIXED — a beam the response catalog does not know warns when it records only `globalForce` (#2119)

A line-station read names a `globalForce`-only bucket, since global end
forces are never station values. That warning fired only for the beam
classes the catalog knows. A `ModElasticBeam3d`, `MixedBeamColumn3d`,
`ForceBeamColumnCBDI3d`, `Pipe` or other uncatalogued beam that recorded
only `globalForce` read no stations, with no warning, on both MPCO and
`.ladruno`.

The gate is now a deny-list of non-beam classes, derived from the
catalog's family tables: the continuum, shell and truss classes of
`RESPONSE_CATALOG`, the shells of `LAYER_CATALOG`, the springs of
`ZEROLENGTH_CATALOG`, and a new `NON_BEAM_LINE_CLASSES` table of bearings,
MVLEM walls, links and springs. A bucket of one of those classes stays
silent. Any other class warns by name, as does a bucket key whose class
does not parse. That includes joints and cables (`BeamColumnJoint3d`,
`Joint3D`, `CatenaryCable`): they record `Mz_1` too and are not on the list.
