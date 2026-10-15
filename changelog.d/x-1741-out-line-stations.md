### FIXED — beam line stations on `.out` and `.ladruno`, elastic beams included (program slice X G3, #1741)

`Results.from_recorders` no longer raises `KeyError: 'No group with tag 5'` on a line-stations record:
the transcoder looked up each element's length by physical-group tag instead of by element id. It
now selects by element id and refuses an id the bound FEMData does not hold.

An elastic beam (`ElasticBeam2d/3d`, `ModElasticBeam2d`, `ElasticTimoshenkoBeam2d/3d`) answers neither
`section force` nor `integrationPoints`, so its line-stations recorder wrote only the time column. A
line-stations record whose `element_class_name` is an elastic beam now emits one `localForce` recorder
(`<name>_line_stations_local_force.out`), and the transcoder builds the two-station slab at
xi = -1, +1 from it, with end j negated into the section-force sign convention. That is the same
synthesis DomainCapture and the MPCO reader apply. Records without `element_class_name` keep the
`section force` pair.

The `.ladruno` reader takes each element's line stations from one source: `section.force` first, then
`localForce`, then any other element-level bucket. A force beam that also records `localForce` or
`basicForce` no longer gains extra end stations. It also reads a 2-D `localForce` bucket's `V`/`M`
columns as `shear_y`/`bending_moment_z`, which it used to drop, so a 2-D elastic girder now has all
three components.
