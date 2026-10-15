### FIXED — MPCO reads nodal forces on force- and disp-based beams, and names the class of a bucket it leaves unread (#2072)

An MPCO read of `nodal_resisting_*` on a `forceBeamColumn` or
`dispBeamColumn` returned an empty slab with no warning. The fork's MPCO
recorder writes a frame beam's `globalForce` / `localForce` under a
custom-rule bucket key, for example `74-ForceBeamColumn3d[1000:1:0]`. The
reader skipped every custom-rule bucket, and then every class outside
`NODAL_FORCE_CATALOG`, which listed the elastic beams only.

The catalog now lists `ForceBeamColumn`, `DispBeamColumn` and
`LadrunoDispBeamColumn`, in 2-D and 3-D and in both frames. Their end-force
layout is the elastic beam's. The MPCO reader reads a nodal-force bucket at
any custom-rule index, because the META check already guards its layout. The
catalog also gates DomainCapture and the `.out` transcoder, so those two
routes now record these classes too. Their line stations still come from
`section force`: the `localForce` synthesis stays limited to elastic beams.

A bucket the reader cannot decode now raises
`NodalForceBucketSkippedWarning` naming its element class and the reason. That
covers an uncatalogued class, a second response header, and a key that does
not parse. A bucket that holds none of the requested ids stays quiet.
