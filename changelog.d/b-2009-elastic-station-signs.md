### FIXED — elastic-beam line stations carry the force-beam section signs on every route (program slice B, #2009)

An elastic beam's two line stations (xi = -1, +1) are synthesised from its `localForce` end vector.
Every route took end i as recorded and negated end j, which gave N and M (and, in 3-D, Vz and T) the
opposite sign of an adjacent force beam's `section force`: a force/elastic junction jumped from +N to
-N and from -M to +M with no warning. Only Vy agreed.

The stations now follow the force-beam convention: station 1 is `s * end_i` and station 2 is
`-s * end_j`, with `s = -1` for `axial_force`, `shear_z`, `torsion`, `bending_moment_y` and
`bending_moment_z`, and `s = +1` for `shear_y`. One table in the response catalog,
`local_force_station_signs`, holds the signs, and DomainCapture, the MPCO reader, the `.ladruno`
reader and the `.out` transcoder all apply it. An MPCO `localForce` bucket of a force or disp beam is
signed the same way. On a force/elastic/force bar the elastic stations now equal the force beams'
live `section force` at both shared nodes, at `rtol=1e-10`, in 2-D and 3-D, for `ElasticBeam2d/3d`
and `ElasticTimoshenkoBeam2d/3d`.

Line-station values read from an elastic beam's `localForce` change sign for those five components;
a script that compensated for the old sign must drop its correction.

Native results H5 files written before this change keep the old station signs. That covers files
captured by DomainCapture and files transcoded from `.out` recorders. No marker in the file tells an
old one from a new one, so the reader cannot correct them. To get the new convention, re-capture
those runs, or re-transcode them from their `.out` files. MPCO and `.ladruno` files are signed when
they are read, so they need nothing.
