### FIXED — MPCO reads shear and moment line stations of 2-D beams (#2010)

`Results.from_mpco` dropped the shear and moment of every 2-D beam that records `localForce`. The bucket
spells them `V`/`M` (`N_1,V_1,M_1,..`), and the reader knew only the 3-D `Vy`/`Mz`, so an elastic girder
came back with `axial_force` stations only. A `dispBeamColumn` spells its bucket without the underscore
(`N1,V1,M1,..`) and came back with no stations at all. Both now read `axial_force`, `shear_y` and
`bending_moment_z` at xi = -1 and +1.

The reader takes `V`/`M` as 2-D only when all of these hold:
- the file's `/INFO/SPATIAL_DIM` is 2;
- the bucket key names an element class that is not a `...3d` class;
- the data width matches the names.

`ComponentElement3d` and `ForceBeamColumnCBDI3d` record their 12-value vector under the six 2-D names, so
their buckets are refused. So is any other malformed `localForce` bucket. A refused bucket lists no
component, and a line-station read that asks for its elements, or for every element, raises a `ValueError`
naming the bucket. Reads of other elements go on as before. A bucket with no station names at all, such as
a truss's `C1..C4`, is still passed over.
