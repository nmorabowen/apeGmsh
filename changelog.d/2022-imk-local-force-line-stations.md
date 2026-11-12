### FIXED — A LadrunoIMKBeam's line stations read from MPCO and `.ladruno`, and an untagged `localForce` of an unknown class is refused (#2022)

The fork records a `LadrunoIMKBeam` / `LadrunoIMKBeam2d` `localForce`
without component tags, so both formats name its columns `C1..C12`
(2-D: `C1..C6`). Both readers passed such a bucket over: a line-station
read of an IMK beam returned no stations and no warning, while an
`ElasticBeam3d` with the same values read its two end stations.

Now the IMK columns read in the class's `NODAL_FORCE_CATALOG` local-force
order (`N, Vy, Vz, T, My, Mz` at end i, then end j; 2-D `N, Vy, Mz`) as two
end stations, with the same station signs the elastic beams take. A
`localForce` bucket that holds only `C` names and belongs to any other beam
class, or whose key does not parse, is refused: a read of its elements
raises `ValueError` naming the class and the bucket. So is an IMK bucket
whose width or column names do not match its class's order. A non-beam's
`C` bucket (a truss's force vector) is still no beam's stations and is
passed over as before.
