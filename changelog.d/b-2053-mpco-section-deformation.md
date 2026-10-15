### FIXED — MPCO reads line-station section deformations, and refuses when they were not recorded (#2053)

`Results.from_mpco(...).elements.line_stations.get(component=...)` now reads
`axial_strain`, `curvature_z`, `curvature_y`, `torsional_strain`,
`shear_strain_y` and `shear_strain_z` from an `-E section.deformation` bucket.
The MPCO recorder names those columns `eps`, `kappaZ`, `kappaY`, `theta`,
`gammaY` and `gammaZ`, in the section's own order. The reader knew only the
section-force names (`P`, `Mz`, ...), so every deformation read raised on
`eps`. Each bucket is now decoded with the names of its own group, and a
force name under `section.deformation` (or the reverse) raises. Asking for a
deformation component from a file that recorded no `section.deformation`
used to return an empty slab with no warning. It now raises `LookupError`
naming the missing group. A section-force component still reads empty
without `section.force`, because `localForce` can supply elastic beams.
