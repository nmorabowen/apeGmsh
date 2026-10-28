### FIXED — `partition()` counts each element once, in the partition that owns it (#2590)

After #2582 a partitioned FEMData held each mesh element once, but
`g.mesh.partitioning` still counted the elements partitioning adds.
`PartitionInfo.elements_per_partition` included the partition-interface
elements Gmsh meshes on each cut, and with `Mesh.PartitionCreateGhostCells=1`
the ghost copies of the neighbouring partition's elements as well: a 2-D
plate of 198 elements split 2 ways read `{1: 112, 2: 110}` (`{1: 132, 2: 130}`
with ghost cells) against `{1: 100, 2: 98}` in `fem.partitions`. The ghost
copies also inflated `weights_per_partition` on a weighted call. With ghost
cells on, the element-to-partition map behind `uncuttable_elements` sent
every ghost-copied element to the neighbouring partition, so a group already
whole on one partition tallied as split, was moved, and raised the
"uncuttable_elements moved" warning.

Both now walk the entities FEMData extraction keeps (the
`_is_partition_artifact` predicate of #2582), so `elements_per_partition`,
`weights_per_partition` and the map agree with `fem.partitions`, ghost cells
on or off. Every partition id from 1 to `n_parts` is a key of
`elements_per_partition`; one that owns no element reads `0` instead of
being left out. A non-ghost, non-interface entity that names more than one
partition now raises `RuntimeError`, since its elements have no single
owner (not observed on Gmsh 4.15).

The `entity` source-entity note in `architecture/h5-schema.md` now also
names the ghost rows a pre-#2582 model.h5 written with ghost cells on can
carry: `-1` rows that repeat the ids of elements the file already holds.
