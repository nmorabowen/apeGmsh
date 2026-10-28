### FIXED — a partitioned FEMData holds each mesh element once: partition-interface and ghost elements are no longer extracted (#2582)

`g.mesh.partitioning.partition(n)` used to add elements to the FEMData taken
after it. Gmsh's default `Mesh.PartitionCreateTopology=1` meshes every cut
between two partitions with new lower-dimensional elements: lines between
surfaces, triangles between volumes, and points where a cut meets a curve.
They reached FEMData as extra elements with source entity `-1`, whenever
the extraction took the dimension below the top one or every dimension
(`get_fem_data()` with no `dim`). Each partition record also listed them
under both partitions. With `Mesh.PartitionCreateGhostCells=1`, a ghost
entity's copies of the neighbouring partition's elements were extracted
again under the same ids, at every dimension, so element ids repeated and
each ghost element appeared in two partitions. A plane-wave box held 3175
elements partitioned 2 ways and 3482 partitioned 4 ways, against 2925 for
the mesh itself.

Extraction now skips both kinds of entity (`_is_partition_artifact` in
`mesh/_fem_extract.py`), in `extract_raw` and in `extract_partitions`. A
partitioned FEMData has the element ids and connectivity of the
unpartitioned mesh, with ghost cells on or off, and each element lies in
exactly one partition. Partitioned Tcl decks for the plane-wave box and a
2-D plate (2, 3 and 4 partitions) are byte-identical before and after the
fix: no element or node an OpenSees rank receives changes. What changes are
the element counts and ids that FEMData, `fem.partitions`, and selections by
dimension report. A `model.h5` written before the fix still loads, with the
extra elements it carries; re-extract to drop them.
