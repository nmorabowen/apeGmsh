### ADDED — each element records the gmsh entity it was meshed on: `/elements/{alias}/entity`, neutral schema 2.36.0 (#2551 S1)

`FEMData` now carries a per-element **source entity**: for every element,
the tag of the gmsh model entity of its own dimension it was meshed on (the
curve a beam element lies on, the surface a shell lies on, the volume a hex
fills). `fem.elements.entity` holds it per element type as an `int32`
array row-parallel to the group's `ids`, and
`fem.elements.source_entity(eids, dim=...)` returns `(module_label,
entity)` for a batch of element ids. That pair is the key that names one
source entity in a composed model, because entity tags repeat across
modules. Nothing reads it yet: the joint-offset planner (#2551 S2) is the
first consumer.

- **Capture.** `g.mesh.queries.get_fem_data()` and `FEMData.from_msh` read
  the elements one gmsh entity at a time. The ids, connectivity and element
  type order are byte-identical to before, so no deck, golden or
  `snapshot_id` moves. On a partitioned mesh, each partition entity resolves
  to its parent, so partitioning does not change an element's value. The
  interface elements a partition adds get `-1`, because their parent has a
  higher dimension.
- **Schema.** model.h5 writes the stream as `/elements/{alias}/entity`,
  `(E,) int32`, on every element group or on none (neutral schema
  2.35.0 → 2.36.0, additive and optional). Files from 2.10 to 2.35, MPCO and
  `.ladruno` read as having no stream. A file that carries the stream on
  some groups only, with the wrong shape, or with a value that is neither
  `-1` nor a positive tag is refused with `MalformedH5Error`. The stream is
  not part of `snapshot_id`.
- **Compose.** `Assembly` carries the stream row for row without
  rewriting the values. A module without the stream contributes `-1` rows.
- The app reader's neutral target moves to 2.36. The app ignores the new
  dataset.
