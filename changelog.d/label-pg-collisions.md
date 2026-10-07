### FIXED — label and physical-group silent-empty and name-collision siblings (#1364)

- `g.labels.add(dim, [], name)` used to register nothing and return. It now
  raises `ValueError`, as an empty selection does since #1335. An empty
  `name` raises too.
- `g.physical.add(..., name="_label:X")` and
  `g.labels.promote_to_physical(..., pg_name="_label:X")` now refuse the
  reserved label prefix before any gmsh call. Before, gmsh refused the
  duplicate name and an unnamed physical group was left behind.
- `g.model.io.load_dxf` creates its layer physical groups through
  `g.physical.add`, so loading a layer name that already has a group merges
  into that group instead of leaving an unnamed one. A layer name held by a
  group at another dimension, or carrying the reserved prefix, raises before
  anything is imported.
- `promote_to_physical` into a name held at another dimension now says to
  pass a distinct `pg_name=`.
- The compose-anchor error and the `fem.nodes` "no label, physical group, or
  part" error listed the snapshot's `(dim, tag)` keys. They now list names,
  and the node error also lists the physical groups.
