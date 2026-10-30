### FIXED — physical-group and label lookups cost O(1), not a scan of every group

`g.physical.get_tag` and `g.labels.entities` / `has` found a group by
calling `getPhysicalName` on every group, so creating or resolving G
named groups cost O(G²). `g.physical.add_*` called `get_tag` four times
per new group. `g.labels.add` rescanned every group per label, and each
label name passed to a boolean was looked up at four dims, where every
miss built its `KeyError` message from a full `get_all()`. In the YOO
tower (one group or label per member) that was 12.4 s of a 22.7 s mesh
build and 214.6 s of a labelled `fragment` that raw gmsh does in 4.1 s.

Both composites now read through one name index,
`core/_pg_name_index.PGNameIndex` (`name -> {dim: tag}`), kept warm by
apeGmsh's own writes. Measured on one desktop:

| Case | Before | After | Raw gmsh |
|---|---|---|---|
| 10,000 `g.physical.add_curve(..., name=)` | 215 s | 3.3 s | 2.7 s |
| `g.model.boolean.fragment` by label name, 3,916 labels | 103 s | 6.9 s | 1.7 s |

The index never answers stale, even after raw `gmsh.model.*` calls,
such as the boolean remap that recreates every label group. Every
lookup asks gmsh at which dims the name exists (one
`getEntitiesForPhysicalName`, a single C++ call) and confirms each
indexed tag with one `getPhysicalName`. If gmsh disagrees, the index is
rebuilt by one full scan. Behaviour is unchanged: unknown names give
`None` or `KeyError`, label groups stay out of `g.physical`, and a name
at two dims still raises where it did. `_bool_op(label=)` now collects
the inputs' labels in one pass instead of one `labels_for_entity` per
input.

`tests/test_pg_name_index.py` covers these cases and both costs, each as
a ratio to raw gmsh on the same geometry.
