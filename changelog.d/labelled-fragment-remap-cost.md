### FIXED — a boolean over labelled entities costs close to raw gmsh

After the name index, a labelled `g.model.boolean.fragment` still spent
most of its overhead in three places. Every boolean snapshots and
remaps every physical group, and reading a group's entities with
`getEntitiesForPhysicalGroup` scans every entity of that dim, so
reading G groups one by one costs O(G·E). That happened twice: once in
the snapshot and once when resolving the label names passed to the
boolean. The remap also removed and recreated every group, one call
each, even when its entities had not changed. And each label name was
looked up five times: `has()` at four dims, then `entities()`.

- The snapshot and remap read every group in one
  `getPhysicalGroupsEntities` call. A gmsh build without that call
  reads group by group as before.
- `synchronize()` keeps a group on the entities that survive it. The
  remap now leaves a group alone, with its tag, when its remapped
  entities equal what it holds. The changed groups are removed in one
  `removePhysicalGroups` call and then recreated.
- The string resolvers in `core/_helpers.py` make one index lookup per
  name (`Labels._entities_by_dim`). When a list of refs is resolved,
  every label's entities come from one bulk read. Group tags still come
  from the checked index, so the staleness rule is unchanged.

Measured on one desktop, walls crossed by frames, with a labelled point
per wall:

| Labels | Raw gmsh | Before | After |
|---|---|---|---|
| 1,200 | 0.40 s | 0.97 s (2.4×) | 0.61 s (1.55×) |
| 3,920 | 3.4 s | 8.8 s (2.5×) | 5.4 s (1.57×) |

What remains is mostly `addPhysicalGroup`, about 0.24 ms per recreated
group. On this geometry, fragment rebuilds every wall face and splits
every frame, so 3,430 of the 3,920 groups really change. The ratio
assert in `tests/test_pg_name_index.py` drops from `6×raw + 1 s` to
`2×raw + 0.1 s`. New tests cover a kept group's tag, list resolution
after raw gmsh edits, and the path without the bulk call.
