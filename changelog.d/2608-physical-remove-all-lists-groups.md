### FIXED — `g.physical.remove_all()` no longer kills the process at `g.end()` (program slice B, #2608)

`g.physical.add` on an existing `(dim, name)` (the upsert: remove the group, re-add it with the union
of entities), then `g.physical.remove_all()`, then `g.end()` exited 127 with no traceback, 3 of 3. The
upsert is not the cause: in gmsh 4.15.2 the argument-less `removePhysicalGroups()` corrupts the heap
at finalize (`0xc0000374`) once any targeted `removePhysicalGroups([(dim, tag)])` ran in the model,
with or without apeGmsh, and every targeted removal (the upsert, `g.physical.remove`, label upkeep)
armed it. `remove_all()` now lists the groups and removes them by `(dim, tag)`, which also drops their
names: the argument-less form kept them, so a group created later at a freed tag inherited a stale
name and its own `setPhysicalName` was dropped. `tests/test_physical_upsert_remove_all.py` pins the
exit code in a fresh interpreter (`subprocess` marker) for the upsert and the single-remove routes,
and `tests/test_pg_name_index.py` pins the name drop in process.
