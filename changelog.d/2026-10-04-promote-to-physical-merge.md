### FIXED — `g.labels.promote_to_physical` into an existing PG name merges instead of dropping the entities (#1332)

Promoting a second label into a `pg_name` that already existed at the same
dimension called `gmsh.model.addPhysicalGroup` + `setPhysicalName` directly;
gmsh refused the duplicate name, so the second set of entities landed in an
unnamed physical group that no `pg=` consumer could see. The readability
workshop lost half a footing strip this way (its line load came out at 300
kN/m instead of 600) with no warning. `promote_to_physical` now writes
through `g.physical.add`, so it carries that method's upsert contract: the
same name at the same dimension unions the entities into the existing PG
and returns its tag, and a name held by a PG at another dimension raises
`ValueError`. The skill's cheatsheet and workflow note document the merge.
