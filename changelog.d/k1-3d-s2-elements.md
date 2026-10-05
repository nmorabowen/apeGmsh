### CHANGED — Element tags come from the build-time tag plan (K1-3d S2, #1454)

`BuiltModel.emit()` now resolves its emit mode, plans the tags once per
mode with `plan_tags` (memoised on `BuiltModel._tag_plans`) and hands
every emit path a fresh `TagPlan.emit_allocator()` fork. The element
fan-out (`allocate_element_tags`) runs once, inside the planner; the flat,
split, staged and partitioned paths, and the partitioned per-rank buckets,
read `plan.elements` and refuse a plan that does not cover exactly the
specs they fan out. The emit allocator carries its plan, read back with
`tag_plan.plan_of(tags)`, so the `build.py` helpers that take only `tags`
can reach the plan in later slices. Every fork of a `TagAllocator` now
refuses `reset()`, whether or not it froze a kind. The oracle's element
case is a real comparison. Every emitted deck and the 86 golden cells are
byte-identical. One ordering change: an element declaration that fans out
to nothing, or a quad-only element on a triangle group, now raises when
the emit plans its tags, before the emit path's own checks run.
