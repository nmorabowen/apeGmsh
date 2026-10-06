### CHANGED — direct test callers of the two-way emit writers go through the tag plan (K1-3d S6a, #1494)

The tests that call the MP-element, interface, contact and recorder-region
writers of `_internal/build.py` directly now hand them the emit allocator of a
tag plan instead of a plain `TagAllocator()`. A new test helper,
`tests/opensees/_helpers/tag_plan.py` (`emit_tags`, `stub_fem`,
`StageClaims`), plans a FEM snapshot (or a stub of one) through the same
planning walks `plan_tags` runs, so no test reaches a writer's standalone
fallback and K1-3d S6 can delete it without a behaviour change. The
assertions are unchanged. Tests only: no emitted byte, schema or behaviour
changes.
