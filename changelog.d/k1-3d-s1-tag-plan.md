### ADDED — the tag-plan scaffold: `plan_tags`, `TagAllocator.freeze()` / `fork()`, `TagLawError` (K1-3d S1, #1453)

`_internal/tag_plan.py` holds `TagPlan` (frozen, slots), with one sub-plan
per derived tag family (elements, transforms, regions, parameters,
MP elements, interfaces, contacts), the `TagMode` triple
`(split, partitioned, staged)`, and `plan_tags(bm, mode)`. `plan_tags`
seeds a planner allocator as the emit seeds its own, refuses a seed that
disagrees with the registered tags, and freezes it. Every family is still
pending, so nothing reads the plan yet and no emitted byte changes.
`TagAllocator.freeze()` makes `allocate`, `allocate_block`,
`allocate_for`, `reserve_through` and `reset` raise `TagLawError`;
`fork(frozen_kinds)` copies an allocator with chosen kinds frozen (such a
copy refuses `reset`), which `TagPlan.emit_allocator()` uses as the
migration's safety net. A new oracle,
`tests/opensees/contract/test_tag_plan_oracle.py`, attributes every
emitted tag to its family by the function that minted it, and compares
each family's plan with those rows, `xfail` per family until that family
migrates; it also pins the plan's seed, `element_tags="fem"` included,
against the emit's. The tag-stream corpus gains a fixture with two named
regions owned by different ranks plus a filtered MPCO recorder, and a
truss whose FEM carries MP elements, an interface, a contact and a
contact plane, emitted with sequential and with FEM-id element tags.
