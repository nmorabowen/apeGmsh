### CHANGED — the emit holds the tag plan, not an allocator (K1-3d S6, #1458)

`BuiltModel.emit` now hands its memoised `TagPlan` to every emit helper of
`apesees.py`, `_internal/build.py` and `recorder.py`, which take it as
`tag_plan` in place of `tags: TagAllocator`. The plan has no allocation API,
so an emit can read a tag but never mint one. An owner the plan does not hold
raises `TagPlanMiss`, which is both a `BridgeError` and a `TagLawError` and
names the family and the owner.

- **Fallbacks removed.** Each helper used to fall back to planning its own
  tags from a plain `TagAllocator`; those fallbacks are gone, along with
  `plan_or_standalone`, `plan_of` and `TagPlan.emit_allocator`. Three
  standalone replay entry points remain, one for each replay waiver on the
  tag-law ledger: `replay_reinforce_ties`, `replay_initial_stress_global` and
  `replay_activate_absorbing`.
- **Flips and updates resolve their elements once.** The tag plan stores each
  absorbing flip's and `s.update_parameter`'s resolved element tags, and the
  writers read them. `emit_activate_absorbing` and `emit_update_parameters`
  now take `(records, emitter, tag_plan, partition_rank=None)`.
- **The dead `emit_element_spec` writer is deleted.** The bridge had stopped
  calling it. Its duplicate-node-tag check never ran on the live element
  fan-out; #1536 tracks that check.
- **The tag-law lock covers both hubs.** `test_tag_law_lock.py` now also locks
  the emit side of `apesees.py`, `build.py` and `recorder.py`, and every
  `TagAllocator(...)` construction in `src/`.

No emitted byte changes: the goldens are untouched, and 110 Tcl and Py decks
(flat, staged, and partitioned and staged-partitioned over 2 and 4 ranks) are
byte-identical to `main`.
