### CHANGED — Parameter tags come from the build-time tag plan; no tag is minted at emit (K1-3d S5, #1457)

The `parameter` allocation loop moved out of `emit_initial_stress_global`,
`emit_activate_absorbing` and `emit_update_parameters` into
`plan_parameters`, which `plan_tags` runs once per emit mode. The plan holds
every parameter site the mode's emit writes, in emit order: the global
initial stresses, then, stage by stage, its initial stresses, its absorbing
flips and its `s.update_parameter` updates. A partitioned emit writes each
flip and update pass rank by rank, and a record takes a tag only on a rank
that owns one of its elements. The helpers read
`plan.parameters[(record, rank)]` and write through `write_planned_ramps`
and `write_planned_flips`. The plan refuses, when it is made, sites that do
not match the model's records by identity and order. The emit refuses a plan
that lacks a record it writes. The emit fork now freezes `parameter`. Every
family is now planned, so no emit path mints a tag; an emit-time mint of any
kind raises `TagLawError`. The three helpers still accept a plain
`TagAllocator` from a direct caller, and from the compose replay under its
ledger waivers, until K1-3d S6. Every emitted deck is byte-identical: the 86
golden cells, and the initial-stress, absorbing and update decks flat and
over 2 and 4 ranks.

One ordering change: on a flat staged emit, an `s.activate_absorbing` or
`s.update_parameter` that names an element no primitive emits now raises its
`BridgeError` when the emit plans its tags. That is before the emit path's
own checks. The message is unchanged.

The tag-plan oracle now compares region rows as a multiset on a flat deck,
so a region line written twice there fails it. The #1446 region test compares
the ordered list of `region` lines.
