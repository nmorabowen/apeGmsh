### CHANGED — MP-element and interface tags come from the build-time tag plan; `element` is frozen at emit (K1-3d S3c, #1480)

The element-tag allocation of every MP-element writer (rigid bodies,
kinematic couplings, interpolation ties, `g.reinforce` and `g.embed` ties,
auto-emitted rebar cells) moved into `plan_mp_elements`, and the interface
allocation loop into `plan_interface_tags`. `plan_tags` runs both once per
emit mode, in the order that mode's emit writes them: flat and split
interleave the MP passes with the interfaces, and a partitioned emit
numbers the interfaces in one pre-pass, then each rank's MP elements and
reinforcement in its block, then the stage passes. On a partitioned emit
the plan also resolves the global MP-constraint pass's rank routing
(`plan_partitioned_mp_constraints`) once, and
`emit_mp_constraints_partitioned` reads it instead of re-running
`_plan_rank_constraints` for every rank on every emit. The writers read
each element's planned tag by record (a rebar cell by `(pg, i, j)`), and
`allocate_interface_tags` reads the planned interface tags. Every family
that mints `element` is now planned, so the emit fork freezes `element`
and `uniaxialMaterial`: an emit-time mint raises `TagLawError`. A plan
that does not cover its FEM exactly (an element dropped, doubled, or
swapped for another model's record) raises when it is made. The writers
still accept a plain `TagAllocator` from a direct caller until K1-3d S6.
Every emitted deck is byte-identical, including partitioned decks over 2
and 4 ranks with ties, rebar, embedded ties and interfaces. One ordering
change: a partitioned MP-constraint or reinforcement routing refusal (a
split host element or coupling set, an unowned rebar node) now raises when
the emit plans its tags, before any line is written.
