### CHANGED — MP-element and interface tags come from the build-time tag plan; `element` is frozen at emit (K1-3d S3c, #1480)

The element-tag allocation of every MP-element writer (rigid bodies,
kinematic couplings, interpolation ties, `g.reinforce` and `g.embed` ties,
auto-emitted rebar cells) moved into `plan_mp_elements`, and the interface
allocation loop into `plan_interface_tags`. `plan_tags` runs both once per
emit mode, in the order that mode's emit writes them: a flat emit
interleaves the MP passes with the interfaces, and a partitioned emit
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
and 4 ranks with ties, rebar, embedded ties and interfaces.

**Refusal order on a partitioned emit.** The partitioned MP and
reinforcement routing now runs when the emit plans its tags, before any
line is written and before the partitioned path's own checks. Its
refusals are a `ValueError` when a kinematic coupling, an element rigid
body or an embedded tie's host element is split across partitions (global
pass or stage-claimed records), and a `BridgeError` when a rebar node is
owned by no partition or a reinforce tie's host nodes share none. They
now pre-empt these partitioned-path refusals, which used to fire first:

- `equation_constraint` rows on a partitioned emit;
- `g.embed` ties on a partitioned emit;
- the explicit SOFT contact knobs (ADR 0092 INV-3);
- node-pair elements (`nodes=`) on a partitioned emit;
- the staged boundary-condition validators (`_run_staged_bc_validators`);
- an unclaimed interface on a stage-bound node
  (`_validate_interfaces_not_stage_bound`);
- the pattern-`sp` refusal on contact and interface ghosts;
- the contact routing refusals (`_plan_partitioned_contacts`: undecidable
  owner, cut master under auto sizing, staged model), which `plan_tags`
  now runs after the MP routing;
- the interface ownership checks (`_plan_partitioned_interfaces`,
  `_plan_stage_interfaces_partitioned`).

Each fault on its own still raises exactly as on main. In a model with
several faults the first error can change type: an MP split such as
"kinematic_coupling … split across partitions" is a `ValueError`, raised
where main raised a `BridgeError` (a `RuntimeError`), for example the
`g.embed` refusal.

**Emit cost.** Planning the MP elements and checking each plan's coverage
by record identity adds about 40 ms to the 10k-element tet gate cell
(median emit/parse ratio 0.52 to 0.58 in three interleaved runs; the hex
cell is unchanged). The gate passes without a re-baseline.
