### FIXED — partitioned emit no longer keeps a copy of the per-node ndf map, and the emit-profile smoke is green again (#2527)

Since #2421, `fixed_dof_map` has run on the partitioned emit path. Its
first step copied the emit's per-node ndf map, and `FixedDofs.ndf` kept
that copy alive into the rank blocks: 48.5 B per global node for each emit
(1.25 MiB at a 30^3 box). The copy was allocated after the pre-block
transient had been freed. Traced memory at the first `partition_open` then
exceeded the window peak, so the ADR 0100 pre-block attribution
(`--phases`) reported no transient sites. That turned the nightly
Benchmarks job red from 2026-10-22.

`FixedDofs.ndf` is now a read-only view of the inferred map, which is never
mutated after inference. On the smoke cell this brings resident memory at
the first open back to 291 B/gn (from 339 B/gn) and restores 12 transient
sites. The smoke now also fails any `fixed_dof_map` site that holds 16 B or
more per global node resident.
