### ADDED — `--phases` mode in the emit residency instrument (program slice S3-a, #1874, ADR 0100)

`tests/benchmarks/emit_throughput_profile.py --phases` reports, for every rank block of the partitioned
emit, the exact traced peak of each phase (hoisted nodes, owned nodes, elements, fix/mass/regions,
constraints, close) as growth above the block's floor, in MiB and B/node. It also attributes one rank's
element phase (`--attr-rank`, default 0) by tracemalloc traceback, grouped by the allocating
`apeGmsh/<path>::<symbol>` and its retaining caller. The phase boundaries come from wrapping the emitter
and module-level emit helpers; `src/` is unchanged. `--phases` is exclusive with `--mem` and `--rss-only`.
