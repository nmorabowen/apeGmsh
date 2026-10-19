### CI — the nightly program probe flags scheduled lanes red two nights running, and `Slice:` PRs missing the `program` label (#2175)

`scripts/program_probe.py` gains two finding kinds. `red-lane` fires when the
last two completed `schedule` runs on `main` of `live-fork.yml`,
`benchmarks.yml`, `deps-canary.yml` or `orphans.yml` both ended `failure`,
`timed_out` or `startup_failure`. `unlabelled-slice` fires for an open PR whose
body has `Slice: #<n>` but no `program` label. Both go into the existing
tracking issue; `program-probe.yml` gains `actions: read`.
