### CI — the nightly program probe flags scheduled lanes non-green two nights running, keeps one issue per red lane, and flags `Slice:` PRs missing the `program` label (#2175, #2556)

`scripts/program_probe.py` gains two finding kinds. `red-lane` fires when the
last two completed `schedule` runs on `main` of `live-fork.yml`,
`benchmarks.yml`, `deps-canary.yml` or `orphans.yml` were both non-green:
`failure`, `cancelled`, `timed_out` or `startup_failure`. A cancel counts,
because a job that hits its own `timeout-minutes` concludes `cancelled`.
`unlabelled-slice` fires for an open PR whose body has `Slice: #<n>` but no
`program` label. Both go into the existing tracking issue; `program-probe.yml`
gains `actions: read`.

With `--issue` the probe also keeps one issue per red lane,
`[Program probe] lane <workflow> non-green`, labelled `program` and the lane's
chain. It opens the issue after 2 consecutive non-green nights, comments on it
each further night, and closes it once the lane's latest run is green. A later
red streak opens a new issue.
