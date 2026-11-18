### Fork pin: `ladruno-v20261118`, MPM B-spline bases and anti-locking

`FORK_PIN` now names fork release `ladruno-v20261118` (fork commit `3e738308c`). It still uses the `-mkl` tarball. The release is 35 commits past `ladruno-v20261116` (`6ee968060`) and adds these fork PRs:

- nmorabowen/OpenSees#1099 (WP-232i): MPM B-spline bases and the manufactured-solution (MMS) checks.
- nmorabowen/OpenSees#1105 (WP-232j): MPM anti-locking F-bar (`mpm antiLocking`).
- nmorabowen/OpenSees#1104 (WP-234): the recorder node region.
- nmorabowen/OpenSees#1100: CPML late-time stability.
- nmorabowen/OpenSees#1102: the g++ test harness links its runtime statically.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows. The fork floors and capability promises are unchanged; `ladrunoSchema` stays v1.
