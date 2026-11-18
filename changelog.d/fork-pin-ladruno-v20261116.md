### Fork pin: `ladruno-v20261116`, the crash-safe recorder

`FORK_PIN` now names fork release `ladruno-v20261116` (fork commit `6ee968060`). It still uses the `-mkl` tarball. The release is 68 commits past `ladruno-v20261114` (`f0b9a1df6`) and adds these fork PRs:

- nmorabowen/OpenSees#1084 (WP-233): `recorder ladruno` survives a hard kill. It writes under HDF5 SWMR; #1082 adds the crash replay model, kill stress and SWMR layout bench.
- nmorabowen/OpenSees#1092 (WP-233 follow-up): one torn-last-stage rule. A last stage reads as torn only after reader-rule step 1 or 3, which aligns schema 7b.2 with 232a 2.5.
- nmorabowen/OpenSees#1083 (WP-232f): implicit and quasi-static MPM through mode A.
- nmorabowen/OpenSees#1088 (WP-232h, #1085): the MPM particle writer presizes its root and holds the guard lock.
- nmorabowen/OpenSees#1089 (WP-232g): MPM M0b measurements.
- nmorabowen/OpenSees#1094: the MPM contracts' stage report skips the FE stage of a mixed file.
- nmorabowen/OpenSees#1086 (WP-200 follow-up): LadrunoIMKBeam element loads under `-endZones`, and the `-pmBal` message.
- nmorabowen/OpenSees#1093: the fork ledger records WP-232f, WP-232h and WP-233.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows. The fork floors and capability promises are unchanged; `ladrunoSchema` stays v1.
