### Fork pin: `ladruno-v20261114`, the MPM release

`FORK_PIN` now names fork release `ladruno-v20261114` (fork commit `f0b9a1df6`). It still uses the `-mkl` tarball. The release is 84 commits past `ladruno-v20261103` (`6050d78e1`) and adds these fork PRs:

- nmorabowen/OpenSees#1075 (WP-232): the MPM slice, a general MPM driver with the `mpm` command, its LoadPattern host and the `.ladruno` particle writer. Its contract PRs are #1077 (particle set, `.ladruno` particle group), #1076 and #1079 (numpy oracles, interface) and #1078 (explicit slice); #1080 and #1081 add the crash-safe particle file and the write barrier.
- nmorabowen/OpenSees#1065, #1069 (WP-210): the LadrunoHS and HSsmall fixes (exact 3-D Lode invariants, cap normality, PLAXIS dilatancy, tension cutoff).
- nmorabowen/OpenSees#1063 (WP-224): the commit-refusal counter is atomic.
- nmorabowen/OpenSees#1066 and #1068 (WP-226, WP-227): `recorder ladruno` writes RBE3/RBE2 connectivity and `GP_PARAM` for the Ladruno solids.
- nmorabowen/OpenSees#1073 (WP-231): LadrunoSANISAND SAS-ME runs on stack arrays.
- nmorabowen/OpenSees#1064: the fork CI runner is pinned to ubuntu-24.04.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows. The fork floors and capability promises are unchanged.
