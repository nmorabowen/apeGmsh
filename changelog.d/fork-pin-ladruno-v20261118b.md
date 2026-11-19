### Fork pin: `ladruno-v20261118b`, MPM soil models and mode-F gridvel/pforce

`FORK_PIN` now names fork release `ladruno-v20261118b` (fork commit `cfe478254`). It still uses the `-mkl` tarball. The release is 61 commits past `ladruno-v20261118` (`3e738308c`) and adds:

- nmorabowen/OpenSees#1108 (WP-232k): MPM soil models at particles. LadrunoSANISAND and LadrunoHS/HSSmall are admitted; SANISAND in mode F is refused on gimp/linear bases and admitted on bspline2/3. It also adds the `mpm gridvel` forms `affine`, `translate` and `profile`, and the `mpm stage` command.
- Fork issue #1109, fixed by #1108: mode F now applies `mpm gridvel` and `mpm pforce`. Earlier releases ignored them silently in explicit mode.
- Fork issues #1110 (no surcharge/traction carrier for particles), #1111 (`eleLoad` accepts an unknown `-type`) and #1112 (rough-footing `gridvel ... track y` loses settlement) are cited by the fork's tests and stay open.
- nmorabowen/OpenSees#1101 (WP-199): the CPML kernel harness links the GCC runtime statically on Windows. It shipped in `ladruno-v20261118`, and that release's fragment did not list it.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows. The fork floors and capability promises are unchanged; `ladrunoSchema` stays v1.
