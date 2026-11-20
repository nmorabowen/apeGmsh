### Fork pin: `ladruno-v20261120`, ASDShell frames in `recorder ladruno`

`FORK_PIN` now names fork release `ladruno-v20261120` (fork commit `23e0f0002`). It still uses the `-mkl` tarball. The release is 4 commits past `ladruno-v20261119` (`f9864cfec`) and adds:

- nmorabowen/OpenSees#1133 (WP-242): `ASDShellQ4` and `ASDShellT3` answer `localAxes` (their material frame), so `recorder ladruno` now writes a `LOCAL_AXES` `FRAME` row for them. apeGmsh does not read it yet: both layer readers still compute a shell's quaternion from node coordinates (#1995, #1996), and no live test reads a stored shell frame.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows, and the MPM fork-surface snapshot is refreshed at the new sha (the 36 subcommands are unchanged). The fork floors and capability promises are unchanged; `ladrunoSchema` stays v1.
