### CHANGED — `FORK_PIN` moves to `ladruno-v20261029`, whose Linux tarballs ship the Tcl `OpenSees` (#2603)

`FORK_PIN` now names fork release `ladruno-v20261029` (fork commit `6ee6a04fd`), still through its `-mkl`
tarball. The fork source matches `ladruno-v20261028`; only the release workflow changed. Both Linux
tarballs now carry the classic Tcl interpreter `OpenSees` beside `opensees.so`, together with the Tcl
script library it was built against (`tcl8.6/`). Run that `OpenSees` with `TCL_LIBRARY=<unpacked>/tcl8.6`.
The `-mkl` build finds MKL the way `opensees.so` does, so put it beside `opensees.so` in the environment's
site-packages.

The nightly `live-fork` lane installs the exe there and points `APEGMSH_OPENSEES_BIN` at it. It fails if
the exe does not initialise Tcl or does not reach a trailing marker line. The four
`test_solver_stats_live` cases, which drive the Tcl subprocess route, now run in that lane instead of
skipping. The fork build tables accept the new pin.
