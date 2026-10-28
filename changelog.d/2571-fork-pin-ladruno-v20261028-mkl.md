### CHANGED — `FORK_PIN` moves to `ladruno-v20261028`, with FEAST and Pardiso in the fork lane (#2571)

`FORK_PIN` now names fork release `ladruno-v20261028` (fork commit `9aadc76cc`). The nightly `live-fork`
lane uses its `-mkl` Linux tarball, `ladruno-opensees-linux-mkl-ladruno-v20261028.tar.gz`. That build
links FEAST and Pardiso dynamically against MKL and does not ship MKL. Install it in the environment that
imports the build: `pip install "mkl>=2024.2.2,<2026"`. mkl 2026.x ships `.so.3` libraries, which this
build cannot load. The MKL-free tarball is unchanged.

The lane now fails on any `ladruno_mkl:` skip and uploads its junit XML. `BackendInfo` reads
`ladrunoSchema` version 1, which is version 0 plus a `capabilities` list. Version 0 is still read for older
fork builds, and its schema lists no capability. The fork build tables accept the new pin.

On fork pins before fork commit `eaaf07e8b` (nmorabowen/OpenSees#1022, FB-1), a `LadrunoCohesiveHinge` placed with
an interior `-hingeLoc` read the wrong moment: `(1 - xi) q1 + xi q2` instead of the true moment `M(xi)` at
the hinge. On the root element of a tip-loaded two-element cantilever with `hinge_loc=0.5`, it read a
third of the real moment. Re-run any model with an interior `hinge_loc` hinge that ran on such a pin.
