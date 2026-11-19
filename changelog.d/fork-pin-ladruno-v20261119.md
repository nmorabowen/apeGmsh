### Fork pin: `ladruno-v20261119`, `hingeStatus` and the bond-slip reversal fix

`FORK_PIN` now names fork release `ladruno-v20261119` (fork commit `f9864cfec`). It still uses the `-mkl` tarball. The release is 4 commits past `ladruno-v20261118b` (`cfe478254`) and adds:

- nmorabowen/OpenSees#1113 (WP-235): a `hingeStatus` element response on LadrunoDispBeamColumn, LadrunoForceBeamColumn and LadrunoIMKBeam, giving hinge kind and damage per slot (8 columns) under one spelling on all three. It is additive: the bridge has no response table for hinge tokens (`hingeSummary` is read raw), so nothing in apeGmsh consumes it yet. The viewer's hinge view can adopt it later.
- nmorabowen/OpenSees#1115: LadrunoBondSlip unloads and reloads through a friction slider capped at the envelope stress of the committed peak, instead of elastically through a line to that peak. Any analysis whose bond slip reverses now reports different bond stress and dissipates energy on the reversal; monotonic histories are unchanged. No apeGmsh test or golden drives a bond-slip reversal, so no expectation moved.

The fork-build tables in `stage_emit.py` gain the new sha beside the older rows, and the MPM fork-surface snapshot is refreshed at the new sha (the 36 subcommands are unchanged). The fork floors and capability promises are unchanged; `ladrunoSchema` stays v1.
