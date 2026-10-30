### CHANGED — `FORK_PIN` moves to `ladruno-v20261030`, whose `LadrunoConcrete3D` refuses a failed return map (#2622)

`FORK_PIN` now names fork release `ladruno-v20261030` (fork commit `c9004f48d`). It still uses the `-mkl`
tarball. The release is `ladruno-v20261029` plus fork PR nmorabowen/OpenSees#1038, which changes only
`LadrunoConcrete3D`. When the last plane-stress pass of its return map fails, or σ22 stays unconverged,
`setTrialStrain` now returns a negative code, and Newton cuts the step. The old build returned 0 there,
so the analysis accepted a state that was out of equilibrium (#2602).

Under the nightly `live-fork` lane:
- **The Cook–Mitchell corbel pushover still ends in a `PushoverRefusedError`, but later.** With
  `MKL_CBWR=COMPATIBLE` it refuses at step 121 (`"no_convergence"`), where the old pin refused at step 2.
  The step and the reason depend on the MKL code path, so the test no longer pins either. It checks the
  refusal and the equilibrated curve that the error carries. The paper bracket comes back with #2621.
- **The smaller strut-and-tie corbel now completes in equilibrium.** It reaches its target with a
  257 kN capacity, and its curve test is back.
- **The mixed-concrete MPCO header test now takes 100 load steps instead of 20.** With 20 steps, the old
  pin accepted step 8 only because the return map failed silently, and the new pin refuses it.

The fork build tables accept the new pin.
