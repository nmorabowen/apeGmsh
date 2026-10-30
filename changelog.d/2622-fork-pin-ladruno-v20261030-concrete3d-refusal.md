### CHANGED — `FORK_PIN` moves to `ladruno-v20261030`, whose `LadrunoConcrete3D` refuses a failed return map (#2622)

`FORK_PIN` now names fork release `ladruno-v20261030` (fork commit `c9004f48d`), still through its `-mkl`
tarball. It is `ladruno-v20261029` plus fork PR nmorabowen/OpenSees#1038, which changes only
`LadrunoConcrete3D`: when the last plane-stress pass of its return map fails, or σ22 stays unconverged,
`setTrialStrain` now returns a negative code, so Newton cuts the step. Before, it returned 0 and the
analysis accepted a state out of equilibrium (#2602).

The Cook–Mitchell corbel pushover still ends in a `PushoverRefusedError` at this pin, now well past
step 2, where the previous pin refused. The step and the reason (`"unbalanced"` or `"no_convergence"`)
depend on the MKL code path, so its test no longer pins either; it checks the refusal and the
equilibrated curve the error carries. The paper bracket returns with #2621. The fork build tables
accept the new pin.
