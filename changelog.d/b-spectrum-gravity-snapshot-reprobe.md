### FIXED — `spectrum_after_gravity`: the gravity snapshot is the settled state, and round-off is not a kink (program slice #2545)

`spectrum_after_gravity` settles the gravity state with unloaded `Linear` steps before the per-mode
legs, but `result.modal.gravity` kept the snapshot taken before settling, so `gravity_drift()` read
1.5e-7 right after the call under a loose test (`NormUnbalance 1.0`) instead of the documented `0.0`.
The snapshot is now retaken where the driver leaves the domain. The per-element kink check refused a
linear model with near-rigid links (`I` x 1e6, thirty times the gravity) on round-off, blaming a link.
A mode that fails the check is now probed again at 100 times the amplitude and judged there: a kink's
ratio does not depend on the amplitude, while round-off shrinks with it, so no-tension and Concrete01
braces are still refused. On the fork's `-increment` route, a leg that raises is now always reverted.
