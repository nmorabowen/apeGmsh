### ADDED — recorded modes read back from `.ladruno` and `.mpco` (#1946)

`Results.from_ladruno(...).modes` and `Results.from_mpco(..., model_h5=).modes`
now return one `kind="mode"` stage per mode recorded with `-N
modesOfVibration` (and `modesOfVibrationRotational`). Each has `eigenvalue`
from the recorder's `LAMBDA`, `frequency_hz`, `period_s` and `mode_index`,
and reads `displacement_*` / `rotation_*` as the mode shape. That is the
layout `DomainCapture.capture_modes` wrote, so the native recorders now
cover modes (gap G1 of #1767).

- An `.mpco` that records modes beside a multi-step result no longer fails
  with "MPCO stage step-count mismatch".
- A `.ladruno` stage with `KIND=eigen` now reads as `static`. The recorder
  writes `eigen` on every stage once modes are requested, so a gravity stage
  had been listed as a mode.
- Mode stages take no `ops.stage(name=...)` name: the program names pair
  with the `MODEL_STAGE` groups only.
- A mode without `LAMBDA`, or `(U)` and `(R)` groups that disagree, is
  refused. A non-positive eigenvalue warns and reads 0 Hz.

A pip `openseespy` install still has no modes route: it ships no MPCO
recorder, and `.out` has no modes.
