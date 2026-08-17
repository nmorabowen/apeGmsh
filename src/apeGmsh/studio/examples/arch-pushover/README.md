# arch-pushover

A planar "shoebuckle" arch frame — two pinned columns and a circular
arch through the crown — modelled as a 2-D fiber-section beam-column
frame and pushed under a parametrised inward pressure field until
`lambda = 1`.

Teaches:

- Declaring a spatially-varying load through `g.loads` and reading it
  back into the OpenSees deck with `ops.pattern.Plain(...).from_model(...)`
  (the load never reaches the deck without this explicit import).
- A hand-written `LoadControl` ramp with limit-point recovery.
- MPCO recording and `Results.from_mpco()` read-back, including
  `results.assess()`'s `RES.U_VS_DIAG` diagnostic.

Requires: `openseespy` (live in-process run — no Tcl emit).

Run:

```
python arch_pushover.py
python verify.py
```

Both write/read `shoebuckle.mpco` and `model.h5` in the current
directory. See `manifest.json` for the oracle metrics `verify.py`
checks.
