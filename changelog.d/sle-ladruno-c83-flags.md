### ADDED — fork build `c83ee812e`: ASD shell `p_delta=`, `response_spectrum_analysis(scale=)`, `eigen_feast(trace_stop=)` and `EigenResult.modal_properties()`

These features need Ladruno fork build `c83ee812e` (fork WP-202, WP-203,
WP-205 and WP-206) or later.

- `ops.element.ASDShellQ4(..., p_delta=True)` and `ASDShellT3(...,
  p_delta=True)` emit `-pDelta`. The shell gets linear kinematics plus the
  initial-stress stiffness of its current membrane forces, for gravity, then
  `loadConst`, then `eigen`. `p_delta=True` with `corotational=True` is
  refused at construction, as the fork refuses the pair. Stock OpenSees
  accepts the token and ignores it, so the in-process run refuses
  `p_delta=True` off the fork. A cantilever strip at `P = 0.5 Pcr` gives
  `T/T0 = 1.4077`.
- `ops.response_spectrum_analysis(..., scale=s)` emits `-scale s`, which
  multiplies `Sa` under `-combine`. `scale` must be finite and greater than
  0. `scale=2` doubles the combined displacement to 1e-12.
- `ops.eigen_feast(..., trace_stop=True)` emits `-traceStop`, which selects
  MKL's trace stopping criterion. The default stops on the relative residual,
  and both return the same eigenpairs. `EigenResult.modal_properties()` runs
  `modalProperties` on a solve's modes. After `eigen_feast`, its
  participation factors match the ARPACK ones.
- `ElasticTimoshenkoBeam` (3-D) on a `PDelta` transformation now softens:
  its tip drift at `P = 0.5 Pcr` equals `elasticBeamColumn`'s to 1e-6, and
  is within 0.2 % of the exact beam-column value. Its `basicForce` response
  answers `(N, Mz1, Mz2, My1, My2, T)`. The bridge refused neither, so no
  code changed for it; the fork lane now covers it.
