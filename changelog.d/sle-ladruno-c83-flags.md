### ADDED — fork build `c83ee812e`: ASD shell `p_delta=`, `response_spectrum_analysis(scale=)`, `eigen_feast(trace_stop=)` and `EigenResult.modal_properties()`

These features need Ladruno fork build `c83ee812e` (fork WP-202, WP-203,
WP-205 and WP-206) or later; `FORK_PIN` (`ladruno-v20261023`) carries it.
No command marks them, so where an older build would answer wrong or end
the process, apeGmsh checks the `ladrunoBuild` sha against the builds
measured to carry `c83ee812e` (`stage_emit.FORK_BUILD_HAS_C83`), and
refuses stock and every other build by name.

- `ops.element.ASDShellQ4(..., p_delta=True)` and `ASDShellT3(...,
  p_delta=True)` emit `-pDelta`. The shell gets linear kinematics plus the
  initial-stress stiffness of its current membrane forces, for gravity, then
  `loadConst`, then `eigen`. `p_delta=True` with `corotational=True` is
  refused at construction, as the fork refuses the pair. Stock OpenSees and
  older fork builds accept the token and run without the stiffness, so the
  in-process run refuses `p_delta=True` on them, and a Tcl or py deck with
  `-pDelta` checks `ladrunoBuild` at its start and stops on them. A
  cantilever strip at `P = 0.5 Pcr` gives `T/T0 = 1.4077`.
- `ops.response_spectrum_analysis(..., scale=s)` multiplies the `-Sa`
  ordinates by `s` before the call. The fork's own `-scale` is not
  emitted, because older builds parse it and ignore it, so the answer is
  the same on every build. `scale` must be finite and greater than 0.
  `scale=2` doubles the combined displacement to 1e-12.
- `ops.eigen_feast(..., trace_stop=True)` emits `-traceStop`, which selects
  MKL's trace stopping criterion. The default stops on the relative residual,
  and both return the same eigenpairs. An older build refuses `-traceStop`
  as an unknown option. `EigenResult.modal_properties()` runs
  `modalProperties` on a solve's modes. After `eigen_feast`, its
  participation factors match the ARPACK ones. An older build ends the
  process there, so after `eigen_feast` the call is refused on any build
  not measured to carry `c83ee812e`.
- `ElasticTimoshenkoBeam` (3-D) on a `PDelta` transformation now softens:
  its tip drift at `P = 0.5 Pcr` equals `elasticBeamColumn`'s to 1e-6, and
  is within 0.2 % of the exact beam-column value. Its `basicForce` response
  answers `(N, Mz1, Mz2, My1, My2, T)`. The bridge refused neither, so no
  code changed for it; the fork lane now covers it.
