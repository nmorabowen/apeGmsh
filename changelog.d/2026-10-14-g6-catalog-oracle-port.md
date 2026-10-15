### ADDED — real-solver RESPONSE_CATALOG checks off DomainCapture (#1953, #1767 G6)

The catalog's plane, shell, truss and solid layouts now have value tests on the
recorder routes. Each test builds a one-element model, solves it and reads it
back. It checks every component at every Gauss point against the live
`eleResponse` vector at `rtol=1e-10`, and against an anchor that does not share
the element's output path. For planes and solids the anchor is `D B(xi) u` at
the natural coordinates read back. For shells it is the section response. For
trusses it is the applied load.

- The stock lane (`-m live`) runs `tests/test_results_catalog_out_real.py` on
  `.out`. It covers 18 cases, including the upstream DKG and MITC4 shells that
  DomainCapture skipped.
- The fork lane runs
  `tests/opensees/integration_ladruno/test_results_catalog_hdf5_live.py` on
  MPCO, through both `emit_mpco` and a hand-written recorder, and on
  `.ladruno`.

Routes that read back an empty slab or lose their natural coordinates are
strict xfails citing #1970, #1990 (`.ladruno` shell resultants), #1998
(`.ladruno` rule-0 natural coordinates) and
#1991 (`emit_mpco` records shell resultants as `section.force`).
#1992 records a further finding. Under `algorithm Linear` the
`ZERO_GAUSS_PROBE_CLASSES` shells are one step stale on every route, not dead:
they read zero on the first step and lag the converged state after it. Under
Newton every route agrees.
