### ADDED — apeGmshViewer: a results.h5 read spike and its measurements (V4a, #1707)

`apeGmshViewer/src/reader/results.ts` reads a native `results.h5` lazily
with h5wasm: stages, steps and time, one component at one step by
hyperslab, one Gauss-point component, and one node's whole history. It
refuses a file outside results 1.0 to 1.x, a missing dataset, an unknown
layout, non-float data and a file HDF5 cannot open, and it reports a gauss
group's element rows as FEM ids only for `domain_capture` files (#1711).
`--mode=measure-results` times that path end to end, and
`apeGmshViewer/MEASUREMENTS.md` has a new "Results sizes (V4)" section with
rows for three generated files (206 MB to 2.1 GB) and an MPCO point: the
ADR 0112 D5 decision input. No UI uses the reader yet.
