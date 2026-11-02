### apeGmshViewer: the app evaluates User Combinations the way Python does (ADR 0130 slice 4)

The app has a TypeScript twin of the Python combination evaluator, in
`apeGmshViewer/src/combination/`. It covers every rule: `linear`, `srss`, `abs_sum`,
`cqc` over the modes of one set or over one whole spectrum case, `directional` and
`envelope`. It checks in the same order as the Python evaluator and refuses with the
same reasons, including `not-linear` when a summing rule meets a user result that is
not `@linear`. On the shared conformance fixture both evaluators meet the same
goldens to `1e-12` relative.

Sign is derived from the rule and the terms, never read from the file. Only a
signed result gives a deformed shape: an unsigned one (a modal rule, an envelope,
or a `linear` over either) is refused as a shape and is shown as a contour only.
The phase bar still lists combinations without drawing them; drawing them through
this evaluator is a later slice.
