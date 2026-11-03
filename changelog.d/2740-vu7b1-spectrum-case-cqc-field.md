### apeGmshViewer: a spectrum case's CQC as a nodal contour column, through the combination evaluator (ADR 0130 slice 7b-1)

The app can now read the values of one nodal result of a `/spectra` case. Main's
new `user:readCase` returns the per-mode `DATA` and the reference state, after
judging the case with the same read as `user:read`. A User Combination over the
cases, such as a whole-case `cqc` or `srss`, or a `cqc` over chosen increments,
is evaluated by the slice 4 evaluator and gives one value per node, ready for a
contour. On a synthetic nodal San Ramón case these values equal Python's
`Results.combinations().evaluate(..., level="nodes")` to 1e-12 relative.

The result is unsigned, so it can only be drawn as a contour, never as a
deformed shape. A stale `/spectra` is refused unless the caller allows it. So is
a refused or missing case, a result or column the case does not hold, and a
`stage` or `mode` term, which this source does not read. Nothing is drawn yet:
the phase-bar picker, the deformed-toggle refusal and the stills are slice 7b-2.
