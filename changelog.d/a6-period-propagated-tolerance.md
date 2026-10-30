### ADDED — A period-propagated parity tolerance for response-spectrum comparisons (chain A, A6)

A response-spectrum result compared with another program's never sees
exactly the same periods, and a flat tolerance fails on differences the
period mismatch fully explains. The new module
`apeGmsh.opensees.analysis.spectrum_tolerance` turns the mismatch into the
tolerance a parity gate should use. `period_tolerance` handles a quantity
governed by one mode: the exact ratio `Sa(T)/Sa(T_ref)·(T/T_ref)^p`
(`p = 0` for forces, `2` for displacements and drifts), or the first-order
`s·δT/T` and `(s + 2)·δT/T`, with the local slope from `spectrum_slope`
(one-sided, so plateau corners are defined). `combined_period_tolerance`
rescales each mode's value to the reference periods and re-runs the CQC or
SRSS with the reference `ω`, so a combined quantity needs no hand-picked
dominant mode. Both return a `ParityTolerance`, `total = base + period`,
whose `str()` is a one-line report and whose `passes(value, reference)` is
the gate.
