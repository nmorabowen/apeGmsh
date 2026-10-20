### FIXED — `capture_live_modes` remedy keeps the `system BandGeneral` guard (program slice #2365)

On stock openseespy a second eigen solve refuses, and the refusal told the user to call
`ops.wipeAnalysis()`. That also drops the `system BandGeneral` that guards against the stock default
eigen solver's plausible wrong positive eigenvalues on an indefinite stiffness. Following the old advice
silently gave `[1, 1]` for a deck whose true eigenvalues are `[-10, 100]`. The message and the docstring
now say: call `ops.wipeAnalysis()`, then `ops.system("BandGeneral")` again, before `capture_live_modes`.
