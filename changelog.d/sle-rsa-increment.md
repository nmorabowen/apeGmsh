### CHANGED — `spectrum_after_gravity` takes the fork's `responseSpectrumAnalysis -increment` fast path where the build has it (ADR 0124 amendment)

On a Ladruno build with fork WP-207 (`ladruno a8bcc07e`), each mode's per-mode response comes from
`responseSpectrumAnalysis dir -Tn T_k -Sa Sa -mode k -increment -sign ±1`, which leaves gravity ± mode k
as an updated, uncommitted trial state, then `-revert`: no static solves. The capability probe
(`responseSpectrumAnalysis -revert` at the committed state) returns `True` or an inconclusive `None`, and
stock or older fork builds keep the static-solve path. The per-mode values agree with the static path
to `1e-9` on a linear frame. `NodeDisp` now reads the trial displacement, `nodeDisp + nodeResponse(n, d, 4)`,
on both paths (`nodeResponse(n, d, 1)` is the committed one).
