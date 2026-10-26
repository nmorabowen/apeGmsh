### CHANGED — `spectrum_after_gravity` takes the fork's `responseSpectrumAnalysis -increment` route where the build has it (ADR 0124 amendment)

On a Ladruno build with fork WP-207 (`ladruno a8bcc07e`), each per-mode probe leg is imposed with
`responseSpectrumAnalysis dir -Tn T_k -Sa Sa -mode k -increment -sign ±1` and returned with `-revert`,
instead of a Linear load step and a settling back. The legs keep the static route's probe amplitude, so the
settled reference, the effective-mass skip, the per-element kink check and the recorder refusal are the same
on both routes. Each imposed displacement is checked against `± a φ / λ`, and a model with a user-callable
extractor keeps the static route. The capability probe (`responseSpectrumAnalysis -revert`) returns `True`
or an inconclusive `None`, and stock or older fork builds keep the static route. `NodeDisp` now reads the
trial displacement, `nodeDisp + nodeResponse(n, d, 4)`, on both routes.
