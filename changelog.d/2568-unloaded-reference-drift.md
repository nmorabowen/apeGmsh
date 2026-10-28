### FIXED — `spectrum_after_gravity(p_delta=False)` accepts a Corotational frame at rest

Under `p_delta=False` the per-mode legs run about the unloaded state, and the
restore check divided each drift by that state's largest value. A
Corotational column at rest is not exactly zero: it reads `6.7e-16` of
displacement (one ulp of its coordinates) and `3.5e-20` of reaction. A
legitimate 3-D frame was refused with "did not return to its reference
state" at a relative drift of up to `1.8e-01` (displacement) and `1.2e10`
(reaction), for absolute differences of `1e-16` m and `4e-9` N.
`state_drift` now treats a reference whose every displacement is within
`1e-12` of the model's extent as the unloaded state and reports the absolute
drift, as it already did for an exactly zero one. `GravityState` gains
`extent`, the model's largest coordinate span. A domain left at another
equilibrium is still refused about either reference.

`LadrunoMembrane` no longer calls `support_nodes=` or `diaphragm_nodes=`
empty when an empty `NodeSet` sits beside other nodes: the message names the
empty item (`support_nodes[0] is a NodeSet with no nodes`).
