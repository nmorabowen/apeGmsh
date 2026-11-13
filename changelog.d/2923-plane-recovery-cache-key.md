### FIXED — a second model on the same mesh no longer reads the first model's plane types, so its von Mises and its warning are its own (#2923)

The out-of-plane recovery behind von Mises, principal stresses and mean
stress on 2-D results cached each model's plane types and Poisson's ratios
under its `snapshot_id`. That id hashes the mesh only, so a mesh run as
plane strain and then as plane stress, or with another material, shared
one entry. Whichever model was read second used the first model's plane
types, and its `OutOfPlaneRecoveryWarning` was dropped as a repeat. In one
process, the viewer fixture `extrap_plane` read a von Mises maximum of
144.599 with the warning on its own, but 141.086 with no warning after
`elem_mixed`.

The cache now keys on the model as well as the mesh: the lineage
`model_hash`, which hashes the OpenSees zone. A model with no
`model_hash` is keyed on the object itself, and the cache holds it so its
id cannot be reused by a later model while an entry still names it; the
last 8 such models are held. Both reads now return 144.599 with the
warning, in either order.
