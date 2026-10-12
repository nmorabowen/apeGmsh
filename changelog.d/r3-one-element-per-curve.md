### ADDED — g.mesh.structured.one_element_per_curve() (R3-S3, #1780)

`g.mesh.structured.one_element_per_curve(*labels)` meshes each selected
curve as exactly one 1-D element (two nodes), the one-element-per-member
switch for trusses and frames. With names, it takes every curve under each
label, physical group or part. With no names, it takes every *free* curve:
one that bounds no surface, is embedded in no surface or volume
(`g.mesh.editing.embed`), and is not the member of a non-conformal tie
(`g.constraints.embedded`, `g.reinforce`, `g.embed`). So it constrains no
surface edge and never coarsens a tied rebar. It fails closed: an unknown
name raises `KeyError` listing the existing labels. A `ValueError` is raised,
with nothing applied, for a name with no curve, a named curve that is not
free (the message names the surface or tie), a tie whose member cannot be
resolved, or a model with no free curve. To put two nodes on such a curve on
purpose, call `set_transfinite_curve(curve, n_nodes=2)`.
