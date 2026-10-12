### ADDED — g.mesh.structured.one_element_per_curve() (R3-S3, #1780)

`g.mesh.structured.one_element_per_curve(*labels)` meshes each selected
curve as exactly one 1-D element (two nodes), the one-element-per-member
switch for trusses and frames. With names, it takes every curve under each
label, physical group or part. With no names, it takes every *free* curve:
one that bounds no surface, is embedded in no surface or volume
(`g.mesh.editing.embed`), and is referenced, in any role, by no definition
in `g.constraints`, `g.embed` or `g.reinforce` (ties, embedded members,
bars, and kinds added later: definitions are read field by field, with no
list of known classes). So it constrains no surface edge and never coarsens
a tied member. It fails closed: an unknown name raises `KeyError` listing the
existing labels. A `ValueError` is raised, with nothing applied, for a name
with no curve, a named curve that is not free (the message names the surface
or the definition), a definition whose references cannot be enumerated, or a
model with no free curve. Definitions are read when it is called, so call it
after every tie is declared. To put two nodes on a curve that is not free on
purpose, call `set_transfinite_curve(curve, n_nodes=2)`.
