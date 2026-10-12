### ADDED — g.mesh.structured.one_element_per_curve() (R3-S3, #1780)

`g.mesh.structured.one_element_per_curve(*labels)` meshes each selected
curve as exactly one 1-D element (two nodes), the one-element-per-member
switch for trusses and frames. With names, it takes every curve under each
label, physical group or part; with no names, it takes every *free* curve,
one that bounds no surface and is embedded in no surface or volume, so the
slab of a frame-plus-slab model meshes as before. It fails closed: an
unknown name raises `KeyError` listing the existing labels, and a name with
no curve, a named curve that bounds or is embedded in a surface, or a model
with no free curve raises `ValueError`, with nothing applied. To put two
nodes on a surface edge on purpose, call
`set_transfinite_curve(curve, n_nodes=2)`.
