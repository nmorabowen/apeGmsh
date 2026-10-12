### ADDED — g.mesh.structured.one_element_per_curve() (R3-S3, #1780)

`g.mesh.structured.one_element_per_curve(*labels)` meshes each selected
curve as exactly one 1-D element (two nodes), the one-element-per-member
switch for trusses and frames.

- **With names**, it takes every curve under each label, physical group or
  part. It refuses (`ValueError`, nothing applied) a curve that bounds or is
  embedded in a surface, or that a definition ties into a host (embedded,
  reinforce, tie, tied contact: the definition also references a surface or
  volume). A curve referenced only by a node-level definition, such as a BC
  or a diaphragm over beam lines, is allowed.
- **With no names**, it takes every *free* curve: one that bounds no surface,
  is embedded in none (`g.mesh.editing.embed`), and that no definition
  references in any role. Definitions are read from every declaration
  composite on the session, including ones added later (loads, masses and
  decoupled nodes excepted), field by field, with no list of known classes.
  Each curve skipped because of a definition is reported in a
  `CurvesSkippedWarning` (`apeGmsh.mesh._mesh_structured`), with the
  definition that references it.
- **Fails closed:** an unknown name raises `KeyError` listing the existing
  labels; a name with no curve, a definition whose references cannot be
  enumerated, or a model with no free curve raises `ValueError`.
- Definitions are read when it is called, so call it after every tie is
  declared. To coarsen a refused curve on purpose, call
  `set_transfinite_curve(curve, n_nodes=2)`.
