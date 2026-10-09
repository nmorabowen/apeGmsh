### ADDED — Assembly rehydrates element rows whose args vary inside a physical group (ADR 0117 D8, AS2b)

`Assembly.bridge()` no longer refuses an instance whose element rows carry
different args inside one physical group, such as an orientation-derived
transform (`geomTransf.Linear(orientation=Spherical(...))`) that gives each
element of a ring its own transform. Each distinct args row becomes one
element spec on an element group that the bridge registers on the merged FEM
as `{instance}.{pg}#<k>` (`k = 1, 2, ...` in tag order). The groups are written
to `/physical_groups/element_side` of the assembly archive, so the archive
re-bridges as an instance to the same deck. A source physical group whose
name equals a synthesized one (for example `Ring#1` beside `Ring`) is refused
before anything is registered.
