### ADDED — instant read on the live route: `Node.read`, `NodeSet.read`, `ops.elements` (ADR 0115 D1–D2, program slice R2-b, #1695)

After a live `ops.analyze(...)`, the bridge keeps a read session, and a
script reads results by the label it modelled with, with no `ops_raw` call:
`ops.nodes.one(label="tip").read("rotation_z")` returns a float, and
`ops.nodes.get(label="supports").read("reaction_force_y")` returns an
`ndarray (N,)` in `.tags` order, so a sum is `.sum()`. Components use the
results package's canonical names (`displacement_x`, `rotation_z`,
`velocity_y`, `acceleration_x`, `reaction_force_y`, `reaction_moment_z`, …),
mapped per node through the bridge's own DOF layout, so `rotation_z` on a
planar `ndf=3` frame is DOF 3. A reaction read calls `reactions()` first.
Values are in model units at the last committed state.

`ops.elements.get|one(pg=|label=|ids=)` selects FEM elements by FEM id,
beside the `ops.element` constructors. Its `.read` serves per-element
scalars only, the `axial_force` of Truss and CorotTruss; anything else raises
and points to `Results`. A read looks the FEM id up in the tag plan the live
emit used, so it never allocates a tag or builds, and a default build and an
`element_tags="fem"` build read the same values.

A read raises `BridgeError`, naming the cause, when no live analyze has run
(a deck run is read with `ops.results(path)`), and once the session is stale:
after `eigen`, another `analyze`, `run()`, a deck run (`tcl`/`py` with
`run=True`, `run_remote`), or another bridge rebuilt the in-process domain.
A component the node lacks raises, naming the node and its `ndf`. A
rotational component (`rotation_*`, `reaction_moment_*`, `angular_*`) is read
only on a node that a beam-column or shell touches and no other family but a
truss: on a u-p node (`ndm=2, ndf=3`) that slot is pore pressure, and on any
other or unknown family it is undefined, so the read raises instead of
guessing. The staleness guard sees the bridge's own live work only; a raw
`get_ops().wipe()` followed by a raw rebuild of the domain is outside it.
`EigenResult.periods` is documented. `analyze` still returns `int`, and no
verb or Protocol method was added.

`ops.recorder.Element` takes `label=` (one geometry label or a list) on a
bridge built with `element_tags="fem"`, where an element's OpenSees tag is its
FEM id; it resolves at the call, before anything registers. Under the default
`element_tags="sequential"` it raises `BridgeError` naming
`element_tags="fem"`; the sequential form arrives with R2-f (#1661).
