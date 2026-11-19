### FIXED — masses on mixed-ndf models run in openseespy and `build('live')`

openseespy's `mass` command sizes the mass matrix by the builder ndf in
force (`OPS_addNodalMass` reads `OPS_GetNDF`), not by the node's own ndf.
The bridge writes `ops.mass(node, *values)` with one value per DOF of the
node (ADR 0048), so any mass on a node whose ndf differs from
`ops.model(ndm, ndf)` failed in `Node::setMass` with "incompatible
matrices". For example, a solid block (3-dof nodes) beside a frame under
`ops.model(ndm=3, ndf=6)` failed at its first solid-node mass.

The py deck and the live emitter now wrap such a mass in
`ops.model('basic', '-ndm', ndm, '-ndf', <node ndf>)` and then restore the
envelope. Consecutive off-envelope masses in one block share a single
bracket. On openseespy a re-issued `model` only sets ndm and ndf, so the
bracket clears none of the timeSeries, geomTransf, beamIntegration or
damping definitions. Tcl decks are unchanged: Tcl `mass` sizes the matrix
by its argument count, and a Tcl `model` re-issue would clear those
definitions (ADR 0099). Masses on nodes whose ndf matches the envelope are
emitted as before.

`equalDOF` without DOFs also uses the builder ndf, but the bridge never
emits it: the constraint resolver always lists the tied DOFs.
