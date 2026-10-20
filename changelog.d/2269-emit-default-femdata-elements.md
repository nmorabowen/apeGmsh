### FIXED — `emit_default` records a hand-built deck's element results with a FEMData archive on stock openseespy, pairing each FEM element with the live element by fact (#2269)

#2251 let `spec.emit_default(..., model_h5=<fem.to_h5 file>)` record node
results on a build with no MPCO recorder (pip openseespy), but it refused
every element record: a FEMData archive pairs no FEM element id with an
OpenSees tag. The classic-recorder route now pairs each recorded FEM
element with the live element of its own id, and only when the live domain
shows it is that element:

- the live domain holds an element with that tag;
- its `ops.eleNodes` equals the FEM connectivity, in order;
- its nodes pass #2251's node identity check;
- its `ops.eleType` has a bridge element token.

The type token gates an elastic beam's `localForce` line stations, as the
bridge model's tokens do. A record that names its element class (the
response catalog's key on read) must name the live class of every element
it records. `run.results()` carries the pairing into the read, so Gauss
points, line stations and nodal forces read under their FEM ids, and the
file is stamped as FEM-indexed.

On real stock openseespy 3.7.1, these equal the live `eleResponse` at
rtol 1e-12:

- the Gauss stress of every tet in the `column_to_surface` case-study deck;
- a planar portal's force-beam line stations (`section force`);
- its elastic girder's two `localForce` stations and its `globalForce`
  nodal forces.

With a FEMData archive the route refuses before any recorder reaches the
domain:

- by FEM id: an element that is not live under its FEM id, whose live
  nodes differ from the FEM connectivity or its order, whose nodes are
  not the FEM's, or that the FEMData does not hold;
- by FEM id, naming both: an element whose ordered connectivity another
  FEM element shares (a stacked quad pair). `eleNodes` cannot tell the
  two apart, so a deck that swapped their tags would record each one's
  response under the other's id;
- an `elements` record that names no class, or whose live class has no
  nodal-force layout for its frame (for example `ZeroLength`);
- by class name: a live class with no bridge token (for example
  `ModElasticBeam2d`), or a record class that is not the live class;
- by record name: `fibers` and per-Gauss-point material records, which need
  the bridge's model archive.

Unchanged: with a bridge archive the route never asks the live domain, and
its `.out` files and reads are byte-identical to `main`'s.
