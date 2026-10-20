### ADDED — a raw deck over a FEMData archive records nine more catalogued element classes on stock, and the live pairing can tell twins of different classes apart, with no route using it yet (#2290, part C)

`spec.emit_default(..., model_h5=<fem.to_h5 file>)` on a stock build with no
MPCO recorder pairs each recorded FEM element with the live element of its
own id. The pairing typed only the classes the bridge builds. It now also
types nine catalogued classes the bridge has no element for, each by its
family's token: `BbarBrick`, `SSPbrick` and `Twenty_Node_Brick` (solids),
`SSPquad` and `EightNodeQuad` (planes), `ShellDKGT`, `ShellNLDKGQ` and
`ShellNLDKGT` (shells), and `ForceBeamColumnCBDI2d` (a force beam, so the
elastic-beam `localForce` gate still refuses it). The response layout is the
catalog's entry for the class the record names.

`Twenty_Node_Brick` keeps its nodes in upstream's serendipity order, so its
`eleNodes` is compared with the FEM (Gmsh) connectivity in that order. A
deck that builds it straight from the Gmsh connectivity refuses by element
id; on a real build upstream exits the process on that element first.

Each served class is checked on stock openseespy 3.7.1 against the live
`eleResponse` at rtol 1e-12, and each Gauss point read back is checked to sit
where the element sampled it.

Still refused by class name: `BezierTri6` and `BezierTet10` (stock cannot
build them); `ShellMITC9` (stock answers 84 stress values for its 72);
`ElasticForceBeamColumn2d`/`3d` (their `section force` file is empty);
`ForceBeamColumnWarping2d` (no stock section it can invert). `Truss2`,
`CorotTruss2` and `ModElasticBeam2d` are still refused here and come in a
follow-up.

`LiveDomainFacts.pair_elements` also takes `named_classes`, the classes each
recorded element's records name. With it, a recorded element that shares
its ordered nodes with another FEM element pairs when every record of it
names a class, every twin is live, and no twin's live class is a named one.
The class gate then refuses a swapped deck. Without it, twins refuse as
before. No route passes it yet, so `emit_default` still refuses every
recorded element that has a twin. The follow-up wires it in.
