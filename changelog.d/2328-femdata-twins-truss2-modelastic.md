### ADDED — a raw deck over a FEMData archive records named twins, Truss2, CorotTruss2 and ModElasticBeam2d, and long-hinge line stations (#2328)

`spec.emit_default` runs a raw openseespy deck with a FEMData archive
(`fem.to_h5`). Four cases it used to refuse are now recorded.

- **Twins told apart by class.** Two FEM elements on the same ordered
  nodes used to refuse whenever either was recorded. The emit route now
  passes each element's record classes to the pairing. A recorded element
  with twins is accepted when every record of it names its class, every
  twin is a live element, and no twin's live class is a named one. Twins
  of equal classes, a record that names no class, and a twin the deck did
  not build still refuse. The refusal now says how to tell them apart:
  name the record's element class.
- **`Truss2` and `CorotTruss2`** record their Gauss `axial_force`.
- **`ModElasticBeam2d`** records its `localForce` line stations and its
  nodal forces, as the other elastic beams do.
- **Long-hinge line stations.** A HingeRadau rule with `lp_i + lp_j >= L/4`
  writes its stations out of order, legally. Over a FEMData archive, its
  record used to be refused, with advice to pass `model=` that a FEMData
  user cannot follow. Each element's file stations are now checked against
  the section locations the live domain gave before the run. A match is
  read as written; a mismatch is refused by FEM id.

A line-stations transcode handed a pairing that is neither the bridge's
model nor the live facts now raises `ValueError`, as the fibre read does,
not `TypeError`.
