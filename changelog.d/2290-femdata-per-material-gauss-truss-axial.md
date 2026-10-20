### ADDED — a raw deck on stock records per-Gauss-point damage and the Truss axial force over a FEMData archive (#2290)

`spec.emit_default` runs a raw openseespy deck with a FEMData archive
(`fem.to_h5`). On stock openseespy it now records two more kinds of element
result:

- **Per-Gauss-point material records** (`damage_tension`,
  `damage_compression`: one `material <gp> damage` file per point) are now
  read through the live element pairing of #2269. They used to be refused.
  The layout is the record's class's, and the live `eleType` must be that
  class. A `fibers` record over a FEMData archive is still refused by name:
  it counts its stations from each element's `beamIntegration`, which only
  the bridge's archive holds.
- **A Truss `axial_force` Gauss record** now takes the classic recorders.
  MPCO cannot write this record, so `emit_default` used to raise before it
  tried the MPCO recorder. Now such a record skips MPCO, on every stock
  build, and the `DefaultRouteWarning` names the records it reroutes. It
  counts as a Truss record when its class is in the Truss family, or, when
  it names no class, when every element's live `eleType` is. Every other
  MPCO refusal still raises at `__enter__`, with or without a Truss axial
  force beside it: an `axial_force` on any other class (an elastic beam,
  say), a gauss component with no MPCO token, or a section deformation on
  line stations.

The `.out` fibre read now loads the model once. It used to load it twice.
