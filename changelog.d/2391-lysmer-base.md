### ADDED — `lysmer_base`: a Lysmer–Kuhlemeyer viscous base for site-response models, and its input force (#2391)

`apeGmsh.opensees.site_response.lysmer_base(ops, base, ground, rho=,
vs=, vp=None)` builds the compliant base under a labelled base curve
(2-D) or face (3-D). A script no longer assembles it by hand.

- For each base node it registers a `ZeroLength` from that node's ground
  node, a `g.decouple_node_set(base, ...)` node, to the base node.
- The `ZeroLength` carries a `Viscous` dashpot `C = rho Vs A_i` on each
  horizontal DOF. `A_i` is the node's tributary length times `thickness`
  in 2-D, or its tributary area in 3-D.
- On the vertical DOF it carries `rho Vp A_i` when `vp` is given.
  Otherwise the vertical DOF is fixed, except on a node that a
  `periodic` or `equal_dof` tie already slaves.
- It fixes the ground nodes and states their `ndf`.

`LysmerBase.add_input(pattern)` loads each base node with `C_i` along
the shaking direction. The pattern's time series is the outcrop
velocity, so the force is `2 rho Vs A_i v_inc(t)`; `motion="incident"`
doubles it for an incident-wave velocity. A live test checks the result
against a closed form: over a matched half-space, the surface velocity
of a homogeneous column is the outcrop pulse delayed by `H / Vs`.

Scope: linear base elements only (a quadratic base is refused). `ndf=`
sizes the base when the soil's ndf differs from the model's. On a model
reloaded from `model.h5`, `ground=` can be the node set's label, and a
2-D base must be straight; a reloaded 3-D base is refused with a
`lysmer_base` error. The vertical fixity skips any DOF a tie already
constrains: an `equal_dof` / `periodic` record, or an
`ops.equation_constraint` row declared before the call. A
`PartlyFixedTieWarning` flags a tie whose master the helper fixed on
some of the tied DOFs only. The `Transformation` handler mishandles such
a tie (#2396); use `Lagrange`. A viscous base gives a static gravity
stage no lateral restraint, so restrain the base there.
