### FIXED — a rotated instance turns its beam axes and authored loads, not its weight (#1597)

`Assembly.instance(rotate=...)` followed by `bridge` now turns each archived
`geomTransf` `vecxz` (Linear, PDelta and Corotational) by the instance's
rotation. Before, a rotated beam kept its source local axes with no error.

`compose(rotate=...)`, and so a rotated v2 instance, now applies the load
rule from ADR 0117's load note:

- Authored nodal forces and moments, line loads (`beamUniform`) and pressure
  directions turn by `R·v`.
- A self-weight or body force stays global: gravity `g`, `bf`, and the nodal
  loads reduced from them.
- DOF indices never transform.

A rotated module whose nodal load has no `source` (a file older than neutral
schema 2.35.0) raises `ComposeError`: the merge cannot tell whether that load
is a force, which turns, or a weight, which stays. A transform field, load
field or element-load parameter that no frame table classifies raises.
