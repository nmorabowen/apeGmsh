### FIXED — a `g.decouple_node` label names the node on the FEM snapshot, so `ZeroLength(nodes=("ground", n))`, `ops.fix(label=)` and `ops.ndf("ground")` resolve it (#2389)

`g.decouple_node(..., label="ground")` kept its label on the session
declaration only. The snapshot had no such label, so a node-pair
`ops.element.ZeroLength(nodes=("ground", n))` raised
`node label 'ground' not found on FEM snapshot`, and a script had to fall
back to the raw `fem.nodes.decoupled_ids[0]`.

- `get_fem_data` now registers each decoupled label as a node label of
  the snapshot. It holds the decoupled node, or every node of a
  `g.decouple_node_set(..., label=)`, and `model.h5` keeps it.
  `fem.nodes.select(label=)`, `ops.fix(label=)`, `ops.mass(label=)` and
  `ops.nodes.one(label=)` reach it.
- A decoupled label that is already a geometry label raises at
  `get_fem_data`. Merging the two would add the decoupled node to every
  existing use of that label.
- A node-pair endpoint string (`ZeroLength`, `CoupledZeroLength`,
  `TwoNodeLink`) resolves a node label first, then a physical group of
  that name. It must select exactly one node. A name that is both, with
  different nodes, is refused.
- `ops.ndf` also takes a decoupled label. A node-set label sizes every
  node of the set.
