### FIXED — `ops.recorder.declare` node components map through the per-node DOF layout (B5-b, #1723)

In a 2-D frame (`ndm=2`, `ndf=3`), `ops.recorder.declare(nodes=("rotation_z",))`
wrote `-dof 6`, so the rotation column read 0 instead of `PL²/2EI`, with no
warning. Every node component (rotations, angular velocity and acceleration,
reaction moments, `moment_*`) now maps through the bridge's DOF layout, the
same one loads, masses, SPs and `.read` use. A planar rotation is DOF 3.
`ndm=3`, `ndf=6` decks are unchanged.

A component with no DOF at a targeted node now raises a `BridgeError` that
names the node. Before, it recorded another quantity or a 0. This covers a
rotation on a truss or solid node, and a rotation on a u-p node, where DOF 3
is pore pressure. A rotation is recorded only where a beam-column or shell
touches the node. `ModelData.recorders` (hand-written decks) maps through the
deck's declared `(ndm, ndf)`.
