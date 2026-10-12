### FIXED — `ops.recorder.declare` node components map through the per-node DOF layout (B5-b, #1723, #1703)

In a 2-D frame (`ndm=2`, `ndf=3`), `ops.recorder.declare(nodes=("rotation_z",))`
wrote `-dof 6`. OpenSees answers a DOF past the node's ndf with 0, so the
rotation column read 0 instead of `PL²/2EI`, and nothing warned. Every node
component now maps through the bridge's DOF layout for the node's own ndf. That
covers rotations, angular velocity and acceleration, reaction moments and
`moment_*`. It is the layout loads, masses, SPs and `.read` already use. A
planar rotation is DOF 3. A 3-D frame deck writes the same lines as before.

One rotation-role table (`_readback.py`) now decides where a DOF slot is a
rotation. `.read`, the `declare` emit and `ops.results` all share it.

- A rotation is recorded on a beam-column, shell or zero-length node
  (`ZeroLength`, `ZeroLengthSection`, `CoupledZeroLength`).
- It is also recorded on an element-less node, such as a decoupled or diaphragm
  master, when the model holds no u-p or unknown element class.
- A node touched by `LadrunoUP` refuses, because DOF `ndm + 1` there is pore
  pressure. So does a node touched by a class the table does not name, even
  when a beam also touches it.
- A node with no DOF for the component (a truss node, or `displacement_z` in
  2-D) raises a `BridgeError` that names the node.
- `.read` no longer refuses a zero-length node's rotation (#1703).

`ops.results` now re-derives a declared record's columns with the same function
the emit uses, and it reads a declared planar `rotation_z`. Before, it refused
it.

`pore_pressure` in `recorder.declare(nodes=...)` now raises. It used to emit
`-dof -1 pressure`, and OpenSees' `pressure` node response is the PFEM pressure
constraint, not a u-p pressure DOF.

`ModelData.recorders` covers hand-written decks, so it has no element plan. It
maps through the deck's declared `(ndm, ndf)`. At `ndm=2`, `ndf=3` it therefore
cannot tell a hand-written u-p node from a frame node, and it records DOF 3 for
`rotation_z` on either.
