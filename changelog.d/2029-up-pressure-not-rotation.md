### FIXED — a 2-D `LadrunoUP` node's pore pressure no longer reads as `rotation_z`; partitioned `.ladruno` and MPCO reads mask rotations (#2029, #2028)

A 2-D `LadrunoUP` vertex node and a 2-D frame node both have ndf 3, and
both lay their third DOF in the `rotation_z` slot: a rotation at the frame
node, a pore pressure at the u-p node. The rotation masks keyed on ndf and
ndm alone, so the `.out`, `.ladruno` and MPCO reads returned the pressure
as `rotation_z`, with no warning, even with `model_h5=`.

The masks now also ask the model's DOF roles, the table the
`recorder.declare` emit already uses. A rotation reads NaN at a node that
a u-p element touches, or that no beam, shell or zero-length element
touches, or, for an element-less node, in a model that holds a u-p
element. A 2-D frame node still reads its `rotation_z`. The roles are read
from `model_h5` on the first rotation read that needs them.

`Results.from_ladruno` and `Results.from_mpco` on a partitioned set now
pass the model's per-node ndf and roles to every part, so the mask applies
there too. Before, the parts got nothing and each warned
`AbsentDofWarning`. A partitioned read without a model now warns once, not
once per part.
