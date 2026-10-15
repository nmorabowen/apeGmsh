### FIXED — `.ladruno` and MPCO reads: a node without rotation DOFs reads NaN rotation with `model_h5=`, and warns without it (#1971)

Neither file records a node's ndf, so over a mixed model the recorder writes
0.0 into the rotation columns of a node that has none: a tet node under a
shell, a 2-dof quad node beside a 2-D frame (mode stages included), or the
pressure DOF of a 3-D u-p node. `Results.from_ladruno(..., model_h5=...)`
and `Results.from_mpco` now read the model's `/opensees/nodes_ndf` and write
NaN for each rotational component (rotation, angular velocity and
acceleration, reaction and unbalance moment) whose slot the node's DOF layout
lacks. A model that misses a node of the file is not used. Without per-node
ndf (no `model_h5=`, an archive without `/opensees/nodes_ndf`, or a
partitioned read) the values stay as recorded, and the first rotation read
over continuum elements warns `AbsentDofWarning` once. The fork recording
per-node ndf (#2015) is the permanent fix.
