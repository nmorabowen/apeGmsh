### FIXED — `.out` transcode: a node without rotation DOFs reads NaN rotation, not 0.0 (#1971)

One `recorder Node` line over a mixed-ndf model writes every requested DOF
for every node and pads the DOFs a node lacks with 0.0. A tet node beside a
shell, or a 2-dof quad node beside a frame, then read back a rotation of
0.0, a value a viewer drew as a real field. `RecorderTranscoder` now reads
the model's `/opensees/nodes_ndf` and writes NaN for each rotational
component (rotation, angular velocity and acceleration, reaction moment,
unbalance moment) whose slot the node's DOF layout (its ndf and the
model's ndm) does not have. A recorded node missing from `nodes_ndf`
raises. Without a model the transcode is unchanged. `PARSER_VERSION`
moves to 1.2, so a cached transcode is re-made. `.ladruno` and MPCO reads
keep the padded 0.0 for now: their readers have no per-node ndf yet.
