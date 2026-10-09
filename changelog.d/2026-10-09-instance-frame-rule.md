### FIXED — a rotated instance turns its constraint directions (#1593)

`compose(rotate=...)`, and `Assembly.instance(rotate=...)` followed by
`bridge`, now apply the instance frame rule to every constraint record the
merge carries. Points move by `R·x + t` and directions turn by `R·v` only.
Before, a rigid diaphragm's `plane_normal` and `offsets`, a rigid link's
`offset`, a rigid body's `omega`, a reinforcement tie's bar `direction` and a
tie's `projected_point` were copied from the source unchanged. So a rotated
floor kept its source plane (`rigidDiaphragm 3` where `2` was right), with no
error. A constraint record kind that has no entry in the frame table now
raises `TypeError` at compose time.
