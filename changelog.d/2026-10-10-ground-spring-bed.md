### ADDED — ground-spring bed: `g.decouple_node_set` and `ops.spring_bed` (ADR 0119, Proposed)

`g.decouple_node_set(source, offset=..., tie_dofs=...)` declares one
decoupled node per node of a label or physical group, by name, resolved at
`get_fem_data` (ADR 0049 nodes: broker provenance `decoupled`, tags after the
single decoupled nodes). `offset` is a triple or a callable of the source
coordinates; `tie_dofs=(1, 2, 3)` adds one `equalDOF` record per pair, the
3-dof side node a zeroLength needs on a 6-dof shell node. The handle pairs
each source node with its new node (`.pairs()`).

`ops.spring_bed(ground, at=side, k=..., c=..., tributary=..., orient=...)`
puts a grounded `zeroLength` on every node of the set: one shared
`Elastic 1.0` and `Viscous 1.0 1.0`, per spring and direction a
`Parallel -factors k c`, the ground `fix` and the `ndf` statements. `k` and
`c` are arrays, a row for every spring, or a callable of the source
coordinates and the tributary areas (end zones and per-face local
intensities are the callable's arithmetic); `tributary=` shares each 2-D
element's area among its corners; `do_rayleigh` defaults to `False`, so the
dashpots, not the global Rayleigh pair, damp the springs. This is the San
Ramón Tier-2 T2S deck topology; the single-node check of its patch
(ζ = 0.1 with a global `rayleigh 0 0 1.0 0`) is an engine test. Partitioned
emit refuses the bed, as every node-pair zeroLength (ADR 0049).
`spring_bed(..., at=...)` refuses an `at` set that is not tied to the
structure on every DOF in `dirs` (all of 1..3 with `orient`), since such a
bed carries no load; the shared unit `Elastic`/`Viscous` are unnamed, so a
user material called `spring_bed_unit_elastic` is just a user material.
