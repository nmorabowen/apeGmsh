### FIXED — partly fixed retained nodes on staged, partitioned, mixed and `rigidLink` constraints (#2406)

The #2396 rewrite of an `equalDOF` whose retained node is partly fixed,
to `equalDOF(m, s, free DOFs)` + `fix(s, fixed DOFs)`, ran only in the
global, unpartitioned constraint pass. It now also covers:

- the global partitioned pass, on every rank that carries the tie;
- `g.constraints.equal_dof_mixed`, split by retained DOF: a pair whose
  retained DOF is fixed becomes a `fix` of its constrained DOF;
- the `equalDOF` half of `g.constraints.node_to_surface`.

A tie a stage claims (`s.equal_dof`) is checked too, flat or
partitioned, but it is never rewritten. A `fix` added after an analysis
is total while an `equalDOF` is incremental, so a slave that moved in an
earlier stage would snap back to zero. Such a tie keeps its declared form
and warns `PartlyFixedTieWarning`, as do the cases #2396 cannot rewrite.

A `rigidLink` cannot be split this way, because its slave DOFs depend on
the master's rotations. It has the same upstream defect: in a raw
openseespy probe, a `rigidLink beam` from a node fixed on one DOF gives
`[0.5176, 0.2165]` under Transformation against `[0.3574, 0.3574]` under
Lagrange. It now warns `PartlyFixedTieWarning`, once per link or rigid
body, when its retained node is fixed, or varied by a stage or a pattern,
on only some of the DOFs it maps: every DOF for `beam`, the translations
for `rod`.

Rigid links also count as chain links now. An `equalDOF` into a node
that retains a `rigidLink` keeps its form and warns: rewriting it would
leave that node partly fixed under the link.
