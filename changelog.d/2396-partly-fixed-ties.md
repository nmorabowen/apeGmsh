### FIXED — an `equalDOF` whose retained node is partly fixed no longer silently loses its tie under the Transformation handler (#2396)

When an `equalDOF`'s retained node is fixed on some of the tied DOFs and
free on others, OpenSees' `Transformation` handler, the bridge default
for `equalDOF` decks, drops the tie on the free DOF at every step.
`TransformationDOF_Group::enforceSPs(0)` resets the constrained node's
free DOF to an absolute zero, and the incremental MP update never
restores it. This is an upstream defect, present in jaabell/master too.
Nothing warned, and the answer was wrong:

- a 1-D site-response column whose left/right ties meet a vertically
  fixed base had its transfer function come out 45–87 % of the right
  one, with a 4 % longer period;
- a two-square static tie on main differs from Lagrange by 10 %.

The global constraint pass now rewrites such a tie as `equalDOF(m, s,
free DOFs)` + `fix(s, fixed DOFs)`. It is the same constraint set under
every handler, and Transformation now matches Lagrange to round-off. The
added `fix` is archived under the master's fix declaration.

Some ties cannot be rewritten statically. Each keeps its declared form
and warns `PartlyFixedTieWarning`, naming Lagrange / Penalty as the safe
handlers:

- the retained node's tied DOFs carry an SP that varies, either a stage
  `s.fix` / `s.support` / `s.remove_sp` or a pattern `p.sp`;
- a stage fixes the slave on a DOF the rewrite would fix;
- the tie is part of a chain, where its slave retains another tie.

Stage-claimed, partitioned, mixed and `rigidLink` ties are covered by
#2406.
