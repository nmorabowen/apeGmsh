### FIXED — a partitioned MP constraint under Penalty, Lagrange or Auto is refused (#1625)

A partitioned emit writes each `equalDOF`, `rigidLink`, `rigidDiaphragm`,
`rigid_body` chain and `node_to_surface` record on every rank that holds one
of its nodes. Under `constraints Transformation` that is required. Under
`Penalty`, `Lagrange` or `Auto`, each copy assembles its own penalty or
multiplier, so the partitioned model answered stiffer than the serial one,
with no warning. Measured on a rigid diaphragm spanning ranks, the eigenvalues
were off by 4.9e-5 on 2 ranks and 7.3e-5 on 4.

Such a deck now raises `BridgeError` before any rank writes a constraint,
and the stream is removed. The message names three remedies: declare
`ops.constraints.Transformation()`, emit unpartitioned
(`ops.tcl(path, flat=True)`), or repartition so the constraint's nodes stay
on one rank. The same rule applies to a stage-claimed constraint, under the
handler of its stage and of every later stage. A constraint held by one rank
is still emitted. Decks under `Transformation`, `Plain` or an undeclared
handler are byte-identical to before.

Writing each record on one rank instead was measured too. It is exact for a
static solve, but `eigen` under OpenSeesMP collapses to 1.0 when a rank holds
no penalty MP element, so the emit refuses rather than writing that deck.
