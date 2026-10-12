### FIXED — a partitioned MP constraint under Penalty, Lagrange or Auto is refused (#1625)

**Changed (behaviour):** a partitioned emit with an MP constraint
(`equalDOF`, `rigidLink`, `rigidDiaphragm`, a `rigid_body` chain or
`node_to_surface`) that two or more ranks would write, under `Penalty`,
`Lagrange` or `Auto`, now refuses with `BridgeError`. Before, it wrote the
deck.

The partitioned emit writes such a record on every rank that holds one of
its nodes. `constraints Transformation` requires that. Under the other
handlers, each copy assembles its own penalty or multiplier, so the
partitioned model answered stiffer than the serial one, and nothing warned.
On a rigid diaphragm spanning ranks, the eigenvalues were off by 4.9e-5 on
2 ranks and 7.3e-5 on 4.

The check uses the handler the deck actually runs under. That is the
declared one, the `Lagrange` the bridge auto-emits for an equation tie, or
for a staged model each stage's own (a stage-claimed constraint is checked
under its stage's handler and every later stage's). The check runs before the
partitioned emit writes anything, so a deck already at the path is left
untouched.

The message names two remedies, both verified against the serial run:
`ops.constraints.Transformation()`, or running serially with
`ops.tcl(path, flat=True)`. A constraint that only one rank holds is still
emitted. Decks under `Transformation`, `Plain` or an undeclared handler are
byte-identical to before.
