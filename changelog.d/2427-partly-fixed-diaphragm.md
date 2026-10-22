### FIXED — a `rigidDiaphragm` whose master is partly fixed in plane is refused under Transformation (#2427)

Under OpenSees' Transformation handler, a `rigidDiaphragm` master that is
fixed on some of its in-plane DOFs and free on others gets a wrong
response. The in-plane DOFs are ux, uy and rz for a floor normal to z.
The slaves are right, but the master accumulates its **total** in-plane
displacement at every step instead of the increment. In a raw openseespy
probe over four linear steps, the master's ux went 1, 3, 6, 10 × the
increment, where Lagrange gives 1, 2, 3, 4. Nothing warned, so every
recorder or drift read at the master node was wrong.

The bridge now raises `BridgeError` for such a deck when an analysis
runs under Transformation, whether declared, auto-emitted for MP
constraints, or forced by a modal deck. The check covers both a global
`fix` and a stage or pattern SP on the master. The message names
`ops.constraints.Lagrange()` and `Penalty(...)`, or dropping the
master's in-plane fix.

These decks are exact and emit as before:
- a master free in plane;
- a master fully fixed in plane;
- a master fixed only out of plane (`fix 0 0 1 1 1 0` for a floor normal
  to z).
