### ADDED — soil–pile springs `ops.uniaxialMaterial.PySimple1` / `TzSimple1` / `QzSimple1`

The three stock OpenSees p-y, t-z and q-z materials (Boulanger et al. 1999,
`SRC/material/uniaxial/PY/`) are now typed primitives:
`PySimple1(soil_type, pult, y50, Cd, c=0)`, `TzSimple1(tz_type, tult, z50, c=0)`
and `QzSimple1(qz_type, qult, z50, suction=0, c=0)`. The backbone selector is
`Literal[1, 2]` (`PyTzQzType`): `1` is clay (Matlock 1970 for p-y, Reese &
O'Neill 1987 for t-z and q-z), `2` is sand (API 1993, Mosher 1984,
Vijayvergiya 1977). Validation refuses what the C++ handles worse than an
error: an unknown type or a non-positive capacity or `*50` displacement
calls `exit(-1)` in `revertToStart`, and a negative `Cd` or `c`, or a
`suction` outside `[0, 0.1]`, is silently clamped. `QzSimple1` emits
`suction` and `c` together or not at all, because the Tcl parser reads both
once either is present. A live test drives each spring on a `zeroLength` and
gates the sign convention, the approach to capacity, the `*50` point and the
q-z bearing/uplift asymmetry.
