### FIXED — `Path(time=, values=)` with unequal lengths is refused at construction (program slice R0-d, #1583)

`timeSeries.Path(time=(0, 1, 2), values=(0, 1))` used to construct, and
only died at stage close inside numpy (`fp and xp are not of the same
length`); on a route that never interpolates it could have become a
silently wrong load history. `Path.__post_init__` now raises
`ValueError` naming both lengths ("time has 3 samples and values has 2")
and the fix (one time per value, or `dt=`), before the public
`ops.timeSeries.Path(...)` registers anything. An empty `values=` or an
empty `time=` is refused the same way.
