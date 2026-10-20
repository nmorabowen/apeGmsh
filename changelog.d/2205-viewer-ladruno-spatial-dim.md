### FIXED — apeGmshViewer refuses an out-of-range `.ladruno` `INFO@SPATIAL_DIM` by name, as Python's `SpatialDimInvalid` does (#2205)

A `.ladruno` whose `INFO@SPATIAL_DIM` was 7, 0 or -1 used to open in the
viewer as if the attribute were absent. The viewer then took the dimension
from the width of `MODEL/NODES/COORDINATES`, while Python has refused the same
file since #2174. A value of 2.9, `"abc"` or an array of two values was refused
too, but only as "is not an integer".

The viewer now uses Python's rule (`_spatial_dim_attr`). A present attribute
with one value is read as a number, and only 1, 2 or 3 is accepted, so `2.0`
and the string `"2"` read as 2. Every other value refuses the file with a
`layout` error that names the file, the attribute and the value. An absent
attribute, or one with no value, is still inferred from the coordinates. The
check runs for each part of a partition set too. All 18 cases, Python's table
among them, gave the same answer from both readers on the same files.
