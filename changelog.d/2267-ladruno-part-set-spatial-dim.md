### FIXED — a partitioned `.ladruno` set whose parts disagree on the spatial dimension is refused by name (#2267)

`Results.from_ladruno` stitched a `<stem>.part-<N>.ladruno` set without
comparing the parts' spatial dimensions. It now resolves each part's dimension
as it does for a single file: `INFO/SPATIAL_DIM`, else the one coordinate
width. A set whose parts disagree is refused when it opens. The message names
each part, its value and where the value came from. A part of unknown
dimension beside a part of known dimension is refused as well, the rule the
viewer adopts in #2245. A part whose `INFO/SPATIAL_DIM` is not 1, 2 or 3 now
refuses the set at open with `SpatialDimInvalid`, naming the part. A set whose
parts agree, or whose parts are all of unknown dimension, reads as before.

With a model bound, `Results.from_recorders` now checks a line-stations
record's per-element integration-point counts before it divides the
`integrationPoints` columns. A record over Lobatto 3 and Lobatto 4 beams is
refused for its different counts, naming the elements, as the fibre reader
refuses it, where it used to report "not divisible". Without a model the
columns are divided as before.
