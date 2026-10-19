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
