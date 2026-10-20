### FIXED — apeGmshViewer refuses a partitioned `.ladruno` set whose parts disagree on the spatial dimension (#2245)

The app read a `<stem>.part-<N>.ladruno` set with part 0's spatial dimension
and never compared the other parts'. It now resolves each part's dimension as
it does for a single file: `INFO/SPATIAL_DIM`, else the one coordinate width.
A set whose parts disagree is refused. The message names each part's value and
where it came from. A part with an unknown dimension beside a part with a
known one is refused as well, rather than read with part 0's value. This is the rule Python's `_validate_part_set` keeps (#2267).
