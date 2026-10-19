### FIXED — an MPCO nodal-force read refuses an element id that two buckets hold (#2173)

MPCO writes each element to exactly one bucket of a token. In a corrupt
file where one element id sat in two `globalForce` or `localForce` buckets,
`Results.from_mpco(...).elements.get(...)` stitched both buckets and
returned that element twice, with no warning: header 0 holding `[7, 11]` and
header 1 holding `[11, 33]` read back as `[7, 11, 11, 33]`. The read now
raises `NodalForceDuplicateElementError` (a `ValueError`), naming the id,
both bucket paths and the file. Buckets that differ by header index, by
custom rule (`[1000:1:0]` and `[1000:2:0]`) or by class are all checked, and
the read refuses whichever elements it asks for. A bucket the reader leaves
unread is not counted, and `available_components()` still lists the file's
components. A file where each element appears once reads as before.

The Gauss-point, line-station and fiber reads get the same check in a later
slice.
