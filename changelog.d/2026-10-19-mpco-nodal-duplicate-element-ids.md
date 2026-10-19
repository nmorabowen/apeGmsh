### FIXED — an MPCO nodal-force read refuses an element id it would return twice (#2173)

MPCO writes each element once, to exactly one bucket of a token. In a
corrupt file where one element id sat in two `globalForce` or `localForce`
buckets, `Results.from_mpco(...).elements.get(...)` stitched both buckets
and returned that element twice, with no warning: header 0 holding `[7, 11]`
and header 1 holding `[11, 33]` read back as `[7, 11, 11, 33]`. The read now
raises `MPCOElementInTwoBuckets` (a `ValueError`), naming the id, both bucket
paths and the file. Buckets that differ by header index, by custom rule
(`[1000:1:0]` and `[1000:2:0]`) or by class are all checked, and an id
repeated inside one bucket's `ID` (`[7, 11, 11]`) refuses too, naming that
bucket.

The check is filter-aware: it refuses only for the ids the read returns, so
`ids=[7, 33]` on the file above still reads both elements once. It reuses
the ids the read already selects and reads no `ID` a second time. A bucket
the reader leaves unread is not counted, `available_components()` still
lists the file's components, and a file where each element appears once
reads as before.

`MPCOElementInTwoBuckets` and its helper live in the new leaf module
`apeGmsh.results.readers._mpco_refusals`, which the Gauss-point,
line-station and fiber reads adopt in a later slice.
