### FIXED — MPCO reads Gauss and material-state buckets on a custom integration rule (#2115)

MPCO brackets an element it has no fixed rule for as a custom rule and
numbers those rules from 1 across the file (`209-SixNodeTri[1000:2:0]`). The
Gauss and material-state readers dropped every bucket whose custom-rule
index was not 0, so the `IntRule.Custom` catalog rows for SixNodeTri,
BezierTri6 and BezierTet10 were never reached, and
`Results.from_mpco(...).elements.gauss.get(component="stress_xx")` returned
no values for those elements, without a warning. Such a bucket is now read
through the catalog's row for its class at that rule. Its META is checked
against the row's Gauss-point count and widths, and a mismatch is refused.
A custom-rule bucket whose class has no catalog row at its rule is left
unread. It is named, by path and class, in one `MPCOBucketSkippedWarning`
per bucket.
