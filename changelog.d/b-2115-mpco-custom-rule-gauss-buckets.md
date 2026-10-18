### FIXED — MPCO names Gauss and material-state buckets on a custom integration rule (#2115)

The Gauss and material-state readers of `Results.from_mpco` dropped every
bucket with a nonzero custom-rule index (`<tag>-<Class>[1000:2:0]`) without a
word, so their elements returned no values and nothing said why. MPCO opens a
new custom-rule index for each distinct set of Gauss-point positions, so the
catalog's fixed coordinates cannot be trusted for such a bucket. Each one is
still left unread, and is now named, by path, class, custom index and file, in
one `MPCOBucketSkippedWarning` per bucket. Reading them is deferred until
`GP_X` is checked against a real recorder file.
