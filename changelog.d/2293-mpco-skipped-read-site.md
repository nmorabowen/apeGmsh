### FIXED — an MPCO read narrowed with `element_ids` names only the skipped buckets that hold an element it asked for (#2165, #2293)

MPCO discovery at the Gauss, material-state, spring, fiber and
line-station levels emitted `MPCOBucketSkippedWarning` itself, once per
skipped bucket (a custom-rule index, no catalog row, a key that does not
parse), whatever the read asked for. A read narrowed with `element_ids`, or
with a selection that resolves to element ids, was told about buckets
whose elements it never wanted.

Discovery now returns the skipped buckets with their reasons
(`scan_gauss_buckets`, `scan_material_state_buckets`, `scan_fiber_buckets`,
`scan_line_station_buckets`, `skipped_spring_buckets`), and the read names,
once each, only the ones holding a requested element. An unfiltered read
and a component listing still name every skipped bucket, and a bucket
whose `ID` cannot be read is always named. A bucket its decoder refuses at
read time (Gauss material-state and spring META, Gauss META labels)
follows the same rule. The message text is unchanged, and the warning
points at the read's caller; the material-state read used to point inside
`read_gauss`.

The Gauss composite's derived scalars (von Mises, principal, shell) and
censuses list the stored components before they read. Under a filter
that listing no longer names skipped buckets; the filtered read does.

`discover_*_buckets` keep their two-value return and still name every
skipped bucket, for a caller with no element filter.

The nodal-force level already filtered its `NodalForceBucketSkippedWarning`
this way. Not changed here: the layer level's `LayerBucketUnreadWarning`
still names its unread buckets whatever the read asked for.
