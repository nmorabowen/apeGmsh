### FIXED — the MPCO reader names every material-state, spring, line-station and layer bucket it skips, in one warning per read; an out-of-range layer filter raises (#2047, #2093)

The MPCO reader used to drop, with no warning, a material-state, spring or layered-shell bucket its decoder refused
(malformed `META`, a `NUM_COLUMNS` mismatch, mixed layer counts in one bucket, several sub-GPs per layer), and the
line-stations listing dropped a bucket the read raises on (a missing `GP_X`, a section code with no canonical): the
bucket's components left the listing and its elements read no values. Each read and each `available_components`
listing now names the file and every such bucket, with its key and the decoder's reason, in one warning:
`MPCOBucketSkippedWarning` (new, `apeGmsh.results.readers._mpco`) at the gauss material-state, spring and line-stations
levels, `LayerBucketUnreadWarning` at the layer level, where the one warning also names the buckets discovery leaves
out. A partition set warns once per part, each naming its own file. The layer listing applies the read's `NUM_COLUMNS`
check, so a bucket the read refuses is no longer listed. A skip the decoder decides by its own predicate (a spring
index the bucket lacks, a component absent from the bucket's META) stays silent, and `MaterialStateComponentRefused`
still reaches the caller. A `gp_indices` / `layer_indices` beyond a layer bucket now raises `ValueError` naming the file
and bucket, as `read_fibers` and the `.ladruno` reader do; it was a silent skip. The layer listing warns once across
`fiber_stress` and `fiber_strain`, and its reason for a beam or zeroLength bucket names `FIBER_CATALOG` as well as
`LAYER_CATALOG`.
