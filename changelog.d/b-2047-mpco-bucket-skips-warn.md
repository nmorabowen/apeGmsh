### FIXED — the MPCO reader names every material-state, spring and layer bucket it skips, in one warning per read (#2047, #2093)

The MPCO reader used to drop, with no warning, a material-state, spring or layered-shell bucket its decoder refused
(malformed `META`, a `NUM_COLUMNS` mismatch, mixed layer counts in one bucket, several sub-GPs per layer, a
`layer_indices` / `gp_indices` beyond the bucket): the bucket's components left the listing and its elements read no
values. Each read and each `available_components` listing now names every such bucket, with its key and the decoder's
reason, in one warning: `MPCOBucketSkippedWarning` (new, `apeGmsh.results.readers._mpco`) at the gauss material-state
and spring levels, `LayerBucketUnreadWarning` at the layer level, where the one warning also names the buckets
discovery leaves out. The layer listing applies the read's `NUM_COLUMNS` check, so a bucket the read refuses is no
longer listed. A skip the decoder decides by its own predicate (a spring index the bucket lacks, a component absent
from the bucket's META) stays silent, and `MaterialStateComponentRefused` still reaches the caller. The layer listing
warns once across `fiber_stress` and `fiber_strain`, and its reason for a beam or zeroLength bucket names
`FIBER_CATALOG` as well as `LAYER_CATALOG`.
