### CHANGED — shared reader code moves out of the `_mpco*` modules into `results/readers/_buckets.py` (#2761)

ADR 0131 D10, stage 0, first part. The code the `.ladruno` reader and its
consumers import (the slab helpers, the partition merge, the nodal and modal
helpers, the node-name translation and the bucket-skip warnings) now lives in
`_buckets.py`. The `_mpco*` modules import from it, `_ladruno`,
`_ladruno_multi`, `user/_modes`, `user/_spectra_source` and `_composites` no
longer import an `_mpco*` module, and no re-export is left behind.
Behaviour is unchanged. The element-side modules move in the next slice.
