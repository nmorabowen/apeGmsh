### CHANGED — Transform tags come from the build-time tag plan (K1-3d S3a, #1455)

The orientation fan-out's allocation loop moved out of
`emit_transform_specs` into `plan_transform_specs`, which `plan_tags` runs
once per emit mode. The plan holds each transform's `geomTransf` lines and
the per-element override map. `emit_transform_specs` only writes them, and
refuses a plan made for other transforms. The emit fork now freezes
`geomTransf`, so an emit-time `geomTransf` mint raises `TagLawError`.
`emit_transform_specs` still accepts a plain `TagAllocator` from a direct
caller, and plans from it through the same loop, until K1-3d S6 removes
the `tags` parameter. Any other fork raises. The oracle's transform cases
are now real comparisons. Every emitted deck and the 86 golden cells are
byte-identical. One ordering change: `orientation=` on an `ndm=2` model, or
on a group whose elements are not 2-node lines, now raises when the emit
plans its tags, before the emit path's own checks.
