### Combinations refuse a native file's design fields (ADR 0130 A5)

A native `results.h5` can store design fields beside the recorded responses as
ordinary components: the YOO slab's holds `wood_armer_*` and `rib_shear_*`.
`Results.combinations()` summed them, so on `1.2(D+SD) + 1.6L` it drew values
up to 322% wrong with no warning. In a native file a component is now linear
only if it is a recorder spelling (`apeGmsh._vocabulary.is_canonical`). A
`linear`, `srss`, `cqc` or `abs_sum` rule, or a factor other than 1, over any
other name, the empty one included, is refused `not-linear`. An `envelope` of
factor-1 terms and a lone state read still read it. `.ladruno` and `.mpco`
results, `domain` values and `.user.h5` results are unchanged.
