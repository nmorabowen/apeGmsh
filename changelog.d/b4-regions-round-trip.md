### FIXED — region-scoped Rayleigh survives `from_h5 -> to_h5` and `build('tcl')` (B4-e, #1579)

`OpenSeesModel.from_h5` now reads the top-level `/opensees/regions` rows
(`H5Model.regions()`), and the broker replays them verbatim on every target:
`to_h5` writes the same `region_NNN` groups back (the rewrite is a fixed
point again, `model_hash` included), and `build('tcl' | 'py' | 'live')`
emits each `region $tag -ele ... -rayleigh ...` line at the bridge's slot,
after the global `rayleigh` commands and before the patterns. Before, the
rewrite silently dropped the group and the replayed deck was under-damped
compared with what was declared. Every top-level region replays, damping or
not, with or without a K1-6 declaration, because the archived row is the
resolved OpenSees call; a row without its `tag` or `params` raises
`MalformedH5Error`. A model with no regions is byte-identical to before.
