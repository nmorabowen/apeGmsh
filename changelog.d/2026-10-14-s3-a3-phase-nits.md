### Emit profile: per-block floor snapshot name, tested "NONE captured" path

The `--phases` profiler now writes each block's floor snapshot to
`open_r{rank}_b{idx}.tm`, so a later block of a rank no longer overwrites an
earlier one. A smoke test now covers the "NONE captured" attribution line
(#1914, from the #1896 review).
