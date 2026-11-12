### FIXED — native results: `NativeWriter` refuses an id repeated within one partition (#2361)

Since #2360 the native reader refuses, at open, a `results.h5` whose
partition lists a node id twice. `NativeWriter` still wrote such a file
without a word: `write_nodes(..., node_ids=[7, 7])` succeeded, and the
error surfaced only when someone opened the file.

The writer now refuses the write with `NativeWriteRepeatedId` (a
`ValueError`), before any of the call reaches the file. The error names
the file, the stage, the partition, the id dataset, and each repeated id
with its rows. The same refusal covers every id write in a partition:

- `write_nodes` (`nodes/_ids`) and `write_element_ids` (`elements/_ids`);
- the `_element_index` of `write_gauss_group`, `write_line_stations_group`
  and `write_nodal_forces_group`;
- the (element, Gauss point, layer, sub-point) row key of
  `write_layers_group`.

The writer uses the readers' own repeated-id check, so it refuses exactly
the node lists the reader refuses. A node held by two partitions is not a
repeat, and a fibers group still lists one row per fiber, so its element
ids repeat by design. Every native writer in apeGmsh writes distinct ids,
so no in-repo file changes.
