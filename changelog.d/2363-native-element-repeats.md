### FIXED — native element-side reads no longer return an element twice, and the repeated-id check is O(N log N) (#2363)

`read_gauss`, `read_elements`, `read_line_stations`, `read_fibers` and
`read_layers` on a native `results.h5` joined each group's element list
with no check, so one element in two places came back as two columns
without a word. The writer refuses an element twice in one group and a
second partition, but accepts one element in two groups of one partition:
`RecorderTranscoder` writes one group per record, so two records over one
element do this. Hand-made and legacy multi-partition files reached the
other shapes. Now an element twice in one group raises
`NativeRepeatedElementId`. An element in two groups or two partitions
reads once when every copy agrees exactly with the first (values, NaN
equal to NaN, and the per-point data: Gauss points, stations, fiber or
layer rows), as `.ladruno` de-duplicates agreeing Gauss columns; copies
that disagree raise `NativeRepeatedElementId`, naming the stage, the
component and each dataset and row. A fibers or layers group lists one
row per fiber or layer, so there an element repeats within its group by
design; a layers group whose (element, Gauss point, layer, sub-point)
row repeats is refused. An `element_ids` filter that leaves the repeat
out still reads.

The repeated-id check behind `LadrunoRepeatedNodeId`,
`NativeRepeatedNodeId` and the writer's `NativeWriteRepeatedId` scanned the
id list once per repeated id. 400,000 ids that all repeat took 26 s; one
sort now takes about 0.2 s, with the same ids, rows and messages.
