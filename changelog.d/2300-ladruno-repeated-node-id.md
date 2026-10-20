### FIXED — a `.ladruno` that lists one node id twice is refused when it opens, by name, so a SUM part set cannot count that node twice (#2300)

`LadrunoReader` used to return a result group's node `ID` as written. A
repeated id read as two columns of one node, and in a part set whose result
sums the partition copies (`PARTITION_REDUCTION=SUM`, such as a reaction)
both copies were added, so the node was counted twice without a warning.

Opening a `.ladruno` now raises `LadrunoRepeatedNodeId` (a `ValueError`)
when any node-result `ID` repeats an id. The check covers every stage's
`RESULTS/ON_NODES` groups (the modal groups included) and
`RESULTS/ENVELOPES/ON_NODES`. The error names the file, the HDF5 path of
the `ID` dataset, and each repeated id with the rows it sits at. The check
runs per file, so `Results.from_ladruno` refuses a single file and a part
set alike. A node held by two part files is still legal: that is the copy
SUM combines. The viewer refuses the same files (#2299).

The fork is not known to write a repeated id; this guards against a
malformed file. Every committed `.ladruno` fixture still opens.
