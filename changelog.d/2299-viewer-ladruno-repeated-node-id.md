### FIXED — apeGmshViewer refuses a `.ladruno` that lists one node id twice when it opens, by name, so a SUM part set cannot count that node twice (#2299)

The viewer's `.ladruno` reader used to take a nodal result's `ID` as
written. A repeated id read as two columns of one node: the history of that
node read only its last row, and the other row was drawn as that node. In a
part set, a result whose partition copies are summed
(`PARTITION_REDUCTION=SUM`, such as a reaction) added every row of the
repeated id, so the node was counted twice, and a node whose id was
overwritten disappeared, its value folded into the repeated one, with no
warning.

Opening a `.ladruno` now raises a `LadrunoError` of kind `layout` when an
`ID` under a stage's `RESULTS/ON_NODES` repeats an id. Both the nodal
result groups and the modal groups are checked. The error names the file,
the HDF5 path of the `ID` dataset, and each repeated id with the rows it
sits at. Every part of a set is opened through the same check, so the whole
set is refused. A node held by two part files is still legal: that is the
copy SUM combines. The Python reader refuses the same files (#2300).

The fork is not known to write a repeated id; this guards against a
malformed file. Every committed `.ladruno` fixture still opens.
