### ADDED — apeGmshViewer opens a partitioned `.ladruno` run as one model (#2188)

An OpenSeesMP run writes one file per rank, `<stem>.part-<N>.ladruno`. The app
used to refuse a part file by name. It now opens the whole set from any of its
parts, or from the model beside it (opened from the model, `<stem>.ladruno`
comes first, then the part set, then `<stem>.results.h5`). The parts are read
as one run over one node index, so a node on a partition interface appears
once. Each nodal result follows its `PARTITION_REDUCTION` attribute (fork
schema §7.1): `NONE` keeps one copy and `SUM` adds the copies. On a
fork-written two-rank set, the reaction at the interface node equals Python's
stitched read and the closed form, and neither part's copy alone does.

A partial set is never drawn. The set is refused, naming the parts, when a part
is missing, when a part is past `NUM_PARTITIONS`, when the parts disagree on
`NUM_PARTITIONS`, or when a part's step axis is shorter (a rank still writing).
A nodal result is refused by name, while the rest of the set reads, when it has
no `PARTITION_REDUCTION`, when the value is outside the §7.1 table, when it is
`UNSUPPORTED`, or when the parts disagree. It is never assumed `NONE`. The set
pairs with the model only when every part carries the same `model_hash` and
`fem_hash`. An unstamped part, or parts stamped against different models, leave
it unpaired, and the banner names the part indices. A part re-stamped, added or
removed while the set is open makes the next read re-index the set. Element
results of a part set (Gauss points, line stations) are not stitched yet. They
are listed as unread.
