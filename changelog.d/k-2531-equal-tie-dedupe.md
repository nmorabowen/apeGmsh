### FIXED — A partitioned `ops.h5()` keeps both of two equal equation ties, so the archive replays (#2531)

Two equal `enforce="equation"` ties with the same name, as one
`s.tie(name=...)` over two equal ties leaves, write two sets of
`equationConstraint` lines in the bridge deck. A partitioned `ops.h5()`
stored only one set: the capture drops a line another rank already wrote
(ADR 0027 INV-1), and it took the second tie's lines for such copies. The
replay then refused the archive with a message about schema versions, which
was not the cause. This held for global ties and for ties a stage claims,
named or unnamed.

The capture now tells a copy on another rank from a second equal line on
the same rank, so the archive stores one row per line, as a flat archive
does, and replays to the bridge deck. An archive written before this fix
still refuses to replay; write it again to replay it. Equal `equalDOF`,
`rigidLink` and `rigidDiaphragm` records keep the old behaviour, and their
replay still warns that it leaves the second record out.
