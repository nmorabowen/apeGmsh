### FIXED — a stage-claimed equation tie is no longer dropped from a partitioned deck (#1979)

In a partitioned emit, an `enforce="equation"` tie claimed by a stage
(`s.tie(name=...)`) wrote no `equationConstraint` row on any rank. Its
ghost nodes were declared, so the deck ran without the tie and reported
nothing. The stage fan-out now writes the tie the way the global pass does
(ADR 0068 Open item 2): the same rows, on every rank that owns the slave or
a master. Under a declared `Penalty`, a stage-claimed tie spanning ranks is
refused like a global one (#1963).
