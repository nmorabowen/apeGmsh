### FIXED — a partitioned equation tie spanning ranks is refused under Penalty (#1963, #1724 item 1)

An `enforce="equation"` tie whose slave and master live on two ranks is
written as one `equationConstraint` per rank (ADR 0068 Open item 2). Under a
declared `Penalty` handler each copy added its own penalty term, the #1625
mechanism, and the emit succeeded silently. Such a partitioned emit now raises
`BridgeError` before writing. The error names the tie and two remedies: a
serial run (`ops.tcl(path, flat=True)`) or `ops.constraints.Lagrange()`.
`Lagrange`, `LadrunoProjection` and the auto handler keep ADR 0068's
replication, and a tie whose nodes share one rank still emits once.
