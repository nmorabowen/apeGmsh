### FIXED — a partitioned equation tie spanning ranks is refused under Penalty (#1963, #1724 item 1)

An `enforce="equation"` tie whose slave and master live on two ranks is
written as one `equationConstraint` per rank (ADR 0068 Open item 2). Under a
`Penalty` handler, declared at model level or by any stage the tie runs
under, each copy added its own penalty term. That is the #1625 mechanism,
and the emit succeeded silently. Such a partitioned emit now raises
`BridgeError` before writing. The error names the tie and the only verified
remedy, a serial run (`ops.tcl(path, flat=True)`). `Lagrange`,
`LadrunoProjection` and the auto handler keep ADR 0068's replication, whose
live verification is pending on #1724. A tie whose nodes share one rank
still emits once.
