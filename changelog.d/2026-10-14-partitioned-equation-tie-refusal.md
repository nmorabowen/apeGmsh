### FIXED — a partitioned equation tie spanning ranks is refused under a multiplying handler (#1963, #1724 item 1)

An `enforce="equation"` tie whose slave and master live on two ranks was
written as one `equationConstraint` per rank. Under `Penalty`, `Lagrange` or
the auto-emitted `Lagrange`, each copy assembled its own penalty or
multiplier, the #1625 mechanism, and the emit succeeded silently. The #1625
refusal now covers equation ties: such a partitioned emit raises
`BridgeError` before writing, naming the tie and the serial remedy
(`ops.tcl(path, flat=True)`). A tie whose nodes share one rank still emits
once.
