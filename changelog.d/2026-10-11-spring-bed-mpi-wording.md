### CHANGED — docs: the spring bed emits under MPI (ADR 0119 wording)

The `ops.spring_bed` docstring and ADR 0119 no longer say a bed emits
single-process only: ADR 0120 routes each spring, its ground and side nodes
and their `equalDOF` to the rank of the structural node.
