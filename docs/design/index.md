# Design

This section explains how apeGmsh is built — the internal structure and the
reasoning behind it — for maintainers, contributors, and anyone who wants to
predict the library's behavior from its architecture rather than its docs.

If you're here to *use* apeGmsh, you're one level too deep: the
[Concepts](../concepts/index.md) pages teach the same systems from the
outside. Come back when a question starts with "why is it built this way" or
"where would I change this."

Four pages cover the load-bearing internals, in reading order:
[Architecture](architecture.md) — the layers and the data flow;
[Principles](principles.md) — the commitments every change is measured
against; [The broker](broker.md) — how declarations become the frozen
`FEMData` contract; [Parts & assembly internals](parts-assembly.md) — the
instancing registry and fragmentation bookkeeping; and
[Results internals](results.md) — the three-broker read side and the
render seam.

The authoritative record of *decisions* — what was chosen, what was
rejected, and why — is the append-only ADR log in the repository at
[`src/apeGmsh/opensees/architecture/decisions/`](https://github.com/nmorabowen/apeGmsh/tree/main/src/apeGmsh/opensees/architecture/decisions).
These pages cite ADRs by number; the log is where you read them.

---

*Next: [Architecture](architecture.md).*
