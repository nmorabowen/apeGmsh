# Architecture Decision Records (ADRs)

Each ADR captures one significant decision: the **context** (what
was at stake), the **decision** itself, the **alternatives** we
considered and rejected, and the **consequences**.

ADRs are append-only. If a later decision reverses or amends an
earlier one, write a new ADR that supersedes it; do not edit history.

| # | Title | Status |
|---|---|---|
| [0001](0001-decouple-from-gmsh-session.md) | Decouple the bridge from the gmsh session | Accepted |
| [0002](0002-typed-primitives-with-capabilities.md) | Typed primitives carry capabilities | Accepted |
| [0003](0003-namespace-api.md) | Namespace API + static typing | Accepted |
| [0004](0004-section-separated-from-material.md) | Sections live outside `material/` | Accepted |
| [0005](0005-patterns-explicit.md) | Patterns are explicit context managers | Accepted |
| [0006](0006-class-name-apesees.md) | Bridge class is `apeSees` | Accepted |
| [0007](0007-time-series-separated-from-pattern.md) | Time series live outside `pattern/` | Accepted |
| [0008](0008-three-emit-targets.md) | Three emit targets via Emitter Protocol | Accepted |
| [0009](0009-no-backwards-compat-with-solvers.md) | No back-compat with `apeGmsh.solvers` | Accepted |
| [0010](0010-csys-for-frame-orientation.md) | Coordinate systems for frame orientation | Accepted (already shipped in `solvers/`) |
| [0011](0011-h5-as-fourth-emit-target.md) | HDF5 as a fourth emit target | Accepted |
| [0013](0013-records-in-mesh-not-solvers.md) | Resolved records live in `apeGmsh.mesh.records`, not `apeGmsh.solvers` | Accepted |
