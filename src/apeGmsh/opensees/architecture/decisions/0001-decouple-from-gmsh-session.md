# ADR 0001 — Decouple the bridge from the gmsh session

**Status:** Accepted

## Context

The legacy `apeGmsh.solvers.OpenSees` lives on the apeGmsh session as
`g.opensees`. It calls `gmsh.*` directly inside `build()` to extract
nodes, elements, and physical groups. Loads, masses, and constraints
declared on the *session* are pulled into the bridge later via
`g.opensees.ingest.X(fem)`.

In practice, the bridge already consumes a `FEMData` snapshot for
loads/masses/constraints. Only `build()` still speaks gmsh, and only
to read mesh data that `FEMData` already carries. The session
coupling is historical, not load-bearing.

## Decision

The new `apeSees` class takes a `FEMData` snapshot at construction
and never imports `gmsh` or holds a session reference:

```python
ops = apeSees(fem)
```

`build()` reads from `fem.nodes`, `fem.elements`, `fem.physical`,
`fem.labels`. The `ingest` step disappears — the bridge reads
`fem.nodes.loads` / `fem.nodes.masses` / `fem.elements.constraints`
directly during build.

## Alternatives considered

1. **Keep the session-coupled bridge as-is.** Rejected — the
   ingest split is artificial and the bridge can't be tested or
   serialized without booting gmsh.
2. **Make the bridge read from gmsh OR FEMData.** Rejected — two
   code paths for the same job, neither cleanly shippable.
3. **Wrap a session inside the bridge.** Rejected — moves the
   coupling, doesn't remove it.

## Consequences

**Positive:**

- Bridge is testable without gmsh.
- Bridge is serializable (declarations + fem-ref are both data).
- Multiple bridges per FEM (try OpenSees, try ANSYS, …).
- Multiple FEMs per session.
- Lifecycle simplifies from `declare → build → ingest → build → export`
  to `declare → build → export`.

**Negative:**

- `g.opensees.X` becomes `ops.X` — slightly more typing per script.
  Mitigated by `g.opensees(fem)` factory shortcut (see ADR sequence).
- Existing apps using `apeGmsh.solvers` need migration.
  Mitigated by ADR 0009 — no back-compat is the intentional choice.

## Reference

- [charter.md P3, P9](../charter.md)
