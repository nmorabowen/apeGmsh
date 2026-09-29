# ADR 0013 — Resolved record dataclasses live in `apeGmsh.mesh.records`, not `apeGmsh.solvers`

**Status:** Accepted (Phase 8.1, PR #119)

## Context

Through years of incremental additions, `apeGmsh.solvers` accumulated
three responsibilities that should not live in the same package:

1. **Record dataclass definitions** — `ConstraintRecord` and the
   four resolver-output subclasses, `NodePairRecord`,
   `NodeGroupRecord`, `InterpolationRecord`, `SurfaceCouplingRecord`,
   `NodeToSurfaceRecord`; the `LoadRecord` hierarchy
   (`NodalLoadRecord`, `ElementLoadRecord`, `SPRecord`); the
   `MassRecord`; and the `ConstraintKind` / `LoadKind` enums that
   classify them.
2. **OpenSees emit helpers** — `_opensees_csys`,
   `_opensees_constraints`, `_opensees_export`, …
3. **The element response catalog** — `_element_response.py`.

The records are produced post-mesh by the resolvers and consumed by
every downstream layer (the bridge, the result reader, the viewer).
They describe **what the broker holds**, not how OpenSees consumes
it. Yet they lived in `apeGmsh.solvers` — forcing the broker (mesh
side) and the consumer (results / viewer) to import from the
OpenSees bridge package. The result was a cycle: producer ↔ broker ↔
consumer all reached into one shared blob.

The Phase 8 untangle plan
([phase-8-untangle.md](../phase-8-untangle.md)) decomposes
`apeGmsh.solvers` into three layers; Phase 8.1 ships the record
relocation.

## Decision

Records, kind enums, the resolvers that produce them, and the
pre-mesh user-facing **definition** dataclasses (`*Def`) all leave
`apeGmsh.solvers` and split across the layers responsible for each
shape of data:

| Concern | New home |
|---|---|
| Resolved record dataclasses + Kind enums | `apeGmsh.mesh.records` |
| Constraint / load / mass resolvers (broker-layer mesh math) | `apeGmsh.mesh._constraint_resolver`, `apeGmsh.mesh._load_resolver`, `apeGmsh.mesh._mass_resolver` |
| FEM shape-function quadrature (consumed by `LoadResolver`) | `apeGmsh.mesh._consistent_quadrature` |
| RCM numberer (broker-side topology op) | `apeGmsh.mesh._numberer` |
| Pre-mesh user-facing `*Def` types | `apeGmsh.core.constraints.defs`, `apeGmsh.core.loads.defs`, `apeGmsh.core.masses.defs` |

`apeGmsh.mesh.records` re-exports the constraint defs and the
resolver so it remains the canonical umbrella — `import
apeGmsh.mesh.records as Constraints` produces the same surface that
`apeGmsh.solvers.Constraints` used to.

Every relocated `apeGmsh.solvers` module is now a thin re-export
shim that emits a one-shot `DeprecationWarning` on import. The
package init (`apeGmsh/solvers/__init__.py`) installs a module-level
`__getattr__` so `from apeGmsh.solvers import Numberer` also warns.

Internal apeGmsh code has been migrated to the canonical paths; only
external callers using the legacy import paths see the deprecation
warning.

## Alternatives considered

1. **Leave the records in `apeGmsh.solvers` and just rename the
   package.** Rejected — the package's identity is the problem, not
   its name. The records aren't OpenSees-specific; pretending they
   are perpetuates the cycle.
2. **Put the records inside `apeGmsh.mesh.FEMData` itself.**
   Rejected — `FEMData` is the **broker container**, not the type
   library. The dataclasses are imported by emitters, viewers, and
   tests independently of any FEMData instance; nesting them inside
   `FEMData` would make the imports awkward.
3. **Keep constraint defs in `mesh/_constraint_resolver/` paired
   with the resolver.** Rejected (after Q&A on PR #119) — for
   symmetry with load and mass defs (which the same plan routes to
   `core/`), constraint defs are also pre-mesh user-facing intent
   and belong in `core/`. The resolver imports them from `core/`,
   the same direction as load/mass resolvers.
4. **Skip the umbrella, force users to import from individual
   submodules.** Rejected — `apeGmsh.mesh.records` as the canonical
   umbrella matches the historical ergonomic of
   `apeGmsh.solvers.Constraints` and limits the migration surface
   for downstream code.

## Consequences

**Positive:**

- The broker (`mesh/`) owns its own data types. The bridge
  (`opensees/`) imports records from `mesh.records`; the results
  layer imports from `mesh.records`; the viewer imports from
  `mesh.records`. One-way dependency, no cycles.
- `apeGmsh.solvers` shrinks toward its eventual deletion (Phase 8.8).
  The remaining content is clearly OpenSees emit machinery, slated
  for Phase 8.2 / 8.3.
- The `apeGmsh.opensees` package can now be moved to its own
  distribution without dragging the mesh-side record library along
  (charter P9 — `apeGmsh.opensees` does not depend on
  `apeGmsh.mesh` internals beyond `FEMData`).

**Negative:**

- Existing apps using `from apeGmsh.solvers.X import Y` see a one-shot
  `DeprecationWarning` and must migrate to the canonical paths
  before Phase 8.8 deletes `apeGmsh.solvers`.
- The relocation touched ~15 source files and ~12 test files; the
  diff is large but mechanical.

## Reference

- [phase-8-untangle.md](../phase-8-untangle.md) — the full untangle plan
- [charter.md P3, P9](../charter.md) — bridge takes a `FEMData`;
  `apeGmsh.opensees` does not depend on `apeGmsh.core` /
  `apeGmsh.mesh` internals
- PR #119 — the Phase 8.1 implementation
