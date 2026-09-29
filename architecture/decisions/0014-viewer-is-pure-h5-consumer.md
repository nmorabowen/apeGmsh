# ADR 0014 — `apeGmsh.viewers` is a pure `model.h5` consumer

**Status:** Accepted (Phase 8.7, May 2026)

## Context

Phase 8.1 (PR #119, ADR 0013) moved the resolved record dataclasses
from `apeGmsh.solvers` to `apeGmsh.mesh.records`, breaking the cycle
between producer (bridge) and broker. Phase 8.4–8.6 then made
`model.h5` the canonical model database: the broker writes the
neutral zone (`/nodes`, `/elements/{type}`, `/physical_groups`,
`/labels`, `/constraints/{kind}`, `/loads/{kind}/{pattern}`,
`/masses`), the bridge writes the `/opensees/...` enrichment zone,
and the two zones compose into one file. Phase 8.8 (PR #159) then
deleted `apeGmsh.solvers/` entirely.

That leaves one open consumer-side dependency: **the viewer still
imports `FEMData` and (until Phase 8.8) the solvers' record types**.
The audit in [phase-8.7-scope.md §2](../phase-8.7-scope.md) lists
~17 distinct accessors the viewer pulls off `FEMData`, spread
across `viewers/scene/fem_scene.py`, ~15 diagrams under
`viewers/diagrams/`, the constraint overlay in `viewers/overlays/`,
and three UI tabs in `viewers/ui/`. Every one of those import sites
is a leak from the consumer layer back into the broker —
structurally the same problem Phase 8.1 solved on the producer
side.

The Phase 8 master plan
([phase-8-untangle.md §6](../phase-8-untangle.md)) sets an explicit
acceptance criterion:

> `apeGmsh.viewers.*` imports nothing from `apeGmsh.opensees.*` or
> `apeGmsh.mesh.*` internals — only `apeGmsh.results` and
> `apeGmsh.opensees.emitter.h5_reader`.
>
> `model.h5` is self-sufficient: the viewer can render any test
> fixture without instantiating a `FEMData`.

Phase 8.7 is the work that closes those criteria.

## Decision

The viewer package depends on two upstream packages and no others:

1. **`apeGmsh.results`** — for *result data* (slabs, components,
   time slices, the existing `Results` API).
2. **`apeGmsh.opensees.emitter.h5_reader`** — for the *model* (the
   neutral and `/opensees/` zones of `model.h5`).

Everything the viewer reads from `FEMData` today is replaced by a
single adapter, **`apeGmsh.viewers.data.ViewerData`**, with two
builders:

```python
ViewerData.from_fem(fem)               # live g.mesh path
ViewerData.from_h5("model.h5")         # post-solve / fixture path
```

`ViewerData` is read-only and exposes only the surface the viewer
exercises today (audited in
[phase-8.7-scope.md §2](../phase-8.7-scope.md)). It is **not** a
1:1 mirror of `FEMData` — it is a viewer-facing slice. Anything
outside that slice stays on `FEMData` and is not migrated.

The two builders produce interchangeable instances. Downstream
viewer code (diagrams, scene builder, overlays, UI tabs) does not
need to know — and does not learn — which source the snapshot came
from.

### One additive schema bump: `/mesh_selections/` (2.3.0 → 2.4.0)

Post-mesh selection sets (`g.mesh_selection` → `fem.mesh_selection`)
become part of the neutral zone in `model.h5`. The on-disk layout
mirrors `/physical_groups/` for symmetry (same compound-row reader
in `H5Reader`). Without this bump, the viewer's `selection=`
selector would silently return empty IDs after a round-trip through
`model.h5`.

### Record decoding lives in `viewers/data/_records.py`

The neutral zone's `/constraints/{kind}`, `/loads/{kind}/{pattern}`,
and `/masses` groups share the symmetric compound contract
(`target_kind`, `target`, `payload_kind`, `payload`) documented in
[h5-schema.md](../h5-schema.md) and implemented by
`mesh/_record_h5.py` (write side). The viewer needs typed row
access for iteration (`constraints.pairs()`,
`constraints.interpolations()`, `loads.by_pattern(...)`, …).

Read-side row dataclasses for that decode live inside
`viewers/data/_records.py`, NOT imported from
`apeGmsh.mesh.records._constraints` etc. Duplicating the field names
(not the logic) in the viewer package decouples the viewer from the
write-side dataclass library while keeping the schema document as
the single source-of-truth for the contract.

### No deprecation shims

`apeGmsh.viewers/` has few external consumers and most of them are
internal apps. Per the same mandate that drove Phase 9 commit 5 and
Phase 8.8 (delete-not-deprecate), this phase ships a clean break:
`fem` kwarg → `view`, `from apeGmsh.mesh.FEMData` import →
`from apeGmsh.viewers.data import ViewerData`. No
`DeprecationWarning` shims, no transitional aliases.

## Alternatives considered

1. **Approach A — viewers consume `Results` (no new adapter).**
   Make the viewer take a `Results` object and pull structural
   data through `results.fem`. Rejected — `Results._fem` is a
   `FEMData` under another attribute name; the coupling is identical
   to the status quo, just under a different import path. It also
   does not solve the "render from `model.h5` alone" case, since a
   `Results` instance requires a result reader (MPCO / native run.h5)
   that the fixtures don't have.

2. **Approach C — bridge `H5Reader` directly through every viewer
   file.** The viewer would read raw `dict[str, ndarray]` views and
   reach into the schema-level structure everywhere. Rejected — the
   schema's symmetric compound rows need a non-trivial decoder; that
   decoder belongs in one place (the adapter), not scattered across
   ~20 viewer files.

3. **Mirror FEMData 1:1 with a new typed proxy.** A `ViewerData`
   that exposes the entire `FEMData` surface. Rejected — most of
   `FEMData`'s surface is producer-side machinery (`from_gmsh`,
   `from_msh`, `to_h5`, resolvers, hash) that the viewer never
   touches. Limiting `ViewerData` to the audited surface keeps the
   adapter small and makes the symmetry guarantee actually
   meaningful.

4. **Skip `mesh_selection` in `model.h5`; let `selection=` work only
   on the live path.** Rejected (per Phase 8.7 brief decision) —
   `mesh_selection` is part of the broker's solver-neutral
   description; symmetric with `physical_groups` and `labels`. Its
   absence from the on-disk schema was an oversight in Phase 8.5,
   not an intentional carve-out.

## Consequences

**Positive:**

- The viewer package becomes self-contained against the
  schema-stable `model.h5` contract. A breaking change to the
  internal FEMData composite shape no longer ripples into the
  viewer.
- The "viewer can render any test fixture from `model.h5` alone"
  property unlocks fixture-driven viewer tests without spinning up
  a Gmsh / FEMData factory in CI. The seven fixtures listed in
  [viewer-integration.md](../viewer-integration.md) become
  end-to-end exercisable.
- Charter principle P9 ("`apeGmsh.opensees` does not depend on
  `apeGmsh.core` or `apeGmsh.mesh` internals") gains its consumer-
  side counterpart: `apeGmsh.viewers` does not depend on either.
- `apeGmsh.viewers/` can be distributed independently in the
  future. The dependency surface narrows to two upstream packages.

**Negative:**

- One additional package (`apeGmsh.viewers.data`) to maintain.
  ~5–8 modules; each is small (the audit drives a tight surface)
  but the schema-vs-FEMData parity needs explicit test coverage.
- Read-side record row dataclasses duplicate field names from
  `mesh.records._constraints`. The schema document is the single
  source-of-truth; the duplication is enforced by the
  `tests/viewers/data/test_viewer_data.py` parity test (which
  asserts that `from_fem(fem)` and `from_h5(fem.to_h5(path))` agree
  on every audited accessor).
- Schema bump 2.3.0 → 2.4.0 obliges any other reader of the file
  to ignore the new `/mesh_selections/` group gracefully. The
  additive-bump policy in
  [viewer-integration.md §"Versioning policy"](../viewer-integration.md)
  already covers this — pre-2.4 viewers ignore the new group and
  lose only the `selection=` round-trip convenience.

## References

- [phase-8.7-scope.md](../phase-8.7-scope.md) — full implementation
  scope, the audit, ViewerData surface, commit decomposition.
- [phase-8-untangle.md](../phase-8-untangle.md) — the master plan;
  §5 (sub-phase 8.7), §6 (acceptance), §7 (open question 4 closed
  in this ADR).
- [decisions/0013-records-in-mesh-not-solvers.md](0013-records-in-mesh-not-solvers.md)
  — Phase 8.1's symmetric decision on the producer side.
- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — `model.h5` as emit target; this ADR makes the viewer its first
  pure consumer.
- [charter.md](../charter.md) — principles P3 (bridge takes a
  `FEMData`, not a session), P9 (`apeGmsh.opensees` does not depend
  on `apeGmsh.mesh` internals); this ADR is their consumer-side
  bookend.
