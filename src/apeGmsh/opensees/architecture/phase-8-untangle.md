# Phase 8 — Untangle `apeGmsh.solvers`

**Status:** Proposal (May 2026).
**Replaces:** the original Phase 8 sketch in
[parallel-execution.md](parallel-execution.md) ("apps migrate" — too
small a frame).

This document scopes the work needed to retire `apeGmsh.solvers` and
turn `model.h5` into the canonical model database. It does not ship
code; it captures the chain, the responsibilities of each layer, the
file-level relocation map, and the sub-phase sequence.

## 1. The problem

`apeGmsh.solvers` is the legacy OpenSees emit path. Through years of
incremental additions, it accumulated three responsibilities that
should not live in the same package:

1. **Record dataclass definitions** — `ConstraintRecord`,
   `NodePairRecord`, `LoadRecord`, `MassRecord`, `LoadKind`,
   `ConstraintKind`. These describe **what the broker holds**, not
   how OpenSees consumes it. The mesh / broker side imports them
   today via `from apeGmsh.solvers.Constraints import ...`.
2. **OpenSees emit helpers** — `_recorder_emit`, `_opensees_csys`,
   `_opensees_constraints`, `_opensees_export`. Bridge-side concerns
   that legitimately live in `apeGmsh.opensees` now that the Phase 0–6
   rebuild has shipped.
3. **Element response catalog** — `_element_response.py`. The mapping
   from element type to the response classes it produces (forces,
   stresses, strains, fiber-level data). This is consumed by
   `apeGmsh.results` to interpret MPCO / .out / .h5 outputs.

Because all three live in one package, every layer of apeGmsh
imports from `solvers/`:

| Layer | Imports from solvers | Why |
|---|---|---|
| `apeGmsh/__init__.py` | `Constraints`, `Numberer` | Re-exported as top-level public API |
| `core/{Constraints,Loads,Masses}Composite.py` | Record types | Broker construction needs the dataclasses |
| `mesh/_record_set.py` | `_kinds.ConstraintKind`, `LoadKind` | mesh-side record types |
| `opensees/transform.py` + `opensees/_internal/build.py` | `_opensees_csys` | Bridge transforms reach back into legacy |
| `results/capture/_domain.py`, `results/readers/_mpco*.py`, `results/transcoders/_*.py` | `_element_response`, `_recorder_emit` | Result interpretation depends on the catalog |
| `viewers/ui/*_tab.py`, `viewers/overlays/constraint_overlay.py` | Records + UI helpers | Viewer reaches across the chain |
| Top-level `tests/test_*.py` | misc | Tests of the legacy paths |

The result is a cyclic mess: the producer (bridge), the broker, the
consumer (results), and the viewer all reach into one shared blob.
There is no clear data flow.

## 2. The chain

The correct layering is a one-way pipeline:

```
Model → Mesh → Broker ──┐
                        ├── model.h5 ──→ Viewer / Results
              Bridge ───┘
                ↑
                └─ analyze → MPCO  ──→ Results (response side)
```

Each layer has one responsibility:

| Layer | Concrete package | Responsibility |
|---|---|---|
| **Model** | `apeGmsh.core.Part`, composites | User-facing description (parts, parameters, material/section composition) |
| **Mesh** | `apeGmsh.mesh.Mesh`, `MshLoader`, partitioning | Gmsh interaction, meshing, partitioning |
| **Broker** | `apeGmsh.mesh.FEMData` | Solver-neutral snapshot: nodes, elements, PGs, labels, constraints, loads, masses |
| **Bridge** | `apeGmsh.opensees.apeSees` | OpenSees-flavored producer of model.h5 + driver of OpenSees execution (Tcl / py / live) |
| **model.h5** | `architecture/h5-schema.md` | Serialized archive — the stable interface between producer side and consumer side |
| **Viewer/Results** | `apeGmsh.results`, `apeGmsh.viewers` | Pure consumers of `model.h5` + MPCO. Never import `solvers/` or `FEMData` internals |

The Bridge is a **side-feeder** into `model.h5` alongside the Broker —
not a separate link in the chain. A future second solver (Code_Aster,
Abaqus, etc.) plugs in at the Bridge layer and writes its own
enrichment zone in the same file.

## 3. `model.h5` — two zones

The schema gains a namespace convention to host multiple solver
enrichments without collision:

```
model.h5
├── /meta                      attrs (schema, ndm, ndf, snapshot_id)
│
├── ── NEUTRAL ZONE ──         Broker-written, solver-agnostic
├── /nodes                     ids + coords + PG / label membership
├── /elements/{type}           per-type ids + connectivity
├── /physical_groups           top-level index for viewer discovery
├── /labels                    apeGmsh-internal labels
├── /constraints/{kind}        MP-style records (NodePair, NodeGroup,
│                               NodeToSurface, Interpolation,
│                               SurfaceCoupling)
├── /loads/{kind}              FEM-side loads (face / line / point / etc.)
├── /masses                    FEM-side masses
│
├── ── /opensees/ ZONE ──      Bridge-written; OpenSees-specific
│   ├── /opensees/materials
│   ├── /opensees/sections
│   ├── /opensees/transforms
│   ├── /opensees/beam_integration
│   ├── /opensees/time_series
│   ├── /opensees/patterns
│   ├── /opensees/bcs           OpenSees-resolved fix / mass
│   ├── /opensees/recorders
│   ├── /opensees/analysis
│   └── /opensees/tag_map       FEM ↔ OpenSees tag translation
│
└── ── /<future_solver>/ ──    same shape, never overlaps
```

**Why namespace.** Today the Bridge zone is at the root
(`/materials`, `/sections`, …) because OpenSees was the only producer.
Moving it under `/opensees/` is a one-time schema reshuffle that
makes room for the Broker to own root-level groups and for a second
solver to coexist without name collision.

**Migration cost of the namespace move.** Phase 6's writers wrote the
Bridge groups at the root; the Phase-6 reference reader (`h5_reader.py`)
walks those root paths. The reshuffle is mechanical (renames in the
writer + matching renames in the reader) but breaks any external
consumer of Phase-6 model.h5 files. The schema_version bump (1.1.0 →
2.0.0) signals the break.

**Symmetry contract.** Every "record set" group (`/constraints/{kind}`,
`/loads/{kind}`, `/masses`, `/bcs/fix`, `/bcs/mass`,
`/patterns/{name}/loads`) uses the same compound-dtype shape so the
viewer has one reader:

```
target_kind  vlen utf-8   "node" | "element" | "pg"
target       vlen utf-8   tag (str) or pg name
payload_kind vlen utf-8   record subtype (e.g. "rigid_beam",
                          "point_load", "lumped")
payload      compound     subtype-specific typed fields
```

Then the viewer reads any record set with one function and dispatches
on `payload_kind`.

## 4. Relocation map

Every file in `apeGmsh.solvers` lands in exactly one of: a new home,
the bridge, deleted (folded into existing code), or moved to a tests
fixture.

### 4a. Records → `apeGmsh.mesh.records`

A new sub-package under mesh, where the broker lives:

| Solvers file | New home | Notes |
|---|---|---|
| `_kinds.py` | `mesh/records/_kinds.py` | `ConstraintKind`, `LoadKind` enums |
| `_constraint_records.py` | `mesh/records/_constraints.py` | The 5 dataclasses |
| `Loads.py` → record types only (`LoadRecord`, `NodalLoadRecord`, `ElementLoadRecord`, `SPRecord`) | `mesh/records/_loads.py` | Strip out the `LoadResolver` — moves separately |
| `Loads.py` → def types (`PointLoadDef`, `LineLoadDef`, ...) | `core/loads/defs.py` | These are user-facing Model-layer types, stay with `core/` |
| `Loads.py` → `LoadResolver` | `mesh/_load_resolver.py` | Resolves user defs into broker records — Broker-layer machinery |
| `Masses.py` → record types | `mesh/records/_masses.py` | Same split as Loads |
| `Masses.py` → def types | `core/masses/defs.py` | Same split |
| `Masses.py` → `MassResolver` | `mesh/_mass_resolver.py` | Same split |
| `Constraints.py` (the re-export module) | `mesh/records/__init__.py` | Becomes the canonical import path |
| `_constraint_defs.py` | `core/constraints/defs.py` | Pre-mesh user-facing intent, symmetric with Loads/Masses |
| `_constraint_geom.py`, `_constraint_resolver.py` | `mesh/_constraint_resolver/_geom.py`, `mesh/_constraint_resolver/_resolver.py` | Broker-layer resolver internals |
| `_consistent_quadrature.py` | `mesh/_consistent_quadrature.py` | FEM shape-fn quadrature; only ever consumed by `LoadResolver`, so it follows the resolver into mesh/ |
| `Numberer.py` | `mesh/_numberer.py` | RCM numberer — Broker-side topology op |

### 4b. OpenSees emit helpers → `apeGmsh.opensees`

These are bridge-side concerns:

| Solvers file | New home | Notes |
|---|---|---|
| `_opensees_csys.py` | `opensees/_csys.py` | `Cartesian` / `Cylindrical` / `Spherical` / `resolve_vecxz`.  Landed in PR #121 (Phase 8.2). |
| `_opensees_build.py` | DELETED | Phase 4 rebuild superseded it.  Removed in PR γ (#130). |
| `_opensees_constraints.py` | DELETED | Bridge fan-out lives in `opensees/_internal/build.py` now.  Removed in PR γ (#130). |
| `_opensees_elements.py` | DELETED | Phase 2 typed elements replaced this.  Removed in PR γ (#130). |
| `_opensees_export.py` | DELETED | Phase 4 TclEmitter / PyEmitter replaced this.  Removed in PR γ (#130). |
| `_opensees_ingest.py` | DELETED | Tied-element ingest moved into the bridge build pipeline.  Removed in PR γ (#130). |
| `_opensees_inspect.py` | DELETED | Phase 5A NodeComposite + accessors replaced this.  Removed in PR γ (#130). |
| `_opensees_materials.py` | DELETED | Phase 1 typed materials replaced this.  Removed in PR γ (#130). |
| `_element_specs.py` | (see §4c) | Originally listed here as DELETE; the actual outcome was relocation to `opensees/_element_capabilities.py` in Phase 8.3b.  See §4c. |

### 4c. Recorder / response → owned by Bridge + Results jointly

The trickiest. Today these live in `solvers/`; both bridge and
results consume them.

| Solvers file | New home | Notes |
|---|---|---|
| `_element_response.py` | `opensees/_response_catalog.py` | The catalog IS OpenSees-specific (it knows which response tokens each OpenSees element produces). Results imports it from the bridge package — that's a one-way dependency, fine.  Landed in PR #123 (Phase 8.3a). |
| `_recorder_emit.py` | `results/spec/_emit.py` | Originally planned for deletion; the [Phase 8.3b scope](phase-8.3b-scope.md) reconsidered and chose Flavor 1 (relocate, don't unify with the typed primitives).  Landed in PR #134 (Phase 8.3b). |
| `_recorder_specs.py` | `results/spec/_resolved.py` | Same — relocated, not deleted (PR #134). |
| `Recorders.py` | `results/spec/declaration.py` | Same — relocated, not deleted (PR #134).  Re-exported as `apeGmsh.results.spec.Recorders`. |
| `_element_specs.py` | `opensees/_element_capabilities.py` | Element-capability map; OpenSees-class metadata so it lives next to the response catalog (PR #134). |
| `OpenSees.py` | DELETED | Phase 4 `apeSees` replaced this; deletion landed in PR γ (#130). |

### 4d. Top-level public API

`apeGmsh/__init__.py` currently re-exports `Constraints` and
`Numberer` from `solvers/`. After the relocation:

```python
# was: import apeGmsh.solvers.Constraints as Constraints
import apeGmsh.mesh.records as Constraints     # back-compat shim
# was: from apeGmsh.solvers.Numberer import Numberer, NumberedMesh
from apeGmsh.mesh._numberer import Numberer, NumberedMesh
```

A deprecation shim `apeGmsh/solvers/__init__.py` keeps re-exporting
the new locations for one release cycle, then is deleted.

## 5. Sub-phase sequencing

Phase 8 cannot land as one PR. It splits into ordered sub-phases:

### Phase 8.0 — Plan (this doc)
Single commit, no code.

### Phase 8.1 — Record relocation (mechanical) — landed in PR #119
Moved `_kinds`, `_constraint_records`, `Loads` records, `Masses`
records, all three resolvers (`LoadResolver`, `MassResolver`,
`ConstraintResolver`), the constraint defs (now in
`core/constraints/defs.py`), `_consistent_quadrature.py`, and
`Numberer.py` to their new homes under `mesh/` and `core/`.
Updated every caller. Shipped the deprecation shims. Apps that do
`from apeGmsh.solvers import X` keep working with a
`DeprecationWarning`.

**Risk:** low. Mostly renames + re-imports. No behavior change.
**Test gate:** every existing test still passes; deprecation warning
shows once per import path.

### Phase 8.2 — Bridge-side helper relocation — landed in PR #121
Moved `_opensees_csys` into `opensees/_csys.py`.  Updated
`opensees/transform.py` + `opensees/_internal/build.py` to import
locally.  The §4b "DELETE" cluster (`_opensees_build`,
`_opensees_constraints`, `_opensees_elements`, `_opensees_export`,
`_opensees_ingest`, `_opensees_inspect`, `_opensees_materials`)
ultimately landed as part of the multi-PR PR γ (#130) bridge
teardown rather than this phase.

**Risk:** low. Internal reorganization with no external surface
change.
**Test gate:** existing OpenSees tests still pass.

### Phase 8.3a — Response catalog relocation — landed in PR #123
Moved `_element_response.py` to `opensees/_response_catalog.py`.
The ~10 `results/` files that import it switched to the new path.

**Risk:** medium — `results/` test suite is sensitive; the response
catalog is a large vocabulary file.
**Test gate:** every `results/` test still passes. MPCO read paths
unchanged.

### Phase 8.3b — Recorder cluster relocation — landed in PR #134
Relocated `Recorders.py`, `_recorder_specs.py`, `_recorder_emit.py`,
and `_element_specs.py` to canonical homes under
`results/spec/` (`declaration.py`, `_resolved.py`, `_emit.py`)
and `opensees/_element_capabilities.py`.  The original deletion
plan was reconsidered in [phase-8.3b-scope.md](phase-8.3b-scope.md);
Flavor 1 (pure relocation) shipped — the typed-primitives-as-
declaration unification (Flavor 2) is deferred to a separate
scoping conversation.  Updated 7 source consumers in `results/`,
9 EOS curriculum notebooks (tagged with the Phase-8.3b TODO),
22 straggler EOS notebooks, and 32 test files.

**Risk:** medium — broad consumer rewire; mitigated by ``__getattr__``
shims at the legacy paths so external code keeps working with a
one-shot ``DeprecationWarning`` for one release cycle.
**Test gate:** every `results/` test still passes; ``import apeGmsh``
+ ``import apeGmsh.solvers`` produce zero ``DeprecationWarning``s.

### Phase 8.4 — model.h5 zone reshuffle (BREAKING)
Move bridge-written groups under `/opensees/` in the schema. Update
H5Emitter writer paths. Update `h5_reader.py` accessors. Bump
schema_version `1.1.0 → 2.0.0`. Update `architecture/h5-schema.md`.

**Risk:** medium-high — the version bump is the signal, but any
external tool reading Phase-6 model.h5 files breaks.
**Test gate:** the Phase 6 fixtures regenerate against the new
schema; `validate()` still returns empty.

### Phase 8.5 — Broker-side groups in model.h5
Implement the neutral-zone writers: `/nodes`, `/elements/{type}`
(if not already at root by the time 8.4 lands), `/physical_groups`,
`/labels`, `/constraints/{kind}`, `/loads/{kind}`, `/masses`.
Implement `fem.to_h5(path)` on FEMData (writes neutral zone only).
Implement the symmetric-record-compound-dtype helper so all record
sets share one shape.

**Risk:** medium — adds the largest amount of new writer code.
**Test gate:** new unit tests per writer; round-trip test from a
real example model (plate_with_edge_crack).

### Phase 8.6 — Bridge enrichment additions
`/opensees/tag_map/`, plus a `set_current_fem_element_id` side
channel in `_internal/build.py` parallel to `set_element_nodes`.
Allows the H5Emitter to record (fem_eid, ops_tag) pairs.

**Risk:** low — additive.

### Phase 8.7 — Viewer migration off FEMData / solvers
Migrate `viewers/ui/*_tab.py`, `viewers/overlays/*.py` to read from
`model.h5` instead of from FEMData broker or solvers records.
After this PR, `viewers/` has zero imports from `solvers/` or `mesh/`.

**Risk:** medium-high — UI changes are user-visible.
**Test gate:** viewer integration tests pass against fixture .h5 files.

### Phase 8.8 — Delete `solvers/`
After two release cycles of deprecation. Remove the shim. Delete
`src/apeGmsh/solvers/`.

**Risk:** medium — the irreversible step. Should land only after every
release-blocker app has migrated.

## 6. Acceptance criteria for Phase 8 as a whole

- `git grep "from apeGmsh.solvers" src/` returns zero matches.
- `git grep "from apeGmsh.solvers" tests/` returns zero matches.
- `apeGmsh/solvers/` directory is empty / deleted.
- `apeGmsh.results.*` imports from `apeGmsh.opensees._response_catalog`,
  not from `solvers/`.
- `apeGmsh.viewers.*` imports nothing from `apeGmsh.opensees.*` or
  `apeGmsh.mesh.*` internals — only `apeGmsh.results` and
  `apeGmsh.opensees.emitter.h5_reader`.
- `model.h5` is self-sufficient: the viewer can render any test
  fixture without instantiating a `FEMData`.
- `schema_version` is `2.0.0` (post-8.4 reshuffle).

## 7. Open questions deferred to implementation

1. **`fem.to_h5` partial vs. full writer.** Should
   `fem.to_h5(path)` write only the neutral zone, or should it
   stub out an empty `/opensees/` zone for symmetry? Probably
   neutral-only — absent zone is the right "no enrichment" signal.

2. **Native FEM HDF5 (`mesh/_femdata_native_io.py`) coexistence.**
   That writer targets `/model/` sub-group for Results-side embedded
   FEM snapshots. Different layout, different consumer, NOT a
   duplicate of `fem.to_h5(path)`. Keep both. Document in the 8.5 PR.

3. **Public re-exports during deprecation.** `apeGmsh/__init__.py`
   currently re-exports `Constraints` and `Numberer`. After 8.1, the
   top-level names should map to the new locations transparently;
   the `solvers/` shim only catches direct `from apeGmsh.solvers
   import X`.

4. **Recorder declaration consolidation.** *Resolved in
   [phase-8.3b-scope.md](phase-8.3b-scope.md):* Phase 8.3b shipped
   Flavor 1 (relocate the legacy `Recorders` class to
   `apeGmsh.results.spec.declaration` rather than delete it).  The
   typed-primitives-as-declaration unification (Flavor 2) is
   deferred to its own scoping conversation; both abstractions
   coexist on `main` for now.

5. **One ADR or several.** This doc proposes a chain-of-decisions
   that probably warrants 2–3 ADRs:
   - **ADR 0012** — model.h5 as canonical model database (zone
     convention, symmetry contract).
   - **ADR 0013** — record relocation: dataclasses live in
     `mesh/records/`, not `solvers/`.
   - **ADR 0014** — viewer is a pure model.h5 consumer (no FEMData
     imports).
   These land alongside the corresponding sub-phase PRs.

## 8. Out of scope

- **Adding a second solver** (Code_Aster, Abaqus, etc.). The zone
  convention makes room for one, but the actual plug-in is its own
  project.
- **MPCO schema changes.** Phase 8 does not touch the MPCO reader
  side beyond moving response-catalog imports.
- **Recipe migration.** `solvers/` has no recipes today; the Phase 7
  recipe work is independent.
- **Live execution path migration.** `LiveOpsEmitter` already exists
  in Phase 4; no work needed here.

## References

- [charter.md](charter.md) — the 14 principles, especially P3
  ("bridge takes a FEMData, not a session"), P8 ("adding an emit
  target is one new file"), P9 ("opensees does not depend on core or
  mesh internals").
- [h5-schema.md](h5-schema.md) — current schema (will be updated by
  Phase 8.4).
- [viewer-integration.md](viewer-integration.md) — current viewer
  contract (will be updated by Phase 8.4 + 8.7).
- [parallel-execution.md](parallel-execution.md) — Phase 0–7 plan
  (Phase 8 in that doc is replaced by this one).
- [decisions/0011-h5-as-fourth-emit-target.md](decisions/0011-h5-as-fourth-emit-target.md)
  — model.h5 as emit target (this proposal extends it).
