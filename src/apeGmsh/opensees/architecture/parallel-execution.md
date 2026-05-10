# Parallel execution plan

This document is the work breakdown for building `apeGmsh.opensees`.
It identifies independent slices that can be executed by parallel
agents, the foundation that must land first to keep them coherent,
and the sync points where the slices come together.

## The dependency graph

```
                       ┌──────────────────────────────────┐
                       │    PHASE 0 — FOUNDATION           │
                       │    (one or two coordinated agents) │
                       │                                    │
                       │    types, base classes, Protocol   │
                       │    must land before anyone else    │
                       │    starts                          │
                       └──────────────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────┬──────────────────┐
              ▼                    ▼                 ▼                  ▼
        PHASE 1A             PHASE 1B           PHASE 1C            PHASE 1D
        Materials            Sections          Transforms           Time series
        (1 agent)            (1 agent)         (1 agent)            (1 agent)
              │                    │                 │                  │
              └────────────────────┼─────────────────┴──────────────────┘
                                   ▼
                              PHASE 2 — Elements
                              (parallel: beam_column, truss, shell,
                               solid, zero_length, joint — 3-6 agents)
                                   │
              ┌────────────────────┼─────────────────────────┐
              ▼                    ▼                         ▼
        PHASE 3A             PHASE 3B                  PHASE 3C
        Patterns +           Recorders                 Analysis primitives
        loads                (delegates to             (1 agent for all)
        (1 agent)            existing system,
                              1 agent)

              ┌────────────────────┼─────────────────────────┐
              ▼                    ▼                         ▼
        PHASE 4A             PHASE 4B                  PHASE 4C
        TclEmitter           PyEmitter                 LiveOpsEmitter
        (1 agent)            (1 agent)                 (1 agent)

                                   │
                                   ▼
                              PHASE 5 — Aggregates
                              (Node, ElementGroup, composites — 1-2 agents)
                                   │
                                   ▼
                              PHASE 6 — H5Emitter
                              (1 agent, after primitives stabilize)
                                   │
                                   ▼
                              PHASE 7 — Recipes
                              (off the critical path; can land any time
                               after Phase 1)
                                   │
                                   ▼
                              PHASE 8 — Migration / cutover
                              (apps move from apeGmsh.solvers)
```

## Phase 0 — Foundation (must land FIRST)

**Critical gate: every agent in later phases is blocked until Phase 0 is merged.**
This is the load-bearing layer. Two coordinated agents max — preferably one.

### What lands

```
apeGmsh/opensees/
├── _internal/
│   ├── __init__.py
│   ├── types.py              ← base classes
│   ├── tag_allocator.py
│   ├── ns.py                 ← namespace base
│   └── registry.py           ← maps type tokens to classes
├── emitter/
│   ├── __init__.py
│   ├── base.py               ← Emitter Protocol (frozen)
│   └── recording.py          ← RecordingEmitter (test fixture)
├── apesees.py                ← apeSees class skeleton
└── __init__.py               ← public exports
```

Plus the test scaffolding:

```
tests/opensees/
├── __init__.py
├── conftest.py
├── fixtures/
│   └── __init__.py
├── unit/, contract/, integration/, parity/, live/, subprocess/, h5/
│   ├── __init__.py
└── (empty test files with TODO markers per phase)
```

### Specific deliverables

1. **`_internal/types.py`** — base classes:
   - `Primitive` (ABC): `_emit(emitter, tag) -> None`,
     `dependencies() -> tuple[Primitive, ...]`, `__repr__`.
   - `UniaxialMaterial(Primitive)`, `NDMaterial(Primitive)`.
   - `Section(Primitive)`.
   - `GeomTransf(Primitive)`.
   - `Element(Primitive)` (or `ElementSpec`; the actual created object
     is an `ElementGroup`).
   - `TimeSeries(Primitive)`.
   - `Pattern(Primitive)` — context manager protocol.
   - `Recorder(Primitive)`.
   - `Analysis Component` bases (one per kind, or one general).

2. **`emitter/base.py`** — the full `Emitter` Protocol from
   [emitter.md](emitter.md), frozen.

3. **`emitter/recording.py`** — `RecordingEmitter` capturing every
   call as `(method_name, args, kwargs)`. Used by everyone in unit
   tests.

4. **`_internal/tag_allocator.py`** — `TagAllocator` per primitive
   kind. Sequential. Resettable.

5. **`_internal/ns.py`** — base classes for namespace objects:
   - `_BridgeNamespace(bridge)` — every namespace inherits.
   - Helper for the `_register_and_return` pattern used by every
     namespace method.

6. **`apesees.py`** — `apeSees` class:
   - `__init__(fem)` storing the snapshot.
   - Stub methods: `model`, `fix`, `mass`, `analyze`, `tcl`, `py`, `run`, `h5`.
   - Stub namespaces: `uniaxialMaterial = _UniaxialMaterialNS(self)`, etc.
     The namespace classes themselves are stubs (no concrete type
     methods yet — those come in later phases).
   - `_register(primitive)` — adds to internal list, allocates tag,
     returns the primitive.
   - `build()` — returns a `BuiltModel` skeleton.

7. **`__init__.py`** — re-exports `apeSees`.

### Tests that ship in Phase 0

- `unit/test_apesees_class.py` — bridge construction, `set_model`,
  empty build.
- `unit/test_tag_allocator.py` — sequential allocation, per-kind
  isolation.
- `unit/test_emitter_protocol.py` — `RecordingEmitter` records
  calls correctly.
- `contract/test_primitive_base.py` — every typed class to land
  later will satisfy this contract; we enshrine it now with a
  parametrize-list that starts empty and grows.

### Acceptance criteria for Phase 0

- `from apeGmsh.opensees import apeSees` succeeds.
- `mypy --strict apeGmsh/opensees` passes.
- All Phase 0 tests pass.
- The `Emitter` Protocol covers every method enumerated in
  `emitter.md`.

**Until these are green, no Phase 1+ work merges.**

## Phase 1 — Primitive families (parallel)

Four agents, four files. Each agent owns one family end-to-end:
typed classes + namespace methods + unit tests + contract list
update.

### 1A — `material/uniaxial.py`

**Classes:** `Steel01`, `Steel02`, `ASDSteel1D`, `Concrete01`,
`Concrete02`, `ASDConcrete1D`, `Hysteretic`, `ElasticMaterial`,
`ENT`, `Cable` (start with the first 6, others later).

**Namespace methods:** matching `_UniaxialMaterialNS` methods on
the bridge.

**Tests:** `unit/primitives/test_materials_uniaxial.py`. Add each
class to `ALL_UNIAXIAL` in
`contract/test_uniaxial_material_contract.py`.

**Reference:** mirror parameter shapes from `apeSees/materials/`
(Steel02, Concrete02 already have validated typed classes there).

### 1B — `material/nd.py`

**Classes:** `ElasticIsotropic`, `J2Plasticity`, `DruckerPrager`,
`PressureIndepMultiYield`, `PM4Sand`, `ASDConcrete3D`.

**Namespace:** `_NDMaterialNS`.

**Tests:** mirror 1A.

### 1C — `section/`

Three files: `fiber.py`, `plate.py`, `beam.py`.

**Classes:** `Fiber` / `GeneralFiberSection`,
`ElasticMembranePlateSection`, `LayeredShell`,
`LayeredShellFiberSection`, `ElasticSection` (1-D scalar).

**Namespace:** `_SectionNS`.

**Tests:** `unit/primitives/test_sections_*.py`. Cross-cutting
test: a Fiber section's `dependencies()` returns its materials.

### 1D — `transform.py`

**Classes:** `Linear`, `PDelta`, `Corotational`. Re-export
`Cartesian`, `Cylindrical`, `Spherical` from
`solvers/_opensees_csys.py` (already shipped — see
[ADR 0010](decisions/0010-csys-for-frame-orientation.md)).

**Namespace:** `_GeomTransfNS`.

**Tests:** verify CS rule integration produces correct vecxz for
horizontal beams, columns, the shoe-buckle arch case.

### 1D-extra — `time_series/time_series.py`

Can run parallel with 1A-1C. **Classes:** `Linear`, `Constant`,
`Path`, `Trig`, `Pulse`, `ASCE41Protocol`, `FEMA461Protocol`,
`ATC24Protocol`.

**Namespace:** `_TimeSeriesNS`.

**Reference:** mirror `apeSees/timeseries/protocols.py` for the
loading-protocol classes — those are already validated and well
designed.

## Phase 2 — Elements (parallel after Phase 1)

Six files in `element/`. Multiple agents possible — each takes one
or two files:

| Agent | File | Classes |
|---|---|---|
| α | `beam_column.py` | `elasticBeamColumn`, `forceBeamColumn`, `dispBeamColumn`, `ElasticTimoshenkoBeam`, `CatenaryCable` |
| β | `truss.py` + `zero_length.py` | `Truss`, `CorotTruss`, `InertiaTruss`; `ZeroLength`, `ZeroLengthSection`, `ZeroLengthContact` |
| γ | `shell.py` | `ShellMITC3`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4`, `ASDShellT3` |
| δ | `solid.py` | `FourNodeTetrahedron`, `TenNodeTetrahedron`, `stdBrick`, `bbarBrick`, `SSPbrick`, `FourNodeQuad`, `Tri31`, `SSPquad` |
| ε | `joint.py` | `Joint2D`, `Joint3D` (rare; can be deferred) |

**Namespace:** `_ElementNS` aggregates all of them.

**Each element class returns an `ElementGroup` from its namespace
method** (apeGmsh-native return type, mirroring
`mesh._element_types.ElementGroup`). Phase 2 ships the basic
ElementGroup; Phase 5 expands it with capabilities.

## Phase 3 — Patterns / recorders / analysis (parallel)

Three independent slices, after primitives stabilize.

### 3A — `pattern/pattern.py`

**Classes:** `Plain`, `UniformExcitation`, `MultiSupport`,
`Earthquake` (subset of MultiSupport).

`Plain` is a context manager: `__enter__` returns the pattern
instance with `.load()`, `.eleLoad`, `.sp()` methods. `__exit__`
finalizes.

**Tests:** verify pattern-explicit error handling per
[ADR 0005](decisions/0005-patterns-explicit.md): calling
`ops.load()` outside a pattern raises with a clear message.

### 3B — `recorder.py`

Thin wrapper over the existing `Recorders.py` system. Surfaces
typed namespace methods: `Node`, `Element`, `MPCO`. Each delegates
to the existing resolution + emit machinery; the new layer is
typing + namespace integration.

### 3C — `analysis/`

Seven files mirroring OpenSees subfolders:
`constraint_handler.py`, `numberer.py`, `system.py`, `test.py`,
`algorithm.py`, `integrator.py`, `analysis.py`.

One agent owns all seven — they share patterns and the volume per
file is small.

## Phase 4 — Concrete emitters (parallel)

After Phase 3, three files:

| File | Agent | Notes |
|---|---|---|
| `emitter/tcl.py` | A | Tcl string accumulation; `pattern_open` writes `pattern Plain N tsTag {`; `pattern_close` writes `}` |
| `emitter/py.py` | B | `ops.X(...)` strings; `pattern_open` writes `ops.timeSeries(...) ; ops.pattern(...)` |
| `emitter/live.py` | C | Direct `ops.X(...)` calls; only emitter that imports openseespy |

Each agent ships their emitter + parity tests for the model
fixtures.

## Phase 5 — Aggregates

Two slices:

### 5A — `Node` aggregator

Built on top of `mesh.NodeComposite`. Lives in `apesees.py` (or a
new `node.py`). Methods: `.fix(dofs)`, `.mass(values)`,
`.load(forces)` (inside pattern), introspection accessors.

### 5B — `ElementGroup` capabilities

Capabilities on the existing `ElementGroup` from Phase 2:
`.plot()`, `.summary()`, `.dependencies()` queries.

## Phase 6 — H5Emitter

After everything else stabilizes. One agent, one file:
`emitter/h5.py`. Implements the schema in [h5-schema.md](h5-schema.md).
Ships test fixtures listed in
[viewer-integration.md](viewer-integration.md).

## Phase 7 — Recipes (off critical path)

Can start any time after Phase 1. Recipes are pure composition;
they don't unlock other phases.

| Recipe | Phase 1 dep |
|---|---|
| `RectangularConfinedColumn` | Materials + Sections |
| `IShape` | Sections |
| `RC_Beam` | Materials + Sections |

## Phase 8 — Migration cutover

Out of scope for the parallel agent plan. Once `apeGmsh.opensees`
is functionally complete, apps migrate from `apeGmsh.solvers`.

## Sync points

The points where parallel work converges. Each one is a checkpoint
where we verify cross-agent coherence:

| Sync | What's checked | Owner |
|---|---|---|
| **End of Phase 0** | Foundation merged; types and Protocol locked; CI green | Coordinator |
| **End of Phase 1** | All 4 primitive families pass contract tests; `mypy --strict` clean across all four | Coordinator |
| **End of Phase 2** | All elements pass contract tests; integration test "build a frame with mixed element types" passes (RecordingEmitter) | Coordinator |
| **End of Phase 4** | Parity tests pass: same model through TclEmitter / PyEmitter / LiveOpsEmitter / RecordingEmitter produces equivalent output | Coordinator |
| **End of Phase 5** | Full integration test passes: build the moment-frame example end-to-end, write Tcl, write py, run live, all match | Coordinator |
| **End of Phase 6** | H5 fixtures validate against schema; viewer team can start work | Coordinator + viewer team |

## Conflict avoidance rules

These keep parallel agents from stepping on each other:

1. **One agent per file.** Two agents do not edit the same file
   simultaneously. If a file naturally has two concerns, split it
   first.
2. **Foundation is read-only after Phase 0.** Adding a new emit
   method to the Protocol is a real architecture event that
   requires coordination — no agent does it unilaterally.
3. **Contract `ALL_*` lists are append-only within a phase.**
   Agents add their classes to the list; they don't reorder or
   remove. (Final reordering happens at the coordinator's
   discretion at sync points.)
4. **Fixtures are immutable** (per `testing.md`). New fixtures
   are new files; no agent mutates a shared fixture.
5. **One PR per slice.** Reviews stay focused. PR title format:
   `opensees: phase-1A-materials-uniaxial`.
6. **CI runs the full unit + contract + integration suite on every
   PR.** The first agent's PR establishes the suite; subsequent
   agents' PRs must keep it green.

## Estimated parallelism

If we run with the maximum parallel agents the dependency graph
allows:

| Phase | Agents | Sequential cost (rough) |
|---|---|---|
| 0 | 1 (foundation) | days |
| 1A-1D | 4 in parallel | days each, days total |
| 2 | 4-5 in parallel | days each |
| 3A-3C | 3 in parallel | day each |
| 4A-4C | 3 in parallel | day each |
| 5 | 1-2 | day |
| 6 | 1 | day |

A coordinator agent reviews each PR before merge to enforce the
sync-point checks.

## Outputs of this plan

This document is a *living plan*. As phases land, each section
gets:
- ✅ marker for completed phases
- A pointer to the merge commit / PR
- Notes on any deviations from the plan

The coordinator agent owns the upkeep of those markers.
