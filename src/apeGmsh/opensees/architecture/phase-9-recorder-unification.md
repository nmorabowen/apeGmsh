# Phase 9 — Recorder declaration unification

**Status:** Scoping (May 2026). Implements the
[Phase 8.3b Flavor 2](phase-8.3b-scope.md) deferred unification.
Additive on `main` post-Phase-8.6; introduces a minor schema bump.

This phase collapses the **two recorder declaration systems** that
exist on `main` today into one bridge-side typed declaration. The
file-based emit path consumes the new declaration directly. Domain
capture (results-side) stays a separate system with its own
declaration shape, decoupled from the bridge.

## 1. The problem

`main` carries two parallel ways to declare a recorder:

1. **Bridge-side typed primitives** — `ops.recorder.Node(...)`,
   `ops.recorder.Element(...)`, `ops.recorder.MPCO(...)`. Frozen
   dataclasses in [opensees/recorder.py](../recorder.py); 1:1 with
   OpenSees commands; raw OpenSees response tokens
   (`response="disp"`, `response=("globalForce",)`).

2. **Results-side fluent helper** — `apeGmsh.results.spec.Recorders`.
   Verbose fluent API
   ([results/spec/declaration.py](../../../results/spec/declaration.py));
   canonical vocabulary (`"displacement_x"`, `"axial_force"`);
   resolves into `ResolvedRecorderSpec`; consumed by both file emit
   (`results/spec/_emit.py`) and `DomainCapture`
   ([results/capture/_domain.py](../../../results/capture/_domain.py)).

Pain points the dual system creates:

- **User confusion.** "Should I call `ops.recorder.X(...)` or
  `Recorders().nodes(...)`?" The answer depends on which output you
  want (file vs domain) — but that pairing isn't obvious from the
  API name.
- **Vocabulary mismatch.** The two systems speak different
  vocabularies. Static typing is harder than it should be.
- **No shared archive.** Model.h5 captures the typed primitives
  under `/opensees/recorders/`; the `Recorders` helper's
  declarations don't land in model.h5 at all.
- **Migration friction.** Phase 8.7 (viewer migration) will want one
  canonical declaration to read; two parallel surfaces double the
  consumer work.

[Phase 8.3b](phase-8.3b-scope.md) explicitly deferred the
unification (Flavor 2). Phase 9 lands it.

## 2. The chain (revised)

Today's relationship:

```
                  ┌─ ops.recorder.Node/Element/MPCO ──▶ BuiltModel ──▶ file emit
user declares ───┤
                  └─ Recorders().nodes/elements/...    ─▶ ResolvedRecorderSpec
                                                          │
                                                          ├─▶ _emit.py ──▶ file emit
                                                          └─▶ DomainCapture (native h5)
```

After Phase 9:

```
                  ┌─ ops.recorder.Node/Element/MPCO ──┐
user declares ───┤                                    ├─▶ RecorderDeclaration ─▶ file emit
                  └─ ops.recorder.declare(...)        │     (bridge-owned)
                                                      │
                  ┌─ Recorders().nodes/elements/...   ┘     (fluent builder for
                  │   = re-exported builder, same effect    ops.recorder.declare)
                  │
                  └─ DomainCaptureSpec().nodes/...   ──▶ DomainCapture (native h5)
                                                          (results-side, sibling)
```

Two systems with a clean boundary, both validating against the same
canonical vocabulary.

## 3. The change

### 3a. `RecorderDeclaration` — bridge-side typed container

New module: `apeGmsh.opensees.recorder` gains a `RecorderDeclaration`
class plus per-category typed records.

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class RecorderRecord:
    """One category-level declaration entry."""
    category: str                            # nodes | elements | line_stations |
                                             # gauss | fibers | layers | modal
    components: tuple[str, ...]              # canonical names
    raw: tuple[str, ...] = ()                # raw OpenSees token escape hatch
    pg: tuple[str, ...] = ()                 # selectors (same vocabulary as
    label: tuple[str, ...] = ()              # apeGmsh.mesh.queries)
    selection: tuple[str, ...] = ()
    ids: tuple[int, ...] | None = None
    dt: float | None = None                  # cadence (mutually exclusive with n_steps)
    n_steps: int | None = None
    name: str | None = None                  # auto-generated if absent
    n_modes: int | None = None               # modal-only

    def __post_init__(self) -> None:
        # Validate components against canonical vocabulary (per-category)
        # Validate cadence (at most one of dt / n_steps)
        # Validate selector union (ids vs named selectors mutual exclusion)
        # Raw tokens skip canonical validation (pure escape hatch)


@dataclass(frozen=True, slots=True)
class RecorderDeclaration:
    """A bundle of recorder records — what to record + how."""
    records: tuple[RecorderRecord, ...]
    name: str = "default"                    # multiple named declarations allowed
```

Validation against `_response_catalog` happens at construction. The
`raw=` escape hatch sidesteps validation for users who need a
non-catalogued OpenSees token.

### 3b. `ops.recorder.declare(...)` namespace method

```python
def declare(
    self,
    *,
    nodes:        tuple[str, ...] | None = None,
    elements:     tuple[str, ...] | None = None,
    line_stations: tuple[str, ...] | None = None,
    gauss:        tuple[str, ...] | None = None,
    fibers:       tuple[str, ...] | None = None,
    layers:       tuple[str, ...] | None = None,
    modal:        bool | int = False,        # True = default n_modes; int = explicit
    raw:          dict[str, tuple[str, ...]] | None = None,
                                             # {category: (raw_token, ...)}
    pg:           str | tuple[str, ...] | None = None,
    label:        str | tuple[str, ...] | None = None,
    selection:    str | tuple[str, ...] | None = None,
    ids:          tuple[int, ...] | None = None,
    dt:           float | None = None,
    n_steps:      int | None = None,
    name:         str = "default",
) -> RecorderDeclaration:
    """Declare recorders in one call. Returns the registered declaration."""
```

Calling `declare(...)` constructs a `RecorderDeclaration` from the
kwargs, registers it on the bridge, returns it.

**Coexists with typed primitives.** `ops.recorder.Node(...)`,
`.Element(...)`, `.MPCO(...)` continue to work; each registers a
single typed primitive on the bridge. The internal representation is
the same `RecorderRecord` shape (typed primitives are recorded as
one-component records with `name=` set).

### 3c. `Recorders` fluent helper — relocated to bridge

Per F1, [`apeGmsh.results.spec.declaration.Recorders`](../../../results/spec/declaration.py)
moves to `apeGmsh.opensees.recorder.Recorders` (or a sibling module
in opensees/). The fluent style stays exactly the same:

```python
ops.recorder.builder()              # → Recorders, registered on the bridge
    .nodes(components=["displacement_x"])
    .elements(components=["nodal_resisting_force_x"])
    .fibers(components=["fiber_stress"])
    .build()                        # returns the RecorderDeclaration
```

Or as a one-shot via `ops.recorder.declare(...)` — same outcome,
different style.

A deprecation shim at `apeGmsh.results.spec.declaration.Recorders`
re-exports for one release cycle.

### 3d. File-emit path consumes `RecorderDeclaration`

The translation logic in [results/spec/_emit.py](../../../results/spec/_emit.py)
(canonical → ops recorder commands) lifts up to the bridge as part
of `RecorderRecord._emit(emitter, tag)`. Concretely:

- `RecorderRecord` gains an `_emit(emitter, tag, fem)` method.
- For typed-primitive records (single `MPCO`/`Node`/`Element`-shape
  entry), the emit is verbatim today's behavior.
- For declared records, the emit walks per-component → ops token
  translation, fans out to multiple `emitter.recorder(...)` calls
  where one declaration produces multiple OpenSees commands (the
  ``nodes(components=["displacement_x","velocity_x"])`` → two
  recorder commands case).

The existing `_emit.py` helpers (`emit_logical`, `to_ops_args`,
`mpco_ops_args`, `line_station_gpx_path`) are reused — they just
get re-homed to `apeGmsh.opensees.recorder` (or a sibling internal
module). One-way dep direction matches the existing flow.

### 3e. `DomainCapture` decouples — `DomainCaptureSpec` sibling

After Phase 9, `DomainCapture` no longer consumes
`ResolvedRecorderSpec` (which moved to bridge). Instead it consumes a
sibling `DomainCaptureSpec` that lives in `apeGmsh.results.capture.spec`:

```python
# apeGmsh.results.capture.spec
class DomainCaptureSpec:
    """Sibling of bridge's Recorders, pruned to domain-capture needs.

    Holds the same kind of per-category records (nodes, fibers,
    layers, line_stations, gauss, modal) but is owned by the
    results-side capture pipeline. Validates against the same
    canonical vocabulary as the bridge.
    """
    def nodes(self, *, components, ...) -> DomainCaptureSpec: ...
    def fibers(self, *, components, ...) -> DomainCaptureSpec: ...
    # ...
    def resolve(self, fem, ndm, ndf) -> ResolvedDomainCaptureSpec: ...
```

`ResolvedDomainCaptureSpec` is a renamed `ResolvedRecorderSpec` —
same shape today, separate identity going forward so the two systems
can diverge cleanly.

`DomainCapture.__enter__` accepts a `ResolvedDomainCaptureSpec` only;
the legacy `ResolvedRecorderSpec` path is removed (caught by mypy at
the seam).

### 3f. Vocabulary module — neutral location

`apeGmsh.results._vocabulary` is the source of canonical name lists
today (`NODAL_KINEMATICS`, `NODAL_FORCES`, `STRESS`, `STRAIN`,
`LINE_DIAGRAMS`, `FIBER`, etc.). Both the bridge and the results
side need to import from it.

Two options:
- **A.** Move to `apeGmsh.opensees._vocabulary`; results imports
  from there. Bridge becomes vocabulary owner.
- **B.** Move to top-level `apeGmsh._vocabulary` (neutral). Both
  bridge and results import from it.

**Recommendation: B.** The canonical vocabulary is solver-neutral
(response naming, not OpenSees-specific). Future second-solver work
(Code_Aster, Abaqus) will want the same vocabulary without
importing from opensees. Top-level neutral location signals shared
ownership. One-line change for results; bridge gains a fresh import.

The actual file move:
- `src/apeGmsh/results/_vocabulary.py` → `src/apeGmsh/_vocabulary.py`
- `apeGmsh.results._vocabulary` becomes a deprecation shim
  (`from apeGmsh._vocabulary import *`).

### 3g. Source-side surface

| File | What changes |
|---|---|
| `src/apeGmsh/_vocabulary.py` (NEW) | Move canonical name lists here from `results/_vocabulary.py` |
| `src/apeGmsh/results/_vocabulary.py` | Becomes deprecation shim `from apeGmsh._vocabulary import *` |
| `src/apeGmsh/opensees/recorder.py` | Add `RecorderRecord`, `RecorderDeclaration`, `Recorders` (fluent builder relocated from results); keep `Node`, `Element`, `MPCO` typed primitives |
| `src/apeGmsh/opensees/_internal/ns/recorder.py` | Add `declare(...)`, `builder()` methods; existing `Node/Element/MPCO` methods unchanged |
| `src/apeGmsh/opensees/_internal/build.py` | Extend `emit_recorder_spec` to dispatch on `RecorderDeclaration` records (fans out per-component into multiple `emitter.recorder` calls) |
| `src/apeGmsh/opensees/emitter/h5.py` | Update `_write_recorders` to handle the unified records (with `kind=("typed"|"declared")` attr); schema 2.2.0 → 2.3.0 |
| `src/apeGmsh/opensees/emitter/h5_reader.py` | Reader returns per-record kind + declaration metadata |
| `src/apeGmsh/results/spec/declaration.py` | Deprecation shim re-exporting from `apeGmsh.opensees.recorder` |
| `src/apeGmsh/results/spec/_resolved.py` | `ResolvedRecorderSpec` stays for back-compat one release; sibling `ResolvedDomainCaptureSpec` lands in `results/capture/spec.py` |
| `src/apeGmsh/results/spec/_emit.py` | Translation logic moves to `apeGmsh.opensees.recorder._emit` (private to bridge); shim re-exports for one cycle |
| `src/apeGmsh/results/capture/spec.py` (NEW) | `DomainCaptureSpec` + `ResolvedDomainCaptureSpec` (sibling of bridge's declaration) |
| `src/apeGmsh/results/capture/_domain.py` | Accept `ResolvedDomainCaptureSpec` only; legacy path removed |

### 3h. Test-side surface

| File | What changes |
|---|---|
| `tests/opensees/unit/primitives/test_recorders.py` | Existing Node/Element/MPCO tests stay; new `test_recorder_declaration.py` covers `RecorderDeclaration` + `ops.recorder.declare(...)` + the `raw=` escape hatch |
| `tests/opensees/unit/primitives/test_recorder_builder.py` (NEW) | Fluent `ops.recorder.builder().nodes(...).build()` mirrors existing `Recorders` tests |
| `tests/opensees/integration/test_recorder_unification.py` (NEW) | End-to-end: declare via both APIs, build, emit through file path, assert ops command stream is equivalent |
| `tests/opensees/h5/test_h5_recorders_unified.py` (NEW) | Unified `/opensees/recorders/` archive round-trip |
| `tests/test_results_*_recorder*.py` | Update results-side imports to point at new locations |
| `tests/test_results_domain_capture*.py` | Switch to `DomainCaptureSpec` |
| `tests/test_recorders_resolve_*.py` | If they test `Recorders().resolve(...)` the path stays identical (deprecation shim) |

### 3i. Doc-side surface

| File | What changes |
|---|---|
| `architecture/h5-schema.md` | Document the unified `/opensees/recorders/` shape; bump 2.2.0 → 2.3.0 |
| `architecture/viewer-integration.md` | Note the new record format; one paragraph |
| `architecture/phase-8.3b-scope.md` | Add note: "Flavor 2 landed in Phase 9" |
| `architecture/phase-8-untangle.md` | Mention Phase 9 as the deferred 8.3b Flavor 2 |
| `architecture/parallel-execution.md` | Add Phase 9 row to the deferred-items list; ✅ marker when landed |

## 4. Locked-in design decisions

Settled with the architect this session (May 2026):

| # | Choice | Rationale |
|---|---|---|
| D1 (vocabulary) | **Canonical primary** + `raw=` per-record escape hatch | Single primary vocabulary; escape hatch covers non-catalogued tokens |
| D2 (location) | **Bridge-side typed container** | Bridge owns FEM snapshot + emit pipeline; declaration belongs here |
| D3 (strategy) | **No strategy kwarg on `ops.analyze`** | Domain capture is its own thing (see D3 dissolution below) |
| D4 (typed primitives) | **Coexist with `declare(...)`** | Both work; both feed the same internal `RecorderDeclaration` store |
| D5 (model.h5) | **One unified `/opensees/recorders/`** group with `kind=("typed"|"declared")` attr | Single viewer code path |
| D6 (`Recorders` fate) | **Relocate to bridge as `apeGmsh.opensees.recorder.Recorders`** | Few users → low-blast-radius relocation |
| G (DomainCapture inputs) | **Sibling `DomainCaptureSpec` on results side** | Decoupled from bridge; same vocabulary, different package, different responsibilities |

### D3 dissolution

User feedback indicated domain capture should stay its own thing,
not share a verb with file recorders. That dissolves the strategy
kwarg on `ops.analyze` — there's nothing to name. `ops.analyze(steps=N)`
is always file-based. Domain capture is invoked via its own
DomainCapture context manager (unchanged shape, just sources from
DomainCaptureSpec instead of Recorders).

## 5. Sub-commits

Seven commits, similar shape to Phase 8.5 / 8.6.

### Commit 1 — `_vocabulary` move to top-level

- Create `src/apeGmsh/_vocabulary.py` with the canonical name lists,
  copied verbatim from `results/_vocabulary.py`.
- Replace `results/_vocabulary.py` body with `from apeGmsh._vocabulary import *`.
- Update internal imports inside `apeGmsh.results.*` to point at the
  new path (one-shot find/replace).
- Existing `apeGmsh.results._vocabulary` re-export keeps external
  consumers working.

**Test gate:** existing `tests/test_results_*.py` unchanged behaviour;
mypy --strict clean.

### Commit 2 — `RecorderDeclaration` typed container + tests

- Add `RecorderRecord` and `RecorderDeclaration` to `opensees/recorder.py`.
- Validation against `apeGmsh._vocabulary` per-category; `raw=`
  escape hatch.
- New unit test file `tests/opensees/unit/primitives/test_recorder_declaration.py`.
- No bridge wiring yet — just the types.

**Test gate:** new unit tests pass; mypy --strict clean.

### Commit 3 — `ops.recorder.declare(...)` namespace method

- Wire `_RecorderNS.declare(...)` to construct + register a
  `RecorderDeclaration`.
- Update `_internal/build.py` `emit_recorder_spec` to handle
  declarations (fans out per-component into emitter calls).
- Extend integration test `test_recorder_unification.py`.

**Test gate:** declare-via-bridge end-to-end; both typed primitives
and declared records produce expected `emitter.recorder(...)` calls.

### Commit 4 — `Recorders` fluent helper relocates to bridge

- Move `apeGmsh.results.spec.declaration.Recorders` → `apeGmsh.opensees.recorder`.
- The class becomes a builder that constructs a `RecorderDeclaration`
  on `.build()` (and registers it on the bridge if attached).
- Surface `ops.recorder.builder()` (or equivalent) as the bridge
  entry-point.
- Deprecation shim at the old path.

**Test gate:** existing `tests/test_recorders_*.py` pass via shim;
mypy --strict clean.

### Commit 5 — DomainCapture decouples to `DomainCaptureSpec`

- New `src/apeGmsh/results/capture/spec.py` with `DomainCaptureSpec`
  + `ResolvedDomainCaptureSpec`.
- `DomainCapture.__enter__` accepts `ResolvedDomainCaptureSpec` only.
- Update `tests/test_results_domain_capture*.py` to use the sibling
  type.
- `ResolvedRecorderSpec` stays as a deprecation alias for one cycle
  but is no longer a DomainCapture input.

**Test gate:** every `tests/test_results_domain_capture*.py` passes
against the new spec.

### Commit 6 — Model.h5 schema 2.3.0

- Update `_write_recorders` to write per-record `kind` attr and the
  declaration metadata (category, components, cadence, selectors).
- Update `h5_reader.py` accessor: `model.recorders()` returns rich
  per-record dicts.
- Schema bump 2.2.0 → 2.3.0.
- Add `tests/opensees/h5/test_h5_recorders_unified.py`.

**Test gate:** end-to-end H5 round-trip through both APIs; schema
validator accepts 2.3.0.

### Commit 7 — Documentation + deprecation envelope

- `architecture/h5-schema.md` — `/opensees/recorders/` section
  rewritten; version history adds 2.3.0.
- `architecture/viewer-integration.md` — one paragraph on the
  enriched record format.
- `architecture/phase-8.3b-scope.md` — "Flavor 2 → see Phase 9" note.
- `architecture/phase-8-untangle.md` — Phase 9 referenced.
- `architecture/parallel-execution.md` — Phase 9 row.
- `architecture/_DEFERRED.md` — remove the dual-system note if
  present.
- Deprecation shim docstrings include "removed in version X.Y".

**Test gate:** `import apeGmsh`, `import apeGmsh.results.spec`,
`import apeGmsh.solvers` all complete with zero unexpected
DeprecationWarnings; `pytest -m "not live and not subprocess"`
green.

## 6. Verification gates (per commit)

Same envelope as Phase 8.4 / 8.5 / 8.6:

- `mypy --strict src/apeGmsh/`
- `ruff check src/apeGmsh/ tests/`
- `pytest -m "not live and not subprocess" --ignore=tests/acad --continue-on-collection-errors`

Re-measure at PR-open time. Phase 9 special checks:

- Both APIs produce equivalent `emitter.recorder(...)` call streams
  for the same logical declaration.
- Model.h5 archive round-trips: `apeSees.h5(path)` → reader →
  validate → recover the original `RecorderDeclaration`.
- DomainCapture tests still pass against the new sibling spec
  (no behaviour change for the user; just internal rewire).
- `apeGmsh.results.spec.Recorders` continues to work via shim with
  one DeprecationWarning per import path.

## 7. Open questions for the implementing session

1. **Fluent builder vs declare kwargs — keep both?** D6 says relocate
   `Recorders` to bridge; user can use either `ops.recorder.declare(...)`
   one-shot OR `ops.recorder.builder().nodes(...).build()`. Both
   produce a `RecorderDeclaration`. Decision: keep both surfaces; the
   builder is just sugar over `declare(...)`.

2. **`raw=` granularity.** Per-record raw escape hatch vs per-component.
   Probably per-record (`raw=("custom_token",)` adds extra ops args
   to the recorder), but per-component would let the user mix
   canonical + raw in one record. Implementing session picks.

3. **Selector vocabulary.** The bridge's existing typed primitives
   use `pg=` (string or None). The new declaration supports
   `pg/label/selection/ids` per the Recorders helper. Should the
   typed primitives gain the extended selectors too? Recommend yes
   (consistency), but defer to implementing session.

4. **Modal — n_modes default.** `declare(modal=True)` should pick
   a reasonable default for `n_modes`. The Recorders helper today
   defaults to 1; some users want 10. Implementing session picks
   (probably 5).

5. **MPCO record promotion.** Today `MPCO` is one of three typed
   primitives; under the unified shape, MPCO could become a category
   of `declare(...)` (e.g. `declare(mpco_file="run.mpco", nodal=...)`).
   Recommend keeping `MPCO` as a typed primitive (its parameter
   shape is too different from file-per-record). Confirm during
   implementation.

6. **Element-class-name override.** The legacy `Recorders.elements`
   method has `element_class_name=` for .out transcoder
   disambiguation (tri31 vs SSPquad). Carry through to
   `RecorderRecord`? Yes — verbatim.

7. **DomainCapture's "no Recorders" deprecation path.** When the
   shim is removed in commit 5 cycle+1, `apeGmsh.results.spec.Recorders`
   stops re-exporting. Confirm timing in the deprecation envelope
   (one release cycle = next minor version?).

## 8. Out of scope

- **Replacing `ops.analyze` strategy.** D3 dissolved — `ops.analyze`
  stays file-only.
- **Domain capture as a bridge verb.** Domain capture stays an
  independent results-side system. A future phase could add an
  `ops.live_session(...)` or similar bridge verb that orchestrates
  DomainCapture, but it's not Phase 9.
- **Multi-solver vocabulary.** `apeGmsh._vocabulary` is positioned
  neutrally but Phase 9 doesn't add a second solver. That's a
  future architecture event.
- **MPCO post-processing changes.** The `apeGmsh.results` consumer
  side of MPCO files is unchanged.
- **Viewer migration off FEMData.** Phase 8.7 work — not Phase 9.

## 9. Risk assessment

**Medium.** Larger than the recent additive H5 phases (8.4–8.6) but
smaller than Phase 8.1 (records relocation). Most of the diff is
plumbing — moving things to new homes — with one substantive new
feature (the unified `RecorderDeclaration` types).

Specific risks:

- **Test-suite coverage gap.** `tests/test_recorders_*.py` (results-
  side) is broad; the `results.spec.Recorders` shim must keep all
  tests passing. Mitigation: shim covers the full API surface.
- **DomainCapture seam.** Migrating `DomainCapture` from
  `ResolvedRecorderSpec` to `ResolvedDomainCaptureSpec` is a typed
  rename; mypy catches drift. Mitigation: ship commit 5 with the
  full `DomainCaptureSpec` API matching today's `Recorders` shape;
  the spec class can shrink in a later cleanup.
- **Schema bump consumer breaks.** Any external tool reading
  Phase-8.6 model.h5 files keeps working (the unified shape is a
  superset — old reader sees `kind=typed` records as the old
  shape). New `kind=declared` records will be invisible to old
  readers. Mitigation: schema bump signals the additive change;
  internal reader (`h5_reader.py`) updated in commit 6.

Estimated effort: **6–8 days for the implementing session**,
dominated by the integration tests in commit 3 and the test-suite
migration in commits 4–5.

## 10. References

- [phase-8.3b-scope.md](phase-8.3b-scope.md) — Flavor 1 vs Flavor 2;
  this phase implements Flavor 2.
- [phase-8-untangle.md §6](phase-8-untangle.md) — recorder
  declaration consolidation noted as a deferred Phase-8 question.
- [emitter.md](emitter.md) — Emitter Protocol's `recorder` method.
- [h5-schema.md](h5-schema.md) — current `/opensees/recorders/`
  spec; will be updated by commit 6.
- [results/capture/_domain.py](../../../results/capture/_domain.py)
  — `DomainCapture` consumer that decouples in commit 5.
- [results/spec/_emit.py](../../../results/spec/_emit.py) —
  translation helpers that lift to the bridge in commit 3.
