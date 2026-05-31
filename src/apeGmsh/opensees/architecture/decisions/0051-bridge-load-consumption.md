# ADR 0051 — Bridge load consumption: opt-in import, load cases, stage-scoped patterns

**Status:** Proposed (2026-05-31). The **bridge-consumption half** of the
loads work. Sibling to [ADR 0050](0050-dimension-indexed-loads-and-displacements.md)
(the *authoring-surface* half). **Supersedes** ADR 0050's LOAD-1 framing
(auto-emit) and its §5 element-form / cross-dim-gravity *emit* decisions —
see "Consequences". Builds on the staged-analysis orchestrator
([0029](0029-staged-analysis-context-manager.md) /
[0034](0034-stage-bound-bcs-and-recorders.md)) and the explicit-pattern
doctrine ([0005](0005-patterns-explicit.md) /
[0007](0007-time-series-separated-from-pattern.md)).

## Context

[ADR 0050](0050-dimension-indexed-loads-and-displacements.md) restructured
the `g.loads` **authoring** surface and split prescribed motion into
`g.displacements`. It deferred the bridge-emit half to **LOAD-1**, assuming
the bridge would (a) **auto-emit** the resolved records and (b) support
**element-form** loads (`beamUniform` / `surfacePressure` / `bodyForce`)
for beam fixed-end moments and cross-dimensional gravity.

A read of the bridge as it stands (post-#493) shows:

- The resolver **already produces every record type** —
  `NodalLoadRecord` (point/line/surface/gravity/volume in nodal form) and
  `ElementLoadRecord` (`target_form="element"`), plus `SPRecord`
  (carrying `is_homogeneous`).
- The bridge **auto-emits only nodal `g.loads`** (`_emit_broker_loads`,
  `apesees.py:3299`), synthesizing one `Plain` + `Linear` per distinct
  `pattern` label. Element loads, broker SP, and broker masses have **no
  consumer**.
- The bridge **already has** the explicit authoring surface this ADR
  needs: `ops.fix` / `ops.mass` (model-level, no auto),
  `ops.timeSeries.*`, and `ops.pattern.Plain(series=)` with `.load(...)`
  / `.sp(...)`. `Plain.sp`'s own contract already states *"homogeneous
  SPs (value=0) are model-level — use `ops.fix(...)`"*.

Two problems with the auto-emit assumption surfaced in design:

1. **It is implicit.** The user must know *which* broker channels silently
   reach the deck — exactly the **DOC-1** guide⇄skill contradiction.
2. **It conflates the geometry's load grouping with an OpenSees pattern.**
   Under auto-emit the `pattern` label became a 1:1 proxy for an emitted
   `pattern Plain`. But a pattern's temporal identity — its `timeSeries`,
   its owning **stage**, its `loadConst` freeze — is an *analysis-time*
   concern the geometry cannot express. The label is not a pattern.

Meanwhile apeGmsh already has a mature staged orchestrator (`ops.stage`),
but **loads/patterns were never made stage-aware** — ADR 0029 INV-7
explicitly anticipated "stage-bound patterns" as future work.

## Decision

### 1. Load **cases** on the geometry, not "patterns"

Rename the geometry grouping verb `g.loads.pattern(name)` →
**`g.loads.case(name)`** (and `patterns()`/`by_pattern()` →
`cases()`/`by_case()`); same for `g.displacements`. A **case** is a
grouping label with **no temporal meaning** — no series, no stage, no 1:1
mapping to a bridge pattern. The vocabulary becomes honest: **case on the
geometry, pattern on the bridge.** They differ in name because they are
different things — this kills the conflation by construction.

The serialized `LoadRecord.pattern` **field** is unchanged (authoring-
surface rename only) so `model.h5` round-trip and compose stay stable —
the field rename is a deferred open question.

### 2. Loads are **opt-in**: `p.from_model(case)`

Delete the `_emit_broker_loads` (and `_emit_broker_loads_partitioned`)
**auto-emit**. Resolved nodal records reach the deck **only** through an
explicit import into a bridge pattern:

```python
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("dead")        # pull resolved nodal records tagged case "dead"
    p.load(node=99, forces=(50, 0, 0))   # ...mix with ad-hoc bridge-authored loads
```

`from_model(case)` reads `fem.nodes.loads` (and imposed `fem.nodes.sp`)
where the record's `pattern` field equals `case`, and replays each as a
`load` / `sp` line inside the pattern. This **closes DOC-1** with a precise
answer: *`g.loads` does not auto-emit; loads are imported or authored
explicitly on the bridge.*

### 3. All loads are **nodal** (element-form dropped)

Every `g.loads` verb (point / line / surface.pressure/traction/shear /
gravity-on-solids / volume) resolves to **`NodalLoadRecord`** and emits as
`load`. The bridge has **no `eleLoad` consumer** — `ElementLoadRecord` is
not consumed "until further notice."

This **reverses ADR 0050 §5** ("element form mandatory" for beam fixed-end
moments and cross-dim gravity). The two things genuinely lost are logged,
not solved: (a) beam **fixed-end moments** (nodal lumping of a UDL drops
them), and (b) beam/shell **self-weight** via `beamUniform` (needs the
section A/t seam). Both are **deferred**. dim-3 **solid** gravity already
works nodal and is unaffected.

### 4. The model-definition vs load-case split

| Category | Source | OpenSees | Where | Brought in |
| --- | --- | --- | --- | --- |
| **Model definition** | `g.constraints.bc` (homog.) | `fix` | global, pre-stage | **explicit** `ops.fix` |
| | `g.masses` | `mass` | global, pre-stage | **explicit** `ops.mass` |
| | `g.constraints.*` (MP) | equalDOF/rigid… | global, pre-stage | auto *(pre-existing, unchanged)* |
| **Load case** | `g.loads` | `load` | pattern → stage | **opt-in** `from_model` / `p.load` |
| | `g.displacements` (imposed) | `sp` | pattern → stage | **opt-in** `from_model` / `p.sp` |

Homogeneous fixes are domain-level and time-invariant (correct that they
are *not* pattern-bound); imposed motion is an `sp` that **must** live in a
pattern and be scaled by a series (correct that it *is*). The split is
keyed on `SPRecord.is_homogeneous` and is already embodied in the bridge.

### 5. Two execution modes, with a hard **no-mixing** rule

- **Non-staged** (single load case): a global `ops.pattern.Plain` + the
  analysis chain on the bridge + `ops.analyze(...)` / `ops.eigen(...)`.
  In-process. No `loadConst`.
- **Staged** (multi-phase): `with ops.stage(...)`, **every pattern lives
  inside a stage** (§6), `s.run(...)` → `ops.tcl`/`ops.py`. Per-stage
  `loadConst`/`wipeAnalysis` cleanup (ADR 0029).

A model **may not mix** a global pattern with stages → `BridgeError`. This
captures the safety of "always stage" (no load reaches the solver
un-staged in a phased model) **without** the cost of universal stages —
which is blocked on deferred staged-live execution (ADR 0029 INV-5) and
does not fit `eigen` (no analyze-loop / `loadConst`).

### 6. Stage-scoped patterns

`s.pattern(series=)` on `_StageBuilder` creates a pattern **owned by the
stage** — it emits inside the stage block and is frozen by that stage's
`stage_close` `loadConst`. The existing **global** `ops.pattern.Plain`
("fires in every stage's analyze loop", ADR 0031) remains the non-staged
path; using it alongside stages is the §5 error.

### 7. Reconciliation warning

At build, `WarnUnconsumedModelLoads` fires when the geometry declared a
case (loads or imposed displacements) that **no** bridge pattern imported,
or a `g.constraints.bc` / `g.masses` not mirrored by `ops.fix` / `ops.mass`.
It **warns, does not fail** (per user decision). `ops.ignore_model_loads(case)`
silences a deliberately-dropped case.

### 8. `s.remove_bc` alias

`s.remove_bc` ≡ `s.remove_sp` (the latter kept for shipped decks/tests).
Reads correctly against `g.constraints.bc`. The DOF-convention gotcha is
unchanged and documented: `remove_bc` `dofs=` are **1-based indices**;
`ops.fix`/`s.fix` `dofs=` is a **0/1 flag vector**.

## Rationale / alternatives

| Alternative | Why rejected |
| --- | --- |
| **Keep auto-emit** (ADR 0050 LOAD-1) | Implicit; the user must memorise which broker channels flow. Reinstates the case→pattern 1:1 the design severs. DOC-1 stays unanswerable. |
| **Build the element-load `eleLoad` consumer now** | Real gaps: `surfacePressure` is normal-only (can't carry `traction`/`shear` global vectors); a brick's body force comes from constructor args, not `eleLoad` magnitude (`Brick.cpp:653-674`). Nodal covers the common cases; defer until a concrete need. |
| **Always-stage (universal)** | Blocked on deferred staged-live execution; `eigen` has no stage shape; massive migration of the non-staged corpus. The no-mixing rule gets the safety benefit cheaply. |
| **Keep the name "pattern" on the geometry** | Collides with the OpenSees pattern and was the documented root of the in-design confusion. `case` is engineering-correct (load case → combination/pattern). |
| **Bind the time series at load declaration** | The series is a property of the *pattern*, not the load; the same case can ramp in one stage and be held constant in another. The label/series separation is the point. |

## Consequences

- **Breaking:** removing `_emit_broker_loads` means existing non-staged
  scripts/notebooks/tests that relied on nodal `g.loads` auto-emit must add
  an explicit `p.from_model(case)`. Mechanical migration.
- **DOC-1 closes** precisely (loads are opt-in). **DOC-2** (namespace
  docstrings) folds into the docs phase.
- **ADR 0050 superseded in part:** LOAD-1 redesigned (opt-in, not
  auto-emit); §5 element-form + cross-dim gravity *emit* deferred. The
  authoring surface (P1–P3) and `g.displacements` stand unchanged.
- **BRIDGE-1 reframed:** model-level `fix`/`mass` are already explicit; only
  an *import symmetry* (`ops.fix.from_model` / `ops.mass.from_model`) + the
  reconciliation warning remain — the **masses/constraints follow-up round**.
- **Persistence untouched:** the `pattern` field is unchanged; only the
  authoring verb is renamed.

## Open questions

1. **Serialized `pattern` → `case` field rename.** Deferred — touches H5
   round-trip + compose tag-rewrite. Authoring rename ships first.
2. **`from_model` granularity.** By case only (lean), or also by `pg=` /
   target? Case-only matches how the geometry already groups.
3. **Masses / bc import symmetry.** `ops.mass.from_model(...)` /
   `ops.fix.from_model(...)` — designed in the follow-up round.
4. **Single-stage sugar.** A future `ops.run_static(case=…, steps=…)`
   wrapping one implicit stage, if one-mental-model ergonomics is wanted —
   sugar, not a forced universal. Not built now.

## Related

- [ADR 0050](0050-dimension-indexed-loads-and-displacements.md) — the
  authoring half this ADR consumes; superseded in part (LOAD-1, §5 emit).
- [ADR 0029](0029-staged-analysis-context-manager.md) /
  [0034](0034-stage-bound-bcs-and-recorders.md) — the stage orchestrator
  `s.pattern` extends; the `loadConst` cleanup contract loads rely on.
- [ADR 0005](0005-patterns-explicit.md) /
  [0007](0007-time-series-separated-from-pattern.md) — explicit patterns +
  series-separated-from-pattern, the doctrine this ADR carries to the
  geometry⇄bridge seam.
- `internal_docs/plan_bridge_load_consumption.md` — phased implementation.
- `internal_docs/todo_apesees.md` — LOAD-1 (redesigned here), DOC-1/DOC-2,
  BRIDGE-1 (masses/constraints follow-up).
