# Plan — Bridge load consumption (ADR 0051)

Implementation plan for [ADR 0051](../src/apeGmsh/opensees/architecture/decisions/0051-bridge-load-consumption.md).
Closes **LOAD-1** (redesigned: opt-in, not auto-emit) and **DOC-1**;
reframes **BRIDGE-1** as a masses/constraints follow-up round.

**Guiding invariant:** this is the *bridge-consumption* half — the
`g.loads` / `g.displacements` authoring surface (ADR 0050 P1–P3) is done
and untouched. The serialized `LoadRecord.pattern` field is **unchanged**
throughout (only the authoring verb renames), so `model.h5` round-trip and
compose tag-rewrite stay stable.

**Verify gotcha:** bare `python -c "import apeGmsh"` resolves the MAIN-repo
editable install (no worktree changes). Verify via `pytest`
(`pythonpath=["src"]` → worktree) using the `opensees_venv` python. All
work here is deck-level (Tcl/py emit) + solved-reaction checks — no viewer,
no GPU. Assert on captured deck lines (as `tests/opensees/unit/test_emitter_*`).

## Scope (ADR 0051)

- **In:** case rename, opt-in `from_model`, delete auto-emit, all-nodal,
  stage-scoped patterns, two-mode + no-mixing guard, `remove_bc` alias,
  docs reconciliation. (`WarnUnconsumedModelLoads` + `ops.ignore_model_loads`
  shipped in BL-4 then **removed** — see §7 / BL-4 below.)
- **Out / deferred:** element-load (`eleLoad`) consumer; beam fixed-end
  moments; beam/shell cross-dim self-weight + section A/t seam; serialized
  field rename; masses/bc import symmetry (follow-up round).

## Phases

### BL-1 — `case` rename (authoring-surface only) ✅ shippable alone
Pure rename, zero behavior change. `g.loads.pattern` → `g.loads.case`;
`patterns()` → `cases()`; `by_pattern()` → `by_case()`. Same on
`g.displacements`. The context manager still sets the record's `pattern`
field (unchanged) — only the verb name changes.
→ verify: parametrized old-call≡new-call producing byte-identical records;
`pytest tests/` loads + displacements suites green; grep shows no
`.pattern(` / `.patterns()` / `by_pattern` survivors in `src/` docs / the
apegmsh-helper skill / notebooks.

### BL-2 — `from_model` import + delete auto-emit (LOAD-1 core)
1. Add `Plain.from_model(case: str)` — read `fem.nodes.loads` (+ imposed
   `fem.nodes.sp` where `is_homogeneous` is False) filtered by
   `rec.pattern == case`; replay each as a recorded `load` / `sp` entry on
   the pattern. Reuse the existing `_broker_load_components` 3D→ndf map.
2. Delete `_emit_broker_loads` + `_emit_broker_loads_partitioned` and their
   call sites (`apesees.py:877` single-process, `:1718` partitioned).
→ verify: a nodal `g.loads` reaches the deck **only** via `from_model`
(assert deck has no `load` line without an import); migrate the existing
auto-emit tests to explicit import; a `from_model` of a known case yields
the expected `load` lines (deck-line assert) and the model actually moves
(solved-reaction smoke).

### BL-3 — stage-scoped patterns
Add `_StageBuilder.pattern(series=)` returning a stage-owned `Plain`
(context manager). Emits inside the stage block (after the chain, before
`analyze`, alongside the existing stage-bound emit order); claimed patterns
are excluded from any global pass. Mirror in `_emit_stages_partitioned`
(per-rank `load`/`sp` fan-out, reuse the rank-owned-node filter).
→ verify: a stage pattern's `load`/`sp` lines land inside that stage's
block and are frozen by its `loadConst`; a second stage's pattern is
independent; partitioned fixture routes per rank.

### BL-4 — two-mode no-mixing guard (~~+ reconciliation warning~~)
1. `BridgeError` at build when a model has **both** a global
   `ops.pattern.Plain` registration **and** `stage_records`. **SHIPPED + KEPT.**
2. ~~`WarnUnconsumedModelLoads` (UserWarning subclass) at build: diff the
   session cases against the set imported by any pattern; warn per
   un-imported case.~~ **SHIPPED then REMOVED** — with loads opt-in the
   explicit deck is authoritative; a completeness audit re-coupled the
   geometry case-list to the deck (the `case` vs `pattern` link §1
   severed) and could not tell "forgot" from "deliberately not in this
   deck". See ADR 0051 §7. The masses/`bc` mirror variant never shipped.
3. ~~`ops.ignore_model_loads(case)`~~ — removed with the warning.
→ verify (no-mixing only): mixing raises with a message naming both;
stage-only patterns + global-without-stages pass. (The unconsumed-case
warning tests were removed with the feature.)

### BL-5 — `remove_bc` alias
`_StageBuilder.remove_bc` delegates to `remove_sp` verbatim.
→ verify: `s.remove_bc(...)` and `s.remove_sp(...)` produce identical
`SPRemovalRecord` / deck lines; docstring states the 1-based-index vs
flag-vector distinction.

### BL-6 — docs reconciliation (DOC-1 / DOC-2)
Rewrite `guide_loads.md` + `guide_opensees.md` to the true opt-in behavior
(case vocabulary; both execution modes; `from_model`; no auto-emit). Refresh
the apegmsh-helper skill. Fill the `ops.*` namespace docstrings (DOC-2).
→ verify: skill ⇄ guide agree on emit; grep clean of stale "auto-emit"
claims; `todo_apesees.md` marks LOAD-1 / DOC-1 DONE.

## Sequencing

- **BL-1 shippable alone** (rename) — unblocks skill/doc churn early.
- **BL-2 depends on BL-1** (uses `case`). The keystone (LOAD-1 close).
- **BL-3 depends on BL-2** (`from_model` exists, now stage-scoped).
- **BL-4 after BL-3** (guard needs both modes wired).
- **BL-5 trivial** — any time after the stage builder exists.
- **BL-6 last** — documents the landed behavior.
- Each phase is its own PR off `main` (per the PR-base rule), not stacked.

## Follow-up round (separate ADR/plan) — masses + constraints

After loads land, a short round designs the **import symmetry** for the
model-definition channels (BRIDGE-1 remainder):
- `ops.mass.from_model(...)` / `ops.fix.from_model(...)` — opt-in import of
  `g.masses` / `g.constraints.bc` mirroring `p.from_model`.
- Whether masses ever auto (user: "with more usage perhaps, not yet").
- Per-DOF mass mapping review.

## Deferred (logged, not in this plan)

- Element-load (`eleLoad`) consumer — `beamUniform` / `surfacePressure` /
  `bodyForce`; the all-nodal scope drops it "until further notice".
- Beam fixed-end moments + beam/shell cross-dim self-weight + the
  element→section (A/t)→material (ρ) introspection seam.
- Serialized `LoadRecord.pattern` → `case` field rename (H5 + compose).
- Single-stage sugar (`ops.run_static`).
