# ADR 0029 — Staged analysis via `ops.stage(name)` context manager (Phase SSI-2.A)

**Status:** Accepted (Phase SSI-2.A, May 2026). Second of the SSI
four-ADR set ([0028](0028-initial-stress-via-parameter-ramping.md)
/ [0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0031](0031-ssi-convenience-helpers.md)). Widens the `Emitter`
Protocol with two stage-bracketing methods.

## Context

[ADR 0028](0028-initial-stress-via-parameter-ramping.md) ships the
in-situ stress ramp but only addresses a **single-stage**
analysis. Real SSI workflows are multi-stage: install in-situ
stress against the undisturbed rock; excavate; install lining;
load; observe. Each stage typically needs a **different analysis
chain** — coarser tolerance during in-situ install, tighter
tolerance during excavation, line-search algorithm during lining
activation — and the inter-stage cleanup (`loadConst` to freeze
accumulated loads, `wipeAnalysis` to free the prior chain) is
load-bearing for convergence.

The STKO reference decks (`SSI/Interaccion/analysis_steps.tcl`)
fan stages out as a flat sequence of:

```
# stage N
parameter ...; addToParameter ...; proc ...; lappend ...
constraints ...; numberer ...; system ...; test ...; algorithm ...
integrator ...; analysis ...
analyze N
loadConst -time 0.0
wipeAnalysis
# stage N+1
...
```

apeSees did not have a way to express stages declaratively. The
bridge held one global pool of analysis-chain primitives — the
moment two stages needed different `test` settings, the user had
to:

1. Construct two `apeSees(fem)` instances and emit them
   independently, then string-concatenate the resulting Tcl files;
   or
2. Drop to live mode and hand-roll the `ops.analyze` + cleanup +
   chain-rebinding sequence in Python.

Neither composes with the bridge's "one model, one emit" contract.
The SSI-1 ramp from ADR 0028 is moot if it can only fire in one
stage.

## Decision

### A new bridge method `apeSees.stage(name)` returning `_StageBuilder`

```python
def stage(self, name: str) -> "_StageBuilder":
    ...

with ops.stage(name="insitu") as s:
    s.add(rock_insitu)                  # bind an InitialStressRecord
    s.analysis(
        test=..., algorithm=..., integrator=...,
        constraints=..., numberer=..., system=..., analysis=...,
    )
    s.run(n_increments=10, dt=0.1)
```

The context manager collects per-stage records inside the `with`
block; on clean `__exit__` it validates that the analysis chain
and `run` parameters are both set and appends a frozen
`StageRecord` to the bridge's `_stage_records` tuple. Stage
ordering follows registration order — the order the `with`
blocks exited, which is lexical order for non-nested usage. Nested
`with ops.stage(...)` blocks are refused (post-merge red-team M4)
because the inner builder's `__exit__` would fire first and
register the inner stage **before** the outer in `_stage_records`,
which is the opposite of what readers expect.

`_StageBuilder` API surface:

| Verb | Required | Behaviour |
|---|---|---|
| `s.add(rec)` | optional | Binds an `InitialStressRecord` to this stage. Removes from the bridge's global pool. Other record types raise `TypeError`. Double-`add` (same record on two stages) raises `ValueError`. |
| `s.activate(pgs=[...])` | optional | Marks element-PG names as activated by this stage. See [ADR 0030](0030-stage-bound-topology-activation.md). |
| `s.analysis(test=, algorithm=, integrator=, constraints=, numberer=, system=, analysis=)` | **required** | All seven kwargs. Each must already be registered with the bridge. Second call raises `ValueError`. |
| `s.run(n_increments=, dt=None)` | **required** | Analyze-loop length + step size. Second call raises `ValueError`; `n_increments < 1` raises. |
| Clean `__exit__` | — | Validates `analysis_set` + `run_set`; emits `StageRecord` to bridge. |
| Exception in body | — | Drops the in-progress stage; user's exception propagates. |

The eight-kwarg ceremony on `s.analysis(...)` is **intentional** —
see "Alternatives considered" below for why we did not auto-pull
the bridge's last-set primitive of each kind.

### Widen the `Emitter` Protocol — two stage-bracketing methods

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def stage_open(self, name: str) -> None: ...
    def stage_close(self) -> None: ...
```

Per-emitter shape:

| Emitter | `stage_open(name)` | `stage_close()` |
|---|---|---|
| `TclEmitter` | `# === Stage: <name> ===` banner at outer indent | `loadConst -time 0.0` + `wipeAnalysis` + (if hooks registered) `set _apesees_before_step_hooks {}; set _apesees_after_step_hooks {}` + reset `_step_hooks_registered` |
| `PyEmitter` | `# === Stage: <name> ===` Python comment | `ops.loadConst('-time', 0.0)` + `ops.wipeAnalysis()` + Python equivalents of the dispatcher list reset |
| `LiveOpsEmitter` | **Raises `NotImplementedError`** | **Raises `NotImplementedError`** — staged live execution deferred |
| `H5Emitter` | No-op — H5 archival of staged structure is deferred | No-op — deferred |
| `RecordingEmitter` | Capture `("stage_open", (name,), {})` | Capture `("stage_close", (), {})` |

### `apeSees.analyze` and `apeSees.eigen` refuse staged models

```python
def analyze(self, *, steps, dt=None) -> int:
    if self._stage_records:
        raise NotImplementedError(
            "apeSees.analyze: live execution does not support "
            "staged models in Phase SSI-2.A ..."
        )
    ...

def eigen(self, num_modes, *, solver="-genBandArpack") -> EigenResult:
    if self._stage_records:
        raise NotImplementedError(
            "apeSees.eigen: live execution does not support staged "
            "models (Phase SSI-2.A) ..."
        )
    ...
```

The single-call `apeSees.analyze` / `apeSees.eigen` shape cannot
express the inter-stage cleanup contract. Users with staged models
emit a Tcl / Py deck via `ops.tcl(p, run=True)` / `ops.py(p,
run=True)` instead — the OpenSees subprocess runs every stage's
analyze loop and inter-stage cleanup as part of executing the deck.

### Stage-close cleanup contract — three load-bearing lines

`stage_close()` MUST emit, in order:

```
loadConst -time 0.0
wipeAnalysis
set _apesees_before_step_hooks {}        # only if hooks registered
set _apesees_after_step_hooks {}
```

Each line is required:

- **`loadConst -time 0.0`** — freezes the prior stage's accumulated
  loads as the new permanent baseline and resets pseudo-time to 0.0
  for the next stage's analyze loop. Without it, the next stage's
  load steps double-apply the prior stage's loads.
- **`wipeAnalysis`** — drops the prior stage's analysis-chain
  binding. Without it, the next stage's `constraints / numberer /
  system / test / algorithm / integrator / analysis` lines are
  silently shadowed by the first stage's chain.
- **Hook-list reset** — clears the dispatcher's `lappend`
  registrations so the next stage's `analyze` does not re-fire the
  prior stage's ramp procs. The proc *definitions* persist; they
  become unreachable until a future stage explicitly `lappend`s
  them. The Tcl + Py emitters also flip
  `_step_hooks_registered = False`, so the next stage's bare
  `analyze` emits a flat `analyze N` line unless that stage itself
  registers a new ramp.

The contract is **the** load-bearing detail of this ADR. A stage
emitter that omits any of these lines silently produces an
incorrect multi-stage analysis — convergence regressions that look
like material instability but are actually `wipeAnalysis` drift.

### Build-time fan-out

In `BuiltModel.emit`:

1. **Skip analysis-chain primitives in the pre-element pass** when
   `stage_records` is non-empty. Each stage re-emits its own chain.
2. **Validate `initial_stress` name uniqueness** across the global
   pool + every stage's pool (post-merge red-team H2). Duplicate
   names would produce two `proc <name> {...}` definitions; the
   second overrides the first but was built with different
   parameter tags + cumulative-state keys, so the surviving proc
   would reference an uninitialised `${name}_state(cum_<tag>)`
   array element and crash at the first analyze step of the later
   stage.
3. **Validate that `fix` / `mass` / `region` directives don't
   target stage-bound nodes** (post-merge red-team H1). A node
   owned only by stage 2 doesn't exist when the pre-stage block's
   `fix N 1 1 1` line emits; OpenSees errors at parse time.
4. **`_emit_stages_flat`** runs after the global block:
   `stage_open(name)`; stage-bound topology (per [ADR
   0030](0030-stage-bound-topology-activation.md)); stage's
   `initial_stress` global + addToParameter passes; analysis-chain
   primitives via their `_emit`; `analyze(steps, dt)`; `stage_close`.

The (stages + MP partitioned) combo is supported as of Phase
SSI-2.C (PR #315). `BuiltModel._emit_partitioned` dispatches to
the new `_emit_stages_partitioned` helper when `stage_records` is
non-empty; per-stage topology / `initial_stress` / `analyze` /
`stage_close` blocks interleave with the per-rank fan-out while
preserving cross-rank tag identity. See
[staged-analysis.md](../staged-analysis.md) §"MP partitioned +
initial_stress + stages (Phase SSI-2.C)" for the deck layout this
ADR's `_emit_stages_flat` shape extends.

## Invariants

- **INV-1.** `Emitter.stage_open` and `Emitter.stage_close` are on
  the Protocol; every existing and future emitter implements them.
  `LiveOpsEmitter` raises `NotImplementedError` on both (live
  execution deferred); H5 + Recording capture without side effects.
- **INV-2.** `stage_close()` emits exactly the contract — `loadConst
  -time 0.0` + `wipeAnalysis` + (conditional) hook-list reset, in
  that order. Re-ordering or omitting any line is a regression.
- **INV-3.** Stage emit order follows the order the `with` blocks
  exited. With M4-validated nesting refusal, this equals lexical
  order — readers can scan top-to-bottom and predict deck order.
- **INV-4.** Analysis-chain primitives are emitted **per stage**,
  not globally, when `stage_records` is non-empty. The global
  pre-element emit skips them; each stage's `_emit_stages_flat`
  iteration emits its bound `constraints / numberer / system / test
  / algorithm / integrator / analysis` via their `_emit`. Multiple
  stages can share the same primitive instance — that's fine; each
  stage emits a fresh `<command>` line at run time.
- **INV-5.** `apeSees.analyze` and `apeSees.eigen` refuse staged
  models. Without INV-5 the live emitter would happily drive the
  first stage's analyze loop and the user would silently get a
  truncated single-stage result. Loud refusal is correct until
  staged live execution lands.
- **INV-6.** Nested `with ops.stage(...)` blocks raise
  `RuntimeError`. INV-6 prevents the lexical-order surprise (inner
  registers before outer) the validator was added for in #312.
- **INV-7.** `_StageBuilder.add(...)` accepts **only**
  `InitialStressRecord` in Phase SSI-2.A. Other record types raise
  `TypeError` with a forward-pointing message. INV-7 anchors the
  API surface — future extensions (stage-bound `fix` / `mass` /
  `region`, stage-bound patterns) widen `add` deliberately, not by
  accident.
- **INV-8.** H5 archival is **deferred and fail-loud** (mirrors
  [ADR 0028 INV-7](0028-initial-stress-via-parameter-ramping.md)).
  `H5Emitter.stage_open` / `stage_close` are no-ops; the bridge
  guard `apeSees.h5(path)` (#313) raises `NotImplementedError` when
  `self._stage_records` is non-empty, pointing the user at
  `ops.tcl(path)` / `ops.py(path)`. A future
  `opensees_schema_version` bump (`2.11.0` → `2.12.0` per
  [ADR 0023](0023-per-zone-schema-versioning.md)) would persist
  per-stage primitive lists under `/opensees/stages/`, lift the
  guard, and restore round-trip parity.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Implicit chain inheritance** — `s.analysis(test=...)` only requires the kwargs that differ from the bridge's currently-set chain | Surprising semantics. The user's mental model is "this stage uses *this* analysis chain"; implicit inheritance turns that into "this stage uses *this* + whatever was last set on the bridge". Debugging which primitive landed on which stage becomes a flow-analysis exercise across the file. Requiring all seven kwargs is verbose but unambiguous. |
| **Declarative stage tuple** — `ops.stages([Stage(name=..., analysis=..., ...), ...])` instead of a context manager | Cannot incrementally build a stage's records inline with the rest of the deck. The CM lets the user write `with s: s.add(...); other_setup(); s.analysis(...); s.run(...)` and interleave with the surrounding code. The tuple form forces everything into the constructor call. |
| **Implicit `stage_close` at the next `stage_open`** | Hides the cleanup contract. INV-2 is the most load-bearing thing this ADR ships; making it an emergent property of the next `stage_open` rather than an explicit `stage_close` makes it easy to drop a stage and silently lose the cleanup. The explicit pair surfaces the symmetry. |
| **Drive staged live execution from `apeSees.analyze`** | Would require the live emitter to gain stage_open / stage_close semantics (chain re-binding, loadConst/wipeAnalysis interleave, hook-list clear). Doable but not the immediate need — the SSI use case routes through Tcl/Py subprocess. Refuse loudly (INV-5) and lift in a follow-up when a caller needs it. |
| **One global `_StageBuilder` reused across stages** | The state machine becomes unclear — when does `analysis()` reset? The per-stage builder is cheap (a few slots on the dataclass) and produces a clean lifecycle. |
| **Auto-emit `stage_open` / `stage_close` from `BuiltModel.emit`** without Protocol methods | The build pipeline would need to write Tcl directly via direct emitter-state access — exactly the "using `_internal` side-channels" anti-pattern [ADR 0018](0018-modeldata-vanilla-opensees-enrichment.md) §Alt 2 rejected for `ModelData`. The Protocol methods put the per-dialect concerns where they belong. |
| **Allow stage-bound `fix` / `mass` / `region`** in Phase SSI-2.A | Doable but expands the H1 validator to support stage-scoped emission of those directives. Out of scope for the SSI tunnel migration's first-stage needs; track as a follow-up (see [staged-analysis.md](../staged-analysis.md) §"Deferred work"). |

## Consequences

**Positive:**

- Closes the multi-stage prerequisite for the Cerro Lindo tunnel
  migration. Users can declare a four-stage in-situ /
  excavate / line / load workflow as four `with ops.stage(...)`
  blocks; the bridge handles every inter-stage cleanup detail.
- The SSI-1 ramp from [ADR 0028](0028-initial-stress-via-parameter-ramping.md)
  composes with stages — different ramps per stage, different
  analysis chains per stage, all the same `apeSees(fem)` model.
- Stage validation surfaces wrong models at build time (H1 = fix on
  stage-bound nodes; H2 = duplicate initial_stress names; M4 =
  nested `with` blocks) with offender-list error messages. Without
  these, the OpenSees subprocess would crash mid-run with messages
  that don't point at the apeSees declaration.
- The eight-kwarg `s.analysis(...)` shape pairs symmetrically with
  the standalone `ops.<family>.<Type>(...)` primitive constructors
  — users compose then bind, both at the bridge level and at the
  stage level.

**Negative:**

- Two new Protocol methods. Every concrete emitter (current +
  future) must implement them. The H5 emitter explicitly no-ops
  (deferred archival, INV-8); the Live emitter explicitly raises
  (deferred execution, INV-5). The Tcl + Py emitters carry the
  cleanup contract (INV-2).
- H5 archival of staged structure is deferred and fail-loud
  (INV-8). `apeSees.h5(path)` on a staged model raises
  `NotImplementedError` (#313); the user keeps their Python
  deck-builder around to reproduce the staged model. A future
  schema bump (`opensees_schema_version` `2.11.0` → `2.12.0`)
  would persist stages under `/opensees/stages/` and lift the
  guard.
- Live execution of staged models is deferred (INV-5). The error
  message points users to `ops.tcl(p, run=True)` / `ops.py(p,
  run=True)`. Workable but a context switch — users testing a
  single-stage model can call `ops.analyze(...)` directly; adding a
  second stage forces them through subprocess.

## Cross-references

- ADR [0028](0028-initial-stress-via-parameter-ramping.md) — the
  initial-stress mechanism this ADR builds on (`s.add(record)`
  binds an `InitialStressRecord` declared via `ops.initial_stress`).
- ADR [0030](0030-stage-bound-topology-activation.md) — the
  per-stage topology activation that ships in the same SSI feature
  set and consumes `s.activate(pgs=[...])`.
- ADR [0023](0023-per-zone-schema-versioning.md) — the per-zone
  schema policy any future archival of stages will bump.
- [staged-analysis.md](../staged-analysis.md) — the internals
  walkthrough: deck layout, ownership computation, hook-dispatcher,
  cleanup contract.
- [api-design.md](../api-design.md) §"Staged analysis" — the user
  surface walkthrough.
- [emitter.md](../emitter.md) §"Phase SSI-2 — staged emit" — the
  per-emitter dialect divergence table.
- `tests/opensees/unit/test_stages.py` — `_StageBuilder` lifecycle,
  `StageRecord` shape, per-stage analysis-chain re-emit.
- `tests/opensees/unit/test_ssi_post_merge_cleanup.py` — the
  H1/H2/M4 validator coverage.
- `tests/opensees/subprocess/test_stages_subprocess.py` — Tcl + Py
  subprocess smoke for end-to-end multi-stage deck execution.
