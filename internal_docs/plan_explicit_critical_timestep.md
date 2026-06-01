# Plan — `criticalTimeStep()` query + auto-`dt` explicit-run helper

**Status:** proposed (2026-06-01) · **Owner:** nmora · **Scope:** apeGmsh-side
SLICE-2 of the Ladruno *implied 6th* feature — the runtime `dt_cr` query and a
driver helper that sub-steps an explicit run stably. Builds on SLICE-1 (PR #514,
merged): the typed explicit integrators + `system Diagonal`.

## What ships

1. **`LiveOpsEmitter.critical_time_step() -> float`** — thin wrapper over
   openseespy `ops.criticalTimeStep()` (mirrors `LiveOpsEmitter.eigen`).
2. **`apeSees.critical_time_step() -> float`** — build + emit a throwaway live
   model (like `eigen`), **prime one tiny step** to trigger `dt_cr` computation,
   query, and return the usable limit. Raises `ValueError` with a reason on a
   non-usable result (user decision: fail-loud).
3. **`apeSees.analyze_explicit(*, duration, safety=0.9, dt_max=None) -> int`** —
   duration-based driver (user decision). Builds, emits, primes, queries `dt_cr`,
   computes the stable sub-step count, runs `analyze(n, duration/n)`.

## Runtime contract (verified live vs venv build 605affeb)

`ops.criticalTimeStep()` returns:
- `0.0` — **NOT_COMPUTED** (before any `domainChanged`/`analyze`; this build does
  NOT compute it at build time → priming is required).
- `-1.0` — **NOT_APPLICABLE / DISABLED**: integrator has no `-cfl`/`-tangent`/
  `-recompute`; a non-explicit integrator (Newmark etc.); or a model whose
  **elements** produced no finite estimate — notably a **pure nodal-mass model**.
  ⚠ The `dt_cr` eigensolve uses **element** mass+stiffness (ADR D4), NOT `ops.mass`
  nodal mass. A model needs element mass *density* (`-rho` / `-mass`) for a finite
  `dt_cr`; lumped nodal mass alone yields `-1.0`.
- `> 0` — the usable limit (Noh-Bathe bound).

So the accessor must: set an explicit integrator **with `cfl=True`**, register a
`Transient` analysis, build, then `analyze(1, PRIME_DT)` once (PRIME_DT = `1e-12`,
negligible vs any real `duration`), then query. This is exactly the ADR D5 recipe.

## Pure, unit-testable helpers (no openseespy)

- `_dtcr_or_raise(dtcr: float) -> float` — returns `dtcr` if `> 0`; else raises
  `ValueError`. `0.0` → "not computed (prime a step / set cfl=True)"; `< 0` →
  "not applicable: needs an explicit integrator with cfl=True and **element** mass
  density (ops.mass nodal mass does not drive dt_cr)".
- `_explicit_substep_count(duration, dt_cr, *, safety, dt_max) -> tuple[int,float]`
  — `dt_stable = safety*dt_cr`; `dt = min(dt_stable, dt_max)` if `dt_max` else
  `dt_stable`; `n = max(1, ceil(duration/dt))`; returns `(n, duration/n)`.
  Validates `duration>0`, `0<safety<=1`, `dt_max>0` if given.

Live methods compose these; the math/sentinel logic is covered by unit tests
without a fork build.

## Files

- `src/apeGmsh/opensees/emitter/live.py` — `+ critical_time_step()`.
- `src/apeGmsh/opensees/apesees.py` — `+ _dtcr_or_raise`, `+ _explicit_substep_count`
  (module-level), `+ apeSees.critical_time_step()`, `+ apeSees.analyze_explicit()`.
  Both reject staged models (mirror `analyze`/`eigen`) and call
  `_check_analysis_chain_for_analyze()`.
- Tests: unit for the two pure helpers; `tests/opensees/live/test_critical_timestep_live.py`
  (fork-gated): truss with `-rho` → finite `dt_cr` > 0; `analyze_explicit` returns 0
  and integrates ≈ `duration`; non-usable model raises.
- Docs: `skills/apegmsh/references/ladruno.md` — flip the "deferred" note to
  "shipped" for the query + helper; `sync_skill.py --check`.

## Verification

mypy clean · unit pure-helper tests + targeted `test_analysis`/contract green ·
live: `dt_cr>0` on a `-rho` truss, `analyze_explicit(duration=…)` ret 0 with
`getTime()≈duration`, and `ValueError` on a pure nodal-mass / no-cfl model · no
regression on `tests/opensees/unit`+`contract` · `sync_skill.py --check` exit 0.

## Post-review hardening (6-agent expert panel, 2026-06-01)

A multi-agent review (opensees-expert + fem-mechanics-expert lenses + adversarial
verification against the fork C++) assessed the helpers + system choice. Applied:

- **`analyze_explicit` now returns `ExplicitRunResult(n, dt, dt_cr)`** (was bare
  `int`) and **raises `RuntimeError`** on a non-zero `analyze` (no longer swallows a
  divergence / `-cflAbort` code).
- **Unguarded-run warning** — `analyze_explicit` warns
  (`OpenSeesExplicitSolverWarning`) when the integrator has neither `cfl_abort` nor
  `recompute`, since the one-shot `dt_cr` is blind to a stiffening tangent (the #1
  risk). The fix is realized as *introspect-and-warn* (the integrator owns the flags;
  the helper does not emit/mutate it) rather than the panel's literal "drive
  `-recompute -cflAbort`" — the apeGmsh-correct form.
- **`c_mass=True` + `Diagonal`/`MPIDiagonal` → `BridgeError`** (`_check_explicit_solver_compat`,
  called from `analyze`/`analyze_explicit`/`critical_time_step`): silently-wrong
  combo (off-diagonal consistent mass dropped; the `-lumped` salvage is unreachable).
- **Explicit integrator + non-diagonal system → warning** (correct but loses O(N)).
- **`safety=0.9` kept** (NOT lowered): empirically a 3000-step ExplicitBathe run at
  `0.9·dt_cr` stays bounded; the getter returns ~10× the analytical CD limit for a
  truss yet the NB scheme is stable there. Comment locks it to the getter's value.
- **Prime kept** (panel's "drop it" suggestion refuted for the deployed build —
  `criticalTimeStep()` returns `0.0` until a step runs on build 605affeb).
- `_dtcr_or_raise` message clarified (element-loop eigensolve; nodal mass excluded;
  no claim about the direction of the omission).

## Non-goals (unchanged)

EnergyBalance text recorder (energy already in `.ladruno -G energy`). Mass scaling
(roadmap §5.1). Mid-run `dt_cr` re-sizing inside the helper (use the integrator's
`recompute=`/`cfl_abort=` guard instead).
