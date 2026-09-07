# ADR 0104 — The `Substep` rung: an adaptive step controller that runs to criterion

**Status:** Accepted (2026-09-07). Adds `ops.strategy.Substep` and widens
`Ladder` to carry one, in
[`analysis/strategy.py`](../../analysis/strategy.py) and
[`_internal/ns/strategy.py`](../../_internal/ns/strategy.py). No schema
bump: a strategy declaration is not a registered primitive and does not
round-trip H5 yet (ADR 0057 Phase C).

**Completes:** [ADR 0057](0057-solution-strategy-ladder.md) **Phase B**
(line 232 — "`Substep` rung for load stages with exact-λ landing +
regrow"). Phase A shipped algorithm escalation only; the module docstring
said in as many words that `Substep` rungs were Phase B.

**Ported from:** the TIMs response-curve harness,
`Tries/response-curve-matrix/model/harness/stages.py` on branch
`work/ape/response-curve-matrix` of the *TIMs Workbench* repo —
`stage_S5_push` (line 3276, the step controller and its verdict) and
`_try_step` (line 2520, the per-step ladder walk), with the verdict
helpers `_tangent` / `_initial_tangent` / `_rolling_k` / `_verdict`
(lines 3885–4027). Read-only; nothing in that repo was touched.

## Context

### What Phase A left on the table

A `Ladder` escalates the **solution algorithm** and nothing else. The
emitted loop is `for i in range(n_increments)`: the step size is fixed
at declaration time, and the only response to a failed increment is to
re-issue a different `algorithm` command against the *same* increment.

That is the wrong lever for the failure mode this project keeps meeting.
On a push into plasticity the Newton radius of convergence collapses at
the knee, and no algorithm converges at the declared step — the fix is a
smaller step, then a bigger one again once the model is through. ADR 0057
saw this (§2, lines 111–120: a `Substep` rung "halves the current dλ …
and retries", regrows "after each `regrow_after` … consecutive
successes, never above the nominal dλ") and deferred it.

### What the harness already had, in production

The TIMs harness had been running the missing controller for months
against SANISAND strip-footing pushes. Its `stage_S5_push` loop is not
a fixed-increment loop at all: it is a **run-to-criterion** loop with a
step-size policy inside it. Everything below is in that function, and
every piece earns its place from a measured failure recorded in its own
comments:

- **a base step that is cell-, then row-, then default-scoped** (line
  3452), because "measured on cell U-V, the coupled Newton's radius of
  convergence at the reference state lies between 3.125e-6 and 6.25e-6 m
  of control-node settlement, an eighth of the declared 2.5e-5, and above
  it every rung of the ladder returns rc = -3";
- **halve on failure, regrow after `grow_after` good steps, capped at
  `ds_max`, floored at `ds_min`** (lines 3568–3585);
- **a subdivision budget** whose exhaustion ends the leg (line 3576);
- **a wall-clock budget** checked at the top of every attempt (line 3549);
- **run-to-criterion**: a target settlement (line 3547) and a
  tail-tangent plateau test evaluated against the initial tangent
  (`_verdict`, lines 3944–4027);
- **exact landing**: `ds_use = min(ds, smax - s_now)` (line 3552), the
  same "never overshoot" contract ADR 0057 evidence point 4 wrote down
  for `loadConst`.

Porting it is cheaper and far better evidenced than designing it again.

### Why this is not simply "emit a longer loop"

ADR 0057's architectural constraint is **the deck is authoritative**.
Phase A honours it: the rung walk is emitted into py and tcl, and the
live path mirrors it. The substep loop is a different shape — it is not
`for i in range(N)`, it is `while not criterion`, and its plateau test is
two least-squares fits over an accumulating curve. Emitting that into
Tcl is a real piece of work and is **not** in this slice; see D6, which
is the honest cost of shipping the controller now.

## Decision

### D1 — `Substep` declares a policy; a `SubstepDriver` executes it

```python
push = ops.strategy.Substep(
    node=CTRL, dof=3, target=0.15,      # run to 0.15 of advance at CTRL dof 3
    ds=2.5e-5, ds_min=1e-8, ds_max=1e-4,
    regrow=2.0, regrow_after=2,         # ADR 0057 §2's names
    budget=8, wall_budget=6 * 3600.0,
    plateau=0.05, plateau_window=0.01,  # ...or until the curve flattens
)
result = push.drive(driver)
```

The controller owns the **step-size policy and nothing else**. The
driver is a three-method Protocol:

| method | contract |
|---|---|
| `analyze(ds) -> int` | attempt ONE increment of size `ds`; 0 == converged, mirroring `ops.analyze` |
| `disp(node, dof) -> float` | `ops.nodeDisp(node, dof)` |
| `load() -> float` | the load conjugate to the control displacement — under `DisplacementControl` against a unit reference load this is `ops.getTime()` and *is* the push force |

Everything that varies between pushes stays behind that seam: which
integrator carries the increment (`DisplacementControl` on the control
node vs the fork's `sp` platen under `LoadControl(-ds)` — the harness
supports both), and whether a failed `analyze` first walks a Phase A
algorithm ladder before reporting failure. The controller does **not**
re-implement algorithm escalation: `Ladder` already owns that, and
composing beats absorbing.

`target` and every step size are **magnitudes of advance** measured from
the displacement read at entry, not absolute nodal values. A push that
starts from a settled gravity state therefore measures its own advance —
the harness's `s_now = -(nodeDisp(CTRL, 3) - u0)` idiom (line 3546),
generalised and sign-free.

### D2 — Termination is checked at the top, success criteria first

```
target reached?      -> TARGET   (success)
plateau reached?     -> PLATEAU  (success)
wall budget spent?   -> WALL     (failure)
attempt one step of ds_use = min(ds, target - u)
  failed  -> halve; budget spent? -> BUDGET (failure); below floor? -> FLOOR (failure)
  ok      -> commit the row; regrow after `regrow_after` good steps, capped at ds_max
```

Order is load-bearing. A run that reaches its target or plateaus on the
last permitted second is a **success**: the guard cannot pre-empt a
criterion that was already met. And because both success criteria are
tested *before* every attempt, a spent budget always means the run had
not yet got what it came for — which is what makes D4 safe.

`ds_use = min(ds, target - u)` lands **exactly** on the target and never
overshoots (ADR 0057 §2 lines 118–120; unloading a yielded model to
correct an overshoot corrupts the plastic state).

### D3 — The plateau criterion is the harness's, in pure Python

`plateau` is the fraction of the **initial tangent** the tail slope must
fall below; `plateau_window` is the width, in control-displacement
units, of the tail the slope is fitted over. Both tangents are
least-squares fits — the initial one over the first `max(4, n // 50)`
committed rows, the tail one over every row within `plateau_window` of
the last. Three guards, all ported:

- **at least 8 committed rows** before the criterion may fire (the
  harness's `_rolling_k` guard, line 3934) — two 3-point fits off fewer
  rows are noise, not a trend;
- **the window must be covered** by the curve's own range (`covered`,
  line 3967) — a run shorter than the window cannot show a sustained
  tangent either way, and calling it a plateau would be reading a
  verdict off a range the criterion was never evaluated on;
- **a NaN or non-positive initial tangent disables the criterion**
  rather than dividing by it.

Softening needs no separate case: a negative tail slope is below any
positive fraction of a positive initial tangent, so it reads as a
plateau, which is the harness's `limit_point_softening` by another name.

The fit is ~8 lines of pure Python, not `numpy.polyfit` as the harness
has it. That is deliberate: it keeps a solver primitive dependency-free
and it is trivially portable into an emitted deck when D6 lands.

### D4 — A spent budget is a FAILURE verdict, always

`SubstepResult.ok` is `verdict in {"target", "plateau"}` and is derived
from nothing else. `BUDGET`, `FLOOR` and `WALL` are failures even when
the run committed hundreds of good steps first.

This **diverges from the harness on purpose.** `_verdict` (lines
3984–4002) promotes a guard-terminated leg to `limit_point_then_guard`
when the tangent criterion had already been met over a full window,
reasoning that a perfectly plastic collapse ends on the subdivision
budget by construction. That reasoning is sound *for a report* and wrong
*for a controller*: it lets a spent budget return success. Here the same
case is handled by ordering instead (D2) — a run that met the criterion
exits `PLATEAU` before it can spend anything — so nothing is lost and
the #587 fail-loud floor holds unbroken.

### D5 — `budget` is per-increment depth, not a run-wide total

The harness counts every halving against one budget (`nsub`, line 3571).
Ported verbatim, that was **measured to be spent by probing rather than
by failure** — see F1 below. Regrow doubles the step every
`regrow_after` good steps, so past a knee it re-fails *by construction*;
a run-wide total dies after ~2 × `budget` perfectly healthy steps.

`budget` is therefore the number of **consecutive** halvings allowed to
rescue ONE increment — which is exactly ADR 0057's own parameterisation
(`Substep(max_halvings=4, …)`, line 90). `SubstepResult.subdivisions`
still reports the run total, so the probing cost stays visible.

### D6 — Deck emission is deferred, and `to_spec` refuses rather than drops

A `Ladder` may carry **one** `Substep`; it is reachable as
`Ladder.substep` and is skipped when the algorithm rungs are resolved.
But `Ladder.to_spec()` **raises** when a `Substep` is present rather than
returning the Phase A spec:

```
Ladder 'push' carries a Substep rung, which is not emitted into decks yet
(ADR 0104 D6). Emitting the algorithm rungs alone would silently drop the
declared step-size policy and run a fixed-step deck. Drive it in-process
with Substep.drive(driver) instead.
```

This is the one place the ADR 0057 deck-is-authoritative constraint is
not yet met, and it is named rather than papered over. The alternative —
emit the algorithm rungs and quietly ignore the step policy — is the
silent-no-op class this project has a documented history of (ADR 0051 §7,
the unconsumed-loads warning). A loud refusal is worse ergonomics and
better engineering. Emitting the loop into py + tcl is the next slice.

### D7 — Hard exclusions carried forward

- **No tolerance relaxation.** `_try_step` (line 2530) re-issues
  `ops.test(tname, tol * rung["tol_factor"], …)` on every rung — its
  ladder relaxes the *convergence test* as it escalates. ADR 0057 §6
  excludes that outright ("Loosening the test changes the physics, not
  the path"), so only the rc-checking skeleton of `_try_step` was taken
  and the `tol_factor` machinery was left behind.
- **No dynamic-relaxation rung.** `_push_dr` (line 2986) is the
  harness's other push driver and is **not ported**, on two independent
  grounds. Source-verified: it wipes the whole analysis chain to
  `numberer Plain` / `system Diagonal` / `algorithm Linear` /
  `integrator LadrunoDynamicRelaxation` / `analysis Transient` (lines
  3047–3057) — a wholesale integrator *identity* change, which ADR 0057
  §6 excludes ("No integrator identity changes mid-stage beyond the
  `Substep` dλ scaling"). Carried from the program brief, not
  re-measured here: it was defective for **state-memory materials** —
  DR drives a fictitious damped transient to rest, and every pseudo-step
  of that transient is a strain increment a path-dependent material
  commits, so the internal state at rest is not the state the static
  path would have produced.

## Consequences

**Positive**

- The measured failure mode — a Newton radius an eighth of the declared
  increment — now has the lever that fixes it, in typed apeGmsh code
  instead of a per-model notebook loop.
- Run-to-criterion means a push declares *what it wants* (a settlement,
  or a plateau) rather than a step count guessed in advance.
- Every termination has a named verdict and a sentence; nothing about a
  spent budget can read as success.
- The controller is testable without a backend (32 tests, 0.3 s), which
  is why the two runaway defects in F2/F3 were found at all.

**Negative / accepted**

- **`Substep` cannot yet reach a deck** (D6). Until the emission slice
  lands, a substep push is an in-process run, and `to_spec` refuses.
  This is a real gap against ADR 0057's deck-is-authoritative
  constraint and is the top of the follow-up list.
- The driver seam is one more thing to write per push (about ten lines
  wrapping `ops.integrator` + `ops.analyze` + `ops.nodeDisp`). A
  bridge-supplied default driver is deliberately *not* shipped here —
  the integrator idiom differs per push and guessing it is how silent
  wrong answers start.
- `regrow` probing costs one failed `analyze` per `regrow_after` good
  steps once the model is past a knee. Priced and accepted (F1): the
  alternative is a step that never recovers from one hard increment.

## Adversarial probe

A throwaway script executed the controller on degenerate inputs (not
shipped; every finding below became a test in
`tests/opensees/unit/test_strategy_substep.py`).

| # | probe | found | disposition |
|---|---|---|---|
| **F1** | stiff-then-soft driver, budget as a run-wide halving total | **DEFECT.** Regrow re-fails by construction past the knee; the run died `BUDGET` after ~2 × `budget` healthy steps, with a converging step size in hand. Verbatim from the run: `subdivision 4/8 at u = 1.25` … `8/8 at u = 2.25` — eight halvings, seven of them undoing a regrow the controller had just chosen. | **Fixed** (D5): `budget` counts *consecutive* halvings; run total still reported. Pinned by `test_the_regrow_probe_does_not_spend_the_budget`. |
| **F2** | `disp()` returns NaN (a blown-up solve) | **DEFECT — runaway.** `u = nan` fails *every* comparison in the loop: the target test, the floor test and the step-1 tripwire alike. `min(ds, target - nan)` returns `ds`, so the controller stepped forever. The probe's own call cap fired: `RuntimeError: RUNAWAY: 5001 analyze calls`. | **Fixed:** a non-finite control reading raises loud. Pinned by `test_nan_displacement_is_refused_not_spun_on`. |
| **F3** | `load()` returns NaN with a plateau declared | **DEFECT — silent no-op.** The initial tangent fits to NaN, the criterion is disabled by its own guard, and the run continued to the target having quietly ignored the declared plateau. | **Fixed:** a non-finite load raises *when a plateau is declared*, and only then — with no plateau nothing reads the load. Pinned by `test_nan_load_under_a_declared_plateau_is_refused`. |
| **F4** | `budget=0`, driver never converges | Correct: one attempt, `BUDGET`, `ok=False`. Zero budget means no halving is permitted, not "unlimited". | Non-finding; pinned by `test_zero_budget_refuses_the_first_subdivision`. |
| **F5** | floor above the base step | Refused at construction: "the floor would refuse the very first halving, so the controller could never subdivide". | Non-finding; pinned. |
| **F6** | target already met | `TARGET`, `steps=0`, **zero** `analyze` calls. Note the datum is relative, so a model arriving already displaced is *not* already met — it has its full `target` of advance to make. | Non-finding; both halves pinned. |
| **F7** | `wall_budget=0` (and negative) | Refused at construction — a zero budget is a run that cannot start, which is a mistake and not a policy. | Non-finding; pinned. |
| **F8** | driver that never converges, default budget | 9 `analyze` calls (budget + 1), `BUDGET`, `steps=0`. | Non-finding; pinned. |
| **F9** | converging driver that moves the DOF **backwards** after the first step (snap-back) | Terminates: the step-1 tripwire passes, later reversals are legitimate physics, and the run still reached the target. With a wall budget declared it ends `WALL` at the right displacement. | Non-finding — deliberately *not* guarded past step 1, since snap-back is real. |
| **F10** | converged step that never moves the control DOF | Refused at step 1: "The control DOF the driver steps and the one this Substep measures are not the same DOF." This is the harness's step-1 fidelity assertion (lines 3654–3687), generalised. | Non-finding; the guard was already there and the probe confirmed it fires. |
| **F11** | `ds_min == ds` | Accepted, and the first failure lands on `FLOOR` immediately. Honest — it is "no subdivision at all", reported as a floor rather than a budget. | Non-finding. |
| **F12** | plateau on a curve with zero initial tangent | Criterion silently disabled, run continues to the target. Kept: "5 % of zero" is not a threshold, and the harness does the same (`if k0 and …`). Distinct from F3, where the tangent is NaN because the *reading* is broken. | Non-finding, documented. |
| **F13** | `plateau_window` wider than the whole run | Criterion never fires; the run reaches its target instead. This is the `covered` guard doing its job. | Non-finding; pinned. |
| **F14** | huge `regrow` against a cap | `ds` jumps to `ds_max` and stops there; never exceeded. | Non-finding; pinned. |
| **F15** | exact landing | Final displacement is the target to the last bit (`1.0`, not `1.0000000000000002`). | Non-finding; pinned. |
| **F16** | `regrow <= 1`, `regrow_after < 1`, `plateau = 0`, `dof = 0` | All four refused at construction with a sentence saying which knob and why. | Non-finding; pinned. |

## Follow-ups

1. **Emit the substep loop into py + tcl decks** and lift the D6
   refusal. This is what closes ADR 0057 Phase B against the
   deck-is-authoritative constraint.
2. **H5 round-trip** of a `Substep` declaration, with the rest of the
   ADR 0057 Phase C schema bump.
3. **Stress-injection stages** (ADR 0057 §4, lines 145–151) still take
   algorithm rungs only; a `Substep` there remains Phase D.

## Cross-references

- [ADR 0057](0057-solution-strategy-ladder.md) — the ladder, the
  profiles, and the phasing this completes Phase B of.
- [ADR 0103](0103-sanisand-integrator-deck-contract.md) — `max_substeps`
  on the *material*: the fork's fix for an integrator that force-accepts
  a degraded substep and tells the step controller nothing. That gate
  and this controller are the two halves of the same problem, and D4's
  refusal to dress a spent budget as success is the same discipline one
  level up.
