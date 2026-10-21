# ADR 0057 — Solution-strategy ladder: emitted per-increment escalation with established profiles

**Status:** Accepted (2026-06-10; evidence base = the Cerro Lindo zoned-twin
campaign of the same date — mesh-sensitivity, element-comparison and
integrator-comparison runs, all archived in the model folder's
`HANDOFF_tunnel_twin.md`)

**Accepted 2026-10-04** (T week 1 status batch, #1202).

## Context

### What exists today

A stage (or a flat run) declares exactly **one** solution chain:
`s.analysis(test=..., algorithm=..., integrator=..., ...)` followed by
`s.run(n_increments=N, dt=...)`. Since the fail-loud loops shipped
(#587), the emitted deck wraps every increment:

```python
for _apesees_i in range(40):
    if ops.analyze(1, 0.025) != 0:
        raise SystemExit("apeGmsh: analyze FAILED at increment %d/40 of stage 'Gravity' ...")
```

This is the right *floor* — no silent partial states, the failure names
its stage and increment — but it is also the whole ceiling: one stalled
increment aborts a multi-stage run, and the only recourse is re-running
the entire deck with a different chain.

### The evidence that motivates this ADR

One day of staged Mohr-Coulomb tunnel runs (zoned excavated twin,
2026-06-10) produced the following, each reproducible from the archived
run folders:

1. **The folk robustness ordering inverted.** NewtonLineSearch — the
   chain conventionally reached for *because* it is "more robust" —
   stalled mid-ramp in five independent configurations (CST lc 0.35 at
   increment 7/20 and lc 0.175 at 13/20 in stage 1 plus 48/100 in
   stage 3; LST and Bézier at lc 0.25 at increments 17/20 and 4/20),
   while **plain Newton at the identical tolerance converged every one
   of those cases**. On non-smooth multi-surface plasticity the line
   search oscillates across the yield-surface kink instead of damping.
   Any hardcoded "fast → robust" ladder built from folklore would have
   escalated *into* the failure.
2. **Whole-run retry is the only rescue today, and it is absurdly
   expensive.** The study notebooks had to grow an escalating-retry
   loop that re-runs the *entire deck* (minutes of already-converged
   increments) to rescue one stalled increment. A per-increment ladder
   retries seconds of work instead.
3. **Equivalent chains agree within a measurable noise band.** Plain
   Newton vs NewtonLineSearch, where both converge, differ ≤1% at
   stage 1 and ~2% at stage 3 on every probed metric. Escalation does
   not change the physics; it changes the path within a band the model
   already exhibits between nominally-identical runs.
4. **Exact-λ landing is load-bearing.** Stages end with
   `loadConst`, which freezes the load at whatever λ was reached — and
   unloading a yielded model to correct an overshoot corrupts the
   plastic state. The integrator-comparison runner already prototyped
   the landing contract (switch to a `LoadControl` tail step once the
   remaining λ is within one max-step of the target).
5. **Step-size adaptivity composes with, but does not replace,
   algorithm escalation.** `LadrunoArcLength` (Ramm, jd=8) covered the
   gravity stage in 14 steps vs LoadControl's 40 — continuous *in-
   integrator* step laddering. It does nothing for an algorithm that
   fails to converge, and it has no λ to adapt in stress-injection
   stages.
6. **Prior art.** STKO's emitted decks run a per-increment custom-
   function loop (`STKO_VAR_analyze_done` machinery) designed exactly
   so adaptation hooks can intervene per increment. Upstream OpenSees
   has no built-in ladder; every production user hand-rolls one.

### Constraints inherited from the architecture

- **The deck is authoritative.** Whatever strategy exists must be
  *emitted* into the py/tcl deck, not implemented as live-run-only
  orchestration (the reversal that killed the unconsumed-loads warning,
  ADR 0051 §7, applies: behavior the deck cannot reproduce does not
  exist).
- **Fail-loud is non-negotiable** (#587). Escalation must be loud,
  auditable, and bounded — exhausting the ladder aborts with the same
  named banner, never limps.

## Decision

### 1. A typed `ops.strategy` namespace, opt-in at `run`

```python
ladder = ops.strategy.Ladder(
    rungs=[
        ops.algorithm.Newton(),
        ops.strategy.Substep(max_halvings=4, regrow=2.0),
        ops.algorithm.ModifiedNewton(tangent="initial"),
        ops.algorithm.KrylovNewton(),
    ],
    name="my_ladder",
)
s.analysis(**chain)
s.run(n_increments=40, dt=0.025, strategy=ladder)   # staged
ops.run(..., strategy=ladder)                        # flat runs too
```

Default (`strategy=None`) is today's behavior, unchanged. A `Ladder` is
a registered primitive like any other (named, H5-serializable in
phase C, folded into `model_hash` when present).

### 2. Rung semantics — per-increment, reset-on-success

- An increment is attempted with **rung 0** (the `s.analysis` chain's
  own algorithm is rung 0 implicitly; listing it is optional sugar).
- On `analyze(1) != 0`, the loop walks to the next rung: an
  **algorithm rung** re-issues `ops.algorithm(...)` and retries the
  same increment; a **`Substep` rung** halves the current dλ (load
  stages only — see §4) and retries.
- A converged increment **resets to rung 0** — the fast chain gets
  first shot every increment, so the ladder costs nothing on the happy
  path.
- `Substep` regrows the step by `regrow` after each `regrow_after`
  (default 2) consecutive successes, never above the nominal dλ, and
  always lands λ **exactly** on the stage target via a final
  `LoadControl(min(remaining, dλ))` step (the loadConst contract,
  evidence point 4).
- Exhausting all rungs (and all halvings) aborts with the existing
  fail-loud banner **plus the rung trace** for the fatal increment.

### 3. Established profiles — named, documented, editable

`ops.strategy.profile(name)` returns a pre-built `Ladder`. Profile
names are a stable contract; each profile documents *why* its ordering
is what it is. The returned object is a plain `Ladder` — callers may
append/remove rungs before use.

| profile | rungs (after implicit rung 0 = declared algorithm) | rationale |
|---|---|---|
| `"standard"` | `Substep(4)` → `ModifiedNewton(initial)` → `NewtonLineSearch(Bisection)` | general default: subdivision rescues more plasticity failures than algorithm swaps; line search last |
| `"non-smooth"` (aliases `"geotech"`, `"mohr-coulomb"`) | `Substep(4)` → `ModifiedNewton(initial)` → `KrylovNewton` — **no line-search rung** | the 2026-06-10 evidence: line search oscillates on MC/DP yield-surface kinks and *is the failure mode*; escalating into it is harmful |
| `"smooth-hardening"` (alias `"metal"`) | `NewtonLineSearch(Bisection)` → `Substep(4)` → `KrylovNewton` | smooth J2-type response is where the line search genuinely helps first |
| `"penalty-stiff"` | `ModifiedNewton(initial)` → `Substep(6)` → `KrylovNewton` | embed/contact penalties (K≈1e8) poison the current tangent; the initial tangent + deep subdivision is the classic remedy |
| `"exhaustive"` | `Substep(6)` → `ModifiedNewton(initial)` → `NewtonLineSearch(Bisection)` → `KrylovNewton` → `BFGS` | last resort; slowest; for "get me *a* converged state to debug from" |

Profiles are **never auto-selected** — material-driven hints (e.g. a
note suggesting `"non-smooth"` when MohrCoulombSoil is declared) are an
open question, not part of this decision.

### 4. Stage-type awareness

- **Load stages** (a pattern with a load factor — e.g. gravity): full
  ladder, including `Substep` with the exact-λ landing.
- **Stress-injection stages** (`initial_stress` ramps — `updateParameter`
  per increment, no λ): **algorithm rungs only**. A `Substep` rung is
  skipped with a loud one-line note (subdividing the *injection
  increment* itself touches the initial-stress emit machinery and is
  deferred to phase D).

### 5. Loud, auditable provenance

- Every escalation prints one line:
  `apeGmsh strategy: increment 17/40 of stage 'Gravity' -> rung 2 (ModifiedNewton initial)`.
- The emitted loop accumulates per-increment provenance
  `(increment, rung_carried, iterations, halvings)`; the live runner
  surfaces it after the run, and phase C persists it next to the
  results so a converged state is auditable as "increment 17 was
  carried by ModifiedNewton after 2 halvings".
- The strategy *declaration* (rungs, profile name) round-trips H5 with
  a schema minor bump and folds into `model_hash` — two decks with
  different ladders are different models for caching purposes.

### 6. Hard exclusions

- **No tolerance-relaxation rung, ever.** Loosening the test changes
  the physics, not the path. Same for swapping the test type.
- **No integrator identity changes** mid-stage beyond the `Substep`
  dλ scaling and the λ-landing `LoadControl` tail (which the
  integrator-comparison work already validated as state-safe).
- **No silent success.** The provenance line is not suppressible below
  warning level.

## Emission

Both the py and tcl emitters enrich the existing #587 loop. Sketch (py):

```python
_rungs = [...]                       # emitted from the Ladder primitive
_dlam, _halv, _streak = 0.025, 0, 0
_t, _prov = 0.0, []
while _t < _TARGET - 1e-9:
    for _r, _rung in enumerate(_rungs):
        _apply(_rung)                # ops.algorithm(...) or dlam halving
        if ops.analyze(1) == 0:
            break
        print("apeGmsh strategy: increment ... -> rung %d (%s)" % ...)
    else:
        raise SystemExit("apeGmsh: analyze FAILED ... after exhausting "
                         "strategy ladder %r (trace: %s)" % (_name, _prov[-1:]))
    _t = ops.getTime()               # load stages; increment counter otherwise
    _prov.append((_inc, _r, ops.testIter(), _halv))
    _reset_to_rung0()
```

The instrumented λ-loop shipped in the integrator-comparison runner
(`twin_zoned_runner.py`, 2026-06-10) is the working prototype of the
load-stage variant, including the landing step and `testIter()`
harvesting.

## Consequences

**Positive**

- One stalled increment no longer costs the whole run; the rescue costs
  seconds, in-deck, identically in py and tcl.
- Profiles turn a folklore-driven choice into a documented, evidence-
  revisable contract (`"non-smooth"` exists *because* the folk ordering
  measurably inverted).
- The notebook-side whole-run retry machinery becomes obsolete for new
  decks.
- Fail-loud gets *stronger*: the exhaustion banner carries a rung trace
  instead of a bare increment number.

**Negative / accepted**

- Path-dependence: a laddered run and a fixed-chain run differ within
  the ±1–2% band the model already shows between equivalent chains.
  Accepted and documented; provenance makes it auditable.
- Emitted decks grow a loop preamble (~20 lines/stage with a ladder;
  zero when `strategy=None`).
- One more registered-primitive family to round-trip (phase C schema
  bump).

## Phasing

- **A** — `ops.strategy.Ladder` + `profile()` + algorithm-rung walking
  emitted into flat + staged decks (py + tcl), provenance print + live
  harvest, fail-loud exhaustion banner. No `Substep` yet.
- **B** — `Substep` rung for load stages with exact-λ landing + regrow.
- **C** — H5 round-trip of the declaration + provenance persistence +
  `model_hash` fold (schema minor bump).
- **D** (demand-gated) — injection-ramp subdivision for stress-control
  stages.

## Alternatives considered

- **Live-run-only orchestrator** (retry logic in `ops.run`, nothing in
  the deck) — rejected: violates deck-is-authoritative; a tcl/STKO-side
  run of the same deck would behave differently.
- **Whole-run retry** (status quo, notebook-side) — wasteful (re-runs
  converged increments), not portable, lives outside the model.
- **Adaptive integrators only** (`LadrunoArcLength` Ramm) —
  complementary, not sufficient: no help for algorithm failure, no λ in
  injection stages.
- **Tolerance relaxation as a last rung** — rejected outright (changes
  physics; the silent-no-op history of this project is the cautionary
  tale).

## Open questions

1. Material-driven profile *hints* (a build-time note when MC/DP
   materials are present and no strategy was declared) — useful or
   noise?
2. Transient analyses: out of scope for v1 (static only); the loop
   shape generalizes but dt-subdivision semantics differ.
3. Should `s.analysis` accept `strategy=` directly (chain-adjacent)
   instead of / in addition to `s.run`? Leaning `s.run` (it owns the
   loop).

## Amendment 1 (2026-10-21) — the transient `Subdivide` rung (#2403)

**Status:** Proposed. The maintainer ruled the direction on 2026-10-21
(option (a) in #2403, relayed by T), and ratifies this text before any
code.

**Context.** Open question 2 left transient analyses out of v1. A staged
transient deck cannot cut its time step on a failed increment: the ladder
carries algorithm rungs, and `Substep` is a load-stage dλ policy that
decks refuse (ADR 0104 D6). A 1-D site-response column, a 12 000-step
shaking stage of multi-yield soil, aborts at step 1793 when every
algorithm rung fails. The same model converges when the failed step is
retried in pieces, as its original openseespy driver did
(`analyze(4, dt/4)`, then `analyze(16, dt/16)`). With that retry the
bridge model reproduces the original amplification to 1e-5 (measured on
the YOO Guayaquil column, base760_nsmesh, 2475 y).

**Decision.**

1. **A new rung, `ops.strategy.Subdivide(n)`** (an integer `n >= 2`).
   It is one entry in `Ladder(rungs=...)`, placed anywhere among the
   algorithm rungs, and a ladder may hold several (`Subdivide(4)`,
   `Subdivide(16)`).
2. **Semantics: cover the rest of the failed increment in `n` equal
   pieces.** When the loop reaches a `Subdivide(n)` rung, it re-issues
   rung 0's algorithm and runs `analyze(1, h)` up to `n` times, where
   `h = (t_target - getTime()) / n` and `t_target` is the time at which
   the increment ends. A converged piece is committed. A failed piece
   hands the *remaining* time to the next rung, which recomputes its own
   `h` the same way. So the increment always ends exactly at
   `t_target`: model time never drifts off the step grid, a recorder at
   the stage's `dt` stays aligned, and a stage of `N` increments ends at
   `N·dt`. A converged increment resets to rung 0, as in §2.
3. **Scope: transient stages only.** A `Subdivide` rung in a static
   (load or stress-injection) stage is refused when the stage closes;
   `Substep` stays the static policy. A `Subdivide` rung in a ladder
   attached to a flat live run (`ops.analyze(strategy=)`) is refused
   until the live emitter learns the loop. `emitter/live.py` is
   hub-locked, and a live runner that silently ignored the rung would
   run a fixed-step solve.
4. **Exhaustion stays fail-loud.** When every rung, `Subdivide` rungs
   included, has failed, the stage aborts with the existing banner. No
   rung skips a step or counts a failure and moves on. The original
   driver did that, and it is the silent success §6 excludes.
5. **Provenance (§5).** Each escalation to a `Subdivide` rung prints
   the same one-line trace, with `n` and the number of pieces it
   committed.
6. **§6 is amended, not loosened.** Cutting the transient step changes
   only the `dt` argument of `analyze`. The integrator's identity and
   parameters, the test and its tolerance are untouched. Newmark-family
   integrators commit a valid state at every converged piece, so a
   subdivided increment is the same physics, integrated more finely.
   No tolerance rung and no test swap are added.

**Emission.** The py and tcl emitters gain the piece loop inside the
existing #587 per-increment loop. `StrategySpec` encodes a `Subdivide`
rung as a reserved non-algorithm tuple (`("__apegmsh_subdivide__", n)`):
an emitter that does not know the tuple passes it to `ops.algorithm`,
which fails loudly instead of silently running fixed-step.

**Not changed.** No schema change: ladders are not archived (the H5
stage writer drops `strategy`, `_h5_stages.py`). `model_hash` is
unaffected. The `Substep` rung, the profiles and their orderings stay
as they are; no profile gains a `Subdivide` rung.

**Consequences.** A soil-dynamics deck declares, for example,
`Ladder(rungs=(Newton(), Subdivide(4), Subdivide(16)))` on its shaking
stage. A model that needs smaller steps throughout should still declare
a smaller `dt`: the rung is a per-increment rescue, not a step-size
policy.
