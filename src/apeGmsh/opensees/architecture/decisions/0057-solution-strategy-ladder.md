# ADR 0057 — Solution-strategy ladder: emitted per-increment escalation with established profiles

**Status:** Proposed (2026-06-10; evidence base = the Cerro Lindo zoned-twin
campaign of the same date — mesh-sensitivity, element-comparison and
integrator-comparison runs, all archived in the model folder's
`HANDOFF_tunnel_twin.md`)

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
