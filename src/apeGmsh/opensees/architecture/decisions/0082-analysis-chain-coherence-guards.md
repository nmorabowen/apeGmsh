# ADR 0082 — Analysis-chain coherence guards: the bridge refuses a chain OpenSees would silently rewrite

**Status:** Proposed (2026-07-25) — **amended same-day by adversarial
review.** G1's original set-membership rule was **refuted by probe**
(it false-positives on the legitimate two-chain non-staged model) and
is replaced by the slot-lookup rule below; the ADR-40c speedup
citation was an overclaim and is corrected; G2's equivalence
assumption is now stated rather than asserted. See *Review findings*.

Two independent findings, both source-verified against the fork, turn
out to be the same defect wearing different clothes: **the OpenSees
analysis chain accepts a configuration it then does not honour, and
says so only in a suppressible warning — or not at all.** The bridge
already owns one guard of exactly this shape
(`_check_explicit_solver_compat`, `apesees.py:453`); this ADR adds the
two that the push-to-failure work needs.

Scope is deliberately small: **two guards on the bridge, no new
knobs, no fork change.** The fork-side hardening is named as a
follow-up, not bundled.

## Context

### D1 — `analysis Transient` silently discards a static integrator

`LadrunoArcLength` is a `StaticIntegrator` (fork
`SRC/analysis/integrator/LadrunoArcLength.h:91`), so
`OpenSeesCommands::setIntegrator` files it under `theStaticIntegrator`
(`SRC/interpreter/OpenSeesCommands.cpp:575-579`). `analysis Transient`
then tests `theTransientIntegrator == 0`, finds it null, and installs
`Newmark(0.5, 0.25)` in its place
(`OpenSeesCommands.cpp:908-914`). The arc-length object stays alive,
fully configured, and is never called.

The mirror case is identical: `analysis Static` with only a transient
integrator registered installs `LoadControl(1,1,1,1)`
(`OpenSeesCommands.cpp:657-663`).

Both diagnostics sit behind `if (!suppress)`. With warnings suppressed
— routine on a long HPC run — the substitution is *completely* silent.
The failure mode is not a crash: it is a load-controlled Newmark run
that produces a plausible load-displacement curve right up to the
limit point it structurally cannot pass. For push-to-failure work this
is a wrong-answer path, and it is the reason arc-length is currently
blocked.

apeGmsh does not catch it either. `Static` / `Transient` /
`VariableTransient` ([`analysis.py:48-85`](../../analysis/analysis.py))
are pure emitters, and no bridge check pairs the registered integrator
against the registered analysis.

### D2 — `Newton -initial` pays for a matrix that cannot have changed

`NewtonRaphson::solveCurrentStep` places `formTangent` **inside** the
iteration loop for every tangent flavour, `INITIAL_TANGENT` included
(fork `SRC/analysis/algorithm/equiSolnAlgo/NewtonRaphson.cpp:163-191`).
`IncrementalIntegrator::formTangent` opens with `theSOE->zeroA()`
(`IncrementalIntegrator.cpp:98`), which resets the SOE `factored`
flag — so the fork's UMFPACK numeric-persist (fork ADR-40c) can never
fire under `Newton -initial`. Every iteration re-assembles and re-
factorizes a matrix bit-identical to the one it factored moments
earlier.

The capability to avoid this **already exists and needs no code**:
`ModifiedNewton -initial` *is* initial-stiffness Newton. Both solve
`K₀·δu = −R` each iteration; ModifiedNewton merely forms `K₀` once per
step, so the iterates coincide (the only difference is the
algorithm-recorder iteration index — `NewtonRaphson.cpp:216`
increments before `record`, `ModifiedNewton.cpp:194` after).

**The equivalence has a premise, and it is worth stating.** It holds
only while `getInitialStiff()` is state-independent *between
iterations of a step*. That is the universal convention but is not
enforced by the `Element` API. Spot-checked: `CorotTruss::getInitialStiff`
(`SRC/element/truss/CorotTruss.cpp:661`) builds its rotation `R` in
`setDomain` from the **undeformed** geometry and never touches it in
`update()`, so even a corotational element yields a fixed `K₀`. Where
an element *did* violate this, `Newton -initial` would not be
factoring a constant matrix either — the premise of the whole finding
would fail, not just the swap. This is the reason G2 is a warning that
names a replacement and **never** a rewrite: the user, not the bridge,
owns that judgement.

**Magnitude — not yet measured for this swap.** Fork ADR-40c's
−57.8 % (1.247 s vs 2.953 s, 10³ LadrunoBrick + LadrunoJ2) is
*indicative only*: its gate script runs `ops.algorithm("ModifiedNewton")`
with the **default current tangent**
(`tests/test_umfpack_numeric_persist.py:166`), and the two columns are
numeric-persist on vs off — not `Newton -initial` vs
`ModifiedNewton -initial`. The real swap differs on both axes: it also
elides the per-iteration *assembly*, and initial-stiffness iteration
counts are much higher than current-tangent ones. Direction is certain
(strictly less work per step, identical iterates); the number is not.
An A/B on the arc-length lane is a **gate for G2**, not a citation.

This matters most in exactly the lane D1 blocks. `LadrunoArcLength`
overrides `formTangent` to assemble the real `K` then poke
`cOverDt·M*` onto the diagonal
(`LadrunoArcLength.cpp:227-238`), so forming once per step still
yields the correctly stabilized matrix — ModifiedNewton is valid under
arc-length. And initial-stiffness iteration near a limit point burns
many iterations per step, which is precisely where N assemblies +
N factorizations per step is the wrong bill.

apeGmsh's own house default already does the right thing
([`strategy.py:140`](../../analysis/strategy.py)) — the gap is that
nothing stops a user hand-writing the wasteful spelling.

## Decision

### G1 — an `analysis` whose integrator slot is empty is a `BridgeError`

**The rule models OpenSees' slot lookup, not set membership.**
`setIntegrator` writes to one of *two* slots — `theStaticIntegrator`
or `theTransientIntegrator` — chosen by the C++ base class
(`OpenSeesCommands.cpp:575-579`). `analysis Transient` reads the
transient slot; `analysis Static` reads the static one. The silent
substitution is exactly "the slot this `analysis` reads was never
filled".

So the guard walks the registered primitives **in order**, maintaining
the same two slots, and raises at any `Analysis` whose slot is still
empty. Hard error, not a warning: the consequence is silently wrong
results, and registering a specific integrator is unambiguous intent.

This fires precisely where upstream's silent default fires, and
nowhere else. In particular it **passes** the ordinary non-staged
gravity-then-dynamics model — `LoadControl → Static → Newmark →
Transient` — which the naive "a static integrator coexists with a
Transient analysis" rule would have wrongly rejected (see *Review
findings* F1). Registration order is preserved into the deck
(probe-verified), so the slot state the guard computes is the slot
state OpenSees will see.

Slot assignment is **by the fork's C++ base class**, source-verified —
not by whether the method "feels" static:

| Primitive | C++ base | Class |
|---|---|---|
| `LoadControl` | `StaticIntegrator` | static |
| `DisplacementControl` | `StaticIntegrator` | static |
| `ArcLength` | `StaticIntegrator` | static |
| `LadrunoArcLength` | `StaticIntegrator` | static |
| `LadrunoIndirectControl` | `StaticIntegrator` | static |
| `Newmark`, `HHT` | `TransientIntegrator` | transient |
| `LadrunoHHT` | `HHT` | transient |
| `LadrunoGeneralizedAlpha` | `GeneralizedAlpha` | transient |
| `CentralDifference`, `ExplicitDifference` | `TransientIntegrator` | transient |
| `ExplicitBathe` (+`-lnvd`), `CentralDifferenceLadruno` | `TransientIntegrator` | transient |
| `CentralDifferenceSMS`, `ExplicitBathe*SMS` | explicit bases | transient |
| **`LadrunoDynamicRelaxation`** | **`TransientIntegrator`** | **transient** |

`LadrunoDynamicRelaxation` is the trap and the reason the table is
source-derived: it is a *quasi-static* solver but a `TransientIntegrator`
by dispatch, so it legitimately requires `analysis Transient`. Fork
ADR-21 confirms this independently — it prescribes `analysis Transient`
plus a relax loop and **refutes** the `StaticIntegrator` route outright
(`21_ladruno_dynamic_relaxation_adr.md:145-147, 443-468`). Classifying
it "static" by intent would break a documented workflow.

`VariableTransient` reads the transient slot like `Transient`. A chain
missing an integrator or an analysis entirely stays the existing
`_check_analysis_chain_for_analyze` incomplete-chain error — G1 speaks
only to *pairing*.

### G2 — `Newton(tangent="initial")` warns and names its replacement

A new tagged `UserWarning` subclass (sibling of
`OpenSeesExplicitSolverWarning`, `apesees.py:209`) fires when a
registered `Newton` carries `tangent="initial"`, naming
`ModifiedNewton(tangent="initial")` as the identical-iterates,
one-factorization-per-step replacement.

Warning, not error, and **no silent rewrite**: the run is correct, and
this ADR does not hold that the bridge may substitute one algorithm
for another behind the user's back — that is the very behaviour D1
exists to outlaw. Tagged so the pytest filter ignores it while
interactive users see it.

`tangent="initialThenCurrent"` and `"hall"` are *not* flagged: both
genuinely need a per-iteration tangent.

## Where the guards live

Both ride the existing pre-emit check pass, next to
`_check_explicit_solver_compat` and
`_check_analysis_chain_for_analyze` (`apesees.py:8497`) — the one
place that already sees the whole registered chain.

The staged path needs the same treatment: `StageRecord` carries its own
`integrator` and `analysis` slots (`build.py:1196-1200`), so G1 runs
per stage as well as on the global chain. G2 is per-primitive and
needs no staged special-casing.

## What this deliberately does NOT do

- **No fork edit.** A hard error at `OpenSeesCommands.cpp:908` is the
  belt-and-braces version of G1 and is worth doing, but it protects
  only fork users; the bridge guard also covers decks run on vanilla
  OpenSeesPy. Named as a follow-up.
- **No `-factoronce` work.** An earlier draft proposed fixing the
  `OPS_ModifiedNewton` single-token parser (`ModifiedNewton.cpp:60`
  is an `if`, not a `while`, so `-initial -factoronce` silently drops
  the second flag) to reach whole-analysis factorization reuse.
  **Rejected for this lane:** `LadrunoArcLength` adapts `cOverDt` at
  runtime — retargeted from the viscous-work ratio
  (`LadrunoArcLength.cpp:556-560`) and decayed toward zero by the
  driver past the limit point (`:712-722`) — so `K₀ + cOverDt·M*`
  genuinely changes step to step. Factoring once would freeze a stale
  stabilization during exactly the softening branch being traced.
  Reuse is capped at once per step, which is what G2 steers to.
  (`ModifiedNewton` also never overrides `SolutionAlgorithm::domainChanged`,
  so `factorOnce` survives a `setSize` — a live vanilla hazard that
  would have to be fixed first anyway.)
- **No new user-facing knob.** Both guards are refusals and
  diagnostics. There is nothing to expose.

## Slices

1. **G1** — slot table + the ordered slot-lookup walk on the global
   chain and per stage; `BridgeError` naming the `analysis` and the
   slot it found empty.
2. **G2** — tagged warning class + the `Newton(tangent="initial")`
   check, plus the arc-length-lane A/B that the warning's wording is
   gated on.

## Validation

- G1 positive: each static integrator followed by `Transient` /
  `VariableTransient` (with no transient integrator registered) raises;
  each transient integrator followed by `Static` raises. Every matching
  pair passes.
- **G1 no-false-positive (the F1 regression):** `LoadControl → Static
  → Newmark → Transient` on one bridge **passes**. This is the case the
  refuted rule broke; it is the highest-value test in the set.
- G1 order sensitivity: `Transient` registered *before* its `Newmark`
  raises (the slot was empty at that point) — matching upstream.
- G1 classification: `LadrunoDynamicRelaxation` + `Transient` asserted
  **valid** (the misclassification regression).
- G1 staged: a stage whose own integrator/analysis disagree raises, and
  the error names the stage.
- G1 negative: an incomplete chain still produces the existing
  incomplete-chain error, not a pairing error.
- G2: `Newton(tangent="initial")` warns; `"tangent"`,
  `"initialThenCurrent"`, `"hall"` and `ModifiedNewton(tangent="initial")`
  do not.
- **G2 magnitude gate:** an interleaved median-of-≥5 A/B of
  `Newton -initial` vs `ModifiedNewton -initial` on the arc-length
  push-to-failure lane, run before the warning text quotes any number.
  Until then the warning names the mechanism, not a percentage.
- Emitted decks are unchanged for every chain that passes the guards —
  these add no lines.

## Review findings (adversarial pass, 2026-07-25)

**F1 — G1's original rule was wrong, and would have broken working
models.** The first draft raised whenever a static integrator and a
`Transient` analysis coexisted. Probe: one `apeSees` bridge accepts
`LoadControl → Static → Newmark → Transient` (`_register` appends;
it does not replace chain singletons, `apesees.py:8238`), and the
emit pass preserves registration order, so that deck is *valid
OpenSees* — each `analysis` reads the slot filled just before it.
The set-membership rule would have rejected the single most common
gravity-then-dynamics model in the library. Replaced by the
slot-lookup rule, which is both safer and a more faithful model of
upstream.

**F2 — the ADR-40c speedup was mis-cited.** The −57.8 % was presented
as "the prize" for the `Newton -initial` → `ModifiedNewton -initial`
swap. It is not: the gate script runs plain
`ops.algorithm("ModifiedNewton")` on the **current** tangent, and the
comparison is numeric-persist on vs off. Downgraded to indicative, and
an A/B on the actual lane is now a gate rather than a citation.

**F3 — G2's equivalence had an unstated premise.** "Identical
iterates" holds only if `getInitialStiff()` is state-independent
between iterations — the universal convention, not an API guarantee.
Spot-checked on `CorotTruss` (holds: `R` is `setDomain`-built from
undeformed geometry). Now stated explicitly, and reinforces
warning-never-rewrite.

**F4 — the DR classification is independently corroborated.** Fork
ADR-21 prescribes `analysis Transient` for `LadrunoDynamicRelaxation`
and refutes the `StaticIntegrator` route, confirming the one table row
a future reader would most plausibly "correct" the wrong way.

**Not found:** no defect in the D1 mechanism itself. The Tcl path
(`SRC/tcl/commands.cpp:2893, 3019, 3119`) carries the same
suppressible-warning substitution as the interpreter path, so the
finding holds for both emitters apeGmsh targets.

## Follow-ups (fork, separate)

- Hard-error the integrator substitution at
  `OpenSeesCommands.cpp:657` / `:908` / `:829` instead of a
  suppressible warning.
- `ModifiedNewton::domainChanged` override resetting `factorOnce`.
- The `OPS_ModifiedNewton` single-token parser and the `-factorOnce`
  casing that matches neither accepted spelling — only if a lane with
  a genuinely constant tangent (linear-elastic transient, fixed `dt`,
  mass/initial-stiffness Rayleigh only) turns up to justify it.

## References

- Fork ADR-40c (`40c_umfpack_numeric_persist_design.md`) — the
  `factored` reuse contract and the −57.8 % ModifiedNewton measurement.
- Fork ADR-20 (`20_ladruno_arclength_stabilized_adr.md`) — the
  stabilized arc-length whose adaptive `cOverDt` bounds reuse to one
  step.
- ADR 0057 — the solution-strategy ladder whose rungs G2's guidance
  composes with.
