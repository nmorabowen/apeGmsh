# ADR 0103 — The SANISAND integrator deck contract

**Status:** Accepted (2026-09-06). Changes the `LadrunoSANISAND.tan_type`
default `0 → 2`, makes its 5-positional tail unconditional, adds
`max_substeps`, and adds three build-time gates in
[`_internal/build.py`](../../_internal/build.py). No schema bump: the
material is a primitive, and `max_substeps` is an additive field with a
behaviour-preserving default.

**Extends:** [ADR 0101](0101-sanisand-materials-and-material-stage.md),
which typed `ManzariDafalias` / `SAniSandMS` and the stage flip.
`LadrunoSANISAND` itself landed later against fork ADR-86 without an
apeGmsh ADR of its own; this one covers the integrator surface for the
whole Manzari family and retroactively carries that warrant.

**Fork-side sources:** `Ladruno_implementation/90_ladruno_regularization_tims_report.md`,
`86_ladruno_sanisand_apegmsh_emitter_guide.md`, fork PR #792. Working
memory and the measurements are in
[`internal_docs/guide_ladruno_sanisand_integrator.md`](../../../../../internal_docs/guide_ladruno_sanisand_integrator.md).

## Context

The fork's ADR-86b reworked the SANISAND *integrator*. It shipped no new
material — the viscoplastic wrapper was measured and rejected, so classTag
33022 stays reserved and unbuilt — but three of its consequences land on
decks apeGmsh generates.

**1. The tangent default moved underneath us.** The fork parser's
`$TanType` default went `0 → 2` (PR #792) while vanilla `ManzariDafalias`
stayed at `0`. Our emitter omitted the whole 5-positional tail whenever
every field was default, which for `LadrunoSANISAND` was the common case.
So a deck that named no tangent at all would have **silently changed
tangent on a fork upgrade** — from elastic to consistent, from symmetric
to unsymmetric — with no diff in apeGmsh and nothing checking the solver.

`TanType 0` is the elastic tangent: `algorithm Newton` runs as *modified*
Newton. Invisible on a single-element calibration deck (there is no global
solve) and expensive on a real BVP — the fork measured **800 vs 283**
Newton iterations on a drained triaxial (2.83×), and at a tighter
tolerance the elastic leg could not finish a push the consistent one
completed. The converged answer does not change with the tangent (the fork
gated that on a free-DOF BVP); the iteration count and the *solver
requirement* do.

**2. The consistent tangent of a non-associated model is unsymmetric.**
Pairing it with a half-storage solver is the ADR-80 silent-wrong-answer
class: the solve reads only the `col >= row` half of each element matrix,
no averaging and no detection, and converges to a plausible wrong answer.
Nothing in the bridge related a material's tangent to the deck's `system`.

**3. Uncapped `ModifiedEuler` lies about failing.** It substeps toward
`dT_min = 1e-6` with no bound on substep count, and on reaching the floor
it **force-accepts** a degraded substep and reports success — the step
controller is told nothing is wrong. Measured on a strip footing: one
`analyze(1)` took **34.3 minutes** while the controller sat idle, 0 of 80
subdivisions used. The fork added an opt-in cap that makes the material
refuse instead: 2.1–2.6× deeper reach for the same wall clock, worst step
759 s → 94 s, same answer.

Separately, and found in our own suite rather than the fork's:
**`NormDispIncr` is unreachable on this material.** The
displacement-increment residual stalls and never meets a tight tolerance,
and it is not mesh-neutral, so the same number means different things at
different `h`. Our own live A/B asked for `tol=1e-8` on a single-hex
triaxial driver, with a comment asserting a floor "around 1e-9"; the
measured floor is **4.2587e-08** and the leg never converged. That test
was red on `main`.

## Decision

### D1 — `tan_type` defaults to `2`, and the tail always emits

`LadrunoSANISAND.tan_type = 2`. `ManzariDafalias` and `SAniSandMS` are
untouched (vanilla's parser default has not moved, so their existing
all-or-nothing tail omission stays correct).

`LadrunoSANISAND` emits its 5-positional tail **unconditionally**, unlike
its base. Leaving the tail off delegates the tangent to whichever parser
happens to read the deck, and the two parsers no longer agree — an
implicit tail means the same deck integrates differently depending on
which material name it carries and which fork build runs it. Written out,
the tangent is a fact of the deck.

### D2 — Warn when an unsymmetric tangent meets a symmetric solver

`validate_manzari_tangent_solver` runs at the same emit seam as the
[ADR 0074](0074-ladruno-up-porous-element.md) D4 u-p gate and shares its
allow-list, `_UNSYMMETRIC_SAFE_SYSTEMS` — the physical fact (which solvers
hold a full unsymmetric matrix) is not u-p-specific. Same scope rules: a
DECLARED symmetric system is wrong whether or not this emit solves; the
missing-system branch is gated on there being an analysis chain, and
skipped for a partitioned deck riding the ADR-0027 INV-5 general auto-emit;
staged decks are checked per stage, since `wipeAnalysis` re-defaults each
stage to ProfileSPD.

**Fail-soft, unlike D4.** The deck is still runnable, and there are
legitimate reasons to take the symmetrized tangent knowingly — a
calibration deck with no global solve, an associated-flow parameter set.
It is the *answer* that is not trustworthy, so this is a warning the model
author weighs, not a build-stopper.

### D3 — Warn when a SANISAND deck drives on the displacement increment

`validate_manzari_convergence_test` warns on `NormDispIncr` /
`RelativeNormDispIncr` under any Manzari-family material, pointing at
`NormUnbalance` scaled to the model's own load or weight.

Deliberately **not** keyed on `tan_type`: the stall belongs to the
integrator, not the tangent, and the leg that actually failed in our suite
was `ManzariDafalias` at vanilla's `tan_type=0`. A tan_type-keyed gate
would have stayed quiet on exactly the deck that was red.

Only a *declared* test is checked; a missing one belongs to the
analysis-chain validation. `EnergyIncr` is not flagged — it was measured to
converge on the same deck.

### D4 — `max_substeps` is opt-in, and refused where it would backfire

A new `max_substeps: int = 0` field emits `-maxSubsteps N` when non-zero.
`0` means uncapped, which is vanilla's behaviour, and the flag is then not
emitted at all — this is the one flag on this class that does **not** always
emit, so a deck that never asks for a cap stays byte-identical to the one
it produced before the field existed. (The other three flags always emit
because their defaults differ from vanilla's; this one's default *is*
vanilla's.) It warns, like `honor_tol_r`, on a scheme that never reaches
`ModifiedEuler`.

`validate_sanisand_substep_cap` **raises** if a capped material reaches any
element not known to propagate a material refusal. This is an
**allow-list** — currently `{LadrunoBrick}` — not the deny-list the fork
guide sketches, for two reasons. The guide's operative finding is positive
("only `LadrunoBrick` propagates on every path today"), and the failure
direction is asymmetric: under an element that discards the return code, a
capped material hands back a *partially integrated* stress with a partial
tangent and the analysis converges on it, which is **worse** than the
uncapped force-accept it replaces — that at least integrated the whole
increment. A wrong bearing capacity is not a runnable-deck problem, so
this one raises where D2 warns. `LadrunoBrick20` is a separate C++ class,
not a formulation, and is not on the list until someone checks its return
paths.

The material graph is walked transitively, so a `PlaneStrain` or
`LogStrain` wrapper cannot hide a capped model one level down.

## Consequences

**Positive.**
- The tangent, the solver requirement and the convergence metric are now
  properties of the emitted deck rather than of the build that reads it.
- Two silent-wrong-answer classes (unsymmetric-tangent-on-symmetric-solver,
  refusal-swallowed-by-element) are surfaced at build.
- The fork's substep cap is reachable from typed apeGmsh without a raw
  `ops` escape hatch.

**Breaking.**
- Every `LadrunoSANISAND` deck line grows the tail `1 2 1 1e-07 1e-07`.
  Goldens recording that line move; goldens recording converged nodal
  values should not move outside tolerance.
- A deck that was implicitly running the elastic tangent now runs the
  consistent one, and may need a general solver — which D2 now says so.

**Neutral / forward-looking.**
- `_UNSYMMETRIC_SAFE_SYSTEMS` is the shared extension point for any future
  gate that needs "a solver that holds a full unsymmetric matrix".
- `_REFUSAL_PROPAGATING_ELEMENTS` is a one-line addition per element as the
  fork's return-code plumbing spreads.
- No new material and no new classTag: fork ADR-90 rejected the
  viscoplastic wrapper on measured grounds, and `33022` stays reserved and
  unbuilt.

## Cross-references

- [ADR 0027](0027-cross-partition-mp-constraints.md) — INV-5 general-solver auto-emit
  that D2's missing-system branch defers to.
- [ADR 0074](0074-ladruno-up-porous-element.md) — the D4 u-p solver gate
  D2 mirrors, and whose allow-list it now shares.
- [ADR 0101](0101-sanisand-materials-and-material-stage.md) — the typed
  Manzari family and the stage flip this extends.
- The three ADR-86 deck rules still hold and are unchanged: do not shear
  during the elastic stage, emit `updateMaterialStage` per tag at every
  stage boundary, and a proportional strain ramp at `p_r = 0` never yields.
