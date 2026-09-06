# Ladruno SANISAND — what the integrator work (ADR-86b / ADR-90) changes for the emitter

Working memory for the `LadrunoSANISAND` primitive after the Ladruno fork's ADR-86b
(integrator) and ADR-90 (regularization) landed on `ladruno`. Everything here is about
what apeGmsh **emits**; none of it changes apeGmsh's own model or snapshot.

Fork-side sources, if you need to check a claim: `Ladruno_implementation/90_ladruno_regularization_tims_report.md`
(the consumer report), `86_ladruno_sanisand_apegmsh_emitter_guide.md` (the deck rules that
still stand), `LEDGER_implementations.md` (ADR-86b row), fork PR #792.

## 0. One-paragraph summary

The fork did **not** ship a viscoplastic wrapper — that was measured and rejected, so there
is no new material to add and no new classTag to map (33022 is reserved and unbuilt). What
shipped is a fix to the SANISAND *integrator*: an opt-in substep cap that makes the material
refuse an increment it cannot integrate instead of force-accepting it, plus a changed tangent
default. Two of those touch decks apeGmsh generates today.

## 1. What shipped on the fork

| change | deck surface | default |
|---|---|---|
| substep-count cap on `ModifiedEuler` | `-maxSubsteps N` | `0` = uncapped = today's behaviour |
| `TanType` parser default 0 → 2 | positional `$TanType` | fork parser now `2`; **vanilla `ManzariDafalias` still `0`** |
| refusal propagation | none (element-side) | — |
| `LoadPath` / `ArcLength` now honour a failed update | none | — |

## 2. Change today: `tan_type`

`LadrunoSANISAND.tan_type` (and `ManzariDafalias.tan_type`) default to **`0`** in
`src/apeGmsh/opensees/material/nd.py`. `TanType 0` is the **elastic** tangent: a deck built
from our defaults runs `algorithm Newton` as *modified* Newton. It is invisible on a
single-element, fully-prescribed calibration deck (there is no global solve) and expensive on
a real BVP — the fork measured **800 vs 283 Newton iterations** on a drained triaxial (2.83×),
and at a tighter tolerance the elastic-tangent leg could not finish a push the consistent one
completed.

**Do:** default `tan_type = 2` on `LadrunoSANISAND`, and keep emitting it explicitly rather
than relying on either parser's default — the fork's is now `2` and vanilla's is still `0`, so
an implicit tail means the same deck integrates differently depending on which material name
it carries.

**Check while you are there:** whether our emitter always writes the 5-positional tail or omits
it when every field is default. It decides whether existing goldens move on a fork upgrade
(tail always written → nothing moves, decks stay elastic; tail omitted → decks silently switch
to the consistent tangent). Either way the fix is the same; only the golden-file story differs.

**Coupled constraint — this one can produce a wrong answer.** The consistent tangent of a
non-associated model is genuinely **unsymmetric**. A deck that pairs `tan_type=2` with a
symmetric solver is the ADR-80 silent-wrong-answer class. When any Manzari-family material in
the deck has `tan_type != 0`, the emitted `system` must be one of `FullGeneral`, `UmfPack`,
`BandGeneral`, or `Pardiso -matrixType 0` — never `ProfileSPD` / `SProfileSPD` /
`Pardiso -matrixType 1|2`. We already carry exactly this sentence for a different reason in
`ConstraintsComposite.py`; the same check belongs on the material/analysis seam. A warning at
`build()` is enough — the deck is still runnable, it is the *answer* that is not trustworthy.

The converged answer does **not** change with the tangent (the fork gated that on a free-DOF
BVP); only the iteration count and the solver requirement do.

## 3. New optional flag: `-maxSubsteps N`

`ModifiedEuler` substeps toward `dT_min = 1e-6` with no bound on substep count, and on reaching
the floor it *force-accepts* a degraded substep and reports success. Measured on a strip
footing: one `analyze(1)` took **34.3 minutes** while the step controller sat idle, having been
told nothing was wrong (0 of 80 subdivisions used). With a cap, the material refuses the
increment, the element propagates the refusal, the integrator cuts the load step — 2.1–2.6×
deeper reach for the same wall clock, worst step 759 s → 94 s, same answer.

If we expose it (a plain `max_substeps: int = 0` field on `LadrunoSANISAND`, emitted as
`-maxSubsteps N` when non-zero), three rules apply:

1. **Default `0` = uncapped = today.** A deck that does not ask for it must be byte-identical.
2. **It is inert on schemes that never reach `ModifiedEuler`** — exactly the condition
   `honor_tol_r` already checks. Reuse `_SCHEMES_REACHING_MODIFIED_EULER` and emit the same
   shape of warning. (Note `int_scheme=7` is *correctly* excluded: `MaxStrainInc` has no case
   for it and falls through to `ForwardEuler`, despite the `INT_MAXSTR_MFE` name.)
3. **The element must propagate a material refusal, or the cap makes things worse.** Under an
   element that discards the return code, a capped material returns a *partially integrated*
   stress with a partial tangent, and the analysis accepts it as converged — worse than the old
   force-accept, which at least integrated the whole increment. Today only `LadrunoBrick`
   propagates on every path.

   - **Safe:** `LadrunoBrick` (all formulations).
   - **Refuse to emit:** `stdBrick`/`Brick`, `BrickUP`, `BBarBrickUP`, the `QuadUP` family
     (`FourNodeQuadUP`, `Nine_Four_Node_QuadUP`, `Twenty_Eight_Node_BrickUP`) — these have no
     return channel at all (`setTrialStrain` is called inside a `void formResidAndTangent`).

   Since we resolve materials against element assignments at `build()`, this is checkable
   there: if `max_substeps != 0` and any element carrying that material is in the refuse list,
   raise rather than warn — a silently wrong bearing capacity is not a runnable-deck problem.

## 4. Convergence test on SANISAND decks

`NormDispIncr` is **unreachable** on this material — the displacement-increment residual stalls
at a median 6.6e-7 m and never meets a tight tolerance, and it is not mesh-neutral, so the same
number means different things at different `h`. Emit `NormUnbalance` scaled to the model's own
weight (`γ'V`) for SANISAND decks; if a caller asks for `NormDispIncr` on one, warn.

## 5. `-Presidual` — unchanged, but emit it explicitly

The fork did **not** change the default. There is a measured onset — at `-Presidual 0` a
free-surface Gauss point gets clamped at `s/B ≈ 0.0153` on a coarse dense leg — and a decision
(declare a non-zero `p_r`; add a surcharge; accept the clamp and disclose it) that belongs to
the model author, not to us. Our existing warning about `p_residual=0.0` being the less
forgiving value stands unchanged; keep emitting `-Presidual` explicitly so an upgrade cannot
move it.

## 6. What did *not* change

- **No new material, no new classTag.** ADR 90 rejected the viscoplastic wrapper on measured
  grounds; `33022` stays reserved and unbuilt. Nothing to add to the type map.
- **The three ADR-86 deck rules still hold** — do not shear during the elastic stage (or `M_c`
  silently inflates ~50 %), emit `updateMaterialStage` per tag at every stage boundary, and a
  proportional strain ramp with `p_r = 0` never yields at all. See
  `86_ladruno_sanisand_apegmsh_emitter_guide.md`; nothing here supersedes them.
- **Element choice for failure legs** is still `LadrunoBrick -formulation bbar`; tetrahedra are
  prohibited (volumetric locking, measured against an exact collapse load).

## 7. Goldens

- Decks that omit the positional tail **and** run on a post-#792 fork build change tangent
  (answer unchanged, iteration count and required solver change). Regenerate any golden that
  records iteration counts or solver choice; a golden that records converged nodal values
  should not move outside its tolerance.
- Decks that write the tail explicitly are unaffected until we flip our own default.
- Nothing about `-maxSubsteps` moves a golden while it is unset.
