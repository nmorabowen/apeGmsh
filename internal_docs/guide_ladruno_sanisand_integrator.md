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

## 2. `tan_type` — DONE

`LadrunoSANISAND.tan_type` now defaults to **`2`**, the 5-positional tail is **always
emitted**, and a build-time warning covers the solver coupling. `ManzariDafalias` is
untouched — vanilla's `0`, and its all-or-nothing tail. The reasoning is kept below
because the numbers are the warrant.

`TanType 0` is the **elastic** tangent: a deck built from the old defaults ran
`algorithm Newton` as *modified* Newton. That is invisible on a single-element,
fully-prescribed calibration deck (there is no global solve) and expensive on a real
BVP — the fork measured **800 vs 283 Newton iterations** on a drained triaxial (2.83×),
and at a tighter tolerance the elastic-tangent leg could not finish a push the
consistent one completed. Emitting the tail explicitly, rather than leaning on a
parser default, is what keeps the two apart: the fork's default is now `2` and
vanilla's is still `0`, so an implicit tail means the same deck integrates differently
depending on which material name it carries and which build runs it.

**The open check is answered, and it was the bad branch.** The emitter *omitted* the tail when
every field was default (`if tail != _MANZARI_TAIL_DEFAULTS`), so a stock `LadrunoSANISAND`
deck wrote no tail at all and would have picked up the fork's new `TanType 2` **silently** on
an upgrade — the consistent, unsymmetric tangent, with nothing checking the solver. That is why
the tail is now unconditional on this class: the tangent is a fact of the deck, not of the
build. (`ManzariDafalias` still omits its tail; vanilla's parser default has not moved.)

**Second site, easy to miss:** `ops.nDMaterial.<Type>` in `_internal/ns/nd.py` re-states every
default in its own signature and passes them all through explicitly, so a dataclass default the
wrapper does not mirror is dead on the public surface. Nothing in the suite compared the two
until `test_namespace_wrapper_mirrors_the_tan_type_default`, which covers the Manzari family
only — the general parity gap is still open.

**Coupled constraint — this one can produce a wrong answer. Now gated.** The consistent tangent
of a non-associated model is genuinely **unsymmetric**. A deck that pairs `tan_type=2` with a
symmetric solver is the ADR-80 silent-wrong-answer class. When any Manzari-family material in
the deck has `tan_type != 0`, the emitted `system` must be one of `FullGeneral`, `UmfPack`,
`BandGeneral`, or `Pardiso -matrixType 0` — never `ProfileSPD` / `SProfileSPD` /
`Pardiso -matrixType 1|2`. `validate_manzari_tangent_solver` in `_internal/build.py` warns
at the same emit seam as the ADR-0074 D4 u-p gate, with the same scope rules (a declared symmetric system is wrong whether
or not this emit solves; the missing-system branch — OpenSees' no-`system` default *is*
ProfileSPD — is gated on there being an analysis chain and skipped for a partitioned deck,
which rides the ADR-0027 INV-5 general auto-emit; staged decks are checked per stage). Both
gates now share one allow-list, `_UNSYMMETRIC_SAFE_SYSTEMS`. Fail-soft, unlike D4: the deck is
still runnable, it is the *answer* that is not trustworthy, and there are legitimate reasons to
take the symmetrized tangent knowingly.

The converged answer does **not** change with the tangent (the fork gated that on a free-DOF
BVP); only the iteration count and the solver requirement do.

## 3. `-maxSubsteps N` — DONE

`ModifiedEuler` substeps toward `dT_min = 1e-6` with no bound on substep count, and on reaching
the floor it *force-accepts* a degraded substep and reports success. Measured on a strip
footing: one `analyze(1)` took **34.3 minutes** while the step controller sat idle, having been
told nothing was wrong (0 of 80 subdivisions used). With a cap, the material refuses the
increment, the element propagates the refusal, the integrator cuts the load step — 2.1–2.6×
deeper reach for the same wall clock, worst step 759 s → 94 s, same answer.

Exposed as `max_substeps: int = 0` on `LadrunoSANISAND`, emitted as `-maxSubsteps N` when
non-zero. All three rules below are implemented; the third is a **raise**, not a warning.

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

   `validate_sanisand_substep_cap` raises at the emit seam. It is an **allow-list**
   (`_REFUSAL_PROPAGATING_ELEMENTS = {LadrunoBrick}`), not the refuse-list sketched above — the
   operative sentence is the positive one, "only `LadrunoBrick` propagates on every path today",
   and most of the refuse-list elements are fork elements apeGmsh does not expose at all.
   `LadrunoBrick20` is a separate C++ class rather than a formulation, so it is off the list
   until someone reads its return paths. The material graph is walked transitively, so a
   `PlaneStrain` / `LogStrain` wrapper cannot hide a capped model one level down.

## 4. Convergence test on SANISAND decks — confirmed in our own suite

`NormDispIncr` is **unreachable** on this material — the displacement-increment residual stalls
at a median 6.6e-7 m and never meets a tight tolerance, and it is not mesh-neutral, so the same
number means different things at different `h`. Emit `NormUnbalance` scaled to the model's own
weight (`γ'V`) for SANISAND decks; if a caller asks for `NormDispIncr` on one, warn.

**We had one of these decks and did not know it.** The live A/B
`test_ladruno_sanisand_live.py` asked for `NormDispIncr(tol=1e-8)` on a single-hex triaxial
driver, with a comment asserting a residual floor "around 1e-9". The measured floor on that
build is **4.2587e-08** — 4.3× the tolerance, and ~40× the floor the comment claimed — and the
deviatoric leg never converged, on the *ManzariDafalias* reference leg. The whole file has moved to `NormUnbalance` at a tolerance
expressed as a fraction of each deck's own applied load
(`_RESIDUAL_REL = 1e-4`, `tol = _RESIDUAL_REL * ref_force`).

Two numbers worth carrying forward:

- **The force residual is reachable but not tight.** Sweeping that constant and running the
  four live tests at each value: all pass at `1e-4`, `3e-5` and `1e-5`; three of four fail at
  `3e-6` and `1e-6`, every one on a `CTestNormUnbalance` stall. The floor is between `3e-6` and
  `1e-5` **relative to the applied load** — so a deck asking for a 1e-6 relative force residual
  on this material is asking for something it cannot have.
- **The floor depends on the tangent.** A probe built on `ManzariDafalias` (elastic tangent)
  reaches `1e-6` on the same decks that stall there with the consistent tangent. Measure with
  the material the deck actually carries, or the number is meaningless.

**Done:** `validate_manzari_convergence_test` warns at the emit seam on `NormDispIncr` /
`RelativeNormDispIncr` under any Manzari-family material, per stage on a staged deck. It is
keyed on the MATERIAL, not on `tan_type` — the stall is the integrator's, and the leg that
was actually red ran the elastic tangent, so a tan_type-keyed gate would have stayed quiet on
exactly the deck that failed. `EnergyIncr` is not flagged: it was measured to converge on the
same deck.

## 5. `-Presidual` — unchanged, but emit it explicitly

The fork did **not** change the default. There is a measured onset — at `-Presidual 0` a
free-surface Gauss point gets clamped at `s/B ≈ 0.0153` on a coarse dense leg — and a decision
(declare a non-zero `p_r`; add a surcharge; accept the clamp and disclose it) that belongs to
the model author, not to us. Our existing warning about `p_residual=0.0` being the less
forgiving value stands unchanged; keep emitting `-Presidual` explicitly so an upgrade cannot
move it.

All of the above is warranted by
[ADR 0103](../src/apeGmsh/opensees/architecture/decisions/0103-sanisand-integrator-deck-contract.md).

## 6. What did *not* change

- **No new material, no new classTag.** ADR 90 rejected the viscoplastic wrapper on measured
  grounds; `33022` stays reserved and unbuilt. Nothing to add to the type map.
- **The three ADR-86 deck rules still hold** — do not shear during the elastic stage (or `M_c`
  silently inflates ~50 %), emit `updateMaterialStage` per tag at every stage boundary, and a
  proportional strain ramp with `p_r = 0` never yields at all. See
  `86_ladruno_sanisand_apegmsh_emitter_guide.md`; nothing here supersedes them.
- **Element choice for failure legs** is still `LadrunoBrick -formulation bbar`; tetrahedra are
  prohibited (volumetric locking, measured against an exact collapse load).

## 7. Goldens — what actually moved

- `_api_index.json` (the committed signature harvest, gated in two CI lanes) — rebuilt with
  `python -m apeGmsh.studio.lookup --build`.
- Every `LadrunoSANISAND` deck line grows the 5-positional tail `1 2 1 1e-07 1e-07`.
- The live A/B `test_i1_pinned_ladruno_sanisand_reproduces_manzari_bit_identically` now has to
  pin `tan_type=0` alongside the low-stress constants, or it A/Bs two different code paths.
- Nothing about `-maxSubsteps` moves a golden while it is unset.

- The same file's `_bind_chain` moved from `NormDispIncr` to `NormUnbalance` — it was red on
  `main` for exactly the §4 reason. See §4 for the measurement.
