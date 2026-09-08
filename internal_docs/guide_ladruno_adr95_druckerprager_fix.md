# Ladruno ADR-95 — what the DruckerPrager return-map fix changes for the emitter

Working memory for apeGmsh after the Ladruno fork's ADR-95 landed on `ladruno` (fork PR #803,
fix commit `31322a47a`, merged 2026-09-08). Everything here is about what apeGmsh **emits** and
what it can now **read back**; none of it changes apeGmsh's own model or snapshot. Sibling of
`guide_ladruno_sanisand_integrator.md` (ADR-86b/90) and `guide_ladruno_asdplastic.md` (ADR-94).

Fork-side sources, if you need to check a claim: `Ladruno_implementation/95_prandtl_reissner_quadratic_root_cause.md`
(the results note), `95_prandtl_reissner_campaign_report.md` (illustrated report),
`_adr95_corner_tangent_review.md` (the defect, line by line), `_adr95_asd_crosscheck_results.md`
and `_adr95_sanisand_crosscheck_results.md` (the other two materials on the same deck),
`LEDGER_vanilla_files.md` (the DruckerPrager rows), `LEDGER_quirks.md` (ADR-95 entries).

## 0. One-paragraph summary

The vanilla UW `DruckerPrager` (the `nDMaterial DruckerPrager` apeGmsh emits from
`material/nd.py::DruckerPrager`) had a dead branch in its two-surface return map: the tension-cutoff
residual was never assembled, so a Gauss point crossing the cutoff kept an unreturned stress and a
pathological consistent tangent. On a strip-footing deck that killed **every quadratic element**
(LadrunoBrick20, TenNodeTetrahedron, BezierTet10) on the step floor at 30–77 % of the Prandtl
load while the linear b-bar hex plateaued at the exact answer. The fork repaired the return map.
**No apeGmsh API changes** — the `DruckerPrager` primitive, its argument order and validation are
untouched — but three things follow for the emitter: results on any deck that reaches the cutoff
change (they were wrong before), quadratic solid elements are now usable with DruckerPrager on
footing/collapse decks, and a new read-only material response exists that apeGmsh can expose.

## 1. What the fork changed (PR #803)

| change | where | emitter consequence |
|---|---|---|
| Two-surface return map: both residual rows assembled in every active-set combination (`Jact` index-driven); the radial-return term of the consistent tangent divides by ‖η_trial‖, not the returned norm | `SRC/material/nD/UWmaterials/DruckerPrager.cpp` (vanilla, marked `// Ladruno ADR-95`) | **Results change.** Cone-only paths move by ≤ 1.3e-5 relative (the tangent denominator was wrong on the cone too — path converges to the same state faster: fork gate leg 1390 s → 181 s). Any path that reached I1 ≥ T is different, because before the fix it was not on the yield surface at all. |
| New read-only material response `ladrunoBranch` → 8 floats `[branch, gamma0, gamma1, f1_trial, f2_trial, forcedAccept, I1, detAmin]`; branch 0 elastic / 1 cone / 2 cutoff / 3 corner; `detAmin` = min over ~200 directions of det(n·C_ep·n)/(2G)³ | same file; forwarded by LadrunoBrick, LadrunoBrick20, BezierTet10, TenNodeTetrahedron through `eleResponse <e> material <gp> ladrunoBranch` | A per-Gauss-point census apeGmsh can read back (§3). Empty list on pre-#803 builds — doubles as a capability probe. |
| `tests/test_r3_prandtl_collapse_gate.py`: the associated-flow control now asserts the two flow rules give **distinct** answers (`ASSOC_MIN_SEPARATION = 0.20`); its old "associated must not plateau" premise was the defect | fork Zone-A slow tier | Only matters if an apeGmsh live gate copied that premise. None does today. |
| Build stamp | `ops.ladrunoBuild()` | Post-fix builds: any hash at or after the merge of #803 into `ladruno`. Pre-fix installs keep failing quadratic DP decks exactly as before. |

Not changed: parameter list, parsing, class tags, send/recv layout, `updateMaterialStage`
semantics, the `theta` tension-softening parameter, hardening.

## 2. Deck rules that now hold (and one that still does)

1. **A small `sigmaY` as an apex regulariser works as intended.** `sigmaY = 0.2 kPa` puts the
   cutoff at I1 = √(2/3)·σ_y/ρ ≈ 0.8 kPa, i.e. essentially "no tension". Before the fix that
   cutoff was decorative; now the first tensile Gauss points beside a footing edge are returned to
   it and the run continues. Keep using a small, explicitly non-physical `sigmaY` for weightless
   or lightly confined frictional decks; document it as a regulariser, not cohesion.
2. **Quadratic solids are usable with DruckerPrager on collapse decks.** Measured on the fork's
   Prandtl–Reissner strip (h0 = 1.0 m, two elements across B, s/B 0.15, q/q_exact):
   `LadrunoBrick -bbar` 1.085 (unchanged), `LadrunoBrick20 -formulation uri` 0.976,
   `BezierTet10 -bbar` 1.040, `BezierTet10` std 1.182, `TenNodeTetrahedron` 1.170 — all
   plateaus. Prefer the **b-bar** variants: they are exactly isochoric at ψ = 0 and plateau
   tightest; standard-integration tets over-shoot (locking), and the reduced-integration H20 loses
   rank once all eight Gauss points yield (9.5 % spurious volumetric increment in its collapse
   mechanism, ~1 000 failed Newton attempts per push). These are coarse-mesh numbers, not
   converged values.
3. **Bernstein-consistent loads on Bézier elements remain mandatory** (apeGmsh ADR 0091,
   `basis="bernstein"` on `g.loads.surface.*`). Lagrange-consistent loads on control points
   reproduce the resultant exactly and put a 190 % oscillation into the surface stress; the
   `WarnLoadBasisMismatch` guard at `build()` catches the common case.
4. **Non-associated DP tangents are unsymmetric.** Same rule as the concrete guide: `UmfPack` /
   `Pardiso` / `Mumps` / `FullGeneral`, never `ProfileSPD`/`BandSPD`.

## 3. Read-back: the `ladrunoBranch` census (ask, not yet done)

apeGmsh's `Results` layer has no token for this response yet. A useful, cheap addition:

```python
# raw, works today on a post-#803 build via the bridge
vals = ops.eleResponse(e, 'material', gp, 'ladrunoBranch')   # [] on old builds, 8 floats on new
branch, gamma0, gamma1, f1, f2, forced, I1, detAmin = vals
```

Suggested surface: `Results.from_native(...).gauss_point_branch(elements=...)` (or a
`material_response("ladrunoBranch")` generic) returning per-element/per-GP arrays, plus two derived
fields the fork's harness found decisive on collapse decks:

- **tension census**: count of Gauss points with mean stress ≥ 0 — generic, from the existing
  `stress` response; the first such points beside a footing edge are where every material on this
  deck has its trouble;
- **corner census**: count of `branch == 3` points and their locations.

Capability probe for `backend-capabilities`: an empty `ladrunoBranch` reply on a single-element
`DruckerPrager` model means a pre-#803 engine.

## 4. The other two materials on the same deck (cross-references)

The trigger is implementation-independent — the first tensile Gauss points beside the footing
edge, which only quadratic elements resolve — but each material responds differently:

| material | what happens at that spot | apeGmsh guidance |
|---|---|---|
| `DruckerPrager` (UW) | was the dead cutoff branch — **fixed** | use it for zero-dilatancy frictional collapse decks |
| `ASDPlasticMaterial3D` + `DruckerPrager_YF` | no cutoff; the PR #815 apex projection is live but its region test `p − p_apex ≥ η·q` is Euclidean; at ψ = 0 the exact test is `p ≥ p_apex`, so over-apex states with small shear go to a flank map that cannot move p → the quadratic leg still walls at the same station (fork prediction on record, confirmed) | ADR 0105's contract stands; **add** the caveat: ASD-DP with `etabar = 0` on a footing/heave deck walls at the apex until the fork's ADR-94 follow-up (elastic-metric classification or flank-then-apex fallback). Its linear control matches UW-DP to the printed digit after #815. |
| `LadrunoSANISAND` | no apex to return to; the substepper's cost explodes as p → 0 (implicit legs drown at ~10 s per attempt); IMPL-EX finishes with `-Pmin` holding the edge points at +0.1 kPa | as in `guide_ladruno_sanisand_integrator.md` / ADR 0103: `maxSubsteps` (the fork's CP1 deck uses 1000; 0 = uncapped lets one `analyze()` block for 15–45 min), `-implex` for footing decks, keep `-Pmin`. Read the late part of an IMPL-EX curve with the floor in mind. |

## 5. A live-gate suggestion (`ladruno_fork` marker)

One discriminating assertion separates pre- and post-#803 engines in a minute: the fork's Prandtl
deck (or any weightless surcharged strip with `sigmaY = 0.2`, ψ = 0, ν = 0.45, φ_txc = 20°) on
`LadrunoBrick20 -formulation uri` at h0 = 1.0 must pass s/B = 0.05 (pre-fix: floor at 0.011,
q/q_exact 0.77; post-fix: plateau, 0.976 at 0.15). The linear b-bar leg is not a discriminator
(1.085 before and after). If the deck is re-hosted in apeGmsh, apply the Bernstein load rule on
any Bézier variant and quote termination mode with every number.

## 6. What is not claimed

Mesh convergence of the quadratic plateaus; the tet penalty on more than one mesh pair; any
shear band (two elements across the footing cannot hold one); the late SANISAND IMPL-EX curve.
Upstream OpenSees master still carries both DruckerPrager defects; an upstream PR is planned on
the fork side.
