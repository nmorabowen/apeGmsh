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
| New read-only material responses `ladrunoBranch` → 8 floats `[branch, gamma0, gamma1, f1_trial, f2_trial, forcedAccept, I1, detAmin]`; branch 0 elastic / 1 cone / 2 cutoff / 3 corner; `detAmin` = min over ~200 directions of det(n·C_ep·n)/(2G)³ | same file; forwarded by LadrunoBrick, LadrunoBrick20, BezierTet10, TenNodeTetrahedron through `eleResponse <e> material <gp> ladrunoBranch` | A per-Gauss-point census apeGmsh can read back (§3). Empty list on pre-#803 builds — doubles as a capability probe. A second response, `ladrunoTangent` → 36 floats (responseID 96, the consistent tangent), ships on the same branch; the guide does not mention it. apeGmsh refuses its bare token too but names no columns for it. |
| `tests/test_r3_prandtl_collapse_gate.py`: the associated-flow control now asserts the two flow rules give **distinct** answers (`ASSOC_MIN_SEPARATION = 0.20`); its old "associated must not plateau" premise was the defect | fork Zone-A slow tier | Only matters if an apeGmsh live gate copied that premise. None does today. |
| Build stamp | `ops.ladrunoBuild()` | #803 merged into `ladruno` as **`61b3efa04`** (2026-09-08); that hash is the floor, recorded as `DruckerPrager`'s `DP_ADR95_MIN_FORK_BUILD`. Pre-fix installs keep failing quadratic DP decks exactly as before — our venv build `1652f945c` is one of them until the fork is rebuilt. |

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

5. **Budget the push, not the element.** Measured on the fork's Prandtl deck (plain push to
   s/B 0.15, idle box, PARDISO threaded, ladder budget 200): `LadrunoBrick -bbar` 1 386 DOF 31 s
   (0.10 s per attempt); `LadrunoBrick20 -uri` 4 659 DOF 481 s, of which 40 % is 1 051 failed
   attempts around the corner Gauss points (0.19 s per attempt); `BezierTet10` std / `-bbar` 7 749
   DOF 466 / 533 s with zero failed attempts (0.31 / 0.35 s per attempt); `TenNodeTetrahedron`
   727 s (0.48 s). SANISAND IMPL-EX on the H20 costs the same per attempt as Drucker–Prager
   (296 s to target); SANISAND implicit costs 9.9 s per attempt, fifty times more, and stops at
   s/B 0.008 after 20 min — on footing decks IMPL-EX is the affordable path, not a convenience.
   Any per-Gauss-point census (`ladrunoBranch`, tangent SVD) at every station multiplies the wall
   by ~5–8×; sample at stations, not at steps.

## 3. Read-back: the `ladrunoBranch` census (recorder/file half ADOPTED)

**Adoption status (apeGmsh, 2026-09-08).** The recorder/file half is done. `ladrunoBranch` and
`ladrunoTangent` joined `_MATERIAL_ONLY_ELEM_TOKENS` (`opensees/recorder.py`), so a bare token is
refused at construction naming `material.<token>` — the bare spelling records nothing on the fork.
The reader (`results/readers/_ladruno_element_io.py`) names the eight `ladrunoBranch` columns by
position — `dp_branch`, `dp_gamma_cone`, `dp_gamma_cutoff`, `dp_f1_trial`, `dp_f2_trial`,
`dp_forced_accept`, `dp_i1`, `dp_det_a_min` — pinned to the fill site
`DruckerPrager::getLadrunoBranch()` (`DruckerPrager.cpp:960-970`), because the fork returns a bare
`MaterialResponse(this, 95, Vector(8))` with no ResponseType and the file therefore writes `C1..C8`.
The 36-entry `ladrunoTangent` takes the documented unknown-bucket route rather than inventing names.

**Not adopted.** The LIVE path. `Results` is a pure file/array reader with no ops handle, so the
`Results.from_native(...).gauss_point_branch(...)` sketched below cannot exist as written; the only
live seam is `DomainCapture`, whose per-material routing still hard-codes `catalog_token ==
"strain"` (`opensees/_response_catalog.py::needs_per_material_strain`). Generalising that table is
its own slice, and it needs a post-`61b3efa04` build to verify. The tension and corner censuses
below are likewise not built.

The original ask, for reference:

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
edge, which only quadratic elements resolve — but each material responds differently. **Updated
2026-09-08:** the ASD row's defect is fixed (fork PR #832); only the SANISAND cost story stands
as a deck rule.

| material | what happens at that spot | apeGmsh guidance |
|---|---|---|
| `DruckerPrager` (UW) | was the dead cutoff branch — **fixed** | use it for zero-dilatancy frictional collapse decks |
| `ASDPlasticMaterial3D` + `DruckerPrager_YF` | cone + apex projection; the region test is now in the ELASTIC metric (fork ADR-94 wp/94f, PR #832, merged 2026-09-08) with a flank-first apex fallback behind it. PR #815 had made the projection live but classified in the EUCLIDEAN metric (`p − p_apex ≥ η·q`), which at ψ = 0 sent over-apex states with small shear to a flank map that cannot move p | **Usable on zero-dilatancy footing decks from #832 on.** Measured on the ADR-95 deck after the fix: `h20uri` TARGET at s/B 0.15, q/q_exact 0.9758 against the repaired UW-DP 0.9757, zero flank refusals (pre-fix: FLOOR at 0.01122 with 435). Linear control 1.0850, unchanged. On a pre-#832 engine the caveat stands: the quadratic leg walls at the footing edge. |
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
