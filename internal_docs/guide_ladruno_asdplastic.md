# Ladruno ASDPlasticMaterial3D — what fork ADR-94 changes for the emitter

Working memory for the `ASDPlasticMaterial3D` family (`MohrCoulombSoil`,
`MohrCoulombTensionCutoffSoil`, `HoekBrownRock`, the generic class, `PlaneStrain`) after the
Ladruno fork's ADR-94 review-and-fix wave landed on `ladruno` (at or after `bbf657d49`,
2026-09-07). Everything here is about what apeGmsh **emits**; none of it changes apeGmsh's own
model or snapshot. The apeGmsh side is ADR 0105.

Fork-side sources, if you need to check a claim: `Ladruno_implementation/94_asdplastic_review_plan.md`
(plan + log), `reviews/adr94_verdict.md` (findings B1–B5, M1–M10), `_adr94_inventory.md` (the 46
registered combinations), `LEDGER_quirks.md` (twelve `ASDPlasticMaterial3D` entries), fork PRs #804,
#806, #809, #815, #816. Every number in §9 was measured against fork build
`3622d6214ef4cdeb8cf65a102ee35f6cd9973337` (the ADR-94 closeout, ASDP-equivalent to `bbf657d49`).

## 0. One-paragraph summary

The fork parser now **fails loud**: an unknown model-parameter name aborts the `nDMaterial`
command, and every parameter of the instantiated combination except `MassDensity` / `InitialP0`
is required. That refused every deck `MohrCoulombSoil` produced, because the helper zero-filled a
21-name superset. apeGmsh now carries the per-combination parameter schema and emits exactly it,
turns `strict_convergence` on and the `Continuum` tangent on by default, refuses the two
integrators the fork refuses, warns when the material sits on an element that swallows its
refusals, and adds the two helpers the SSI decks actually use. A deck that used to finish with
wrong stresses now fails at the step. That is the point.

## 1. What shipped on the fork

| change | deck surface | apeGmsh response |
|---|---|---|
| unknown parameter / option / IV name aborts the command; unset parameters are refused (B1) | `Begin_Model_Parameters` block | §2 — schema table + build-time `ValueError` |
| `strict_convergence` effective on every integrator and failure path (ADR-84 P2a + wp/94a) | `strict_convergence 0\|1` | §3 — default **on** |
| `f_relative_tol` (M5) | `f_relative_tol x` | §3 — exposed, default off |
| `Backward_Euler_LineSearch` (M7) and `Runge_Kutta_45_Error_Control_old` (M8) refused by name | `integration_method` | §4 — client-side `ValueError` with the fork's reason |
| per-instance tangent (M1); `Continuum` measured 5.3× fewer global iterations than `Secant` (M3) | `tangent_type` | §4 — default `Continuum` |
| `LADRUNO_MATERIAL_REFUSED` from all 15 failure sites; `stdBrick` / `BrickUP` / `QuadUP` still swallow it (B2, by design) | none (element side) | §5 — swallowing-host warning |
| Hoek–Brown yields at the textbook tensile strength `−s·σci/mb` (#806); DP gradients fixed; per-GP state | results | §9 — the plateau is pinned live |

## 2. The parameter schema — DONE

The fork's `list` verb prints the four type strings of each of the 46 registered combinations,
never the parameter names, so the schema cannot be discovered at emit time. It is composed
from the component headers (`using parameters_t` in `SRC/material/nD/ASDPlasticMaterial3D/**`)
as `EL ∪ YF ∪ PF ∪ (one set per IV hardening policy) ∪ {MassDensity, InitialP0}` and lives in
`material/nd.py` as `_ASDP_PARAMS_BY_COMPONENT`; `asdp_parameter_schema(yf, pf, el, iv)` parses
the IV string (`Name(Policy):…` — the same IV name may recur with different policies).

`ASDPlasticMaterial3D.__post_init__` validates when every component is in the table: a foreign
name is a `ValueError` naming it and the schema; a missing required name is a `ValueError`
listing the missing names. A combination with a component outside the table (today: the
StiffSoil family, 3 of the 46) is accepted unchanged and the fork validates — the escape hatch.

The 46 combinations are pinned as a fixture captured from the fork's `list` verb
(`tests/opensees/fixtures/asdp_registered_combinations_3622d6214.txt`), and the unit test resolves
every one of them. The table is a maintenance liability until the fork prints parameter names
(`asdplastic_fork_asks.md`, ask 1).

`MohrCoulombSoil` emits exactly `YoungsModulus, PoissonsRatio, MC_phi, MC_c, MC_ds, MC_psi,
MassDensity, InitialP0`. The twelve foreign names (`AF_*`, `DP_*`, `Dilatancy`, `DuncanChang_*`,
`Reference*`, `*LinearHardeningParameter`, `TC_min_stress`) are gone. The old block is kept
verbatim as a fixture in the unit test and in the live battery, because it is what the fork refuses.

## 3. `strict_convergence` and `f_relative_tol` — DONE

Both are first-class on every helper and typed in the generic class's option list.

- `strict_convergence: bool = True`. A non-converged or inadmissible state is refused instead of
  committed. apeGmsh's contract is that a silently wrong deck does not exist (ADR 0103 D2/D3
  reasoning). It changes convergence behaviour for decks that today commit inadmissible states —
  §9 has the numbers.
- `f_relative_tol: float = 0.0`. Off = the fork's own default, byte-identical behaviour. When set,
  the tolerance becomes `max(f_absolute_tol, f_relative_tol × strength scale)`; the strength scale
  is `c·cos φ` for Mohr–Coulomb and `|TC_min_stress|` for the cut-off branch. Rock-scale decks
  should set it: the absolute default is a verdict on the unit system (§9, case 6).

Both need a fork build at or after `bbf657d49`; an older parser drops the tokens silently
(`ASDP_MIN_FORK_BUILD` in `material/nd.py`).

## 4. Tangent and integrators — DONE

`tangent_type` defaults `Secant → Continuum` on every helper and its namespace mirror (the
"second site" the SANISAND guide warns about; `test_namespace_wrapper_mirrors_the_helper_defaults`
covers all three helpers). Results are identical at convergence; only the iteration count moves.

Client-side token validation in `__post_init__`, against the fork's lists after ADR-94 and
ADR-97. The methods split by kind, because the distinction is load-bearing:
**implicit** `{Backward_Euler, Closest_Point}` — both supported, neither warns — and
**explicit** `{Forward_Euler, Forward_Euler_Subincrement, Modified_Euler_Error_Control,
Runge_Kutta_45_Error_Control}`, which warn `ASDPlasticIntegrationWarning` and are REFUSED by the
fork at or after `7e93e4381` without `experimental_integrator=1`.
`Backward_Euler_LineSearch` and `Runge_Kutta_45_Error_Control_old` raise with the fork's reason
(M7 / M8); anything else raises naming the valid set. `tangent_type` and
`return_to_yield_surface` are validated the same way.

`Closest_Point` (fork ADR-97, ADR 0107) is the second implicit map — a true closest-point
projection with an exact consistent tangent, `tangent_type Algorithmic`. It is the only tangent
apeGmsh cross-checks: `Algorithmic` without `Closest_Point` raises (ADR-97 D2). Which YF/PF
combinations support the map (23 of 46, matched-family pairs only) is left to the fork's parser
— all three helpers build matched pairs, so all three are supported. **Defaults do not move**;
see `guide_ladruno_asdp_closest_point.md` and ADR 0107 for why, and for the fork's measurement.

## 5. The swallowing-host gate — DONE

`validate_asdplastic_host` in `_internal/build.py`, next to the ADR 0103 gates, **warns once per
deck** (`ASDPlasticHostWarning`) when an `ASDPlasticMaterial3D` — directly or through
`PlaneStrain` / any wrapper — sits on an element measured to discard material return codes. It is
keyed on a new tri-state `_ElemSpec.propagates_material_refusal` (`True`: `LadrunoBrick`,
`TenNodeTetrahedron`; `False`: `stdBrick`; `None`: unmeasured, silent), not on element names.
`BrickUP` / `QuadUP` are in the fork's B2 list but apeGmsh has no typed element for them, so there
is nothing to mark. A warning, not a raise: a vanilla-host deck is legal and was the SSI-1 default;
it is the fail-loud contract that never reaches it.

## 6. The two helpers — DONE

`MohrCoulombTensionCutoffSoil(c, phi, psi, tension_cutoff, E, nu, …)` — the fork's ADR-84
composite, Cerro Lindo's material. `tension_cutoff` is `TC_min_stress`: a Rankine limit on the
major principal stress, **tension positive**, `≥ 0`; the fork caps the effective cut-off at the
Mohr–Coulomb apex `min(tension_cutoff, c·cot φ)`.

`HoekBrownRock(E, nu, sigci, mb, s, a, mb_psi=None, ds=0.0, …)` — `mb_psi=None` means `mb`
(associated). Deriving `mb, s, a` from `mi, GSI, D` is the caller's job; the fork's
`HoekBrown_Utils.h` formulas are one-liners and are not duplicated. `HB_ds` is the numerical
derivative's perturbation; the fork's own rigs use `0.0`.

No DruckerPrager / VonMises helpers — the generic class covers them (the live battery's two-cube
model drives a VonMises combination through it).

## 7. What did *not* change

The four-string dispatch header and the three keyed blocks; the `PlaneStrain` wrapper;
`InitialP0` and the `setParameter` initial-stress workflows; the readers.

**Recorder side — the `material.` prefix.** Plastic strain and the other ASDP scalars are
**material-level** responses. The fork recorder forwards a dotted `material.<token>` request to
every Gauss point's material (`LadrunoRecorder.cpp` splits it into `material <k> <token>` and
iterates `k`); the material tags the columns (`epsP11 … epsP13`, `eqpstrain`, `p`, `J2stress`,
`epsVol`, `J2strain`, `BackStress_1..6`) and the reader lands `material.pstrain` on
`plastic_strain_*` and `material.eqpstrain` on `equivalent_plastic_strain` — verified
engine-checked in `test_ladruno_gauss_generic_columns.py`. A **bare** `pstrain` token never
reaches the material and records nothing, silently (the fork's own `setResponse` comment says
so). `ops.recorder.Ladruno` / `MPCO` now refuse the bare spelling of the known material-only
tokens (`pstrain`, `pstrains`, `eqpstrain`, `PStress`, `J2Stress`, `VolStrain`, `J2Strain`) at
construction, naming the `material.<token>` form. Resolved since, in ADR 0105 Amendment 1: the
reader now keeps those five buckets, mapped by BUCKET TOKEN (never by column label —
`material.PStress` labels its column `p`, which collides case-insensitively with the section
axial force `P`): `material.PStress` → `material_mean_stress`, `material.J2Stress` →
`material_j2_stress`, `material.VolStrain` → `material_volumetric_strain`,
`material.J2Strain` → `material_j2_strain`, `material.BackStress` →
`back_stress_{xx,yy,zz,xy,yz,xz}` by column position — kept provenance-distinct from the
tensor-derived `mean_stress` / `j2_stress` / `volumetric_strain` / `j2_strain` (measured equal
on build `3622d6214`, but that is a measurement, not a contract). The scalar internal-variable
buckets (`material.YieldStress`, `material.DP_cohesion`, `material.CapPressure`,
`material.EpsQpShear`) map too, and a `material.<Token>` bucket nothing can name now raises a
`GaussColumnDroppedWarning` naming the bucket instead of vanishing.

## 8. Goldens — what actually moved

- `_api_index.json` (the committed signature harvest, gated in two CI lanes) — rebuilt with
  `python -m apeGmsh.studio.lookup --build`.
- No golden deck under `tests/opensees` carried a `MohrCoulombSoil` line, so no golden file
  moved. Every emitted `MohrCoulombSoil` deck did: 8 parameter lines instead of 21,
  `f_relative_tol 0.0`, `strict_convergence 1`, `tangent_type Continuum`.
- Unit tests that built the generic class with a partial MC parameter block now supply the full
  schema (they were exercising the old silent-zero contract).
- A recorder deck that asked for a bare `pstrain` element token was recording nothing; it is now
  refused at construction — spell it `material.pstrain`.

## 9. Measurements — the live battery (`test_asdplastic_live.py`, 7/7 on `3622d6214`)

Each case is a fresh subprocess with `stdin=DEVNULL`; the child prints `ops.ladrunoBuild()`.
Driver: one apeGmsh-meshed unit hex, 1/8-symmetry restraints, `sp`-prescribed normal strains
under `LoadControl`, `system UmfPack`.

| # | case | measured |
|---|---|---|
| 1 | `MohrCoulombSoil` (schema deck) on `LadrunoBrick`, deviatoric leg `ε = (+0.002, +0.003, −0.010)` in 20 steps | 20/20; `|f_MC|` from the committed stress ≤ **1.03e-8** every step (tolerance 1e-6); last state `σ = (−3218, −3245, −10000)` kPa on the surface |
| 2 | pre-ADR-0105 superset deck (fixture) | refused: `unknown model parameter 'AF_cr'` … `REJECTED … (ADR-94)` |
| 3 | schema deck without `MC_phi` | refused: `2 required model parameter(s) were never given a value: MC_phi, MC_phi` (listed once per declaring component — YF and PF both declare it) |
| 4 | `strict_convergence=True`, same MC problem ×1e9 under the absolute tolerance | `LadrunoBrick` refused at step 1 (`analyze() = −3`, "REFUSED the trial strain"); `stdBrick` **20/20**; the D4 gate warned on the stdBrick deck at build time |
| 5 | `HoekBrownRock`, σci 50 MPa, mi 10, GSI 60, D 0 (`mb` 2.3965, `s` 0.011744, `a` 0.50284), uniaxial-stress tension to 3·εt in 60 steps | strict **off**: last committed σxx = **245.0152 kPa** = `s·σci/mb` to 7 figures, next step refused at the corner; strict **on**: the corner step is refused, last committed 232.76 kPa (0.95 σt), nothing above σt ever committed |
| 6 | `f_relative_tol = 1e-7`, same MC problem in kPa and ×1e9 | both 20/20; final stresses equal to 1e-6 relative after the ×1e9 scale |
| 7 | two disjoint VonMises cubes (fork rig: E 70000, ν 0.3, σy 30, H 7000; loads −60 / −10), 4 load steps, `NormDispIncr 1e-9` | Newton iterations **Continuum 72, Secant 139** (1.9×); stresses equal to 1e-7 relative |

Two things measured on the way that shape the deck rules below:

- With `MC_ds = 0` the same deviatoric leg is refused at step 17 on the triaxial-compression
  corner; the helper default `1e-5` completes it. Keep the rounding.
- A uniaxial-stress compression leg (lateral faces free) stalls the **global** Newton at the
  corner (`NormDispIncr` floor 2.8e-6 against 1e-10) — a global-solver fact, not a material
  refusal; the prescribed-strain drivers the fork uses do not have that problem.
- A starved `n_max_iterations = 1` does **not** provoke a refusal on the confined path
  (Backward_Euler converges it in one iteration); the ×1e9 deck is the reproducer.

## 10. Deck rules

1. **Exact schema.** Emit the combination's parameter names and nothing else; use the helpers.
   A hand-built generic deck for a table-covered combination is validated at construction.
2. **`strict_convergence` on** (the default). Expect decks that used to finish to fail at a step;
   read the fork's "REFUSED the trial strain" line, then cut the step or fix the deck.
3. **`Continuum` tangent** (the default). Same answer, fewer iterations; `Secant` is available.
4. **Host on `LadrunoBrick` or `TenNodeTetrahedron`** for a fail-loud deck. `stdBrick` swallows
   every material refusal — the build warns.
5. **Rock scale: set `f_relative_tol`** (`1e-8` is the fork's suggested start for Hoek–Brown at
   50 MPa; `1e-7` completed the MC problem at ×1e9 in the battery).
6. **Keep `MC_ds > 0`** on Mohr–Coulomb decks that reach a corner.
7. **Implicit only** — `Backward_Euler` (default) or `Closest_Point` (fork build
   `7e93e4381`+; pair it with `tangent_type="Algorithmic"`, `algorithm KrylovNewton` and an
   unsymmetric solver). The four explicit schemes warn and the fork now refuses them without
   `experimental_integrator=1`; the two refused ones raise.
8. **Plastic strain is `material.pstrain`** (and `material.eqpstrain`) in `elem_responses`; the
   bare token is refused because the fork records nothing for it. Same rule for
   **`material.cp_iterations`** (ADR 0107): a recorder token, NOT an `eleResponse` —
   `eleResponse(tag, "cp_iterations")` returns an empty list. It reads 0 while the Gauss point
   is elastic and 1 once MC yields, and is meaningless unless the deck selected `Closest_Point`.
9. **Minimum fork build `bbf657d49`**; older parsers silently drop `strict_convergence` and
   `f_relative_tol`. The battery prints the build hash it ran against.
