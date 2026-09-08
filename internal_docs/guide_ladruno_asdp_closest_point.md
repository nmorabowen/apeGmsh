# Ladruno ASDPlasticMaterial3D closest-point return map (ADR-97) — what it changes for the emitter

Working memory for `ASDPlasticMaterial3D` / `MohrCoulombSoil` after the Ladruno
fork's ADR-97 landed on `ladruno` (2026-09-08, closeout build `7e93e4381`,
`ops.ladrunoBuild()` re-verified on the banner build `33c670bf5`). Everything
here is about what apeGmsh **emits**; the fork shipped this strictly opt-in,
so no existing apeGmsh deck changes behaviour by upgrading the fork build.

Fork-side sources, if you need to check a claim:
`Ladruno_implementation/97_ladruno_asdp_closest_point_adr.md` (plan +
decisions D1–D6 + implementation log), `Ladruno_implementation/reviews/
adr97_verdict.md` (§0 verdict, §3 the P6 measurement + default-flip
recommendation, §4 owner items, §5 residual risks), `LEDGER_implementations.md`
(the four ADR-97 rows), `LEDGER_quirks.md` (Hoek-Brown / Mohr-Coulomb
header-bug entries), the parser itself:
`SRC/material/nD/ASDPlasticMaterial3D/OPS_AllASDPlasticMaterial3Ds.cpp`. PR
stack: #817 (plan+oracles) → #819 (P1, smooth families) → #824 (P2,
Mohr-Coulomb) → #825 (P3, Hoek-Brown) → #827 (P5, explicit-integrator gate +
StiffSoil fixes) → #829 (P4, `Numerical_Algorithmic_*` re-point) → #826 (P6,
mesh-scale measurement) → the P7 closeout PR (verdict + ledgers + banner
line). This guide is scoped to ADR-97 only; the broader ADR-94 fail-loud
parser adoption (unknown tokens, required parameters, `strict_convergence`,
`f_relative_tol`) is a separate apeGmsh change tracked in the proposed
`0104-asdplastic-deck-contract-after-ladruno-adr94.md` ADR — read it alongside
this one before touching `material/nd.py`, since both land in the same file
and the same `Begin_Integration_Options` block (§9).

## Adoption status — ADOPTED as ADR 0107 (2026-09-08)

`Closest_Point` / `Algorithmic` are accepted values on
`ASDPlasticMaterial3D` and all three D5 helpers, and
`material.cp_iterations` is a readable Gauss component; see
`src/apeGmsh/opensees/architecture/decisions/0107-asdplastic-closest-point-return-map.md`.
**Verified live, 4/4, on fork build `ff47275fd`** (the `origin/ladruno`
tip). Measured: CP commits at `max|f_MC|` 8.24e-13 against
`Backward_Euler`'s 1.03e-08 at equal iterations; the maps agree to 0.82 %,
differing only where CP pins the exact MC corner BE rounds.

**The finding this guide does not have** (§0 says CP "converges strictly
more of a load history"): apeGmsh's own `strict_convergence=True` default
REFUSES a CP leg at ordinary kPa soil scale. The fork tests a TRIAL
state's residual against the ABSOLUTE `f_absolute_tol`, a mid-Newton
trial reaches `f = 1.37e-06` against the `1e-06` default (~1.6e-08
relative), and the step is rejected — `analyze() == -3` at step 1 of 20.
The MORE accurate map trips a threshold the coarser one misses. Set
`f_relative_tol`, or strict off. Pinned in
`case_strict_refuses_closest_point`.

Four things below were written against a base 30 commits stale and are
now WRONG — corrected here rather than edited in place, so the fork-side
record stays as it was handed over:

- **§9's prerequisite is satisfied.** The "proposed ADR 0104" shipped as
  **ADR 0105** (PRs #1110–#1115). Its D3 already built the client-side
  enum validation §5 item 1 asks for; this adoption only extends the
  tables.
- **§0 / §4's "`Secant` (apeGmsh's emitted default)" is stale.** ADR 0105
  moved every helper's default to `Continuum` on measurement. The
  57/80/103 % tangent-error figures still stand; the one that describes
  our decks is `Continuum`'s **57 %**, not `Secant`'s 80 %.
- **§5 item 1's "`MohrCoulombSoil` (`:1191-1328`)" is one of three.**
  `MohrCoulombTensionCutoffSoil` and `HoekBrownRock` also exist (ADR 0105
  D5) and are also matched YF/PF pairs, so all three support the map.
  Their docstrings delegate to `MohrCoulombSoil`'s, so one edit covers
  all three; the `_internal/ns/nd.py` mirror delegates too, making §5
  item 2 a genuine no-op.
- **§5 item 4's premise is false, but `cp_iterations` shipped anyway.**
  The results layer does NOT expose an arbitrary named material response
  generically: `_MATERIAL_BUCKET_TOKENS` is a closed table (ADR 0105
  Amendment 1) and an unmapped bucket is dropped with a warning. It took
  one table entry, whose shape was OBSERVED rather than guessed — a
  scalar per Gauss point (`NUM_COMP` 1, `MULTIPLICITY` 1, `FIBER_ID` -1,
  one column per `GAUSS_ID`), reading 0 while elastic and 1 once MC
  yields. §6's "per-Gauss-point response" is right; its `ops.eleResponse(
  1, "cp_iterations")` example in §8 is NOT — that returns an empty list.
  The token only reaches you through a recorder, as
  `elem_responses=("material.cp_iterations",)`, the same
  material-level rule ADR 0105 found for `pstrain`.

One thing the guide does not mention that mattered: `__post_init__`
warned on `integration_method != "Backward_Euler"`, so `Closest_Point`
would have raised a bogus "experimental, no drift correction" warning.
The warning is now keyed on the explicit four (ADR 0107 D3).

## 0. One-paragraph summary

`Backward_Euler` (apeGmsh's only integrator today, via `MohrCoulombSoil`'s
default and every generic `ASDPlasticMaterial3D` call) is an Ortiz–Simo
cutting-plane map, and no tangent the fork ever shipped for it was the tangent
of that map — measured against a central difference of the material's own
committed response: `Continuum` 57 % off, `Secant` (apeGmsh's emitted
default) 80 % off, `Elastic` 103 % off. ADR-97 adds a second, fully implicit
integrator, `Closest_Point`, with its own exact consistent tangent,
`tangent_type Algorithmic`. It is a **new opt-in pair**, not a change to
`Backward_Euler`'s behaviour: every existing `MohrCoulombSoil`/
`ASDPlasticMaterial3D` deck apeGmsh emits today is byte-identical on the new
fork build. Coverage is **23 of the 46 registered YF/PF/hardening
specializations** — VonMises, Drucker–Prager (including the apex),
Mohr-Coulomb, MohrCoulombTensionCutoff and Hoek-Brown, each only where the
yield function and the plastic-flow direction are the SAME family. The fork's
own mesh-scale measurement (P6) found `Closest_Point` converges strictly more
of a bearing-capacity load history than `Backward_Euler`/`Secant` at
equal-or-fewer Newton iterations — on a 24000-DOF strip footing, 12/12 push
steps in 78 s vs 7/12 steps in 504 s — and **recommends** flipping the fork's
shipped default. That flip is explicitly the owner's decision in a separate
fork PR, not part of this warrant, so apeGmsh should not anticipate it
either.

## 1. What shipped on the fork

| change | deck surface | default |
|---|---|---|
| new integrator `Closest_Point` | `integration_method Closest_Point` | fork default stays `Backward_Euler` |
| new consistent tangent `Algorithmic` | `tangent_type Algorithmic` | fork default stays `Secant`; refused unless paired with `Closest_Point` |
| explicit-integrator opt-in gate | `experimental_integrator 1` | default `0` = the four explicit methods refused |
| `Numerical_Algorithmic_{First,Second}Order` re-pointed at the actual committed map | none (value name unchanged) | now differentiates whichever `integration_method` is configured, instead of an unrelated third map |
| `cp_iterations` material response | response token `cp_iterations` | new; local-Newton iteration count of the last `Closest_Point` solve (0 elastic, 1 for the planar MC/MCTC/VM/DP families, up to 5 for Hoek-Brown's curved surface) |

Nothing above touches `Backward_Euler` (D1) — `MohrCoulombSoil` and every
hand-built `ASDPlasticMaterial3D` deck apeGmsh emits today are unaffected
until a caller explicitly asks for the new tokens.

## 2. The exact option syntax apeGmsh must be able to emit

`ASDPlasticMaterial3D.integration_options` (`material/nd.py:1120`) already
passes an arbitrary `(name, value)` sequence straight through — the two new
tokens need no new emitter code, only new accepted values on `str` fields
that are documented today as "see the OpenSees source for valid tokens"
(`material/nd.py:1108`) and on `MohrCoulombSoil`'s keyword docstrings
(`material/nd.py:1234-1242`), which currently do not list `Closest_Point`/
`Algorithmic` at all.

**Tcl** (verbatim card shape, matching the fork parser and
`ASDPlasticMaterial3D._emit`):

```tcl
nDMaterial ASDPlasticMaterial3D 1 MohrCoulomb_YF MohrCoulomb_PF LinearIsotropic3D_EL \
    "BackStress(NullHardeningTensorFunction):" \
    Begin_Internal_Variables \
        BackStress 0.0 0.0 0.0 0.0 0.0 0.0 \
    End_Internal_Variables \
    Begin_Model_Parameters \
        YoungsModulus 30000.0 PoissonsRatio 0.3 \
        MC_phi 32.0 MC_c 15.0 MC_psi 8.0 MC_ds 0.0 MassDensity 0.0 \
    End_Model_Parameters \
    Begin_Integration_Options \
        integration_method Closest_Point \
        tangent_type Algorithmic \
        n_max_iterations 150 \
    End_Integration_Options
```

**openseespy / apeGmsh's `LiveOpsEmitter`** takes the identical flat
positional sequence — this material has no Python-native kwargs form on the
OpenSees side, so `_emit` (`material/nd.py:1144`) needs no Tcl/Python branch
for this change; it is already emitter-agnostic.

**The explicit-integrator opt-in** (only relevant if apeGmsh ever exposes the
four gated methods — it does not plan to, §4):

```
Begin_Integration_Options
    integration_method Forward_Euler
    experimental_integrator 1
End_Integration_Options
```

Without `experimental_integrator 1`, `Forward_Euler`,
`Forward_Euler_Subincrement`, `Modified_Euler_Error_Control` and
`Runge_Kutta_45_Error_Control` are refused (`Backward_Euler_LineSearch` and
`Runge_Kutta_45_Error_Control_old` stay refused outright — an ADR-94 decision
ADR-97 did not revisit).

### Refusal rules a user will hit, in the parser's own wording

`tangent_type Algorithmic` without `integration_method Closest_Point`:

> `tangent_type 'Algorithmic' is the exact consistent tangent of the
> 'Closest_Point' return map and is REFUSED with any other
> integration_method (ADR-97 D2). Set 'integration_method Closest_Point', or
> pick a tangent_type that the chosen integrator defines: Secant, Continuum,
> Elastic, Numerical_Algorithmic_FirstOrder, Numerical_Algorithmic_SecondOrder.`

`integration_method Closest_Point` on an unsupported specialization (mixed
YF/PF pairing, StiffSoil, …):

> `integration_method 'Closest_Point' is not available for this model (YF
> <name>, PF <name>, IV <name>). ADR-97 D3: the closest-point map is added
> family by family. … a mixed pairing (MohrCoulomb_YF with VonMises_PF or
> DruckerPrager_PF, … HoekBrown_PF with anything) is verified by no oracle
> and stays refused. … StiffSoil is P5. Use Backward_Euler.`

An explicit method without the opt-in:

> `integration_method is an EXPLICIT integrator with no active drift
> correction … it is REFUSED by default (ADR-97 D5). The supported implicit
> pair is Backward_Euler and Closest_Point. Set 'experimental_integrator 1'
> … to opt in anyway.`

Both refusal classes name the offending token and cite the ADR — exactly the
information apeGmsh needs to surface verbatim to a user (§3).

## 3. Support matrix — what `Closest_Point` accepts

**Accepted (23 of 46 registered specializations), matched-pair only:**

| Family | Rule |
|---|---|
| VonMises | `VonMises_YF` × `VonMises_PF` |
| Drucker–Prager | `DruckerPrager_YF` × `DruckerPrager_PF`, including the apex |
| Mohr-Coulomb | `MohrCoulomb_YF` × `MohrCoulomb_PF` only |
| MohrCoulombTensionCutoff | `MohrCoulombTensionCutoff_YF` × `_PF` only (reuses ADR-84's `special_return` geometry) |
| Hoek-Brown | `HoekBrown_YF` × `HoekBrown_PF` only — the ONE registered pairing where both functors of the family exist |

**Refused at parse time, unconditionally:** StiffSoil (`StiffSoilCap` and
`StiffSoilShear` — a two-surface composite active set the fork has not
built); `RoundedMohrCoulomb` (not even compiled into `AllYieldFunctions.h`);
every **mixed** YF/PF pairing across families (`MohrCoulomb_YF` is
registered in 7 specializations and `MohrCoulomb_PF` in 6, but only ONE
combination has both — the other six are covered by no oracle and stay
refused; same shape for Hoek-Brown, 7/6/1/11); every explicit integrator
without `experimental_integrator 1` (orthogonal to family support, applies
to `Backward_Euler` too).

**Recommendation: do not duplicate the refusal logic in apeGmsh.** The
family/pairing rule depends on compile-time markers inside the fork's
templated `ASDPlasticMaterial3D<YF,PF,EL,IV>` instantiations, which apeGmsh
cannot introspect from Python; the fork's own parser already fails loud
naming the exact YF/PF/IV and citing ADR-97 D3. The one client-side check
worth adding — because it is a static fact about the deck's own two strings,
not about the templated instantiation — is the D2 cross-check:
`tangent_type == "Algorithmic"` requires `integration_method ==
"Closest_Point"`, raised as a `ValueError` in `__post_init__` before ever
reaching a live fork process. Everything else (family/pairing support,
StiffSoil, RoundedMC) should surface the fork's `opserr` text unaltered
rather than be reimplemented — a Python enum table would have to be kept in
sync with a C++ template registry it cannot see, the same lesson ADR 0104 D8
already draws for the ADR-94 parameter schema.

## 4. Recommended defaults for apeGmsh's typed primitive

**Keep the fork's shipped defaults until the fork flips them.**
`MohrCoulombSoil`'s `integration_method: str = "Backward_Euler"` and
`tangent_type: str = "Secant"` (`material/nd.py:1202-1203`) stay exactly as
they are. The fork's own verdict is explicit that the default flip is **not**
part of this warrant — scoped to the 23/46 supported specializations,
measured on one geometry/mesh/solver combination, and stated as "the owner's
decision, a separate PR". apeGmsh changing ITS default ahead of the fork
would silently change behaviour for every existing `MohrCoulombSoil` caller
on a fork build that has not made that decision either.

**Expose `integration_method` and `tangent_type` as first-class typed
fields with the new enum values** — additive, not a signature change:
`integration_method` gains `"Closest_Point"`; `tangent_type` gains
`"Algorithmic"`. Do not add an `experimental_integrator` field speculatively
— no current apeGmsh consumer asks for the four gated explicit methods
(CLAUDE.md §2).

**One-line recommendation for new models:** for a non-associated
Mohr-Coulomb/MCTC/Hoek-Brown/VonMises/Drucker–Prager deck, prefer
`integration_method="Closest_Point"`, `tangent_type="Algorithmic"`,
`algorithm KrylovNewton` (already the fork's default algorithm), and an
unsymmetric solver (`UmfPack` or `Pardiso -matrixType 0`) — the exact
configuration the fork's P6 measurement found fastest by wall clock and most
robust to convergence at every scale/family it tried (§6, §8). This is
docstring-example guidance, not a new default.

## 5. Required apeGmsh code changes — checklist

1. **`src/apeGmsh/opensees/material/nd.py`**
   - `ASDPlasticMaterial3D` docstring (`:1103-1111`): list `Closest_Point` in
     `integration_method` and `Algorithmic` in `tangent_type`.
   - `ASDPlasticMaterial3D.__post_init__` (`:1122-1142`): add the D2
     cross-check from §3 — scan `integration_options` (a tuple of
     `(name, value)` pairs in caller-chosen order) for `tangent_type ==
     "Algorithmic"` without `integration_method == "Closest_Point"` and raise
     `ValueError` quoting the fork's own reasoning.
   - `MohrCoulombSoil` (`:1191-1328`): extend the `integration_method`/
     `tangent_type` docstrings (`:1234-1242`); no signature change, the
     values already flow straight into `integration_options`.
   - A `HoekBrownRock` helper (already proposed independently in ADR 0104
     D5, not ADR-97-specific) would be a natural place to default to
     `Closest_Point`/`Algorithmic`, since HB's `Backward_Euler` cannot even
     converge the fork's own oedometric test deck (§7).

2. **`src/apeGmsh/opensees/_internal/ns/nd.py`** — `ASDPlasticMaterial3D`
   (`:349-401`) and `MohrCoulombSoil` (`:808-`): no signature change, values
   pass straight through the existing dict → tuple conversion (`:389-392`);
   only docstrings need updating.

3. **`src/apeGmsh/opensees/_internal/build.py`** — optional, orthogonal
   build-time gate: extend whatever swallowing-host gate lands under ADR
   0104 D4 (proposed, not implemented — no `ASDPlasticMaterial3D`-aware gate
   exists in `build.py` today) so it also fires for `Closest_Point` decks;
   `LadrunoBrick`/`TenNodeTetrahedron` propagate a material refusal,
   `stdBrick`/`BrickUP`/`QuadUP` swallow it, for both ADR-94's and ADR-97's
   refusal paths alike (both return `LADRUNO_MATERIAL_REFUSED`) — one shared
   gate, not a second ADR-97-only one.

4. **A `cp_iterations` response hook** — only if apeGmsh's results layer
   already exposes an arbitrary named material response generically (check
   `opensees/_internal/typed_records.py` around line 618 and
   `results/readers/_ladruno_element_io.py`, which today maps only
   `pstrain`/`plastic_strain_*`). If it does, `cp_iterations` needs no new
   code, only documentation that the token exists (§6). If it does not, file
   it as its own ask rather than folding it into this change.

5. **Tests to add**, mirroring the SANISAND precedent
   (`tests/opensees/unit/test_asd_plastic_material_3d.py` /
   `tests/opensees/integration_ladruno/test_ladruno_sanisand_live.py`):
   - **Unit / emit-roundtrip** (no fork needed): the D2 cross-check raises
     with the right message; `MohrCoulombSoil(integration_method=
     "Closest_Point", tangent_type="Algorithmic")` emits the expected Tcl
     card via the existing `str` branch of `_emit` (`:1162-1170`).
   - **Live `ladruno_fork` smoke test**, new
     `tests/opensees/integration_ladruno/test_asdplastic_closest_point_live.py`
     (same shape as `test_ladruno_sanisand_live.py`: module docstring naming
     what each test proves, `pytestmark = pytest.mark.ladruno_fork`, run via
     `tests/run_ladruno_integration.ps1` / `APEGMSH_OPENSEES_BIN`): (a) a
     `MohrCoulombSoil` `Closest_Point`+`Algorithmic` deck on `LadrunoBrick`
     converges a triaxial leg the `Backward_Euler`/`Secant` default also
     converges, committed stresses agreeing within the fork's measured
     sub-1 % gap; (b) `tangent_type="Algorithmic"` with `integration_method=
     "Backward_Euler"` is refused client-side (the new `ValueError`) and, via
     the generic primitive directly, server-side (the fork's `opserr`,
     captured with the `capfd`-cannot-see-a-`.pyd` subprocess idiom the fork
     and `test_ladruno_sanisand_live.py` both use); (c) `cp_iterations` reads
     0 on an elastic step and nonzero once the deck yields.

## 6. Analysis-chain guidance

- **Unsymmetric solver is required for non-associated flow** — the same
  ADR-80/ADR-0103 lesson as SANISAND, and a general ASDP fact, not new to
  ADR-97. `MohrCoulombSoil` decks are non-associated whenever `psi < phi`
  (the physically normal case); use `UmfPack` or `Pardiso -matrixType 0`,
  never `ProfileSPD`/`SProfileSPD`/`Pardiso -matrixType 1|2`.
- **`algorithm KrylovNewton`** is the fork's already-default algorithm and
  was fastest by wall clock at every scale/family P6 measured
  (`CP_Algorithmic_Krylov`); plain `Newton` converges the same steps but
  loses the wall-clock edge.
- **`test NormDispIncr` tolerances**: the fork's measurement used `test
  NormDispIncr 1.0e-8 250 0` throughout. Under that exact tolerance
  `Closest_Point` converged 10/10, 12/12, 9/10, 11/12 (small/cerro ×
  MC/MCTC) steps where `Backward_Euler`/`Secant` converged only 7/10, 7/12,
  6/10, 6/12 of the same load history — the MAP determines whether a step
  converges at all, not the tolerance.
- **Reading `cp_iterations`**: a per-Gauss-point response — the local-Newton
  iteration count of the LAST `Closest_Point` solve there. `0` elastic, `1`
  on a plastic step for the planar families (MC, MCTC, VM, DP), up to `5`
  for Hoek-Brown's curved surface. Meaningless (undefined) under
  `Backward_Euler` — do not read it on a deck that has not selected
  `Closest_Point`.
- **The byte-identity promise**: any apeGmsh deck emitting the current
  defaults (`Backward_Euler` + `Secant`, no `experimental_integrator` token)
  is unaffected on a fork build at or after `7e93e4381` — pinned at 23 decks
  / 282 committed-stress rows in fresh subprocesses, with exactly one
  documented, root-caused, sub-1e-8 exception on a load-controlled free-DOF
  rig exercising `Numerical_Algorithmic_FirstOrder` specifically (not
  `Closest_Point`/`Algorithmic`, and not anything a `MohrCoulombSoil` caller
  reaches without asking for that tangent option).

## 7. Fork caveats users may hit

These affect **`Backward_Euler` decks only** — every existing apeGmsh deck —
and are handed to the fork's owner as separate follow-on PRs (ADR-97 verdict
§4), not fixed under this ADR's D1 byte-identity promise:

- **Hoek-Brown flow direction is a Tresca potential, not the declared
  frame-consistent one.** `HoekBrown_PF::g` is evaluated in the un-negated
  frame while `HoekBrown_YF` negates first, so `g` collapses to Tresca:
  `HB_mb_psi` is inert, flow is exactly non-dilatant. Quantified: plastic
  `eps_vol` +2.208e-05 (frame-consistent, what `Closest_Point` uses) vs
  `Backward_Euler`'s +2.09e-13 — a 5.91e-2 relative stress gap, 26.2 % of
  strength scale. Only matters for `mb_psi != mb`; `Closest_Point` does not
  share the bug.
- **Mohr-Coulomb's shipped analytic Lode-angle gradient is wrong** at
  `MC_ds = 0` (error 2.9e-1, step-size dependent) and agrees with the exact
  return only through the header's own finite difference (`MC_ds > 0`, error
  2.1e-14). Affects `Backward_Euler`'s iteration count, not the converged
  stress. `MohrCoulombSoil` already defaults `ds=1e-5` (`material/nd.py:
  1199`), so apeGmsh callers are on the better-behaved branch by accident.
- **Drucker–Prager cohesion hardening is inert.** The cohesion IV is
  commented out of `f` itself while its hardening derivative is not — a
  DP-with-cohesion-hardening deck is perfectly plastic under `Closest_Point`
  but appears to harden under `Backward_Euler`. Only relevant to a DP deck
  declaring scalar cohesion hardening; apeGmsh has no such helper today.
- **Apex tangent is rank 0 (or rank 1, bulk-only under hardening) by
  construction**, for DP, MC and HB alike — a structural fact about the map,
  not a defect. Do not expect a nonzero deviatoric apex tangent.
- **StiffSoil stays refused under `Closest_Point`** (both `StiffSoilCap` and
  `StiffSoilShear`); `StiffSoil_EL` received two unrelated NaN fixes under
  `Backward_Euler` in ADR-97 P5, orthogonal to `Closest_Point` support.

## 8. Minimal end-to-end example (openseespy, runs on the fork today)

```python
import opensees.openseespy as ops

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 3)

for tag, coords in enumerate([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
], start=1):
    ops.node(tag, *coords)
for tag in (1, 2, 3, 4):
    ops.fix(tag, 1, 1, 1)

ops.nDMaterial(
    "ASDPlasticMaterial3D", 1,
    "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL",
    "BackStress(NullHardeningTensorFunction):",
    "Begin_Internal_Variables",
    "BackStress", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    "End_Internal_Variables",
    "Begin_Model_Parameters",
    "YoungsModulus", 30000.0, "PoissonsRatio", 0.3,
    "MC_phi", 32.0, "MC_c", 15.0, "MC_psi", 8.0, "MC_ds", 1.0e-5,
    "MassDensity", 0.0,
    "End_Model_Parameters",
    "Begin_Integration_Options",
    "integration_method", "Closest_Point",
    "tangent_type", "Algorithmic",
    "n_max_iterations", 150,
    "End_Integration_Options",
)

ops.element("LadrunoBrick", 1, 1, 2, 3, 4, 5, 6, 7, 8, 1)

ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
for n in (5, 6, 7, 8):
    ops.load(n, 0.0, 0.0, -100.0)

ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("UmfPack")
ops.test("NormDispIncr", 1.0e-8, 50, 0)
ops.algorithm("KrylovNewton")
ops.integrator("LoadControl", 0.1)
ops.analysis("Static")

for step in range(10):
    rc = ops.analyze(1)
    assert rc == 0, "step %d failed to converge" % step
    print(step, ops.testIter(), ops.eleResponse(1, "cp_iterations"))
```

Single-hex, fully-prescribed sanity check, not a P6-scale measurement — it
exercises the exact deck shape `MohrCoulombSoil(integration_method=
"Closest_Point", tangent_type="Algorithmic")` would emit once §5 lands, on
`LadrunoBrick`, one of the two hosts that propagate a material refusal
instead of swallowing it (§6, §7).

## 9. Relationship to the ADR-94 deck contract (ADR 0104)

ADR-97 and the proposed ADR 0104 both touch `ASDPlasticMaterial3D`/
`MohrCoulombSoil` in the same file, but answer different fork changes: ADR
0104 is about the ADR-94 fail-loud PARSER (unknown tokens and missing
required parameters now abort the command — breaks `MohrCoulombSoil`'s
current blanket 21-parameter emit, independent of `Closest_Point`), and its
D3 already proposes the client-side `integration_method`/`tangent_type` enum
validation §5 item 1 extends. **Implement them together, ADR 0104 first,
then this guide's §5** — both touch `__post_init__` and the same
`model_parameters`/`integration_options` construction, and validating
against one combined enum (ADR-94's five values + ADR-97's two) avoids
touching `__post_init__` twice. If ADR 0104 has not landed yet,
`Closest_Point`/`Algorithmic` support can still ship standalone — it needs
none of ADR 0104's schema machinery, only the D2 cross-check in §3/§5.
