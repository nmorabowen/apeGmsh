# ADR 0105 — The ASDPlasticMaterial3D deck contract after fork ADR-94

**Status:** Accepted (2026-09-07). The live `ladruno_fork` battery in §D7
is 7/7 green against fork build
`3622d6214ef4cdeb8cf65a102ee35f6cd9973337` (the ADR-94 closeout,
ASDP-equivalent to `bbf657d49`, ladruno, 2026-09-07); measurements in
`internal_docs/guide_ladruno_asdplastic.md` §9. Numbered 0105: 0104 was
taken by the Substep controller ADR (#1107) after this draft was written.

**Extends:** the Phase SSI-1 primitives in
[`material/nd.py`](../../material/nd.py) — `ASDPlasticMaterial3D`,
`MohrCoulombSoil`, `PlaneStrain` — which were typed against the
pre-ADR-94 parser. Sibling of [ADR 0103](0103-sanisand-integrator-deck-contract.md)
(same shape: a fork integrator change becomes an apeGmsh deck contract).

**Fork-side sources** (all in `Ladruno_implementation/` of the fork,
`github.com/nmorabowen/OpenSees`, branch `ladruno`):
`94_asdplastic_review_plan.md` (plan + implementation log),
`reviews/adr94_verdict.md` (findings B1–B5, M1–M10, fix list §7),
`_adr94_inventory.md` (the 46 registered combinations), `LEDGER_quirks.md`
(the twelve `ASDPlasticMaterial3D` entries), fork PRs #804 (review), #806
(Hoek–Brown port), #809 (fail-loud contract), #815 (per-instance state,
Voigt convention, apex, relative tolerance), #816 (closeout).

## Context

The fork reviewed and then rewrote the parts of `ASDPlasticMaterial3D`
that apeGmsh's decks touch. Five of the changes reach this codebase.

1. **The parser fails loud.** An unknown token inside
   `Begin_Integration_Options`, an unknown model-parameter name, an
   unknown internal-variable name, or an unknown `integration_method` /
   `tangent_type` / `return_to_yield_surface` value now aborts the
   `nDMaterial` command with an `opserr` message listing the valid names.
   Every model parameter of the instantiated combination except
   `MassDensity` and `InitialP0` is **required**; a missing one aborts the
   command with the list of missing names. Before ADR-94 all of this was
   silent: unknown names were dropped and unset parameters ran at zero
   (fork finding B1 — a typo'd `MC_phi` ran at φ = 0).

   **This breaks `MohrCoulombSoil` today.** The helper emits a blanket
   superset of 21 parameter names with zeros (`AF_cr`, `AF_ha`, `DP_eta`,
   `DP_etabar`, `DP_xi_c`, `Dilatancy`, `DuncanChang_*`,
   `ReferencePressure`, `ReferenceYoungsModulus`,
   `ScalarLinearHardeningParameter`, `TensorLinearHardeningParameter`,
   `TC_min_stress`, …). Twelve of them are foreign to the
   `MohrCoulomb_YF / MohrCoulomb_PF / LinearIsotropic3D_EL /
   BackStress(NullHardeningTensorFunction)` specialization, so the new
   parser refuses the deck at the first foreign name. Every SSI deck built
   through the helper, and the `ladruno_fork` live tests that use it, go
   red on the new build.

2. **Two integrators are refused by name.** `Backward_Euler_LineSearch`
   (fork M7: less robust than plain BE, ignored its own options) and
   `Runge_Kutta_45_Error_Control_old` (M8) abort the command citing ADR-94.
   The explicit schemes remain selectable but the fork documents
   `Backward_Euler` as the only supported integrator.

3. **Two options matter that the helper does not expose.**
   `strict_convergence` (int, default 0; since fork ADR-84 P2a, now
   effective on every integrator and every failure path — a non-converged
   or inadmissible state is refused instead of committed) and
   `f_relative_tol` (double, default 0 = off; when set, the convergence
   tolerance becomes `max(f_absolute_tol, f_relative_tol × strength
   scale)` so a deck behaves the same in kPa and Pa — fork M5 measured the
   same MC problem passing 20/20 in kPa and refused on step 1 in Pa with
   the absolute default).

4. **Fail-loud depends on the host element.** A refused trial strain now
   returns `LADRUNO_MATERIAL_REFUSED` from all 15 failure sites.
   `LadrunoBrick` and `TenNodeTetrahedron` propagate it and the step
   fails; `stdBrick`, `BrickUP` and `QuadUP` swallow every material return
   code (fork B2, deliberately left, pinned by a fork test). A deck that
   wants the fail-loud contract must not host `ASDPlasticMaterial3D` on
   `stdBrick`.

5. **Results moved.** Hoek–Brown in net tension now yields at the
   textbook tensile strength (`−s·σci/mb`) instead of a plateau a factor
   `mb` too high (#806). Drucker–Prager stresses change by design (its
   gradient was wrong in every convention). Mohr–Coulomb and its
   tension-cutoff composite differ from the old build at FP
   re-association level only on shear-free paths, and by ~2.5e-3
   relative on Hoek–Brown between two admissible states (the map is
   path-dependent; fork M3). `tangent_type Continuum` costs 5.3× fewer
   global iterations than the `Secant` default the helper emits (fork
   M3), and every Gauss point now gets its own tangent (fork M1), so
   iteration counts in golden files will move.

What did **not** change: the four-string dispatch header and the three
keyed blocks; the response tokens (`stress`, `strain`, `pstrain`, …) the
readers in `results/readers/_ladruno_element_io.py` already map; the
`PlaneStrain` wrapper; `InitialP0` and `setParameter` initial-stress
workflows.

## Decision

### D1 — apeGmsh carries the per-combination parameter schema and emits exactly it

The fork's `list` verb prints only the four type strings, not the
parameter names, so the schema cannot be discovered at emit time. It is
composed from the component headers as
`EL params ∪ YF params ∪ PF params ∪ hardening params (one set per IV
policy) ∪ {MassDensity, InitialP0}`:

| Component | Parameters (fork header `parameters_t`) |
|---|---|
| `LinearIsotropic3D_EL` | `YoungsModulus`, `PoissonsRatio` |
| `StiffSoil_EL` | `SS_Eur_ref`, `PoissonsRatio`, `SS_pref`, `SS_m`, `MC_phi`, `MC_c` |
| `VonMises_YF` / `VonMises_PF` | none (yield stress is the `YieldStress` IV) |
| `DruckerPrager_YF` | `DP_xi_c`, `DP_eta` |
| `DruckerPrager_PF` | `DP_etabar` |
| `MohrCoulomb_YF` | `MC_phi`, `MC_c`, `MC_ds` |
| `MohrCoulomb_PF` | `MC_phi`, `MC_c`, `MC_ds`, `MC_psi` |
| `MohrCoulombTensionCutoff_YF` / `_PF` | `MC_phi`, `MC_c`, `MC_ds`, `MC_psi`, `TC_min_stress` |
| `HoekBrown_YF` | `HB_sigci`, `HB_mb`, `HB_s`, `HB_a`, `HB_ds` |
| `HoekBrown_PF` | `HB_sigci`, `HB_mb_psi`, `HB_s`, `HB_a`, `HB_ds` |
| IV policy `TensorLinearHardeningFunction` | `TensorLinearHardeningParameter` |
| IV policy `ScalarLinearHardeningFunction` | `ScalarLinearHardeningParameter` |
| IV policy `ArmstrongFrederickHardeningFunction` | `AF_ha`, `AF_cr` |
| IV policy `NullHardening*` | none |
| always optional | `MassDensity`, `InitialP0` |

Implement as a module-level table in `material/nd.py`
(`_ASDP_PARAMS_BY_COMPONENT`) plus a resolver
`asdp_parameter_schema(yf, pf, el, iv) -> frozenset[str]` that parses
the IV string (`Name(Policy):Name(Policy):…`). `ASDPlasticMaterial3D.__post_init__`
validates when every component is in the table: a model parameter
outside the schema is a `ValueError` naming the offender and the schema
(the fork would refuse it anyway — fail at build time, not at run time);
a required parameter missing from `model_parameters` is a `ValueError`
listing the missing names. Combinations with a component outside the
table are accepted unchanged (escape hatch preserved, documented as
"the fork validates").

`MohrCoulombSoil` emits exactly its schema:
`YoungsModulus, PoissonsRatio, MC_phi, MC_c, MC_ds, MC_psi, MassDensity,
InitialP0` — nothing else. The twelve foreign names go.

### D2 — `strict_convergence` and `f_relative_tol` are first-class, with opinionated defaults

`MohrCoulombSoil` (and every helper added under D5) gains
`strict_convergence: bool = True` and `f_relative_tol: float = 0.0`.
The generic `ASDPlasticMaterial3D.integration_options` already passes
any pair through; add both names to its documented list and to the
bool/float typing branches in `_emit`.

Default `strict_convergence=True` because apeGmsh's contract with its
users is that a silently wrong deck does not exist (the same reasoning
as ADR 0103 D2/D3). It changes convergence behaviour for decks that
today commit inadmissible states — that is the point. `f_relative_tol`
stays off by default (byte-identical to the fork's default); the docstring
tells rock-scale users to set it (`1e-8` is the fork's suggested starting
value for HB at 50 MPa).

### D3 — `tangent_type` defaults to `Continuum`; refused integrators are refused client-side

`MohrCoulombSoil(tangent_type=...)` default `"Secant" → "Continuum"`,
matching fork ADR-84 §9.4 and the ADR-94 measurement (5.3× fewer
iterations; results identical at convergence). `integration_method`
stays `"Backward_Euler"`.

Client-side validation in `ASDPlasticMaterial3D.__post_init__`:
`integration_method` ∈ {`Forward_Euler`, `Forward_Euler_Subincrement`,
`Backward_Euler`, `Modified_Euler_Error_Control`,
`Runge_Kutta_45_Error_Control`}; `Backward_Euler_LineSearch` and
`Runge_Kutta_45_Error_Control_old` raise `ValueError` with the fork's
reason; `tangent_type` ∈ {`Elastic`, `Continuum`, `Secant`,
`Numerical_Algorithmic_FirstOrder`, `Numerical_Algorithmic_SecondOrder`};
`return_to_yield_surface` ∈ {`Disabled`, `One_Step_Return`,
`Iterative_Return`}. A non-`Backward_Euler` choice emits a
`warnings.warn` ("experimental on the fork after ADR-94; no active drift
correction") rather than an error, since the fork still accepts them.

### D4 — A build-time gate warns when an ASDP material sits on a swallowing host

In [`_internal/build.py`](../../_internal/build.py), next to ADR 0103's
gates: for every element whose material resolves to an
`ASDPlasticMaterial3D` (directly or through `PlaneStrain`), if the element
type is `stdBrick`, `BrickUP` or `QuadUP`, warn once per deck:
"material refusals are swallowed by <element> (fork ADR-94 B2); use
`LadrunoBrick` or `TenNodeTetrahedron` for a fail-loud deck". Not an
error: vanilla-host decks are legal and were the SSI-1 default.
`_element_capabilities.py` gets a `propagates_material_refusal` flag so
the gate does not hard-code element names.

### D5 — Two helpers, no more

`MohrCoulombTensionCutoffSoil(E, nu, c, phi, psi, tension_cutoff, …)`
(the fork's ADR-84 composite; Cerro Lindo's material) and
`HoekBrownRock(E, nu, sigci, mb, s, a, mb_psi, …)` (derived `mb, s, a`
from `mi, GSI, D` are the caller's job — the fork's `HoekBrown_Utils.h`
formulas are one-liners; do not duplicate them here). Both built the way
`MohrCoulombSoil` is: exact schema, D2 defaults, `PlaneStrain`-wrappable.
DruckerPrager and VonMises stay on the generic class — no fork consumer
asked for them (apeGmsh CLAUDE.md §2).

### D6 — Golden files and readers

Golden decks under `tests/opensees/**` that contain `MohrCoulombSoil`
output change shape (fewer parameter lines, `strict_convergence 1`,
`tangent_type Continuum`). Regenerate them in the same PR and say so in
the changelog. Readers do not change: `_ladruno_element_io.py`'s
`pstrain`/`plastic_strain_*` mapping is verified by an existing
`ladruno_fork` test (`test_ladruno_gauss_generic_columns.py`); add one
assertion there that the ADR-94 build still labels the same columns.

### D7 — Live battery (`ladruno_fork` marker) is the acceptance gate

New `tests/opensees/integration_ladruno/test_asdplastic_live.py`, each
test a fresh subprocess (the fork's `capfd` cannot see the `.pyd`'s
`cout`; pass `stdin=subprocess.DEVNULL` — both are recorded fork
quirks):

1. `MohrCoulombSoil` deck (D1 schema) is accepted by the ADR-94 parser
   and a 20-step triaxial leg on a `LadrunoBrick` cube commits admissible
   states (`|f_MC| ≤ f_absolute_tol` computed in numpy from the committed
   stress).
2. The pre-D1 superset deck (kept as a fixture string) is refused, and
   the refusal message names the first foreign parameter.
3. A deck missing `MC_phi` is refused naming `MC_phi`.
4. `strict_convergence=True` on a starved `n_max_iterations` refuses the
   step on `LadrunoBrick` (`analyze() != 0`) and is swallowed on
   `stdBrick` (`analyze() == 0`) — pins D4's rationale.
5. `HoekBrownRock` uniaxial tension on realistic parameters plateaus at
   `−s·σci/mb` within 5 % (fork wp/94d number: 245.0 kPa for σci 50 MPa,
   mi 10, GSI 60, D 0).
6. `f_relative_tol` on: the same MC problem completes in kPa and in Pa.
7. Iteration count with `Continuum` ≤ with `Secant` on the two-cube
   heterogeneous model (fork `tests/test_adr94_redblue_blue.py` rig,
   reproduced through apeGmsh).

Unit tests (no fork needed): schema resolver over the 46 registered
combinations listed in the fork's `_adr94_inventory.md`; `__post_init__`
rejections; helper emit shapes; the D4 gate.

### D8 — Fork asks (file as issues on `nmorabowen/OpenSees`, mirror in `contact_2d_fork_asks.md` style)

- `nDMaterial ASDPlasticMaterial3D <tag> list -params` (or a Python
  query) printing the parameter names per registered combination, so D1's
  table can be generated and pinned instead of hand-maintained.
- A Python-visible constant for the refused-integrator list, so D3 does
  not drift when the fork refuses more.
- `stdBrick` propagation is by design on the fork; apeGmsh keeps D4 until
  the fork decides otherwise.

## Consequences

- Every existing `MohrCoulombSoil` user gets a deck that the new fork
  accepts and the old fork also accepts (the schema is a subset of what
  was emitted; the old parser ignores nothing it needs). Users on
  pre-ADR-94 fork builds lose nothing; users who pass `strict_convergence`
  or `f_relative_tol` to an old build get the old parser's silent drop —
  document the minimum fork build (`bbf657d49`) in the helper docstrings.
- Convergence behaviour changes for decks that were committing
  inadmissible states: they now fail at the step instead of finishing
  with wrong stresses. That is a feature; the changelog says so plainly.
- Iteration signatures in golden files change (Continuum default,
  per-instance tangents). Regenerated in the PR.
- The schema table is a maintenance liability until D8's first ask
  lands; the unit test over the 46 combinations is the tripwire.

## Cross-references

- Fork: `Ladruno_implementation/reviews/adr94_verdict.md` §1 (B1, B2, M3,
  M5), §3 (fail-loud table), §7; `LEDGER_quirks.md` entries "Shear-slot
  convention: Voigt everywhere", "`f_absolute_tol` is absolute in stress
  units", "`LadrunoBrick` compares ONLY `== LADRUNO_MATERIAL_REFUSED`",
  "pytest `capfd` cannot see a native `.pyd`'s `cout`"; fork tests
  `tests/test_adr94a_fail_loud.py`, `tests/test_adr94_hlist_hb.py`.
- apeGmsh: ADR 0101 / 0103 (the SANISAND precedent for fork-driven deck
  contracts), `internal_docs/guide_ladruno_sanisand_integrator.md` (the
  measurement-log format to reuse as `guide_ladruno_asdplastic.md`),
  `tests/conftest.py` (`ladruno_fork` auto-skip), `APEGMSH_OPENSEES_BIN`.
