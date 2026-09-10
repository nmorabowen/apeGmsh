# ADR 0109 — Footfall vibration by the FRF method (AISC Design Guide 11, 2nd ed., Chapter 7)

**Status:** Proposed (2026-09-10). Study only — nothing implemented, nothing
run. The study behind it (what Robot Structural Analysis actually computes,
what the 2nd-edition Design Guide asks for, and what apeGmsh already has)
is written into the Context so the decisions can be checked against it.

**Sibling of** [ADR 0075](0075-modal-response-family.md) (the ADR-44
modal-response drivers — `eigen` → `modalProperties` → a dense
post-processor on the mode basis) — this ADR is the first *engineering
evaluation* built on that basis rather than a new solver verb. It is also
the first apeGmsh feature whose reference implementation is a commercial
program the user already drives from Python (`apeRobot`), which makes an
end-to-end cross-check possible and shapes the slices below.

**Sources.** AISC Design Guide 11, *Vibrations of Steel-Framed Structural
Systems Due to Human Activity*, 2nd ed. (2016) — Chapter 1 §1.5–1.6
(Eq 1-1, 1-4 … 1-7, Table 1-1), Chapter 2 §2.1–2.2 (Fig 2-1, Table 2-1),
Chapter 4 (Table 4-1, Table 4-2), Chapter 7 in full (Eq 7-1 … 7-10,
Tables 7-1 … 7-4, Example 7.1 with Table 7-2); PDF at
`C:\nmb\My Libraries\Libros\Codigos\AISC\Design Guide\2nd-edition-2016-Design-Guide-11-…pdf`
(Seafile, cited in place — same convention as the LATBSDC skill). The 1st
edition (1997/2003 reprint, Chapter 2 Eq 2.1–2.6, Table 2.1) sits beside it.
Autodesk: *Description of Footfall Harmonic Analysis* (RSAPRO 2023 help,
GUID-3AD2F89E), *Footfall Analysis Parameters* (GUID-33CEBBC4), support
article *Footfall Analysis Code Basis … in context of 2nd edition of AISC
Design Guide 11* (2025-07-22), *How to run response factor (footfall)
analysis* (2023-10-08). RobotOM typelib as bound by `apeRobot`
(`IRobotFootfallAnalysisParams`, `…ModalParams`, `…NodeSelection`,
`IRobotFootfallResults`, `I_CAT_DYNAMIC_FOOTFALL = 16`). Fork:
`Ladruno_implementation/44_ladruno_frequency_domain_adr.md`,
`LadrunoModalResponse_guide.md` (P2), `SRC/domain/domain/DomainModalProperties.cpp`.

## Context

### What Robot computes

Robot's *Footfall* case (`I_CAT_DYNAMIC_FOOTFALL`) is a linear-only
post-processor on a modal basis. From the help text and the COM surface:

| Robot parameter | Meaning |
|---|---|
| `ModalParams.FrequencyLimit`, `IncludeMassForDirX/Y/Z`, `IgnoreDensity` | the eigen solve feeding it — modes up to a frequency limit, with the mass directions to activate |
| `ExcitationMethod` = `SELF` / `FULL` | *self*: response read at the node the force is applied to, one solve per node; *full*: response at every response node for a force at every excitation node, "applied independently … no interaction between them" |
| `ExcitationNodes`, `ResponseNodes` (`ALL` / `SELECTED_NODES` / `NODES_BELONGING_TO_SELECTED_PANELS`) | the two node sets — in *full* mode the FRF matrix is `len(exc) × len(resp)` |
| `ExcitationForces` = `CONCRETE_CENTRE` (CCIP-016, 1.0–2.8 Hz) / `SCI_P354` (1.8–2.2 Hz; stairs 1.2–4.5 Hz) / *AISC DG11 (2003)* (1.6–2.2 Hz; UI-only, not in the enum) | which walking-force model and which acceptance metric |
| `MinWalkingFrequency`, `MaxWalkingFrequency` | the step-frequency band; "divided to 20 intervals considering additional points for the frequency of eigenvibrations" |
| `WalkersWeight`, `FootstepsNumber` | resonant branch only; the transient branch always uses a 746 N impulse |
| `Damping` (constant / Rayleigh / per-mode) | the modal damping channel |
| Frequency weighting `Wg` / `Wb` | SCI P354 only — BS 6841 weighted RMS |

Per response node it returns `Frequency` (the critical frequency),
`RF_Resonant`, `RF_Transient`, `RF_Overall`, `A` (peak acceleration),
`VRMS`, `VRMQ`, and `ExcitationNode` (the worst excitation node under
*full*). The three code options are three *evaluators on the same FRF*:

- **CCIP-016**: up to 4 walking harmonics; per harmonic the acceleration at
  every sweep frequency against the peak base curve
  `a_{R=1} = 0.0141/√f` (f < 4 Hz), `0.0071` (4–8 Hz), `2.82π·10⁻⁴ f`
  (f > 8 Hz) m/s²; `RF_resonant = √Σ_h RF_h²`. Transient: one impulse at
  the maximum step frequency, velocity history, RMS over the step period,
  `RF = v_RMS / v_{R=1}` with `v_{R=1} = 5·10⁻³/(2π f₁)` (f₁ < 8 Hz) or
  `1·10⁻⁴` m/s (f₁ ≥ 8 Hz). Both run when f₁ ≈ 8–10 Hz; overall = max.
- **SCI P354**: `RF = a_{w,rms} / 0.005` m/s² (BS 6472 / ISO 10137 z-axis).
- **AISC DG11 (2003)**: resonant only — "no results for non-permanent
  vibrations in this method"; `RF = a_p / g` against `a_o/g` = 0.5 %
  (offices, residences, churches) / 1.5 % (shopping).

Autodesk's own 2025 article answers the question this ADR started from:
Robot's footfall analysis "is based on the solutions of the **first
edition**" of the Design Guide. What the second edition adds — the
`0.09e^{−0.075f_n}` dynamic coefficient, the partial resonant build-up
factor ρ, and above all the **high-frequency effective-impulse branch with
the equivalent sinusoidal peak acceleration** — is not in Robot's AISC
option at all, and its only impulsive branch is CCIP-016's velocity metric.

### What the 2nd edition asks for (Chapter 7, walking)

1. **Model** (§7.2): floor + adjacent bays (a 3×3 grid for an interior
   bay), orthotropic shells at `1.35 E_c`, uncracked, members composite and
   continuous, spandrels ×2.5 for cladding, partitions as 2.0 kip/in./ft
   vertical springs, *actual* in-place mass, damping from Table 4-2 as a
   ratio applied to every mode. These are modelling recommendations apeGmsh
   already supports; they belong in the how-to, not the code.
2. **Modes** (§7.3): eigen solve; modes up to ≈ 2× the fundamental for
   low-frequency evaluation; all modes to 20 Hz for the high-frequency
   branch. Ritz vectors explicitly not recommended.
3. **FRF** (§7.3): steady-state acceleration at occupant node *j* per unit
   sinusoidal force at walker node *i*, in `%g/lb`, on a band from 1 Hz
   below f₁ to 1 Hz above the highest mode (20 Hz for the high-frequency
   branch), evaluated *at every modal frequency* plus 20–30 other points.
   The frequency of the maximum FRF magnitude is the **dominant frequency**
   — "not always a natural frequency" when modes are close and damping is
   high.
4. **Low-frequency floor** (dominant frequency < 9 Hz), Eq 7-1 … 7-3:

   ```
   a_p = FRF_max · α · Q · ρ
   α   = 0.09 e^{−0.075 f_n}                       (f_n = dominant frequency, Hz)
   ρ   = 50β + 0.25   (β < 0.01)
       = 12.5β + 0.625 (0.01 ≤ β < 0.03)
       = 1.0           (β ≥ 0.03)
   Q   = 168 lb
   ```

5. **High-frequency floor** (dominant frequency 9–20 Hz), Eq 7-4 … 7-6,
   Table 7-1, Eq 1-6:

   ```
   h        = harmonic for the dominant frequency (Table 7-1: 9–11 → 5, 11–13.2 → 6, 13.2–15.4 → 7, 15.4–17.6 → 8, 17.6–20 → 9)
   f_step   = f_dom / h
   I_eff,m  = f_step^1.43 / f_n,m^1.30 · Q / 17.8                     (Eq 1-6, lb·s with Q in lb)
   a_p,m    = 2π f_n,m · φ_i,m · φ_j,m · I_eff,m                       (Eq 7-4, mass-normalised φ)
   a(t)     = Σ_{m: f_n,m ≤ 20 Hz} a_p,m e^{−2πβ f_n,m t} sin(2π f_n,m t)   (Eq 7-5)
   a_ESPA   = √2 · RMS( a(t), 0 ≤ t < 1/f_step )                     (Eq 7-6)
   ```

6. **Acceptance**: `a_p` (or `a_ESPA`) against Fig 2-1 / Table 4-1:
   0.5 %g offices, residences, churches, schools; 1.5 %g shopping malls,
   dining, indoor footbridges; 5 %g outdoor footbridges — flat between
   4 and 8 Hz, rising outside it.

Example 7.1 works both branches to numbers (tip mode 3.49 Hz,
`FRF_max = 0.0344 %g/lb`, β = 0.025 → `a_p = 0.374 %g`; backspan mode 22
at 12.6 Hz, φ = −3.15 (in./kip·s²)^½, `f_step = 2.1 Hz` → `I_eff = 1.01 lb·s`,
`a_p,22 = 0.206 %g`, total peak 0.865 %g, `a_ESPA = 0.314 %g`), and its
Table 7-2 lists all 38 modes to 20 Hz with their backspan shape values.
That table is a **kernel oracle that needs no finite-element model**.

### What apeGmsh already has

- `apeSees.eigen` / `modal_properties` / `EigenResult.mode_shape` — the
  basis, lazily read through `ops.nodeEigenvector` on the retained live
  domain ([ADR 0075](0075-modal-response-family.md)).
- `apeSees.frequency_response` / `steady_state_dynamics` — the fork's ADR-44
  P2 sweep: complex FRF of **one** response DOF for **one** load pattern,
  and each call **rebuilds the domain and re-solves eigen**
  (`_run_modal_sweep`). Correct, and the right oracle, but the wrong shape
  for an `N_exc × N_resp` matrix.
- `apeSees.modal_response_history` — the exact modal transient with a
  `-load/-series` channel; a validator for the impulse branch, not the
  method.
- `Results` + the native writer's `write_nodes(components={name: (T, N)})`
  — any nodal scalar written there renders in the viewer as a map.
- A units-agnostic bridge: apeGmsh has no unit system. The Design Guide's
  constants are imperial and its FRF is in `%g/lb`.

Two facts found while reading the sources shape D2:

- `modalProperties -return` does **not** export the generalised modal
  masses. `DomainModalProperties` computes `m_generalized_mass_matrix`
  (`V' M V`) and uses it for `Γ = L/m̃` and `partiMass = L²/m̃`, but the
  returned dict carries only `partiFactor*`, `partiMass*`, ratios, totals,
  `eigen*`. The fork's P2 sweep uses `generalizedMasses()` internally
  (guide: "`m̃_a = generalizedMasses()(a)` … identical under default and
  `-unorm`"), which is why its `-load` channel is normalisation-proof —
  and why a Python-side sum has to establish the eigenvector scale itself.
- Both stock eigensolvers return M-orthonormal vectors by construction
  (`-fullGenLapack` is LAPACK `dsygv`, `ZᵀBZ = I`; `-genBandArpack` is
  ARPACK `bmat='G'`, B-orthonormal Ritz vectors). That is a property of the
  solvers, not a documented OpenSees contract — so it is asserted, not
  assumed (D2).

## Decision

### D1 — The kernel is DG11 2nd-edition Chapter 7, as pure functions over an FRF matrix

A new module `apeGmsh.opensees.analysis.footfall` holds the evaluation
kernel with **no OpenSees import**: `dynamic_coefficient(f)` (Eq 7-2),
`resonant_buildup_factor(beta)` (Eq 7-3), `harmonic_for_dominant(f)`
(Table 7-1), `effective_impulse(f_step, f_n, Q)` (Eq 1-6),
`walking_low_frequency(frf_max, f_dom, beta, Q)` (Eq 7-1),
`walking_high_frequency(f_n, phi_i, phi_j, f_dom, beta, Q, dt)`
(Eq 7-4 … 7-6, returning `a_espa`, the raw peak and the sampled `a(t)`),
`tolerance_limit(occupancy, f)` (Fig 2-1 / Table 4-1), and
`dominant_frequency(freq, frf_mag, f_max)`.

Robot's three code options are *other evaluators over the same data*. The
CCIP-016 response factor, the SCI P354 weighted-RMS factor and a
1st-edition "Robot-parity" evaluator can each be added later as one
function taking the same `(freq, H_ij)` and mode tables; none is in this
ADR's scope (D7). The kernel is written so that they can be.

**Units are the caller's.** The driver takes `body_weight` (model force
units) and `g` (model acceleration units) as **required** keyword
arguments — no defaults, because apeGmsh cannot know whether a model is in
N·m, kN·m or kip·in, and a silently wrong `g` is a factor of 386 in the
answer. Every Design Guide constant is used in a form that is dimensionless
in the model's force unit: Eq 1-6 is `I_eff = (Q/17.8) · f_step^1.43 /
f_n^1.30` (force·s for any force unit, frequencies in Hz), Eq 7-1 needs the
FRF in acceleration/force, and accelerations are reported as **fractions
of g** (`a_p / g`; `%g` only in reports). Q = 168 lb ≈ 747 N (Robot's
746 N) is documented as the recommended value, not baked in.

### D2 — The FRF matrix is built in Python from one eigen solve, with the eigenvector scale asserted

```
H_ij(Ω) = Σ_a  φ_ia φ_ja / ( m̃_a (ω_a² − Ω² + 2 i ξ_a ω_a Ω) )      (displacement, e^{+iΩt})
A_ij(Ω) = −Ω² H_ij(Ω)                                              (acceleration per unit force)
```

built from `EigenResult`/`ModalPropertiesResult` — `ω_a` from the
eigenvalues, `φ_a` read once per node into an `(N, p)` array through
`mode_shape` (vertical component only), `ξ_a` from D6 — and evaluated with
numpy over the D4 grid. The `|A_ij|` maximum over excitation nodes streams
per response node, so *full* excitation never materialises the
`(n_freq, N_exc, N_resp)` tensor.

**`m̃_a` is taken as 1 and asserted, not looked up.** The assertion uses
what `modalProperties -return` *does* export: for every mode with
`|partiFactor_C| > 1e-8` in any component `C`,
`partiMass_C / partiFactor_C² = m̃_a` must equal 1 within 1e-6. A mode
with every `Γ_C = 0` (a doubly antisymmetric floor mode on a symmetric
grid — real, not hypothetical) cannot be checked this way; the driver
then requires that at least one checkable mode exists and that all
checkable modes pass, and records `normalization="asserted(k of p)"` on
the result. If any checkable mode fails, the driver **refuses** — there is
no silent rescale. `unorm=True` is refused outright (it changes the basis
`partiFactor` is computed in; see the basis caveat on
`ModalPropertiesResult`).

**Fork ask (small, non-blocking):** export `genMass` in the
`modalProperties -return` dict. When present, the driver uses it and the
assertion becomes a consistency check. Filed against
`Ladruno_implementation/44_…`; this ADR does not wait for it.

**Why not loop the fork's `frequencyResponse`.** `apeSees.frequency_response`
rebuilds the domain per call; at the live-emitter level it is still one
interpreter call per `(i, j)` pair *and* needs a `Plain` pattern with a
unit vertical load per excitation node created on the live domain after
the eigen solve. For *self* excitation on a few nodes that is fine; for
Robot's *full* on a panel's nodes it is `N²` calls. The modal sum is a
`(p × n_freq)` complex product per pair — microseconds in numpy — and it
runs on **stock openseespy**: `eigen`, `modalProperties`, `nodeEigenvector`
are all upstream. The fork is needed only for the oracle in D8.

### D3 — Robot's two excitation methods are one parameter

`excitation="self" | "full"` with `excitation_nodes=` and `response_nodes=`
resolved through the existing selection surface (node tags, `Node`
handles, labels / physical groups). *self* evaluates the diagonal
`(j, j)`; *full* evaluates every `(i, j)` and reports, per response node,
the excitation node giving the largest `a_p` — Robot's `ExcitationNode`.
Default is `self` over `response_nodes`, with the Design Guide's guidance
stated in the docstring: the walker at the mid-length of an unobstructed
path, the occupant at the maximum mode-shape value, both conservatively at
mid-bay. The vertical DOF is `dof=3` by default and explicit for 2-D or
rotated models.

### D4 — Band, grid and basis follow §7.3, not Robot's 20 intervals

- Band: `f_min = max(f₁ − 1 Hz, 0.1 Hz)`; `f_max = 20 Hz` (the high-frequency
  branch needs it; the low-frequency search is then a `< 9 Hz` slice).
- Grid: every modal frequency in band, a ±5 % cluster of 5 points around
  each (the fork's `-biased` idea — a low-damping peak must not be
  stepped over), plus `n_extra=30` linear points. Robot's "20 intervals +
  eigenfrequencies" is a coarser instance of the same rule.
- Basis: all modes with `f ≤ 20 Hz`, plus a margin so the modal sum's tail
  is not truncated exactly at the band edge. `num_modes=` is the caller's
  (as in every ADR 0075 driver); the driver **warns** when the highest
  extracted mode is below 20 Hz — the impulse branch would then be summing
  an incomplete basis — and points at `eigen_feast(0, 22)` on the fork as
  the exact way to get the band.

### D5 — One result object, and a map through the native writer

`FootfallResult` carries, per response node: `f_dom`, `frf_max`
(acceleration per unit force, model units), `a_p_lf` (Eq 7-1, `NaN` when
no dominant frequency lies below 9 Hz), `a_espa_hf` (Eq 7-6, `NaN` when no
mode lies in 9–20 Hz), `a_p` (the governing one), `regime`
(`"low" | "high" | "both"`), `exc_node`, `limit` and `ratio = a_p / limit`
for the requested `occupancy`; the full `(freq, |A_ij|)` table is kept for
inspection (`result.frf(j, i=None)`). Per-mode `(f_n, φ_i, φ_j, a_p,m)`
rows are kept for the high-frequency branch — Example 7.1's Table 7-2 is
exactly that table.

`result.to_results(fem, path)` writes a native results file with **one
frame** whose nodal components are `footfall_ap`, `footfall_ratio`,
`footfall_fdom` (shape `(1, N)`) via `NativeWriter.write_nodes`, so
`Results.from_fem(fem, path, kind="native").viewer()` renders the ratio map
with the ordinary nodal-scalar machinery. Not a recorder and not a
`RESPONSE_CATALOG` entry — it never comes from the engine. Nodes outside
`response_nodes` are `NaN`.

### D6 — Damping is the ADR 0075 channel

Exactly one of `damp=` (uniform β — Table 4-2 is a sum of component
ratios), `modal_damp=` (per mode, absolute mode order) or `rayleigh=(a0,
a1)` (converted per mode, `ξ_a = a0/(2ω_a) + a1 ω_a/2`), validated by the
existing `_damping_channel_args` rules. Eq 7-3 and Eq 7-5 use the ratio of
the mode in question; with `rayleigh=` that is the converted per-mode
value, and the dominant-frequency mode's ratio feeds ρ.

### D7 — Scope of the first version

**In:** walking on level floors and pedestrian bridges (§7.4.1), both
regimes, both excitation methods, the map, the oracle tests, a how-to.

**Out, with the seam named:** running (§7.4.2, Eq 7-7, Table 7-3) and
slender stairs (§7.4.3, Eq 7-8) are the same FRF with a different
coefficient table and build-up rule — one evaluator function each;
rhythmic group loads (§7.4.4, Eq 7-9/7-10) need a *uniform* unit load over
an area (a distributed excitation vector, not a node pair) — one more
excitation kind in D2; sensitive equipment (§7.5 / Chapter 6) is a
different tolerance family; CCIP-016 / SCI P354 response factors and a
1st-edition Robot-parity evaluator are evaluators over the same data.
Lateral footbridge response and group synchronisation are not on the
roadmap.

### D8 — Verification is three oracles, none of which is Robot

1. **Kernel:** Example 7.1 reproduced from Table 7-2 as a pure-function
   test — `α(3.49) = 0.069`, `ρ(0.025) = 0.938`, `a_p = 0.374 %g` from
   `FRF_max = 0.0344 %g/lb`; `I_eff,22 = 1.01 lb·s`, `a_p,22 = 0.206 %g`,
   total peak 0.865 %g, `a_ESPA = 0.314 %g` at `dt = 0.005 s` (the Guide's
   own sampling). Tolerances follow the Guide's three-figure rounding.
2. **FRF matrix:** the tip-mass cantilever of
   `tests/opensees/live/test_modal_sweeps_live.py` against the SDOF closed
   form; and, gated on `capabilities().has_fork`, one `(i, j)` pair of the
   Python matrix against `apeSees.frequency_response(load=unit at i,
   node=j, resp="accel")` on the same model — same `φ`, same `m̃`, so the
   tolerance is 1e-6 relative, not "close".
3. **Impulse branch (optional):** `modal_response_history` with a
   `-load/-series` single-footstep impulse of magnitude `I_eff` (a one-step
   Path series) on a two-mode model against Eq 7-5 — an independent
   integration of the same physics.

**Robot is a cross-check, not a gate.** With `apeRobot` the same floor can
be run under Robot's AISC option (self-excitation, 1.6–2.2 Hz) and its
`Frequency` / `A` read back through `IRobotFootfallResults`. Because Robot
is on the first edition, agreement is expected on the dominant frequency
and on the FRF-shaped part of the answer, not on `a_p`; the deliverable of
that slice is a report of the differences with the first-edition
coefficients (`α_i = 0.5/0.2/0.1/0.05`, Q = 157 lb, R = 0.5) applied to our
FRF, which is also the cheapest way to *measure* what Robot does with `R`
and `FootstepsNumber` — the help does not say.

## Alternatives considered

- **Time-history per walking path** (the Guide's "can be used instead").
  Rejected as the method: the Guide itself prefers the FRF; a moving-load
  transient needs a path, a stride and a phase model, and reproduces Eq 7-1
  only on average. Kept as the D8-3 validator.
- **A fork `footfall` command.** Rejected: this is post-processing on a
  basis the fork already exposes; putting Design Guide constants in C++
  moves the oracle tests to the wrong repo and loses the stock-openseespy
  path. The only fork ask is the one-line `genMass` export.
- **Reproduce Robot (CCIP-016 / SCI P354) first, AISC later.** Rejected:
  the user's reference is the 2nd edition, Autodesk's own note says Robot
  is on the 1st, and CCIP/SCI are evaluators that D1 leaves room for.
- **Looking up `m̃_a` from `partiMass/Γ²` and rescaling silently.**
  Rejected: fails on every mode with `Γ = 0`, which on a symmetric floor
  are exactly the antisymmetric modes the FRF at an off-centre node is
  made of. Assert-and-refuse instead (D2).
- **A `%g/lb` result to match the Guide's tables.** Rejected: apeGmsh has
  no unit system; `a/g` is unit-free once `g` is given, and the Guide's
  numbers convert in the test, not in the API.

## Consequences

**Positive.**
- A Design-Guide-current footfall evaluation, including the
  high-frequency branch Robot's AISC option lacks, on any apeGmsh floor
  model, with the map in the viewer.
- Runs on stock openseespy; the fork only tightens verification.
- The kernel/driver split makes every Guide equation a unit-testable
  function with the Guide's own worked example as its oracle.

**Costs / risks.**
- The eigenvector-scale assertion (D2) is a real refusal path; a model
  whose every in-band mode has zero participation in all six components
  cannot be evaluated until the fork's `genMass` export lands.
- `nodeEigenvector` is one interpreter call per node per mode; 10 k
  response nodes × 40 modes is ~400 k calls (seconds). Fine for floors;
  the distributed harvest path (`ParallelModalResult`) is the escape if it
  ever is not.
- The dominant-frequency search on a `< 9 Hz` slice vs the whole band is a
  judgement the Guide leaves to the engineer ("not always a natural
  frequency"); the result exposes both so the choice is visible.

## Slices

| # | Deliverable | Verify |
|---|---|---|
| S0 | `analysis/footfall.py` kernel (D1) | Example 7.1 oracle, Table 7-1/4-1 pins, ρ and α edge cases |
| S1 | `frf_matrix(...)` from a live eigen + the scale assertion (D2) | SDOF closed form; fork `frequency_response` pair to 1e-6; assertion refuses `unorm=True`; mutation: scale a φ column and see the refusal |
| S2 | `apeSees.footfall_walking(...)` driver + `FootfallResult` (D3–D6) | two-mode plate: `self` vs `full` agree on the diagonal; `regime` flips at 9 Hz; damping channels agree where they should |
| S3 | `to_results` map + viewer smoke; `docs/how-to/footfall-vibration.md`; skill note | viewer renders `footfall_ratio`; docs gate green |
| S4 | apeRobot validation report on one floor (D8) — after S3 is merged | report, not a gate |
| S5 | running / stairs / rhythmic evaluators; CCIP-016 / SCI P354 factors; fork `genMass` ask | own ADR amendment when picked up |

## Owner decisions (2026-09-10)

1. **Limit curve is the default.** `limit="curve"` — Fig 2-1's
   frequency-dependent limit, modelled as the flat Table 4-1 value scaled
   by the ISO 2631-2 base-curve shape `s(f) = √(4/f)` (f < 4 Hz), `1`
   (4–8 Hz), `f/8` (f > 8 Hz). Example 7.1's own words pin it: "slightly
   greater than 0.5 %g" at 8.85 Hz (0.553 %g) and "just over 0.5 %g" at
   9.35 Hz (0.584 %g). `limit="table"` keeps the flat value.
2. **Robot is validation only.** S4 runs after S3 is merged, against the
   finished implementation, and produces a report. It gates nothing.
3. **No fork ask now.** The D2 assertion stands; the `genMass` export is
   filed only if a real model refuses. Recorded here so the refusal path
   is read as intended, not as a gap.

The orchestration program (agents, models, prompt contracts) is in
`internal_docs/handoff_adr0109_program.md`.

## Cross-references

- [ADR 0075](0075-modal-response-family.md) — the drivers, the damping
  channel, and the staleness contract this rides on.
- [ADR 0077](0077-parallel-modal-analysis.md) — why `modalProperties` is serial-only
  and where the harvested mode shapes come from if this ever goes
  distributed.
- Fork ADR-44 (`44_ladruno_frequency_domain_adr.md` §4.4) — the modal FRF
  the D8 oracle computes, and the `e^{+iΩt}` sign this ADR adopts.
- `apeRobot` — `IRobotFootfallAnalysisParams` and `IRobotFootfallResults`
  for S4; enums `FOOTFALL_A`, `FOOTFALL_RF_*`, `FOOTFALL_EXCITATION_NODE`,
  `FOOTFALL_FREQUENCY` in `apeRobot.enums`.
