# ADR 0124 — Gravity-preloaded modal analysis and per-mode response spectrum on one live domain

**Status:** Proposed (2026-10-22). Ships in two PRs: PR 1 the gravity-preloaded
modal driver (D1–D3), PR 2 the per-mode response spectrum and its combinations
(D4–D7).

## Context

A linear service-level (SLE) response-spectrum analysis that must match ETABS
takes P-Delta the way ETABS does with "P-Delta, iterative based on loads":
the gravity combination (for example `D + SD + 0.25 L`) is solved first, its
axial forces stay on, and the modes and every spectrum response are computed
on the tangent `K - Kg` of that state. The responses are increments about the
gravity state, and gravity is combined with them separately (`1.0 D + L_exp ± E`).

The bridge could not do this. Every modal driver on `apeSees` (`eigen`,
`modal_properties`, `response_spectrum_analysis`, ADR 0075) builds a fresh
domain and runs `eigen` on it, so the gravity preload is never there and the
modes see `K` only. `response_spectrum_analysis` combines the nodal
displacements and nothing else (ADR 0075, fork ADR 44 P1b), which is correct
for displacements and wrong for any other quantity: a combined force is not
the force of the combined displacements.

Reading `SRC/analysis/analysis/ResponseSpectrumAnalysis.cpp` settles what the
per-mode command does to a preloaded domain. `responseSpectrumAnalysis ... -mode k`
sets every node's trial displacement to the *absolute* modal displacement
`φ_k Γ_k Sa(T_k) / λ_k` (`Node::setTrialDisp(u, dof)` replaces, it does not add),
then calls `updateDomain` and `commitDomain`. A probe on the fork build
`462cf9cf` confirms it: after `-mode 1` on a gravity-loaded frame the committed
displacement is the modal one, and the gravity displacement is gone. So the
command cannot give a response about the gravity state. The element states it
leaves are those of `u_k` with no gravity axial force, so a PDelta frame's
forces lose the `Kg u_k` term the analysis exists to capture, and
`revertToLastCommit` cannot recover the gravity state because the command
committed over it.

## Decision

### D1 — `apeSees.modal_after_gravity(num_modes, *, p_delta=True, solver="-genBandArpack")`

One live domain, in this order: build and emit; run the declared analysis
chain to `λ = 1` (every registered pattern is the gravity preload); `loadConst
-time 0`; `eigen` on the current tangent; `modalProperties -return`. It
returns `GravityModalResult`: the existing `ModalPropertiesResult` (periods,
participation factors, mass ratios and cumulative ratios, total mass) plus a
`GravityState` snapshot (every node's displacement and reaction right after
`loadConst`). The driver re-reads the state after the eigen solve and raises
if it moved; `GravityModalResult.gravity_drift()` repeats the check later.
The domain is left at the gravity state, and the read session (ADR 0115) opens
there.

This is a flat model on the live route, which ADR 0121 D2 keeps supported. A
model with `ops.stage(...)` blocks is refused, as every live driver refuses it.

### D2 — The gravity chain is the declared one

The driver adds no analysis options of its own (P7). The declared chain must be
`Static` with a fixed-increment `LoadControl(dlam=1/N)`; the driver runs the
`N` steps, and the declared `test` and `algorithm` set the tolerance and the
iteration. An adaptive `LoadControl(num_iter=...)`, a `dlam` that does not
reach `λ = 1` in whole steps, another integrator, or a transient analysis is
refused before the build. The constraint handler is the declared one too:
`Transformation` for rigid diaphragms and links, as the user declares it.

### D3 — `p_delta` says which stiffness the modes see

`p_delta=True` solves the eigenproblem after the preload, on `K - Kg`.
`p_delta=False` solves it before the preload, on the unstressed `K`, and runs
the gravity analysis afterwards, so the gravity state is the same either way.
The geometric stiffness comes from the element formulations (`geomTransf
PDelta` or `Corotational`, a geometrically nonlinear shell such as
`ShellNLDKGQ`), never from the driver. `ShellNLDKGQ` joins the typed shells
for that reason. On fork build `462cf9cf`, `ASDShellQ4 -corotational` showed no
softening at `P = 0.5 Pcr` (`T/T0 = 1.0000`), so it does not carry the wall
P-Delta here; that is recorded as an open question for the fork.

The `eigen` solver flag is passed through. The default `-genBandArpack` is the
`ArpackSOE`, which factors through the analysis's own linear system, so the
declared `system` (`UmfPack` on stock, `Pardiso` on the fork) is the one the
eigen solve uses.

### D4 — Per-mode responses come from a static modal-force solve on the preloaded domain (PR 2)

For each mode `k` the driver applies the modal inertia force `±c M φ_k` (with
`c = max_d |Γ_kd Sa_d(T_k)|` over the requested directions) as a nodal-load
pattern on the gravity domain, solves with the declared algorithm at `dλ = 0`,
and evaluates the user's response extractors at `u_g + u_k` and `u_g - u_k`.
The pattern is then removed and the domain solved back to the gravity
equilibrium. The per-mode quantity is the central difference
`(r⁺ - r⁻) / (2c)` scaled by `Γ_kd Sa_d(T_k)` for each direction `d`: an
increment about the gravity state, linear in the spectrum, and exact to
third order for elements whose forces are quadratic in the displacements
(PDelta), exact for linear ones. Because `K_t φ_k = λ_k M φ_k`, the solve
reproduces the command's modal displacement `φ_k Γ_k Sa / λ_k` on a linear
model to round-off (`1e-14`). On a PDelta model the two differ by about `3e-7`
relative: the PDelta transformation's tangent omits the term from the change
of axial force across the existing gravity sway, and the Newton solve uses the
true linearization of the element forces.

`M` is the nodal mass (`nodeMass`). A model whose `modalProperties` total mass
exceeds the nodal mass (element mass: `rho` in a section, `-mass` on a beam) is
refused, because its modal force cannot be formed from nodal masses.

`responseSpectrumAnalysis -mode k` is not used, for the reason in Context. A
fork option that adds the modal displacement to the committed state and does
not commit would let the driver impose `u_g ± u_k` directly; it is requested in
the fork-request notes, and the solve route stays the reference.

### D5 — Response extractors are opaque callables

An extractor is any callable `extractor(ops) -> 1-D float array` of fixed
length, with an optional `.labels` sequence, where `ops` is the live module.
The driver never looks inside one, so story and pier aggregation (or any other
reduction) plugs in unchanged. Three are built in: nodal displacements, nodal
reactions (they call `reactions` first), and element responses by
`eleResponse` name. An extractor whose length changes between calls is refused.

### D6 — Combination happens in Python, per quantity

CQC uses the Der Kiureghian (1981) `ρ_ij` with per-mode damping ratios; SRSS
is the other rule. Directional combination is 100/30 (the envelope over which
direction carries the 100 %) or the SRSS of the per-direction results. Every
combination works on one quantity's own per-mode values, never on derived
displacements. The spectrum is a `(periods, accels)` table, interpolated the
way the fork does (linear, clamped at both ends), or a function: any object
with a vectorised `.at(T)` (a codeSpectra `Spectrum` satisfies it) or a plain
vectorised callable, normalised by one small function so codeSpectra never
becomes a dependency. It carries a scale factor and a direction; the scale is
applied in Python, not passed as `-scale`.

The driver adds one extractor of its own, `base_shear` (the total support
reaction per translational direction), so the combined `V_t` per direction is
always in the result. Minimum-base-shear scaling is code-agnostic:
`scale_to_base_shear(v_target, names)` returns `max(1, v_target / V_t)` per
direction and applies it to the combined force quantities named. The target
is the caller's (ASCE 7-22 §12.9.1.4.1, NEC-SE-DS, NCh433, …); code maths
stays in codeSpectra / apeETABS, and drifts are not scaled here (apeETABS
`DriftScale` owns that).

### D7 — Gravity is combined separately

The gravity state's extractor values are kept apart from the spectrum. Load
combination helpers build `a·G + b·E` and the envelope over the sign of `E`
(`1.0 D + L_exp ± E`), from the gravity values and a combined spectrum
response.

## Rejected alternatives

- **`responseSpectrumAnalysis -mode k` plus a restore.** The command replaces
  and commits the state, so a restore has nothing to return to, and the forces
  it leaves have no gravity axial force in them (see Context).
- **Imposing `u_g ± u_k` with `setNodeDisp`.** `setNodeDisp` moves the nodes
  but runs no `Domain::update`, so whether an element's response reads the new
  state depends on the element (a PDelta transformation reads its axial force
  from the last update, `ShellNLDKGQ` updates in `update()`).
- **A `Linear` algorithm for the legs.** It reads the tangent at the leg's start
  and leaves the upstream shells' sections stale (`shell_probe`); measured on
  the PDelta frame it left a `1e-5` error and a gravity drift.
- **Deriving the modes from a second, unloaded domain for `p_delta=False`.** One
  domain with the eigen solve placed before the preload gives the same `K`
  without a second build.

## Consequences

- `modal_after_gravity` (PR 1) and the spectrum driver (PR 2) run on stock
  openseespy; nothing here is fork-only. `-combine` stays the fork's, used only
  as a cross-check of the Python combination.
- The per-mode cost is three static solves per mode on the full model. The
  tangent is refactorised for each solve.
- Accuracy is bounded by the declared convergence test: the gravity state is
  restored to that tolerance, not bit for bit, and the driver checks it.

## Cross-references

- ADR 0075 — the modal-response family and its fresh-domain drivers.
- ADR 0115 — the read session the driver opens at the gravity state.
- ADR 0121 D2 — live execution for unstaged models.
- `SRC/analysis/analysis/ResponseSpectrumAnalysis.cpp`,
  `SRC/analysis/analysis/LadrunoModalCombination.h` (fork) — the per-mode
  command and the CQC kernel the Python combination is checked against.
