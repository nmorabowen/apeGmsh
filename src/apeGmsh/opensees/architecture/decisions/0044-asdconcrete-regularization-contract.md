# ADR 0044 — ASDConcrete regularization contract

**Status:** ACCEPTED 2026-05-30 (signed off by N. Mora as the
in-house OpenSees authority). Grounded by a 9-agent design discussion
(ground → adversarial debate → synthesis), then **corrected by direct
source + runtime verification** — the synthesis made three CLI-level
claims that implementation disproved (see **The CLI facts** and
**Alternatives D/F**). A first implementation slice has **shipped**
under this ADR: `ASDConcrete3D` + the owned backbone generator
([_asdconcrete_laws.py](../../material/_asdconcrete_laws.py)) +
`from_fc` + the bridge entry point + tests
(`tests/opensees/unit/primitives/test_materials_asdconcrete.py`).

**Pinned builds.** Reference source for the ported numerics:
OpenSees commit `7c92197`. Runtime apeGmsh tests against: the Ladruno
openseespy build `288f6d0`. These differ; the design tolerates that
divergence by *emitting the explicit backbone* (Decision 1) so the
solver integrates exactly what apeGmsh generates, independent of the
build's own `-fc`.

Builds on / consumes the typed `nDMaterial` primitive pattern in
[nd.py](../../material/nd.py) (`ElasticIsotropic` /
`ASDPlasticMaterial3D` / `PlaneStrain`) and the warn-as-contract idiom
established by `ComposeInterfaceSizeWarning` /
`WarnGeomCoincidentFace`.

## Context

`ASDConcrete3D` / `ASDConcrete1D` (Petracca, ASDEA) is a plastic-damage,
**crack-band-regularized** model for concrete and masonry. apeGmsh's
typed `nDMaterial` layer currently ships only the priority-1 set
(`ElasticIsotropic`, `J2Plasticity`, `DruckerPrager`) and explicitly
defers the ASD damage models. Users now need a physics-first
`from_fc(fc, ft, Gf, Gc, ...)` entry point.

The design question is **how much of the regularization machinery
apeGmsh should OWN versus DELEGATE** to the OpenSees binary — the
softening backbone, the crack-band reference length `lch_ref`,
instance/element granularity, the snapback element-size ceiling, and 1D
confinement.

### The two facts that dictate the design (both source-verified)

**Fact 1 — regularization is per-element, via per-Gauss-point clones.**
Continuum elements clone the material once per integration point:
[`Brick.cpp:190-197`](file) loops `for (i<8) materialPointers[i] =
theMaterial.getCopy("ThreeDimensional")`. Each clone carries its own
`regularization_done` flag and its own `lch`. On the clone's first
`setTrialStrain`, it reads `lch =
ops_TheActiveElement->getCharacteristicLength()`
(`ASDConcrete3DMaterial.cpp:1607-1620`) — i.e. **its own parent
element's** characteristic length — regularizes `ht`/`hc` once, and
freezes. Therefore **one material *tag* shared across a graded mesh
produces N×(GP) independent clones, each correctly regularized to its
own element.** Sharing a tag across differently-sized elements is
*correct*, not a hazard. (See **Alternatives D** for the refuted claim
that this is a per-tag freeze.)

**Fact 2 — the physics is invariant to `lch_ref`.** `regularize()`
rescales the specific fracture energy `gnew = m_fracture_energy *
(lch_ref/lch)` (`ASDConcrete3DMaterial.cpp:947-965`), holding `Gf =
g_input · lch_ref` constant per element. `lch_ref` is a calibration /
bookkeeping length, not a physical knob. When `lch` exceeds the
snapback limit `l_max = 2·E·Gf/ft²`, `gnew` is **clamped** to
`gmin = (peak.y·peak.x/2)·1.01` (`cpp:961-965`) — the binary *floors*
fracture energy at the brittle limit rather than producing a
negative-energy curve. Worst case is an over-brittle, mesh-dependent
(non-objective) answer — a quality defect, **not a malformed model**.

### The CLI facts that reshaped the decision (verified, not assumed)

The 9-agent synthesis recommended "delegate to `-fc`, emit a bare
`-autoRegularization`, forward `-Gc`." Implementation against the actual
parser disproved all three:

* **`-fc` cannot take a user fracture energy.** Source `7c92197` accepts
  exactly these option tokens: `-rho -fc -ft -Kc -implex -implexControl
  -implexAlpha -eta -tangent -autoRegularization -Te -Ts -Td -Ce -Cs -Cd
  -crackPlanes -cdf`. There is **no `-Gf`/`-Gc`/`-Gt`** — the `-fc` path
  *derives* them from `fc` (CEB-FIP `Gt = 0.073·fc^0.18`,
  `Gc = 2·Gt·(fc/ft)²`, `cpp:607-712`). The parser **silently ignores**
  unknown tokens, so a runtime `nDMaterial(... '-Gf' 0.1)` *appears* to
  succeed while dropping the value — a footgun, not support.
* **`-autoRegularization` requires an explicit `$lch_ref`.** The bare
  flag is a parser error (`cpp:547-555`; confirmed at runtime). The
  self-derived `min(hmin_t, hmin_c)` lives in the `-fc` block for *curve
  building* (`cpp:613-630`); it is **not** what a bare flag would
  trigger.

**Consequence.** Because `Gf` is *the* physical regularization input
(the entire premise of this work) and `-fc` cannot accept it, apeGmsh
must own the backbone and emit it explicitly. And emitting the explicit
`-Te/-Ts/...` points means the solver integrates exactly apeGmsh's
curve — there is **no parity-drift surface** (the synthesis's reason to
avoid owning the curve does not apply to explicit emit).

### What is NOT reachable at emit time

Mesh size / target characteristic length does not reach the emit
boundary: `PhysicalGroups` carry only name/tag/entities, `FEMData` has
no per-PG size, and the Gmsh size field is not persisted into the
snapshot. A mesh-**seed**-based `lch_ref` source therefore has no data
to read. Element **class** is only indirectly knowable (emitter
side-channel / node-count). Element **connectivity** *is* reachable
(`expand_pg_to_elements`), so a realized-**geometry** `l_max` check is
computable if node coordinates are threaded to the check site. Material
primitives themselves store no element/mesh/solver context (the
`ElasticIsotropic` precedent) — a material **cannot self-guard
`l_max`**; that diagnostic must live at bind/assembly time.

## Decision

1. **Curve builder — apeGmsh OWNS it, emits explicit points.** A pure
   Python port of `_make_tension`/`_make_compression`
   ([_asdconcrete_laws.py](../../material/_asdconcrete_laws.py)),
   parameterized by `(E, fc, ft, Gf, Gc, lch_ref)`, generates the
   backbone; `from_fc` emits the explicit `-Te/-Ts/-Td/-Ce/-Cs/-Cd`
   card. This is the *only* way to honour a user `Gf`/`Gc` (the `-fc`
   path can't — see CLI facts), and because the explicit points are what
   the solver integrates, there is no parity-drift risk. The generator
   doubles as the read-only `preview_backbone()` surface and is the
   substrate for the invariance / `l_max` unit tests. (This restores the
   author's original lean and reverses the synthesis; the synthesis's
   drift objection applied only to emitting `-fc` while previewing via a
   separate port, which is not what we do.) Inputs are range-checked in
   `__post_init__` / `from_fc`; apeGmsh validates **hard** rather than
   relying on the parser, which silently drops unknown tokens.

2. **`lch_ref` — explicit value, always.** The bare flag is a parser
   error, so `from_fc` always emits `-autoRegularization $lch_ref` with
   a concrete value. The curve and the emitted `lch_ref` share one
   reference length, so the binary's per-element rescale preserves
   `Gf` (`area·lch_ref == Gf`). Default `lch_ref` = the ported native
   self-derivation `min(hmin_t, hmin_c)`; **user-overridable** with a
   representative element size for better-conditioned softening (the
   physics is invariant to the value — Fact 2 — but a representative
   length minimizes rescale shape-distortion). No mesh-seed plumbing
   (the mesh size is not reachable at emit, and `lch_ref` is inert
   anyway).

3. **Granularity — NOT mandated.** Because regularization is per-element
   (Fact 1), **one material tag may serve a whole graded region
   correctly.** apeGmsh imposes no per-region / per-element-class
   instance-minting rule. (This reverses the discussion's draft
   recommendation, which rested on the refuted per-tag-freeze claim.)

4. **Element-class purity — docstring note only.** No regularization
   hazard exists from sharing a tag across element classes — each
   element's clones use their own class-correct
   `getCharacteristicLength()` (solid min-edge, shell `√A`, force-beam
   integration-point length). The only real constraint is that
   `ASDConcrete3D` is 3-D-only; 2-D/shell use goes through the existing
   `PlaneStrain` wrapper precedent. Document; add no runtime gate.

5. **`l_max` guard — WARN, never block, at bind/assembly time.** Emit a
   dedicated, non-default-suppressed `ASDRegularizationWarning`
   reporting offending element(s), realized `lch`, computed `l_max =
   2·E·Gf/ft²`, the ratio, the count over limit, and the remedy — and
   make it CI-promotable via `pytest -W error::ASDRegularizationWarning`
   (the `ComposeInterfaceSizeWarning` idiom). Do **not** raise: the
   binary floors snapback (Fact 2), so over-limit is over-brittle, not
   malformed; a hard raise would let one coarse outlier veto a whole
   emit and forbid coarse exploratory runs. Compute on realized geometry
   where available (`expand_pg_to_elements` + node coords); otherwise
   ship an input-only advisory and state the estimate basis.

6. **1D confinement — document, defer Mander.** `ASDConcrete1D.from_fc`
   produces an **unconfined** backbone and is confinement-blind (no
   stress decomposition / Lubliner surface). State plainly in the
   docstring that confinement is the user's responsibility (Mander into
   the backbone) for now; defer a proper `ConfinedConcrete` helper to a
   forcing-function-driven slice.

## Proposed API surface

```python
# Tier 1 — physical entry point (delegates curve to the binary)
mat = ASDConcrete3D.from_fc(fc=30, ft=3.0, Gf=0.1, Gc=20.0,
                            Kc=2/3, rho=2.4e-9)   # emits -fc ... -autoRegularization

# Tier 2 — explicit backbones (test-calibrated / STKO-matched), future additive
mat = ASDConcrete3D(E=..., v=..., Te=(...), Ts=(...), Td=(...),
                    Ce=(...), Cs=(...), Cd=(...), lch_ref=...)

# 3-D material in a 2-D element → existing wrapper, no new machinery
ops.element.quad(..., material=PlaneStrain(base=mat))

# Fibers / 1-D — confinement-blind; bake Mander into the backbone yourself
mat1d = ASDConcrete1D.from_fc(fc=30, ...)
```

## Alternatives considered

**A. Port the backbone into Python and emit an explicit
`(strain,stress,damage)` curve** — **ADOPTED** (Decision 1). The
synthesis rejected this fearing parity drift, but that fear only applies
if apeGmsh emits `-fc` while *separately* previewing via a port (so
preview ≠ what runs). Emitting the explicit points the port generates
means the solver integrates exactly that curve — no second source to
drift from. And it is the only way to expose `Gf` (the `-fc` path
can't). The residual cost — apeGmsh owns ~100 lines of correctness — is
bounded and unit-tested.

**B. Source `lch_ref` from the mesh seed with element-median fallback**
— rejected: physically inert (Fact 2) and the plumbing does not exist.

**C. Hard-raise on element-class impurity / on `l_max`** — rejected as
default: both guard numerical-quality conditions, not malformed models
(the binary floors snapback); a raise seizes engineering judgment the
repo's `CLAUDE.md §1` says to *surface*. Warn (CI-promotable) instead.

**D. [REFUTED claim, recorded so it is not re-litigated] "Regularization
is per-material-instance, so a tag freezes to the first element's
`lch`."** The 9-agent synthesis flagged this as a "decisive correction."
It is **wrong**: it conflates one apeGmsh primitive / one OpenSees tag
with one C++ runtime instance. Elements clone the material per Gauss
point (`Brick.cpp:190-197`), so there is no per-tag freeze — see Fact 1.
This refutation removes the rationale for mandated granularity (Decision
3) and for the class-purity hazard (Decision 4).

**E. Fixed `lch_ref=1.0` required keyword** (skeptic) — superseded by
the ported self-derivation default (Decision 2), which is a physically
meaningful reference length rather than an arbitrary `1.0`.

**F. [REFUTED CLI claims, recorded so they are not re-litigated]** The
synthesis asserted, without checking the parser, that (i) a bare
`-autoRegularization` is accepted, (ii) `-fc` forwards a user `-Gc`, and
(iii) delegating to `-fc` is sufficient. All three are false against
source `7c92197` and the runtime build (see **The CLI facts**): the bare
flag errors, no `-Gf`/`-Gc` token exists, and `-fc` can't carry `Gf`.
These are *why* Decisions 1 and 2 reversed the synthesis. Lesson:
verify CLI surface against the parser, not the discussion.

## Consequences

**Positive.** Custom `Gf`/`Gc` are expressible (impossible via `-fc`);
the solver integrates exactly what apeGmsh emits, so there is no
parity-drift surface and behavior is decoupled from the build's `-fc`;
the backbone is inspectable/plottable and the invariance/`l_max` claims
are CI-testable without a live OpenSees build; one tag serves a graded
region (per-element regularization, captured as an invariant);
coarse-exploratory and graded-mesh workflows stay unblocked; the
warn-as-contract diagnostic is CI-promotable where teams want a gate.

**Negative.** apeGmsh owns ~100 lines of backbone numerics (bounded,
unit-tested); the *quality* (not bit-equality) of the generated curve
must track upstream's reference shape; the realized-geometry `l_max`
warning still needs node-coordinate threading + a pinned per-class
`getCharacteristicLength()` convention to wire into bind time.

**Neutral.** Behavior at runtime is decoupled from the build's own
`-fc` because we emit explicit points — a feature here, not a risk.

## Resolved questions (2026-05-30)

1. **Owned generator + preview — shipped in the first slice.**
   `_asdconcrete_laws.py` + `preview_backbone()` make the invariance /
   `l_max` claims CI-testable without a live OpenSees build. Done.
2. **`l_max` guard — realized geometry is the target.** The
   material-level formula (`l_max()`) and per-element warning
   (`check_element_size`, emitting `ASDRegularizationWarning`,
   `-W error`-promotable) shipped. **Follow-up:** wire the realized
   per-element call into bind/assembly (thread node coords; pin the
   per-class characteristic-length convention) — tracked as the next
   slice.
3. **Sign-off + version pin — locked.** Accepted by N. Mora as the
   in-house OpenSees authority; reference source `7c92197` and runtime
   `288f6d0` pinned in Status.
4. **Confinement — covered by the explicit-curve path.** The raw
   `Te/Ts/Td/Ce/Cs/Cd` constructor *is* the user's channel for a
   Mander-confined (or any test-calibrated) curve — pass it through, no
   `-fc`, no parity concern. A dedicated `Mander`/`ConfinedConcrete`
   helper is therefore sugar, deferred until a confined-column project
   forces it.

## Follow-up slices

* `ASDConcrete1D` (fibers) — same owned-curve pattern, no
  `Kc`/`cdf`/`crackPlanes`; document confinement-blindness.
* Bind-time realized-geometry `l_max` sweep (Resolved Q2 follow-up).
* Optional golden-parity test of the port vs the runtime build's `-fc`
  readback (`Te/Ts/Ce/Cs` responses) — gated on build alignment.
* `Mander`/`ConfinedConcrete` convenience helper (Resolved Q4).

## References

* OpenSees source `7c92197` (verified):
  `SRC/material/nD/ASDConcrete3DMaterial.cpp` (option tokens 498-601;
  `-fc` builder 607-712; `bezier3` 345-365; `regularize` 947-965;
  `setTrialStrain`/`regularization_done` 1607-1620; hardening-law
  responses `Te/Ts/Ce/Cs` 2158-2170),
  `SRC/material/uniaxial/ASDConcrete1DMaterial.cpp`,
  `SRC/element/brick/Brick.cpp:190-197` (per-GP `getCopy`),
  `SRC/element/Element.cpp:682`,
  `SRC/element/shell/ASDShellQ4.cpp:1858`,
  `SRC/element/forceBeamColumn/ForceBeamColumn3d.cpp:3682`.
* Runtime: Ladruno openseespy build `288f6d0` (silently ignores unknown
  option tokens — apeGmsh validates hard).
* apeGmsh: [nd.py](../../material/nd.py) (`ASDConcrete3D`,
  `ASDRegularizationWarning`),
  [_asdconcrete_laws.py](../../material/_asdconcrete_laws.py) (owned
  generator), `_internal/ns/nd.py` (bridge entry point),
  `tests/opensees/unit/primitives/test_materials_asdconcrete.py`.
* Discussion: workflow `asdconcrete-adr-discussion` (9 agents,
  ground → debate → synthesize), 2026-05-30 — corrected by source
  verification (see Alternatives D/F).
