# Nonlinear concrete solver guide (apeGmsh → OpenSees / Ladruno fork)

How to set up and **solve** a nonlinear concrete model in apeGmsh: pick the constitutive model,
regularize it, choose the element, build an analysis chain that converges through cracking/softening,
and read the capacity curve back.

> **The golden rule:** concrete plastic-damage tangents are **non-symmetric** (non-associated flow +
> damage). Use an **unsymmetric linear solver** — `ops.system.UmfPack()` (or `FullGeneral` / `Mumps`).
> A symmetric solver (`ProfileSPD`/`BandSPD`) under-converges or diverges. `LadrunoConcrete3D` prints a
> runtime warning to this effect. Everything below assumes you obeyed it.

> **API note (verified against source):** the OpenSees bridge is **`ops = apeSees(fem)`** — *not* the
> old `g.opensees` facade (removed in PR γ; `_core.py:90`). Build the model with `g`, extract a
> `FEMData`, then drive OpenSees through `apeSees`. All code below uses the current API.

Cross-refs: the `opensees-concrete` skill (constitutive theory + numerics), the integrator study
(ADR-49 scorecard / ADR-52), and the canonical `docs/examples/pushover-steel-frame.md`.

---

## 1. Pick the constitutive model

| Your model is… | Material | Why |
|---|---|---|
| **3-D solid** concrete (column, joint, deep beam) — triaxial/confinement | **`LadrunoConcrete3D`** (CDPM2: Menétrey-Willam + dual damage) | the fork flagship; real return map + ductility |
| 3-D solid, robust general-purpose / where CDPM2 is overkill | **`ASDConcrete3D`** (Petracca) — use `.from_fc()` | auto CEB-FIP backbone, very robust |
| **RC wall / panel / shell layer** (in-plane shear) | **`LadrunoRCConcrete`** (ASDConcrete3D spine + MCFT softening + cyclic interlock) | membrane shear is constitutive; shell `PlateFiber` view |
| **Fiber section** (beam-column) | uniaxial `Concrete02`/`04`/`ConcreteCM` (+ steel) | 1-D σ–ε; see the `concrete-uniaxial` skill |

The three nDMaterials are **fork-only** — a deck built with them runs only against a Ladruno build,
and the error bites at run time (emission always succeeds).

---

## 2. Define the material

apeGmsh materials are typed dataclasses on the `ops.nDMaterial.*` namespace; each call returns a
**handle** you pass to elements via `material=`.

```python
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results.capture.spec import DomainCaptureSpec

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)

# CDPM2 solid (positive magnitudes; the fork REQUIRES ft < fc):
conc = ops.nDMaterial.LadrunoConcrete3D(
    E=30.0e9, nu=0.2, rho=2400.0,
    fc=30.0e6, ft=3.0e6, Gf=120.0, Gc=30.0e3,   # Gf/Gc = fracture energy (N/m); Gc is the WEAKEST knob
    kupfer=1.16,                                 # eccentricity from fcc/fc (or e=… directly, in (0.5,1])
    auto_regularize=True,                        # each element scales its softening to its own size (§3)
    # implex=True, eta=…,                        # robustness tiers (§5.5)
)
```
Other kwargs map 1:1 to the parser flags: `Df=`, `As=`, `hardening=(qh0,Hp)`, `ductility=(Ah,Bh,Ch,Dh)`,
`lch=`, `ct_temper="none"|"alphat"|"proj"`, `hoop_k=/hoop_fy=` (the **BeamFiber** confined-fiber view only).

**ASDConcrete3D — the easy button** (auto backbone from `fc`):
```python
from apeGmsh.opensees.material.nd import ASDConcrete3D
conc = ASDConcrete3D.from_fc(E=30.0e9, v=0.2, fc=30.0e6, lch_ref=50.0e-3)   # note v=, not nu=
conc.check_element_size(lch=actual_h)   # apeGmsh-side warn if h exceeds l_max (see §3)
reg = ops.register(conc)                # register a pre-built instance → handle
```

---

## 3. Regularize — or your results are mesh garbage

Softening concrete **localizes**; without crack-band regularization the dissipated energy → 0 as you
refine. apeGmsh + the fork handle the scaling, but respect the **element-size ceiling**:
```
l_max = 2·E·Gf / ft²          (Bažant-Oh snapback limit)
```
- **`auto_regularize=True`** — each element scales softening to its own `getCharacteristicLength()`
  (or pass a fixed `lch=` calibrated for a known size).
- **Keep `h ≲ l_max`** in the cracking zone. `ASDConcrete3D.l_max()` / `.check_element_size()` are
  **apeGmsh-side** helpers that warn when you exceed it (above it the energy is floored → no longer
  mesh-objective). For the §6 numbers `l_max = 2·30e9·120/(3e6)² ≈ 0.8 m`, so `h ≲ 0.8 m`.
- Refine the crack zone with a `g.mesh.field` (distance + threshold) rather than globally; brittle `ft`
  shrinks `l_max` fast.

> This is **energy-objective**, not band-width-objective — the right trade for structural analysis.
> Don't reach for nonlocal/gradient (descoped, fork ADR 59).

---

## 4. Elements

| Use | Element | Notes |
|---|---|---|
| solid concrete, small strain | `ops.element.LadrunoBrick` | the fork host; `geom="linear"` (default) or `"corot"` |
| solid, tets from CAD | `FourNodeTetrahedron` / `TenNodeTetrahedron` | tets lock in bending — prefer hexes |
| solid, stock | `stdBrick`, `bbarBrick`, `SSPbrick` | `bbarBrick` reduces volumetric locking |
| **RC shell / wall** | `ASDShellQ4` (+ `LayeredShellFiberSection`) | the `LadrunoRCConcrete` host; `-corotational` for large rotation |

`g.mesh.partitioning.renumber(dim=…, method="rcm", base=1)` **before** `get_fem_data` — dense 1-based
IDs + RCM bandwidth for the direct solver.

---

## 5. The analysis chain — by regime

The **solver** is fixed by the golden rule; the **integrator** is chosen by the *event time scale*.
Set the chain primitives on `ops`:

```python
ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.UmfPack()                                  # UNSYMMETRIC — the golden rule
ops.test.NormDispIncr(tol=1e-6, max_iter=20)          # tol is in MODEL length units (m here); EnergyIncr is better post-peak
ops.algorithm.KrylovNewton()
```

### 5.1 Quasi-static with softening (pushover, panel) → displacement / arc-length
`LoadControl` fails at the limit point. Use displacement control, or arc-length for snap-back:
```python
ops.integrator.DisplacementControl(node=ctrl_id, dof=1, dU=0.1e-3)
# snap-back (post-peak unloads in BOTH load and disp):
# ops.integrator.LadrunoArcLength(s=…, alpha=…, stabilize=True)   # adaptive: jd=/ell_min=/ell_max=
ops.analysis.Static()
```

### 5.2 Seismic dynamic → HHT / generalized-α
Numerical damping suppresses the spurious high-frequency content from cracking:
```python
ops.integrator.HHT(alpha=0.9)     # OpenSees convention: alpha∈[2/3,1]; alpha=1 = Newmark (no damping), lower = more
ops.analysis.Transient()
```
Get the frequencies for Rayleigh damping from an eigen solve, then fit two-target coefficients:
```python
w = ops.eigen(num_modes=3)                                  # angular freqs (rad/s)
from apeGmsh.opensees.analysis.rayleigh import rayleigh_from_ratio
rc = rayleigh_from_ratio(f_i=w[0]/(2*3.14159), f_j=w[2]/(2*3.14159), zeta=0.05, stiffness="initial")
# apply rc.alpha_m / rc.beta_k (mass/stiffness proportional). NB: there is NO `ops.rayleigh(...)` method —
# use rayleigh_from_ratio(...) + the coefficients, or the fork frequency-band `ops.damping.*`.
```

### 5.3 Blast / impact / fast collapse → explicit (no tangent at all)
Explicit needs **no global solve**, so the softening that breaks implicit Newton is free — but it is
**conditionally stable**: `Δt ≤ h_min/c_d`, `c_d = √((K+4G/3)/ρ) ≈ 3700 m/s` for concrete ⇒ ~**µs**
steps. Only economical for short, high-rate events.
```python
ops.integrator.CentralDifferenceLadruno()     # or ExplicitBathe / ExplicitBatheLNVD
ops.analysis.Transient()
# Mass comes from material rho on the element (set rho=2400; don't ALSO add g.masses → double count).
# Explicit wants a LUMPED mass matrix.
```

**Selective mass scaling (SMS) — lift the µs CFL step.** The fork ships SMS: it inflates the mass of
sub-target (tiny) elements so the *bulk* timestep reaches `dt_target` (added mass capped by
`max_added_mass`), turning an otherwise intractable explicit run economical. **apeGmsh exposes it
directly** (PR #730):
```python
# lumped (DT2MS-style) — cheapest:
ops.integrator.CentralDifferenceSMS(dt_target=1e-5, max_added_mass=0.05)
# consistent (Olovsson) — PCG mass-solve; PRESERVES the fundamental frequency
# (f1 ≈ −0.2% vs lumped ≈ −53%); use when the scaled response must stay accurate:
ops.integrator.CentralDifferenceSMS(dt_target=1e-5, consistent=True)        # or pcg_tol=/pcg_max_it=
# Noh-Bathe hosts also take it:
ops.integrator.ExplicitBatheSMS(p=0.54, dt_target=1e-5, consistent=True)
ops.integrator.ExplicitBatheLNVDSMS(p=0.54, alpha=0.1, dt_target=1e-5)      # + FLAC damping
```
`dt_target` sets the floor the explicit step is lifted to; `max_added_mass` (default 0.05) caps the
distortion. ADR-36 (lumped) / ADR-38 (consistent). Pair with **bulk viscosity**
(`LadrunoBrick -bulkViscosity b1 b2`, ADR-52) to damp the volumetric shock noise mass scaling can excite.

### 5.4 Robustness — the escalation ladder + constitutive tiers
Softening + non-symmetry makes plain Newton fragile.
- **Algorithm ladder** (staged API): on a failed increment the loop walks down the rungs, then
  restores the fast one:
  ```python
  ladder = ops.strategy.Ladder(rungs=[
      ops.algorithm.KrylovNewton(),
      ops.algorithm.NewtonLineSearch(line_search="Bisection"),    # NB kwarg is line_search=, not type=
      ops.algorithm.ModifiedNewton(tangent="initial"),
  ])   # or ops.strategy.profile("non-smooth")
  ```
- **Constitutive tiers** (set on the material): `implex=True` (Tier-2 IMPL-EX — extrapolated stress +
  degraded-elastic secant, robust through snap-through, bounded Δt error; *still* unsymmetric solver,
  the dual-damage secant is only single-sign SPD) and `eta=…` (Duvaut-Lions; needs a positive `dt`;
  `eta=0` is byte-identical to inviscid). **They don't compose** — under `-implex`, `-eta` runs inviscid
  and the material warns.

---

## 6. Worked recipe — concrete column pushover (3-D solid), gravity then push

> Called a *plain* concrete column on purpose: confinement is **not** a keyword on a solid. It comes
> from **modeled transverse steel** (the embedded-rebar API, `g.reinforce(...)`, non-matching mesh) plus
> the CDPM2 triaxial response under real confining stress. `-hoop`/`hoop_k` is the **BeamFiber** (1-D
> fiber) view only — irrelevant to a `LadrunoBrick` solid.

```python
import numpy as np
import openseespy.opensees as opspy        # the live handle; on a Ladruno build: `import opensees as opspy`
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results.capture.spec import DomainCaptureSpec

with apeGmsh(model_name="column") as g:
    # … geometry + PGs "Body","Base","Top" … then:
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.LadrunoConcrete3D(
        E=30e9, nu=0.2, rho=2400, fc=30e6, ft=3e6, Gf=120, Gc=30e3,
        kupfer=1.16, auto_regularize=True)
    ops.element.LadrunoBrick(pg="Body", material=conc)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ctrl_id = int(fem.nodes.get_ids(pg="Top")[0])         # the control node

    # --- STAGE 1: gravity / axial pre-load (concrete capacity is axial-sensitive) ---
    tsg = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=tsg) as pg_:
        pg_.load(pg="Top", forces=(0.0, 0.0, -1.0e6))     # axial N on the column head
    ops.constraints.Transformation(); ops.numberer.RCM(); ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-6, max_iter=20); ops.algorithm.KrylovNewton()
    ops.integrator.LoadControl(incr=0.1); ops.analysis.Static()
    ops.run(wipe=False)                                   # build + leave the domain live
    opspy.analyze(10)                                     # apply gravity in 10 increments
    opspy.loadConst('-time', 0.0)                         # hold gravity, reset pseudo-time

    # --- STAGE 2: lateral push under displacement control, capturing the curve ---
    tsp = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=tsp) as pat:
        pat.load(pg="Top", forces=(1.0, 0.0, 0.0))        # UNIT reference load (sets push direction)
    n_steps, dU = 600, -0.05e-3
    ops.integrator.DisplacementControl(node=ctrl_id, dof=1, dU=dU)

    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(pg="Top",  components="displacement")
    spec.nodes(pg="Base", components="reaction_force")
    with ops.domain_capture(spec, path="pushover.h5") as cap:
        cap.begin_stage("pushover", kind="static")
        for k in range(n_steps):
            if opspy.analyze(1) != 0:
                print(f"non-convergence at step {k}"); break
            cap.step(t=(k + 1) * abs(dU))
        cap.end_stage()

# --- read the capacity curve back, by name ---
om = OpenSeesModel.from_h5("pushover.h5", fem_root="/model")
with Results.from_native("pushover.h5", model=om) as r:
    ux = np.asarray(r.nodes.get(pg="Top",  component="displacement_x").values)[:, 0]
    V  = -np.asarray(r.nodes.get(pg="Base", component="reaction_force_x").values).sum(axis=1)
# (ux, V) is the pushover curve.
```

### Cyclic RC shear wall (variant)
Swap `LadrunoConcrete3D`+`LadrunoBrick` → `LadrunoRCConcrete`+`ASDShellQ4` (via
`LayeredShellFiberSection`), and drive a **prescribed cyclic displacement** with a protocol series +
an `sp` history on the control node (not monotonic `DisplacementControl`):
```python
proto = ops.timeSeries.ASCE41Protocol(...)    # or ModifiedATC24Protocol / FEMA461Protocol
with ops.pattern.Plain(series=proto) as pat:
    pat.sp(node=ctrl_id, dof=1, value=1.0)     # unit-scaled cyclic drift history
# keep the unsymmetric solver + the Ladder; LadrunoArcLength helps through the cyclic snap-backs.
```
> ⚠️ `LadrunoRCConcrete`'s aggregate-interlock block is **SI-only** (`fc` in MPa, `w`/`a_g` in mm) — a
> US-units model silently mis-scales the crack-shear cap.

---

## 7. Convergence troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| diverges immediately, residual explodes | symmetric solver on a non-symmetric tangent | `ops.system.UmfPack()`/`FullGeneral()` |
| diverges right at the **peak** load | `LoadControl` at the limit point | `DisplacementControl` / arc-length |
| stalls in **post-peak** softening | indefinite tangent | the `Ladder` (line search → initial-stiffness); then `implex=True` or `eta=…`; prefer `EnergyIncr` |
| results change with mesh refinement | no/over-coarse regularization | `auto_regularize=True`; keep `h ≲ l_max=2EGf/ft²` |
| `eta`/`implex` "does nothing" | needs a positive `dt`; off by default | set `dt`; `eta>0`/`implex=True` (not both) |
| noisy seismic response | undamped Newmark | `HHT(alpha<1)` + Rayleigh (via `eigen` + `rayleigh_from_ratio`) |
| explicit blows up after a few steps | Δt above CFL | shrink `dt ≤ h_min/c_d`; mass-scale; bulk viscosity |
| pushover runs but gives a wrong/flat curve | no reference load pattern on the control DOF | add the `pattern.Plain` + unit `pat.load` before `DisplacementControl` |

---

## 8. Pitfalls

1. **Unsymmetric solver is mandatory** — the most common failure. (§golden rule.)
2. **`ops = apeSees(fem)`**, not `g.opensees` (removed). Materials → `ops.nDMaterial.*`, elements →
   `ops.element.*`, BCs → `ops.fix(pg=, dofs=)`.
3. **`DisplacementControl` needs a reference `pattern.Plain` with a unit load on the controlled DOF** —
   `fix` + geometry loads alone do not push anything.
4. **Gravity first.** Concrete lateral capacity is axial-sensitive — run a gravity stage and
   `loadConst` before the push.
5. **Fork-only materials** error at run time, not at emit — run against a Ladruno build.
6. **`ft < fc`, positive magnitudes** — the fork rejects `ft ≥ fc` (the apex `√3·fc/m0` blows up).
7. **Mesh size vs `l_max`**; **`renumber(method="rcm")` before `get_fem_data`**; **`Gc` is the weakest
   CDPM2 knob** (sensitivity-check it); **RC interlock is SI-only**; **mass from `rho` OR `g.masses`,
   not both**.

---

## 9. See also
- `opensees-concrete` skill — constitutive theory + numerics (`cdpm2_grassl.md`, `ladruno_concrete3d.md`,
  `ladruno_rcconcrete.md`, `concrete-numerics`).
- `docs/examples/pushover-steel-frame.md` — the canonical pushover (this recipe mirrors it).
- Integrator study (ADR-49 / ADR-52); `opensees-expert` for hand-edited decks.
