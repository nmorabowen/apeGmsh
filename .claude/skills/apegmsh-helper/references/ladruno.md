# OpenSees fork (Ladruno) integration
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

apeGmsh can target the **Ladruno fork** of OpenSees (`nmorabowen/OpenSees`,
branch `ladruno`) in addition to stock `openseespy`. The fork adds features
apeGmsh emits/reads that stock OpenSees does **not** have. Stock `openseespy`
stays first-class — the fork is **opt-in**; gate fork-only features at the
point of use, never force the fork.

## Targeting a build + gating fork features

*Which* OpenSees runs (`OpenSeesTarget`) and the precedence/inert-live
rules are in `opensees-bridge.md` → "Which OpenSees runs" — not repeated
here. The fork-specific idiom: **never infer capability from the path**;
branch on the **live** probe, and use `require_fork=True` to fail loud at
the live boundary instead of three primitives deep.

```python
ops = apeSees(fem, opensees=OpenSeesTarget(require_fork=True))   # live-path assertion
if ops.capabilities().has_fork:      # OpenSeesCapabilities(has_fork=, has_profiler=, version=)
    ops.element.BezierTet10(pg="Body", material=m)
else:
    ops.element.FourNodeTetrahedron(pg="Body", material=m)
```

`has_fork` (via `ops.capabilities()`) tracks the fork-only `profiler`
command. The **live backend resolver** (`opensees/emitter/live.py`) detects
the fork separately, by the fork-only `criticalTimeStep` symbol, and now
**prefers the Ladruno fork**: with `$APEGMSH_OPENSEES_BIN` set it adds that
dir to the DLL path and imports the fork; else a bare `import opensees` that
exposes `criticalTimeStep` is taken as the fork; else it falls back to stock
`openseespy.opensees`. `get_backend_name()` → `"ladruno-fork"` /
`"stock-openseespy"`. Running a fork-only element (`_FORK_ONLY_ELEMENTS`:
`LadrunoBrick`, `LadrunoDispBeamColumn`, `LadrunoIMKBeam`, `LadrunoRigidBody`,
Bézier, …) on a stock build fails loud at the live boundary; deck emission
(`.tcl`/`.py`) works on any build.

### Which *build* — pinning the engine (not just the flavour)

`get_backend_name()` says fork-or-stock; it does **not** say which fork
build. `get_backend_build()` (and `ops.capabilities().build`) returns the
40-char git hash the engine binary was compiled from — the fork's
`ladrunoBuild` command (fork PR #718) — or `None` on stock / a fork build
predating it.

```python
from apeGmsh.opensees.emitter.live import get_backend_build

stamp = get_backend_build()          # '0bf66bbb…' or None
if stamp != expected_hash:           # fail loud, do not measure a mystery build
    raise RuntimeError(f"wrong engine build: {stamp}")
```

**The stamp is baked at CMake *configure* time, not at compile time — so it
can be STALE.** Check out a new commit in an already-configured build tree and
rebuild incrementally, and the binary contains the new code while
`ladrunoBuild` still reports the hash it was configured at. Measured
2026-08-24: a worktree binary reporting `e7555f2c9` demonstrably contained the
ADR-85 F1 winding lane that only landed in `52938956b`, emitting F1's own named
FATAL verbatim. So a mismatch does **not** tell you the binary is older than
the stamp — it may be newer, which is the more dangerous direction, because a
measurement then gets attributed to a build that never produced it. Treat the
stamp as trustworthy only after a `build.bat clean` / `rebuild` (both wipe
`build/`, forcing reconfigure). To identify a suspect binary, probe a
*behaviour* only the newer code has — a named refusal is ideal, since it needs
no converged run.

Use it in any harness whose results are attributed to a specific engine
(A/B comparisons, validation gates, defect probes). Before this command the
only build identifier was the splash banner's `Ladruno OpenSees build:`
line — and apeGmsh **suppresses that banner** (`LADRUNO_OPENSEES_QUIET=1`,
set by the resolver), which is precisely how the fork's TIMs T1 incident
attributed a probe to the wrong build. It also catches a stale
`opensees.pyd`: rebuild, and if the hash did not change, the build did not
take.

## Fork-only features apeGmsh touches

| Feature | Kind | Notes |
|---|---|---|
| **BezierTri6 / BezierTet10** | elements | fork-only Bézier (Bernstein) continuum elements — typed primitives `ops.element.BezierTri6/BezierTet10` (Tet10 also takes `-geom linear/corot/finite` + `-fbar`) |
| **LadrunoBrick** | element | fork-only unified 8-node hex (tag 33002) — typed `ops.element.LadrunoBrick` with `-formulation`/`-geom`/`-hourglass`/`-damp` |
| **ExplicitBathe / ExplicitBatheLNVD / CentralDifferenceLadruno** | explicit integrators | not in stock OpenSees |
| **EnergyBalance** | recorder | fork-only |
| **`.ladruno` recorder** | recorder | `recorder ladruno` — note `.ladruno`, a sibling of the vanilla `.mpco` |
| **LadrunoPorousOverlay family** | load pattern + driver + recorder channels | fork-only persistent-fluid staggered u-p (PATTERN tag 33022, fork ADR-73, shipped incl. both explicit lanes) — **no typed primitive yet** (follow-up ADR; see the section below) |
| **LadrunoShellModifier** | section | fork-only ETABS-style stiffness modifiers wrapping any order-8 plate section (fork ADR 91) — typed `ops.section.LadrunoShellModifier` |
| **stack profiler** | control command | `ops.profiler.*` — brackets the analyze loop; writes `profile.h5` |

The three **explicit integrators** are emittable via typed primitives:
`ops.integrator.ExplicitBathe(p=0.54, cfl=True, ...)`,
`ops.integrator.ExplicitBatheLNVD(p=0.54, alpha=0.8, ...)`, and
`ops.integrator.CentralDifferenceLadruno(cfl=True, ...)`. They share an order-free
option grammar (`cfl` / `cfl_abort` / `tangent` / `recompute=N` /
`lump="rowsum"|"diagonal"` / `verbose` / `divergence=f`). Emission works on **any**
build (it's just an `integrator <Type> ...` line); the fork is required only to
*run* the deck — stock OpenSees raises "unknown integrator" at `ops.analyze(...)`.
Defaults: Bathe `p∈(0,1)`=0.54, LNVD `alpha∈[0,1)`=0.80; `lump` defaults to RowSum
on the Bathe schemes and Diagonal on CentralDifferenceLadruno (omit to inherit).
Pair with `ops.system.Diagonal()` (lumped diagonal mass) for explicit runs.

## LadrunoShellModifier (cracked-section stiffness for shells)

`section LadrunoShellModifier $tag $innerSecTag <-f11 v> <-f22 v> <-f12 v>
<-m11 v> <-m22 v> <-m12 v> <-v13 v> <-v23 v> <-mass v>` — a **decorator**
section that scales the tangent, resultants and density of any order-8 plate
section. This is the ETABS/ACI cracked-section idiom expressed without
disturbing the wrapped section's constitutive law:

```python
slab = ops.section.ElasticMembranePlateSection(E=25e6, nu=0.2, h=0.3, rho=2.4)
# Cracked shear wall per ACI 318-25 6.6.3.1.1.
wall = ops.section.LadrunoShellModifier(inner=slab, f11=0.35, f22=0.35, f12=0.35)
ops.element.ASDShellQ4(pg="Walls", section=wall)
```

- **`inner` should be `ElasticMembranePlateSection` — that is the supported
  case.** The fork accepts any order-8 plate section (it refuses anything else
  at parse time), but wrapping `LayeredShell` / `LayeredShellFiberSection` or
  any other path-dependent section raises
  `ShellModifierNonlinearInnerWarning`. Modifiers are a stiffness fiction: the
  fork drives the wrapped section at a scaled deformation `S·e`
  (`scale[i] = sqrt(mod[i])`) and scales its resultants back by `S`. For an
  elastic inner that is exactly a stiffness scale. For a path-dependent inner
  the constitutive law integrates at a fictitious strain, so yield and damage
  land in the wrong place, and per-layer stresses read from the inner materials
  are the response at `S·e` — no post-hoc scaling recovers the physical values.
  Model a nonlinear stiffness reduction constitutively instead.
- Every flag defaults to `1.0` and **only non-default flags are emitted**, so an
  all-defaults wrap is a no-op and a generator can wrap unconditionally.
- `0.0` is accepted (ETABS-legal) but leaves the section singular in that
  response mode; the fork warns once per `section` command. Negative is refused.
- Modifiers apply as a congruence `D' = S·D·S`, `S = diag(√f11 … √v23)`, so the
  Poisson coupling moves as `√(f11·f22)`. Indistinguishable from a plain block
  scale whenever `f11 == f22`. A diagonal-only rescale was rejected (ADR 91 §4):
  it destroys positive definiteness at exactly the cracked-wall values.
- **A wall cracked with `m11` is a SILENT NO-OP.** In-plane bending of a wall or
  deep beam is carried by MEMBRANE action (`sigma_xx` over the depth), so the
  modifier that softens it is **`f11`**. The `m` modifiers are *out-of-plane*
  plate bending and the `v` modifiers *out-of-plane* (transverse) shear —
  neither is in the in-plane load path, and neither warns. Reaching for "the
  bending ones" to crack a shear wall gets you no cracking at all. The fork
  pins this as gate G10 rather than leaving it to prose. Corollary: for a member
  loaded in its own plane, cracking all eight is indistinguishable from cracking
  only `f11`/`f22`/`f12`.
- **No weight modifier.** OpenSees derives shell self-weight from the same
  `getRho()` that builds the mass matrix, so a weight flag could only alias
  `mass`. Scale self-weight at the load level instead.
- `Ep_mod` on `ElasticMembranePlateSection` (upstream, optional 5th arg, now
  exposed) is exactly equivalent to `m11=m22=m12=v13=v23=r`. Prefer this
  section — it is strictly more expressive.

**Validation status.** The fork's G9/G10 gates cross-check a cracked shell
against the equivalent cracked FRAME member (a flexure-controlled cantilever,
`L/d = 10`, built both ways): the shell/frame deflection RATIO is identical
gross and cracked — 2.8571 = 1/0.35 at every mesh density from 20x2 to 120x24 —
so the modifier scales straight through the membrane-locking discretisation gap.
That is the accepted validation. Bit-parity with ETABS itself is **not**
established and is not being pursued: CSI does not document whether ETABS uses
the same `D' = S*D*S` congruence, so for strongly unequal `f11`/`f22` the
Poisson coupling term may differ from ETABS. Equal `f11 == f22` — every standard
cracked-wall recipe — is unaffected.

The ETABS import path (`apeGmsh.interop`) consumes area modifiers
automatically; see `references/interop.md`.

## More fork bridge clusters (all emit on any build, RUN only on the fork)

These typed primitives landed after the original five-feature ledger. Names
are exact; full per-field signatures are in source (`grep` the class) — read
those before relying on exact kwargs.

**Concrete / J2 nDMaterials** (`material/nd.py`, `ops.nDMaterial.<Type>`):
`LadrunoJ2` (:890, combined Voce+Chaboche von Mises, `-iso`/`-kin`/`-damage`/
`-autoRegularization`/`-implex`), `LadrunoJ2Finite` (:996, finite-strain
J2 — no damage/regularization), `LadrunoConcrete3D` (:1290, plastic-damage
concrete, `E`/`nu`/`fc`/`ft`/`Gf`/`Gc` + regularization + `-implex`),
`LadrunoRCConcrete` / `LadrunoRCFiniteStrain` (:1807/:1833, RC plastic-damage
+ MCFT), `LadrunoCohesiveHingeBiaxial` (:1861). These follow the ASDConcrete
"apeGmsh owns the curve, always `-autoRegularization $lch_ref`" idiom.

**ASDPlasticMaterial3D geotechnical/rock materials** (`material/nd.py`,
`ops.nDMaterial.<Type>`, ADR 0105 / fork ADR-94). Three typed helpers wrap
the generic templated class, each emitting exactly its combination's
parameter schema — a foreign or missing model-parameter name is a
`ValueError` at construction, not a run-time fork refusal:
`MohrCoulombSoil(*, c, phi, psi, E, nu, rho=0.0, ds=1e-5, ...)` (8 names:
`YoungsModulus, PoissonsRatio, MC_phi, MC_c, MC_ds, MC_psi, MassDensity,
InitialP0` — the pre-ADR-94 21-name superset is refused by the fork),
`MohrCoulombTensionCutoffSoil(*, ..., tension_cutoff, ...)` (adds
`TC_min_stress`; `tension_cutoff` is **tension-positive**, `>= 0`, capped
by the fork at the Mohr-Coulomb apex `c·cot(phi)`), and
`HoekBrownRock(*, E, nu, sigci, mb, s, a, mb_psi=None, ds=0.0, ...)`
(`mb_psi=None` → `mb`, associated flow; deriving `mb`/`s`/`a` from
`mi`/`GSI`/`D` — Hoek & Brown 2018 — is the caller's job, not duplicated
here; the fork yields at `-s·sigci/mb` in net tension, fork PR #806). All
three default `strict_convergence=True` and `tangent_type="Continuum"`
(was `"Secant"` pre-ADR-0105) and expose `f_relative_tol: float = 0.0`;
PlaneStrain-wrappable. No DruckerPrager / VonMises typed helpers — use
the generic `ASDPlasticMaterial3D(yf=, pf=, el=, iv=,
model_parameters=(...), ...)` for those (and for any other
table-covered combination): a name outside the schema or a missing
required one is a `ValueError` naming the schema; a combination with a
component outside `_ASDP_PARAMS_BY_COMPONENT` (e.g. the StiffSoil
family) is accepted unchanged and the fork validates it instead.

`integration_options` tokens are validated at construction:
`integration_method` ∈ `Forward_Euler, Forward_Euler_Subincrement,
Backward_Euler, Modified_Euler_Error_Control,
Runge_Kutta_45_Error_Control` (`Backward_Euler` is the only one the fork
documents as supported after ADR-94; any other raises
`ASDPlasticIntegrationWarning`) — `Backward_Euler_LineSearch` and
`Runge_Kutta_45_Error_Control_old` **raise** instead (fork ADR-94 M7/M8:
both measured broken — the line-search variant ignores
`n_max_iterations` and returns success for a strain increment the
element never asked for; the RK45 variant's NaN guard calls `exit()` on
the whole process). `tangent_type` and `return_to_yield_surface` tokens
are validated the same way.

**Host gate.** `strict_convergence` only reaches the analysis on a host
MEASURED to propagate a material refusal — `LadrunoBrick` /
`TenNodeTetrahedron`. `stdBrick` swallows it unconditionally (fork
ADR-94 B2: `Brick::update()` returns 0 regardless, measured 20/20
"success" on a deck `LadrunoBrick` refuses 0/20); the build warns
`ASDPlasticHostWarning` once per deck
(`_internal/build.py::validate_asdplastic_host`, keyed on the measured
per-element flag, not on element names). A vanilla-host deck stays
legal — it was the SSI-1 default — it just means the fail-loud contract
never reaches it.

**Minimum fork build** `bbf657d49` (`ASDP_MIN_FORK_BUILD`) for
`strict_convergence` / `f_relative_tol` — an older parser drops both
silently. Rock-scale decks should set `f_relative_tol` (`1e-8` is the
fork's suggested start for Hoek-Brown at 50 MPa; `1e-7` completed an MC
problem at ×1e9 in the battery, fork ADR-94 M5) — the absolute
tolerance alone is a verdict on the unit system. Keep `MC_ds > 0`
(default `1e-5`) on Mohr-Coulomb decks that reach a corner (`ds=0` was
refused at step 17 of the measured deviatoric-leg battery). Behaviour
change stated plainly: a deck that used to commit an inadmissible or
non-converged state now **fails at the step** — that is the ADR 0105
contract, not a regression.

**Recording plastic strain / material-level responses (ADR 0105
Amendment 1).** Plastic strain, `eqpstrain`, and the other ASDP scalars
are answered by the MATERIAL, not the element; `ops.recorder.Ladruno` /
`MPCO` reach them only through `elem_responses=("material.<Token>",
...)`, e.g. `elem_responses=("material.pstrain", "material.eqpstrain",
"material.PStress", "material.J2Stress", "material.VolStrain",
"material.J2Strain", "material.BackStress")`. A bare `pstrain` /
`pstrains` / `eqpstrain` / `PStress` / `J2Stress` / `VolStrain` /
`J2Strain` token is **refused at construction** (naming the
`material.<token>` form to use) because the fork's
`ASDPlasticMaterial3D::setResponse` records nothing for it — no error,
no bucket, silently.

Reader-side (`.ladruno`/`.mpco`, keyed by the **bucket token**, resolved
by column position — never by column label, because
`material.PStress` labels its column `p`, which collides
case-insensitively with a force-based beam's section axial force `P`):
`material.pstrain` / `material.eqpstrain` keep the existing
self-describing path (`plastic_strain_xx..xz`,
`equivalent_plastic_strain`); the five buckets above land on
`material_mean_stress`, `material_j2_stress`,
`material_volumetric_strain`, `material_j2_strain`, and
`back_stress_xx..xz` (6 columns, the material's Voigt order 11, 22, 33,
12, 23, 13 — the same order the `epsP1..` columns already use); the
scalar internal-variable buckets add `yield_stress` / `dp_cohesion` /
`cap_pressure` / `eps_qp_shear`. The `material_*` names are
**provenance-distinct on purpose** from the reader's tensor-derived
`mean_stress` / `j2_stress` / `volumetric_strain` / `j2_strain`
(measured equal on the fork — same sign, both `trace/3`, both
`J2 = 1/2 s:s` — but aliasing them would need a per-material
sign/definition audit, not a measurement, so the two provenances stay
separately named). A `material.<Token>` bucket that neither table can
name now warns `GaussColumnDroppedWarning` once per bucket — naming the
bucket, the unmapped column labels, and the element class — instead of
vanishing silently.

**Beam-column elements** (`element/beam_column.py`, `ops.element.<Type>`):
`LadrunoDispBeamColumn` (:661, displacement-based + crack-band `lch`, optional
`-nl` bowing strain, `-hinge`/`-hingeY`/`-hingeBiaxial` lumped hinges) and
`LadrunoIMKBeam` (:807, concentrated-plasticity IMK). Both take `pg=` +
`transf=` like any beam.

**Analysis cluster** (`analysis/integrator.py`, `ops.integrator.<Type>`):
`LadrunoArcLength` (:612, adaptive Ramm arc-length + viscous stabilisation),
`LadrunoDynamicRelaxation` (:773, matrix-free path-follower),
`LadrunoIndirectControl` (:906, weighted multi-DOF displacement control).

**Selective Mass Scaling (SMS) explicit integrators** (`ops.integrator.<Type>`):
`CentralDifferenceSMS` (:1166), `ExplicitBatheSMS` (:1235),
`ExplicitBatheLNVDSMS` (:1292) — each augments nodal mass to reach `dt_target`
under a `max_added_mass` cap (`-maxAddedMass`, default 0.05), `lump=
"rowsum"|"diagonal"|"hrz"`; a `consistent=True` PCG variant unlocks
`pcg_tol`/`pcg_max_it`.

```python
ops.integrator.CentralDifferenceSMS(dt_target=1e-4, max_added_mass=0.05, lump="rowsum")
```

**Constraints coverage** (fork): `equalDOF_Mixed` (mixed retained/constrained
DOF pairs, ADR 0069 — `ops.equalDOF_Mixed(master, slave, n, rdof1, cdof1, …)`),
`LadrunoRigidBody` as an **element** (`g.constraints.rigid_body(...,
as_element=True, mass=None, omega=(wx,wy,wz))` — `-omega` initial spin, ADR
0071), `LadrunoEmbeddedNode` (the `enforce="penalty_al"` / `g.embed` target),
and two constraint **handlers**: `LadrunoProjection`
(`ops.constraints.LadrunoProjection(verbose=, project_ics=, ic_tol=)` —
momentum-conserving projection, auto-picked for explicit `enforce="equation"`
ties) and `LadrunoContact` (auto-emitted by the bridge whenever
`g.constraints.contact` / `contact_plane` records are present — activates the
fork contact solve: NTS/mortar, the mortar-only edge-edge fallback, the
`-cell` broad-phase knob, and `contact_plane` rigid analytical planes, ADR
0073). On a **flat** deck do **not** also call
`ops.constraints.LadrunoContact()` — that duplicates the auto-emitted line.
Staged contact still requires each stage to declare the handler (ADR 0092).
See `api-cheatsheet.md` constraints + `opensees-bridge.md` for the
`enforce=` routes.

**PARDISO / MUMPS (`ops.system.Pardiso` / `Mumps`, fork ADR-75).** Typed
emit always writes `-matrixType 0|1|2` as an int (including `0` for
`"unsymmetric"`). Contact, friction, non-associated flow, `LadrunoUP`, and
finite-strain F-bar need unsymmetric — use the default or
`matrix_type="unsymmetric"` (deck shows `system Pardiso -matrixType 0`).
`"symmetric"` / `"spd"` are half-storage and wrong on those tangents.
Thread count is env-only (`MKL_NUM_THREADS` / `OMP_NUM_THREADS` before
process start). Optional `krylov=L` (CGS reuse, not with `"symmetric"`)
and `stats=True`.

## DruckerPrager on collapse decks (fork ADR-95)

The vanilla UW `DruckerPrager` apeGmsh emits from `material/nd.py::DruckerPrager`
has a dead branch in its two-surface return map: the tension-cutoff residual row
is never assembled, so a Gauss point crossing the cutoff keeps an unreturned
stress and a pathological consistent tangent (whose radial-return term also
divides by the returned norm instead of ‖η_trial‖). On a Prandtl–Reissner
strip-footing deck that kills **every quadratic element** — `LadrunoBrick20`,
`TenNodeTetrahedron` and `BezierTet10` sit on the step floor at 30–77 % of the
Prandtl load while the linear b-bar hex plateaus at the right answer, a false
collapse rather than a mesh or material problem. Fork PR #803 (commit
`31322a47a`) repaired it; it merged into `ladruno` as **`61b3efa04`** on
2026-09-08, and that hash is the floor for everything below
(`DP_ADR95_MIN_FORK_BUILD` in `material/nd.py`). A build older than it — the
venv's `1652f945c` until the fork is rebuilt — still fails these decks.

Quadratic solids then become usable on collapse decks. Post-fix on the fork's
coarse strip (h0 = 1.0 m, two elements across B, s/B 0.15, q/q_exact):
`LadrunoBrick -bbar` 1.085 — unchanged before and after, so the linear leg is no
discriminator — `LadrunoBrick20 -formulation uri` 0.976 (the fork-only 20-node
hex, `ops.element.LadrunoBrick20(…, formulation="std"|"uri")`, `uri` = the
C3D20R analog), `BezierTet10 -bbar` 1.040, `BezierTet10` std 1.182,
`TenNodeTetrahedron` 1.170. Prefer b-bar (exactly isochoric at ψ = 0, tightest
plateau); standard-integration tets over-shoot (locking) and the
reduced-integration H20 loses rank once all eight Gauss points yield — 9.5 %
spurious volumetric increment in the collapse mechanism, ~1 000 failed Newton
attempts per push. Coarse-mesh numbers, not converged values.

Deck rules. A small, explicitly non-physical `sigmaY` — 0.2 kPa puts the cutoff
at I1 = √(2/3)·σ_y/ρ ≈ 0.8 kPa — is a legitimate **apex regulariser** for
weightless or lightly confined frictional decks; call it a regulariser, never
cohesion (before the fix it was decorative). Bernstein-consistent loads stay
mandatory on Bézier elements (ADR 0091, `basis="bernstein"`; the
`WarnLoadBasisMismatch` guard at `build()` catches the common case).
Non-associated DP tangents are unsymmetric — `UmfPack`/`Pardiso`/`Mumps`/
`FullGeneral`, never `ProfileSPD`/`BandSPD` (see the PARDISO/MUMPS note above).
Budget the push, not the element: `LadrunoBrick -bbar` 1 386 DOF 31 s (0.10 s per
attempt); `LadrunoBrick20 -uri` 4 659 DOF 481 s, 40 % of it 1 051 failed attempts
around the corner Gauss points (0.19 s); `BezierTet10` std / `-bbar` 7 749 DOF
466 / 533 s with zero failed attempts; `TenNodeTetrahedron` 727 s (0.48 s).

Any deck that reached the cutoff answers differently post-fix — it was wrong
before; cone-only paths move ≤ 1.3e-5 relative but converge faster (a fork gate
leg went 1390 s → 181 s). The branch also adds two read-only **material**
responses: `ladrunoBranch` (8 floats — branch 0 elastic / 1 cone / 2 cutoff /
3 corner, gamma0, gamma1, f1_trial, f2_trial, forcedAccept, I1, detAmin) and
`ladrunoTangent` (36 floats), forwarded by `LadrunoBrick`, `LadrunoBrick20`,
`BezierTet10` and `TenNodeTetrahedron`. Record them the ADR 0105 way,
`material.ladrunoBranch` — the bare token records nothing and the recorder
refuses it — and read an empty reply as the capability probe for a pre-#803
engine. A per-Gauss-point census at every station multiplies wall time ~5–8×:
sample at stations, not steps. The live `DomainCapture` path is not adopted.

## LadrunoBrick (unified 8-node hex)

Fork-only 8-node hexahedron (class tag **33002**, Gmsh hex8 / etype 5) that folds
the anti-locking treatment into one `-formulation` selector and an orthogonal
`-geom` kinematics selector — one class reproduces upstream `stdBrick`/`bbarBrick`/
`SSPbrick` where they overlap and adds the cheap explicit hex:

```python
ops.element.LadrunoBrick(pg=…, material=m,
                         formulation="std",      # std|bbar|uri|ssp|eas
                         geom="linear",          # linear|corot|finite
                         hourglass=None,         # uri only: viscous|stiffness|physical
                         hourglass_coeff=None,
                         lumped=False, body_force=None,
                         damp=None)              # element-flag -damp (ADR 0053)
```

```
element LadrunoBrick $tag $n1..$n8 $matTag [-formulation std|bbar|uri|ssp|eas]
    [-geom linear|corot|finite] [-hourglass viscous|stiffness|physical [$coeff]]
    [-lumped] [-b $bx $by $bz] [-damp $dampTag]
```

Key points (apeGmsh fails loud at construction, mirroring the fork's parse guards):

- **`formulation`:** `std` (full integration, default), `bbar` (mean-dilatation),
  `uri` (1-pt reduced + hourglass control), `ssp` (stabilized single-point), `eas`
  (true Simo–Rifai enhanced assumed strain).
- **`geom` `corot`/`finite` ship `std`/`bbar` only** — `uri`/`ssp`/`eas` under
  corot/finite raise (deferred in the fork). `finite` needs a finite-strain
  material; `finite` + `bbar` = F-bar (unsymmetric tangent → `FullGeneral`/`UmfPack`).
- **`hourglass` is `uri`-only** (raises with any other formulation); the optional
  `hourglass_coeff` requires `hourglass` to be set. `viscous` is explicit-only.
- **`lumped`** emits `-lumped` (diagonal mass) — required for explicit integrators.
- **`damp`** attaches a `Damping` object via the element's own `-damp` flag
  (ADR 0053 element-flag attach) — honoured **only** with `std`/`bbar`; apeGmsh
  raises for the other formulations rather than letting the fork silently drop it.
  Defaults (`std`/`linear`) are elided so decks stay byte-clean.
- **Result reads** go through the usual `results.elements.gauss.get(...)`; the
  recorder always returns 8-GP `Vector(48)` (single-point forms mirror slot 0).

## Bézier elements (`BezierTri6` / `BezierTet10`)

Two fork-only Bézier (Bernstein) continuum elements (Kadapa 2018), exposed as
typed primitives:

```python
ops.element.BezierTri6(pg=…, thickness=…, material=m, plane_type="PlaneStrain",
                       bbar=False, consistent_mass=False,
                       pressure=None, rho=None, body_force=None)   # 2D, 6 nodes
ops.element.BezierTet10(pg=…, material=m, bbar=False, consistent_mass=False,
                        rho=None, body_force=None, pressure=None,
                        geom="linear", fbar="centroid")            # 3D, 10 nodes
```

Emit grammar is **flag-prefixed** (each option independently optional), unlike
`SixNodeTri`'s positional tail:

```
element BezierTri6  $tag $n1..$n6  $thick $type $matTag [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]
element BezierTet10 $tag $n1..$n10 $matTag [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p] [-geom linear|corot|finite] [-fbar centroid|mean_dilatation]
```

Key points:

- **`geom` (Tet10): `linear` (default) / `corot` / `finite`.** `corot` = large
  rotation / small strain (EICR); `finite` = large strain (updated-Lagrangian),
  which needs a finite-strain material (`setTrialF(F)`, e.g. `nDMaterial LogStrain`)
  — the fork rejects a small-strain material there at run time. `finite` + `bbar` =
  **F-bar** (volumetric-locking cure; unsymmetric tangent → use `FullGeneral`/
  `UmfPack`). `pressure` is **rejected** under `corot`/`finite`. Defaults elide the
  flag so existing decks stay byte-identical.
- **`fbar` (Tet10): `centroid` (default) / `mean_dilatation`** — only meaningful with
  `bbar=True` + `geom="finite"`; apeGmsh raises if set otherwise.

- **`plane_type` (Tri6 only) accepts ONLY `PlaneStrain` / `PlaneStress`** — not the
  `*2D` spellings `SixNodeTri` tolerates (the fork factory rejects them).
- **B-bar guard (Tri6):** `bbar=True` under `PlaneStress` warns
  (`BezierBBarPlaneStressWarning`) and drops the `-bbar` flag (mirrors the fork's
  D5 warn-and-disable). Tet10 has no plane-stress degeneracy, so B-bar is always
  valid (no guard).
- **Node order is verbatim Gmsh.** On a straight-sided mesh the Gmsh `tri6`
  (etype 9) / `tet10` (etype 11) nodes coincide with the element control points, so
  connectivity passes through unpermuted. The tet10 mid-edge order is
  `(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)` — machine-precision-locked (the O11 test).
- **Fork required only to RUN.** Emission (`ops.tcl` / `ops.py`) works on any build;
  running in-process (`ops.run()` / `ops.analyze()`) on a stock build raises a clear
  *"element BezierTri6 requires the Ladruno fork build … use the direct-drive
  fallback"* error rather than a cryptic openseespy failure.
- **Direct-drive fallback (no apeGmsh change needed).** The elements also run via
  *direct-drive*: mesh a straight-sided domain to T6/T10 on stock py3.11, dump
  `nodes` + `fem.elements.<group>.connectivity` to JSON, and feed those verbatim to
  `ops.element('BezierTri6'|'BezierTet10', …)` on the fork build — Gmsh order is
  byte-identical to the control-point order. See the fork's
  `bezier_apegmsh_integration.md`.

**Result reads** go through the usual `results.elements.gauss.get(...)`. The
`.ladruno` reader is self-describing (`FAMILY="bernstein"` + `QUADRATURE/GP_PARAM`),
so GP stress/strain (axis-form `sigma_xx`/`eps_xx`/`gamma_xy` tokens) and the GP
**world** coordinates both come straight from the file — `slab.global_coords(fem)`
reconstructs `x = B(ξ)·X` via the neutral `apeGmsh._basis` Bernstein evaluator
(never a catalog GP order). The committed pipeline is straight-sided only (no curved
high-order geometry).

The runtime critical-time-step (`dt_cr`) is exposed on the bridge:
`ops.critical_time_step() -> float` (builds, primes one tiny step, queries — needs
an explicit integrator with `cfl=True`, a `Transient` analysis, and **element**
mass density via `-rho`/`-mass`; the eigensolve ignores `ops.mass` nodal mass).
`ops.analyze_explicit(duration=, safety=0.9, dt_max=None)` drives the whole run:
it queries `dt_cr` and sub-steps `analyze(n, duration/n)` with `n=ceil(duration/
(safety·dt_cr))` (ADR D5), returning an `ExplicitRunResult(n, dt, dt_cr)`. Both
raise `ValueError` on a non-usable `dt_cr` (no `cfl`, non-explicit integrator, or
pure nodal-mass model — the eigensolve uses element mass, not `ops.mass`).

**Stiffening caveat:** `dt_cr` is queried once on the initial stiffness. If the
tangent stiffens mid-run (contact, geometric/material) the true step shrinks and a
fixed `dt` can diverge. `analyze_explicit` warns (`OpenSeesExplicitSolverWarning`)
unless the integrator is built with `cfl_abort=True` (and `recompute=N`), and
re-raises a non-zero `analyze` as `RuntimeError` instead of returning it silently.

**System guards (apeGmsh, build/analyze-time):**
- `system Diagonal`/`MPIDiagonal` + an element with `c_mass=True` → **`BridgeError`**:
  the solver keeps only the diagonal, so off-diagonal *consistent* mass is silently
  dropped. Use lumped mass (drop `c_mass`) with a diagonal solver, or a non-diagonal
  system.
- an explicit integrator + a non-diagonal system → **`OpenSeesExplicitSolverWarning`**:
  correct but factors the full mass each step (loses the O(N) point of explicit).
  `Diagonal` (lumped) is the right pairing.

The `.ladruno` recorder **does** write `MODEL/LOCAL_AXES` (per-class quaternion
`FRAME`) for beams — unlike vanilla `.mpco`, which omits beam local axes. Don't
carry the stale "MPCO carries no beam LOCAL_AXES" assumption into `.ladruno`
readers.

`Results.from_ladruno(...)` (model_h5 **optional** — a `.ladruno` is self-sufficient)
surfaces this as **`results.elements.local_axes(...)`** → a `LocalAxes` with per-element
scalar-first quaternions plus `.matrices` / `.x_axis` / `.y_axis` / `.z_axis`. The local
axes are the **rows** of each matrix (OpenSees `quatFromMat` stores the transpose), in
global coords — verified: a beam's `.x_axis` points along node1→node2. So beam
orientation for line/section-force diagrams comes straight from `.ladruno` (wired
classes; ElasticBeam3d today), **not** the native `vecxz` path: `results.plot.line_force(...)`
prefers the recorder frame (true cross-section roll) over the geometric guess. Energy
lands via **`results.energy(region=)`** → a DataFrame `KE/IE/DW/ULW/RES/ERR` (recorder
`-G energy`).

**Element value channels** read through the same `results.elements.*` API as any
backend, with one Ladruno-specific split (the file is self-describing, so component
names come from the file):

- `results.elements.gauss.get(component="stress_xx")` — continuum stress/strain,
  **neutral** vocabulary (handles both `sigma11` and `sigma_xx`/`eps_xx`/`gamma_xy`
  token forms; cross-backend).
- `results.elements.line_stations.get(component="axial_force")` — beam internal-force
  diagrams, **neutral** (`axial_force`/`shear_y`/…; `localForce` end forces get the
  sign-continuity flip, `basicForce` is one station at ξ=0). For **force-based** beams
  this also serves `section.force`/`section.deformation` (`P`→`axial_force`,
  `kappaZ`→`curvature_z`, …) — one station per integration point, its ξ read from the
  element's `GP_PARAM` (not synthesized).
- `results.elements.fibers.get(component="fiber_stress")` — fiber-section stress/strain
  (`fiber_stress`/`fiber_strain`), one row per (element, GP, fiber), with `y`/`z`/`area`/
  `material_tag` from `MODEL/SECTION_ASSIGNMENTS`. (A `.ladruno` has no distinct *layer*
  or *spring* level — layered shells serialise as fiber sections; zeroLength force/material
  state is reachable via the element/gauss reads.)
- `results.elements.get(component="localForce")` — **the fork-only escape hatch**,
  token-driven: the component is the file's `ON_ELEMENTS/<token>` key
  (`basicForce`/`localForce`/`force`/`globalForce`) and the slab is the raw
  `(T, E, NUM_COLUMNS)` block in the file's column order. Prefer the neutral
  sub-composites above (`gauss`/`line_stations`/`fibers`, cross-backend); drop to
  this only for raw fork tokens the neutral views don't expose.

Multi-partition runs (`<stem>.part-N.ladruno`) auto-discover siblings and merge
(node-union + element-concat), like `from_mpco`. Higher-order / Bézier elements are
self-describing: GP world coords are reconstructed from the file's `BASIS` +
`GP_PARAM` via the neutral `apeGmsh._basis` evaluator (shared with the Bézier read
path), since a `.ladruno` from a Bézier element carries no `GLOBAL_GP_COORDS`.

## Live monitor (`ops.recorder.Monitor` + `read_monitor` / `tail_monitor`)

The **Monitor** is a *lightweight live-telemetry sidecar* — distinct from the
canonical `.ladruno` recorder. It streams a few selected nodal scalars to a small
SWMR-HDF5 file (`FORMAT="ladruno-monitor"`: `COLUMNS`/`STEP`/`TIME`/`FRAMES`) that a
viewer process can **tail while the analysis is still running**; the same file is a
valid at-rest result once the run ends. Fork-only — emit on any build, the fork is
needed only to *run*.

Emit:

```python
ops.recorder.Monitor(sink="live.h5", nodes=(roof,), dofs=(1, 2), resp="disp",
                     every=5)         # or pg="roof_nodes"; resp ∈ disp|vel|accel|reaction
```

Channels are nodes × dofs, labelled `node<N>.<resp>.dof<D>` in **node-major** order;
`every=K` (step decimation) and `hz=H` (wall-clock throttle) bound the stream.

Read — **not** a `Results` object (it carries no FEM), a thin time-history instead:

```python
from apeGmsh.results import read_monitor, tail_monitor
m = read_monitor("live.h5")          # at-rest snapshot
m.to_dataframe(index="time")         # DataFrame, one column per channel label
m.channel("node5.disp.dof1")         # one [T] history

for step, t, row in tail_monitor("live.h5", timeout=2.0):   # follow a growing sink
    ...                              # row is [nCols] in m.columns order
```

For a reader in a *separate process* from the solver, set
`HDF5_USE_FILE_LOCKING=FALSE` before `h5py` is imported (the SWMR/libhdf5 quirk).

## LadrunoPorousOverlay (persistent-fluid staggered u-p) — no emitter yet

Fork-only family, fully shipped on `ladruno` (fork PRs #575/#576/#580/#581/
#582/#585): `pattern LadrunoPorousOverlay` owns a pore-pressure field over a
snapshot of EXISTING `ndf = ndm` solid elements (region = element set,
drained = node set) and pushes `+Q*p` back each step — element removal below
the water table, an implicit fs1 lane + `LadrunoStaggeredAnalyze` iterated
driver, and two explicit lanes (`-fsL zero`; `-fluidUpdate explicit` = fully
matrix-free fluid). p-field recording rides `recorder ladruno ... -overlay`
(ordinary `ON_NODES/overlayPressure_<tag>` nodal scalars + `MODEL/OVERLAYS`
topology rows) and `recorder Monitor -overlay`.

**apeGmsh status: direct-drive only.** There is NO typed primitive — drive
the fork build's `openseespy` directly (`ops.pattern("LadrunoPorousOverlay",
tag, "-region", *eles, ...)`, the fork batteries' idiom). A typed
`ops.pattern.LadrunoPorousOverlay(pg=..., drained=<selection>, ...)` emitter
+ `LadrunoStaggeredAnalyze` verb is a **follow-up ADR** (fold into the ADR
0074 porous-media emission family — the overlay is LadrunoUP's element-free
sibling; one division-of-labor decision). Before wiring ANYTHING, read the
fork contract doc's `LadrunoPorousOverlay family` row + implementation-notes
section — it carries the emitter-facing traps (ONE overlay per hydraulically
connected water body; quad `b1 b2` is FORCE/VOLUME = `rho_mix*accel`;
`-moduli` re-set via the parameter route after stage flips) and the
explicit-fluid lane rules (undrained CFL binds the SYNC interval `N*dt`
under `-subcycle`; lumped `CentralDifferenceSMS` is the only supported
mass-scaling combination). Normative detail:
`Ladruno_implementation/LadrunoPorousOverlay_guide.md` in the fork repo.

## Contract lives in the fork repo

The exact emit/read contracts — command grammar, apeGmsh touch-points
(`_ELEM_REGISTRY` / `_response_catalog` / `Results.from_ladruno`), the
class-tag band, and the `.ladruno` schema notes — live in the fork's own
reference doc:

> `Ladruno_implementation/ladruno_apegmsh_contract.md` in
> `nmorabowen/OpenSees@ladruno`
> raw: `raw.githubusercontent.com/nmorabowen/OpenSees/ladruno/Ladruno_implementation/ladruno_apegmsh_contract.md`

**Read it before wiring any fork-only emitter or reader.**

## Profiler (`ops.profiler.*`)

The fork's stack profiler is a **control command** that brackets the analyze
loop — not a model primitive, not a recorder (no class tag, no
`_response_catalog` entry). It writes one `profile.h5`; apeGmsh ships **no
reader** — read it with the fork's out-of-tree
`Ladruno_tools/profiler_viewer/` (the headless `ProfilerResults` API, which is
Jupyter-usable, or the React viewer).

The five verbs map 1:1 to the shipped fork command
(`start|stop|reset|report|memory`):

```python
ops.profiler.start(deep=False, memory=False, per_step=False)  # profiler start [-deep] [-memory] [-perStep]
ops.profiler.stop()                                           # profiler stop
ops.profiler.reset()                                          # profiler reset
ops.profiler.report("profile.h5", run="caseA")               # profiler report profile.h5 -run caseA
ops.profiler.memory()                                         # profiler memory
```

There is **no** `config` verb and **no** `-warmupSteps` (the design doc showed
them but the shipped `OPS_profiler()` never wired them; `-perStep` is a flag on
`start`).

**Deck emit (Tcl / Py) — explicit verbs.** Record the verbs *before* the
`ops.tcl(...)` / `ops.py(...)` call; the bridge brackets the appended `analyze`
line. Bracket side is by **verb**, not call order: `start` / `reset` emit before
`analyze`; `stop` / `report` / `memory` after.

```python
ops.profiler.start(deep=True)
ops.profiler.report("profile.h5", run="caseA")
ops.tcl("deck.tcl", run=True, analyze_steps=200)   # → profiler start -deep / analyze 200 / profiler report ...
```

**Live (`ops.analyze`) — the `profile=` kwarg.** The live single-call has no
"after analyze" seam, so it takes the bracket as kwargs:

```python
ops.analyze(steps=200, profile="profile.h5", profile_run="caseA", profile_deep=True)
```

**Fork gate.** Emitting the deck text works on **any** build. Running needs the
fork: `ops.tcl(run=True)` is the recommended profiled path (the `profiler`
command is registered in the Tcl interpreter). The live / py-deck paths call the
openseespy binding `ops.profiler(...)`; on stock openseespy the live emitter
re-raises a clear *"requires the Ladruno fork build"* error. (Whether the fork
exposes `profiler` in the openseespy **Python** module, not only Tcl, is a
fork-side confirmation — prefer the Tcl-deck path until confirmed.)

**Reading `profile.h5`.** apeGmsh ships no profiler reader, but
`apeGmsh.profiler` is a thin bridge to the fork's out-of-tree viewer:

```python
import apeGmsh
with apeGmsh.profiler.open("profile.h5") as pr:   # → fork's ProfilerResults
    pr.manifest()                                 # run picker rows
    pr.rollup("caseA")                            # flame graph
    pr.series("caseA")                            # per-step time history (the "monitor")
    pr.diff("caseA", "caseB")                     # prove a fix
apeGmsh.profiler.show_web("profile.h5")           # launch the React UI at :8000
```

It **re-exports** `Ladruno_tools/profiler_viewer` (never re-implements). The dir
must be importable — pass `viewer_dir=` , set `LADRUNO_PROFILER_VIEWER`, or have
it on `sys.path`; otherwise a clear install-hint error fires. The one-click
`Profiler_Viewer.bat` / `profiler_viewer.sh` opens a browser with no setup.

## Class-tag band

Fork-only class tags live in the **private `≥33000` band**. Don't hardcode
the dead sub-300 values — read them live from the fork's `classTags.h` /
ledger. (See also `~/.claude/CLAUDE.md`: the OpenSees C++ source is at
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees`.)
