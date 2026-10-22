# Fork-only element and material classes on stock (#2434, phase 1)

Audit for #2434 (chain K, after #2430). It covers every class the bridge
ships under `element`, `uniaxialMaterial` and `nDMaterial`: 35 elements,
24 uniaxial materials and 23 nD materials, 82 classes in all. Each class
gets one category:

- **(a)** both builds give the same answer;
- **(b)** stock refuses loudly;
- **(c)** stock exits 0 with a different answer;
- **(d)** the class cannot be tested in isolation.

Sections, transforms and beam integrations are not covered.

## Method

The method follows #2429/#2430.

- Each class is built with the bridge's own dataclass and emitted through
  `RecordingEmitter`, dependencies first, with a tag resolver attached.
- The recorded tokens are replayed **raw** onto the bound module. The replay
  skips `LiveOpsEmitter`, so its `_FORK_ONLY_ELEMENTS` gate cannot hide what
  the build itself does.
- Each case runs in its own process. It solves a small model and returns its
  answer:
  - uniaxial materials: `testUniaxialMaterial` over a cyclic strain path;
  - nD materials: one `stdBrick`, with optional hydrostatic confinement,
    then `DisplacementControl`;
  - plane and plate wrappers: one `quad`, or a `ShellMITC4` with a
    `LayeredShell`;
  - elements: a minimal static or Newmark model.
- 141 cases cover 82 classes. Option variants exercise each emitted flag that
  a stock parser could silently ignore.
- Two answers count as the same within a relative 1e-6.

Builds (the module and `ladrunoBuild()` were asserted in every run):

| Side | Module | Bound by |
|---|---|---|
| stock | `openseespywin` 3.7.1.2, `site-packages\openseespywin\opensees.pyd` | `APEGMSH_OPENSEES_BIN` and `LADRUNO_OPENSEES_BIN` set to that directory |
| fork | `C:\Program Files\Ladruno\OpenSees\bin\opensees.pyd`, build `79e062367` | `APEGMSH_OPENSEES_BIN` |

Caveats:

- CI's `live-stock` lane pins `openseespy==3.8.0.0` on Linux, and that build
  was not run here.
- `79e06` is older than `FORK_PIN`.
- An unknown `element`, `uniaxialMaterial` or `nDMaterial` type raises
  `OpenSeesError` on **both** builds. This differs from `integrator`, which
  stock silently replaces with its default (#2430). The silent class can
  therefore only come from a **stock-named** class that the fork changed.

## Counts

| | Elements | Uniaxial | nD | Total |
|---|---|---|---|---|
| (a) same | 22 | 18 | 8 | 48 |
| (b) stock refuses at declaration | 12 | 5 | 12 | 29 |
| (b') stock refuses at `analyze` (non-convergence aborts the emitted deck) | 0 | 0 | 1 | 1 |
| **(c) stock exits 0 with a different answer** | **1** | 0 | **1** | **2** |
| (d) not isolatable | 0 | 1 | 1 | 2 |

## (c) Silent wrong answers on stock

| Class | Case | Stock 3.7.1.2 | Fork 79e06 | Oracle |
|---|---|---|---|---|
| `TenNodeTetrahedron` | one tet10, base fixed, apex and edge loads; apex `ux, uy, uz` | 0.616596, 0.289096, 0.079813 | 0.102766, 0.048183, 0.013302 | Independent numpy quadratic-tet `K` (4-point rule): 0.102766, 0.048183, 0.013302. Fork error 8e-17; stock is **6× too soft** |
| `ManzariDafalias` (bridge defaults, no `updateMaterialStage`) | `p = 100` confinement, then `εa = -0.001` | `uz = -0.00188025`; `σa = -45.676` | `uz = -0.00129377`; `σa = -69.564` (exactly linear) | Elastic stage with `G` at `P_atm`, `e_init`: `p/3K = 0.00129377`, `Eε = 69.564`. Matches the fork |
| `ManzariDafalias`, explicit `-stage 0`, then `-stage 1` | same path | `σa = -30.6666` at `εa = -0.0006` | `-30.6529` | none. Elastic stage identical; plastic residual 0.045% |

The fork's ledger names both causes:

- **`TenNodeTetrahedron`:** fork `ee942546c` / PR-692. `xsj` applied the
  tetrahedral 1/6 factor twice, so every stiffness was 6× too soft. The entry
  is marked "upstreamable", so stock 3.8.0 very likely still carries the
  defect; that is unverified here.
- **`ManzariDafalias`:** fork `41cfb81de` / PR-714. Upstream's full
  constructors never set `mElastFlag`, so the material starts
  **elastoplastic**, while the fork starts it elastic. The 0.045% plastic
  residual is the fork's tangent and sub-step work (WP-110, -158, -160).

Every Manzari variant differs from stock, with no warning and exit 0:
`int_scheme` 1 with a loose `tol_r`, and `int_scheme` 5. `SAniSandMS` agrees
on both builds. A non-default `tol_r` is already refused by the bridge
(`NotImplementedError`).

## (b') Loud only at `analyze`

`DruckerPrager` gives the same answer while one surface is active. Two cases
leave that regime:

- tension, where the cone meets the cutoff;
- non-associated flow with hardening, under compression.

In both, stock stops converging at step 1 or 2, and the fork runs on. The
fork-side cause is `31322a47a` (ADR-95 P4, which repaired the two-surface
return map and its tangent). The emitted decks abort when `analyze` fails, so
the failure is loud. No converged wrong answer was found.

## (b) Stock refuses at declaration

- **Elements:** `LadrunoDispBeamColumn`, `LadrunoForceBeamColumn`†,
  `LadrunoIMKBeam`, `LadrunoMembrane`†, `LadrunoBrick`, `LadrunoBrick20`,
  `LadrunoQuad`, `LadrunoCST`, `LadrunoLST`, `LadrunoUP`, `BezierTri6` and
  `BezierTet10`. This is exactly the list in `emitter/live.py::_FORK_ONLY_ELEMENTS`,
  less its constraint-path members.
- **Uniaxial materials:** `ASDSteel1D`‡, `LadrunoBondSlip`,
  `LadrunoUniaxialJ2`, `LadrunoRebarBuckling` and `LadrunoCohesiveHinge`.
- **nD materials:** `ASDPlasticMaterial3D`‡, `LadrunoSANISAND`, `LadrunoJ2`,
  `LadrunoJ2Finite`, `LadrunoConcrete3D`, `LadrunoRCConcrete`,
  `LadrunoRCFiniteStrain`, `LadrunoCohesiveHingeBiaxial`, `LogStrain`,
  `LogStrain2D`, `InitDefGrad` and `StagedStrain`.

† Fork `79e06` does not know this class either (it was added after that
build), so its fork-side run is unverified.

‡ This class exists in newer upstream builds. Stock 3.7.1.2 refuses it, but
3.8.0 may accept it and must be re-run.

No material set exists in code today. The docs' "fork-only surface" list
omits `ASDSteel1D`, `ASDPlasticMaterial3D`, `LogStrain`, `LogStrain2D`,
`InitDefGrad` and `StagedStrain`.

## (d) Not isolatable

- **`InitialStress`:** the bridge emits `uniaxialMaterial InitialStressMaterial`,
  a type **neither** build knows, so it fails loudly on both. The real token,
  `InitStressMaterial`, gives the same answer on both builds. This is a
  separate bridge defect, not part of the fork/stock question.
- **`PlaneStressRebar`:** `PlaneStressRebarMaterial` is Tcl-only by design
  (see the class docstring), so openseespy has no in-process run on either
  build.

## (a) Same on both

- **Elements:** `Truss`, `CorotTruss`, `InertiaTruss`, `ZeroLength`,
  `ZeroLengthSection`, `CoupledZeroLength`, `TwoNodeLink`,
  `elasticBeamColumn`, `forceBeamColumn`, `dispBeamColumn`,
  `ElasticTimoshenkoBeam`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4`,
  `ASDShellT3`, `stdBrick`, `FourNodeTetrahedron`, `FourNodeQuad`, `Tri31`,
  `SixNodeTri`, `ASDAbsorbingBoundary2D` and `ASDAbsorbingBoundary3D`.
- **Uniaxial materials:** `Steel01`, `Steel02`, `Concrete01`, `Concrete02`,
  `Hysteretic`, `MultiLinear`, `PySimple1`, `TzSimple1`, `QzSimple1`,
  `ElasticMaterial`, `ENT`, `ElasticPP`, `ElasticPPGap`, `Viscous`,
  `ViscousDamper`, `Maxwell`, `Parallel` and `ASDConcrete1D`.
- **nD materials:** `ElasticIsotropic`, `J2Plasticity`, `SAniSandMS`,
  `ASDConcrete3D`, `PlaneStrain`, `PlateFiber`, `PlateFromPlaneStress` and
  `PlateRebar`.

Every emitted option flag tested on these classes also agreed:

- `ASDShellQ4`: `-corotational`, `-noeas`, `-drillingNL`, `-drillingStab`,
  `-local`;
- the ASD concrete materials: `-implex`, `-eta`, `-cdf`, `-Kc`, `-rho`,
  `-tangent numerical`, `-autoRegularization`;
- the mass flags `-rho`, `-cMass` and `-mass`, under Newmark;
- `Parallel`'s `-factors`;
- the TwoNodeLink flags `-orient`, `-pDelta` and `-shearDist`.

PR-990, the fork's fix for `ASDConcrete3D` with the same tag after a `wipe`,
did not reproduce with `-autoRegularization`.

## Fix shape (phase 2 input)

1. **(b), the cheap and mechanical part.** Add one `FORK_TYPES` map to
   `caps.py`, `{family: casefolded type tokens}`, covering `element`,
   `uniaxialMaterial` and `nDMaterial`. It folds in #2430's `FORK_CHAIN`, and
   `_FORK_ONLY_ELEMENTS` moves into it so that one source feeds both the
   live gate and the stamp. Then:
   - derive `"fork"` in `value_requires` and `flat_requires` from the archived
     element-meta and material types;
   - pass those types in at the two call sites #2430 names
     (`H5Emitter.solve_stamp`, `opensees_model.py`);
   - pin each token, as `_SYSTEM_TOKENS` does.

   `"fork"` is existing vocabulary, so the schema does not change.
2. **(c) needs a ruling, not just a token.**
   - **`TenNodeTetrahedron`:** stamp `"fork"`, or refuse it on stock. Its
     stock answer is provably 6× wrong.
   - **`ManzariDafalias`:** emitting `updateMaterialStage -stage 0` right
     after the declaration makes stock match the fork exactly through the
     elastic stage, with no token. The 0.045% plastic residual would remain
     a documented difference. The alternative is to stamp `"fork"`.
3. **`DruckerPrager`** fails loudly and is optional to fix. Stamp `"fork"`
   only if a non-convergence on stock should be caught before the run.
4. **`InitialStress`** gets its own card: the token should be
   `InitStressMaterial`.
5. Re-run the (c) and ‡ rows against stock 3.8.0 and against `FORK_PIN`
   before pinning.
