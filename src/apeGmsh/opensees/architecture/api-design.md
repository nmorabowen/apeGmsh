# API design

## Two complementary surfaces

The `apeSees` instance presents two surfaces to the user, both
statically typed:

- **Namespace API** — for **creating** things (`ops.uniaxialMaterial.Steel02(...)`).
- **Composite API** — for **inspecting** things (`ops.materials.uniaxial`,
  `ops.elements.by_pg("Cols")`).

```python
ops = apeSees(fem)

# Namespace API — creates a Steel02, registers it, returns the typed instance
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)

# Composite API — read-only view of what's been registered
ops.materials.uniaxial            # UniaxialMaterialSet (apeGmsh-style)
ops.materials.uniaxial.summary()  # DataFrame, indexed by tag
for m in ops.materials.uniaxial:  # iterable
    print(m)
```

The two are dual: namespaces *create*, composites *inspect*.

## The namespace API in detail

Every OpenSees command that has type variants is a namespace on the
bridge. Each variant is a typed method on that namespace.

```
ops
├─ uniaxialMaterial            (namespace)
│    .Steel02(*, fy, E, b, ...)             → Steel02
│    .Concrete02(*, fpc, epsc0, ...)        → Concrete02
│    .ElasticMaterial(*, E, eta=0.0)        → ElasticMaterial
│    ...
├─ nDMaterial                  (namespace)
│    .ElasticIsotropic(*, E, nu, rho=0.0)   → ElasticIsotropic
│    .J2Plasticity(*, K, G, sig0, ...)      → J2Plasticity
│    ...
├─ section                     (namespace)
│    .Fiber(*, patches, fibers=(), GJ=None) → Fiber
│    .ElasticMembranePlateSection(*, E, nu, h, rho=0.0)
│                                            → ElasticMembranePlateSection
│    ...
├─ geomTransf                  (namespace)
│    .Linear(*, orientation=None)           → Linear
│    .PDelta(*, orientation=None)           → PDelta
│    .Corotational(*, orientation=None)     → Corotational
├─ beamIntegration             (namespace)
│    .Lobatto(*, section, n_ip)             → Lobatto
│    .Legendre / .NewtonCotes / .Radau / .Trapezoidal (same shape)
│    .HingeRadau(*, secI, lpI, secJ, lpJ, secE)        → HingeRadau
│    .HingeRadauTwo / .HingeMidpoint / .HingeEndpoint (same shape)
├─ timeSeries                  (namespace)
│    .Linear(*, factor=1.0)                 → Linear
│    .Path(*, file=None, values=None, dt=None, factor=1.0)  → Path
│    ...
├─ pattern                     (namespace, context-manager-producing)
│    .Plain(*, series)                      → PlainPattern (CM)
│    .UniformExcitation(*, direction, series)  → UniformExcitationPattern (CM)
│    ...
├─ element                     (namespace)
│    .elasticBeamColumn(*, pg, transf, A, E, ...)   → ElementGroup
│    .forceBeamColumn(*, pg, section, transf, n_ip, ...)  → ElementGroup
│    .FourNodeTetrahedron(*, pg, material)  → ElementGroup
│    ...
├─ recorder                    (namespace)
│    .Node(*, file, pg=None, nodes=None, dofs, response, ...)
│    .Element(*, file, pg=None, elements=None, response, ...)
│    .MPCO(*, file, N=(), E=(), dT=None, nsteps=None)
├─ constraints                 (namespace, no varargs at user level)
│    .Plain()
│    .Penalty(*, alpha_sp=1e10, alpha_mp=1e10)
│    .Transformation()
│    .Lagrange()
├─ numberer                    (namespace)
│    .Plain(); .RCM(); .AMD()
├─ system                      (namespace)
│    .BandGeneral(); .UmfPack(); .Mumps(); ...
├─ test                        (namespace)
│    .NormDispIncr(*, tol, max_iter, print_flag=0)
│    .NormUnbalance(*, tol, max_iter, print_flag=0)
│    .EnergyIncr(*, tol, max_iter, print_flag=0)
├─ algorithm                   (namespace)
│    .Newton(); .ModifiedNewton(); .NewtonLineSearch(...); ...
├─ integrator                  (namespace)
│    .LoadControl(*, increment, num_iter=1, ...)
│    .DisplacementControl(*, node, dof, increment, ...)
│    .Newmark(*, gamma, beta)
│    ...
└─ analysis                    (namespace)
     .Static(); .Transient(); .VariableTransient()
```

Commands without type variants are **flat** methods on the bridge:

```
ops.model(*, ndm: int, ndf: int)
ops.fix(*, pg=None, nodes=None, dofs)
ops.mass(*, pg=None, nodes=None, values)
ops.region(*, name, pg=None, nodes=None)
ops.analyze(*, steps, dt=None) -> int
ops.eigen(num_modes, *, solver="-genBandArpack") -> EigenResult
ops.tcl(path, *, run=False, bin=None,
        analyze_steps=None, analyze_dt=None)
ops.py(path, *, run=False,
       analyze_steps=None, analyze_dt=None)
ops.run(*, wipe=True)
ops.h5(path, *, model_name=None, cuts=(), sweeps=())
ops.domain_capture(spec, *, path, ops=None) -> DomainCapture

# SSI helpers (Phase SSI-1 / SSI-3) — declarative wrappers that fan
# out at build time. See "## Initial-stress injection" and
# "## Imposed displacement" below.
ops.initial_stress(*, name, pg=None, elements=None,
                   sigma_xx, sigma_yy, sigma_zz,
                   ramp_steps, lambda_install=1.0) -> InitialStressRecord
ops.convergence_confinement(*, name, pg=None, elements=None,
                            sigma_xx=0.0, sigma_yy=0.0, sigma_zz=0.0,
                            lambda_target, n_steps) -> InitialStressRecord
ops.imposed_displacement(*, pg=None, nodes=None,
                         ux=None, uy=None, uz=None,
                         pattern_factor=1.0, series=None) -> Plain

# Staged analysis (Phase SSI-2.A / SSI-2.B) — context-manager block.
ops.stage(name: str) -> _StageBuilder
```

The `analyze_steps=` / `analyze_dt=` kwargs on `ops.tcl(...)` /
`ops.py(...)` append one `analyze` line at the tail of the emitted
deck. When any `ops.initial_stress(...)` registered a per-step ramp,
that line is automatically wrapped in a for-loop that calls the
hook dispatcher between steps — see [emitter.md](emitter.md) "Phase
SSI-1 analyze hook-wrapping" and the
[staged-analysis.md](staged-analysis.md) §"Hook dispatcher" walkthrough.

### Model-wide defaults at construction

`apeSees(fem, *, default_orientation=Cartesian())` accepts a
model-wide default `Orientation` used whenever the user constructs a
`geomTransf` without supplying either `orientation=` or `vecxz=`. The
default is the structural-engineering convention `Cartesian()`
(Z-up); pass an explicit `None` for 2D models, or a custom orientation
(e.g. `Cartesian(reference_axis=(0,1,0))` for a Y-up CAD import) to
shift the whole model.

```python
# Standard 3D structural model — Z-up implicit
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
trans = ops.geomTransf.PDelta()             # inherits Cartesian(Z-up)

# Y-up CAD import
ops = apeSees(fem, default_orientation=Cartesian(reference_axis=(0,1,0)))

# 2D model — no orientation needed (vecxz omitted at emit time)
ops = apeSees(fem, default_orientation=None)
ops.model(ndm=2, ndf=3)
trans = ops.geomTransf.Linear()             # orientation stays None

# Per-call override always wins
trans = ops.geomTransf.PDelta(orientation=Cylindrical(...))
```

Substitution is skipped for 2D models and when `ndm` has not yet been
set (e.g. tests that construct transforms before `model()`).

## Static typing — no `**kwargs` user-facing

Every namespace method has an explicit, fully-typed signature. No
`**kwargs`, no positional `*args` except where the OpenSees command
genuinely takes a variable-length list (e.g. `dofs` for `fix`).

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class Steel02(UniaxialMaterial):
    fy : float
    E  : float
    b  : float
    R0 : float = 20.0
    cR1: float = 0.925
    cR2: float = 0.15
    a1 : float | None = None
    a2 : float | None = None
    a3 : float | None = None
    a4 : float | None = None

    def _emit(self, emitter: Emitter, tag: int) -> None:
        params: list[float] = [self.fy, self.E, self.b,
                               self.R0, self.cR1, self.cR2]
        if self.a1 is not None:
            params += [self.a1,
                       self.a2 or 1.0,
                       self.a3 or 0.0,
                       self.a4 or 1.0]
        emitter.uniaxialMaterial("Steel02", tag, *params)


# In the namespace class:
class _UniaxialMaterialNS:
    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge

    def Steel02(
        self, *,
        fy : float, E : float, b : float,
        R0 : float = 20.0, cR1: float = 0.925, cR2: float = 0.15,
        a1 : float | None = None, a2 : float | None = None,
        a3 : float | None = None, a4 : float | None = None,
    ) -> Steel02:
        return self._bridge._register(
            Steel02(fy=fy, E=E, b=b,
                    R0=R0, cR1=cR1, cR2=cR2,
                    a1=a1, a2=a2, a3=a3, a4=a4)
        )
```

What pyright sees:

```python
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
# ✓ steel: Steel02

ops.uniaxialMaterial.Steel02(fy="bad", E=200e9, b=0.01)
# ✗ Argument of type "str" cannot be assigned to parameter "fy" of type "float"

ops.uniaxialMaterial.Steel02(yield_strength=420e6, E=200e9, b=0.01)
# ✗ No parameter named "yield_strength"

ops.uniaxialMaterial.Steel02(E=200e9, b=0.01)
# ✗ Missing required argument: "fy"
```

The boundary between typed user surface and OpenSees-vocabulary
varargs lives in `_emit` — see [emitter.md](emitter.md).

## Capabilities on typed instances

Each primitive class carries the operations natural to that primitive:

```python
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
steel.tag                                # auto-allocated by bridge
steel.fy, steel.E, steel.b               # named, typed attributes
steel.summary()                          # DataFrame
steel.plot.backbone(strain_max=0.05)     # ax, MaterialTestResult
steel.test.cyclic(ASCE41Protocol(max_disp=0.05))    # MaterialTestResult
steel.check.parameters()                 # warns on suspect values

sec = ops.section.Fiber(patches=[...], fibers=[...], GJ=1e9)
sec.dependencies()                       # set of materials reached
sec.plot()
sec.area, sec.centroid                   # geometric
sec.moment_curvature(axial_load=-1000e3) # MomentCurvatureResult

trans = ops.geomTransf.PDelta(orientation=Cartesian())
trans.plot_for_pg("Cols")                # vis vecxz on each beam

gm = ops.timeSeries.Path(file="elcentro.txt", dt=0.01, factor=9.81)
gm.plot();  gm.peak_value;  gm.duration
```

The capability target shape is documented in [charter.md](charter.md)
P1.

## Aggregate types

### `Node` — aggregates BC / mass / loads on one node

The deliberate violation of strict P1 (one class = one OpenSees
command). A `Node` instance carries everything OpenSees says about
that node:

```python
roof = ops.nodes.get("RoofNode")     # Node (apeGmsh-style query)
roof.coords                           # (x, y, z)
roof.tag                              # OpenSees node tag

# Model-level operations — flat, no pattern needed
roof.fix(dofs=(1, 0, 0, 0, 0, 0))
roof.mass(values=(50, 50, 50, 0, 0, 0))

# Pattern-level operations — only inside a pattern context
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    roof.load(forces=(100e3, 0, 0))
    # OR equivalently:
    p.load(node=roof, forces=(100e3, 0, 0))

# Inspection
roof.bcs                              # tuple of (dofs,)
roof.loads                            # tuple of (pattern_name, forces)
roof.summary()                        # DataFrame
```

The flat verbs (`ops.fix(...)`, `ops.mass(...)`, `p.load(...)`) **also
work** for multi-node convenience. The aggregate is the convenience,
not the substitute.

### `ElementGroup` — apeGmsh-native return from `ops.element.X(pg=...)`

Every PG-bound element creation returns a typed `ElementGroup`,
mirroring `apeGmsh.mesh._element_types.ElementGroup`:

```python
cols = ops.element.forceBeamColumn(
    pg="Cols", section=col_sec, transf=col_t, n_ip=5,
)

# apeGmsh-native interface
cols.element_type                     # ElementTypeInfo (forceBeamColumn, dim=1, ...)
cols.ids                              # ndarray of OpenSees tags
cols.connectivity                     # ndarray, one row per element
for tag, nodes in cols:               # iteration
    ...

# Tied dependencies (the package the user requested)
cols.section                          # Fiber instance
cols.transf                           # PDelta instance
cols.integration                      # BeamIntegration spec
cols.material_dependencies            # set of materials reached transitively

# Capabilities
cols.plot()                           # highlight in 3D
cols.summary()                        # DataFrame
cols.count                            # int
```

### `*Composite` / `*Set` views

`ops.materials`, `ops.elements`, `ops.nodes`, etc. are read-only
composites with the same shape as `apeGmsh.mesh.FEMData`'s
`NodeComposite` / `ElementComposite`. Iteration, indexing, filtering,
`.summary()`. See `apeGmsh.mesh._group_set.PhysicalGroupSet` for the
precedent.

## Standalone primitives (P11)

Typed primitives are constructable outside a bridge for material
studies, parametric sweeps, and notebooks:

```python
from apeGmsh.opensees.material.uniaxial import Steel02
from apeGmsh.opensees.time_series.time_series import ASCE41Protocol

s = Steel02(fy=420e6, E=200e9, b=0.01)
s.tag                             # None — not registered
s.plot.backbone(strain_max=0.05)  # works without a bridge
s.test.cyclic(ASCE41Protocol(max_disp=0.05))   # spawns isolated ops domain

# Later, register it with a bridge:
ops.register(s)                   # tag is allocated
```

## Standards for adding a new primitive

1. Write the typed class as a `@dataclass(frozen=True, kw_only=True, slots=True)`
   in the appropriate module. Inherit from the right base
   (`UniaxialMaterial`, `Section`, `Element`, etc.).
2. Implement `_emit(self, emitter: Emitter, tag: int) -> None` —
   internal forwarding to the emitter, where `*args` is allowed.
3. Implement `dependencies(self) -> tuple[Primitive, ...]` if the
   primitive composes others.
4. Add a method on the matching namespace class
   (`_UniaxialMaterialNS.Steel02`, etc.) with the same signature
   that calls `self._bridge._register(Cls(...))`.
5. Tests using `RecordingEmitter` to assert the emitted command.

No registry edits, no factory functions. The class IS the registry
entry.

## Initial-stress injection (Phase SSI-1)

`ops.initial_stress(...)` declares a ramped in-situ stress tensor on
`ASDPlasticMaterial3D`-bearing elements. The mechanism mirrors the
STKO `stressControl` pattern: one `parameter` declaration per
component (XX / YY / ZZ), one `addToParameter <tag> element <ele> <response>`
per element / component, and a per-step `proc` body that ramps
`factor = min(count / ramp_steps, 1.0)` linearly 0 → 1, advancing
`updateParameter` between analyze steps via the hook dispatcher.

```python
fem = g.mesh.queries.get_fem_data(dim=2)
ops = apeSees(fem, default_orientation=None)
ops.model(ndm=2, ndf=2)

# Mohr-Coulomb soil — a typed convenience helper that returns a
# fully-configured ASDPlasticMaterial3D (yf/pf/el/iv composition
# locked, model parameters wired). See material/nd.py for the
# generic ASDPlasticMaterial3D escape hatch and the PlaneStrain
# wrapper used to bridge the 3-D constitutive into 2-D quads.
mat_3d = ops.nDMaterial.MohrCoulombSoil(
    c=1014.0, phi=45.95, psi=11.49,
    E=4.08e6, nu=0.18, rho=4.5,
)
mat_2d = ops.nDMaterial.PlaneStrain(base=mat_3d)
ops.element.FourNodeQuad(
    pg="Rock", thickness=1.0, material=mat_2d, plane_type="PlaneStrain",
)

ops.fix(pg="Fixed_All", dofs=(1, 1))
# ... analysis chain (constraints / numberer / system / test /
#     algorithm / integrator / analysis) ...

# Declare the ramp.  Returns the InitialStressRecord so callers can
# pass it to a stage block via ``s.add(record)``; non-staged callers
# can ignore the return value.
ops.initial_stress(
    name="rock_insitu",
    pg="Rock",
    sigma_xx=-6300.0, sigma_yy=-6300.0, sigma_zz=-6300.0,
    ramp_steps=10,
    lambda_install=1.0,        # 1.0 = full install; partial install
                               # produces a convergence-confinement
                               # intermediate (see helper below).
)

# Emit and run.  ``analyze_steps=`` appends an ``analyze`` line at
# the tail of the deck; once a step hook is registered, the emitter
# wraps it in a for-loop with hook-dispatcher calls between steps.
ops.tcl("deck.tcl", analyze_steps=10, analyze_dt=0.1, run=True)
# verified: tests/opensees/subprocess/test_initial_stress_acceptance.py
#           ::test_initial_stress_ramp_matches_fixed_reference_full_apesees
```

Validation (raised by `apeSees.initial_stress`):

| Condition | Outcome |
|---|---|
| `name` empty | `ValueError` |
| `name` not a Tcl identifier (alphanumeric + `_`, can't start with a digit) | `ValueError` — name becomes a Tcl proc name |
| `(pg is None) == (elements is None)` | `ValueError` — XOR required |
| `ramp_steps < 1` | `ValueError` |
| `lambda_install ∉ (0, 1]` | `ValueError` |
| Duplicate `name` across global pool + every stage's pool | `BridgeError` at `build()` time — two `proc <name>` definitions would collide and the surviving one would reference uninitialised state (red-team H2, post-merge hardening) |

Per-axis target is `sigma_<xx|yy|zz> × lambda_install`; the ramp
factor always advances 0 → 1.0. Even when only one component is
non-zero, three `parameter` declarations and three
`updateParameter` calls per step are emitted (one for each axis).
For zero-target axes the delta is always 0.0 — wasteful but
harmless. The STKO reference (Interaccion deck, `_stressCtrl_11`)
allocates fewer parameters when a component is null; this is a
documented divergence.

### `convergence_confinement` helper (Phase SSI-3)

Thin wrapper over `initial_stress` for the tunnelling-mechanics
canonical pattern: ramp a single stress component on a boundary
region to `lambda_target × sigma` over `n_steps` analyze increments.
Mirrors the `_stressCtrl_11`-style proc at
`SSI/Interaccion/analysis_steps.tcl:19753-19767`.

```python
relax = ops.convergence_confinement(
    name="rock_relax_50",
    pg="Rock",
    sigma_xx=-6300.0,           # at least one non-zero required
    lambda_target=0.5,          # 50% relaxation (intermediate)
    n_steps=100,
)
```

The two cosmetic renames vs. `initial_stress`: `lambda_target`
reads more naturally for relaxation contexts than `lambda_install`,
and `n_steps` matches the typical tunneling-spec phrasing.

## Imposed displacement (Phase SSI-3)

`ops.imposed_displacement(...)` declares a `pattern Plain` carrying
prescribed-displacement `sp` entries. Mirrors STKO's
`pattern Plain N tsTag -fact F { sp NODE DOF VAL ... }` (the
fault-slip pattern at `SSI/Interaccion y Falla/analysis_steps.tcl:22832-23253`).

```python
fault_slip = ops.imposed_displacement(
    pg="Fault_Hanging",
    ux=-1.0, uy=-4.0,          # scalar broadcast per targeted node
    pattern_factor=0.001,      # folded into Linear(factor=) (see below)
)
# Equivalent for an explicit node list:
support_settle = ops.imposed_displacement(
    nodes=[105, 107, 109],
    uz=-0.01,
)
```

Where STKO uses `pattern Plain N tsTag -fact F`, this helper folds
the `F` factor into an auto-created `Linear(factor=F)` time series.
The pattern itself is registered with the bridge default factor
`1.0` — numerically identical (`value × F × t`), simpler API. Pass
an explicit `series=` (a pre-registered `TimeSeries`) to override;
`pattern_factor` is then ignored.

Validation:

| Condition | Outcome |
|---|---|
| `(pg is None) == (nodes is None)` | `ValueError` |
| All of `ux` / `uy` / `uz` are `None` | `ValueError` |
| `pattern_factor == 0.0` | `ValueError` (an inert pattern is a typo) |
| `uz=` on an `ndf=2` model (or any DOF index > `ndf`) | `ValueError` at declaration time (red-team H3, post-merge hardening) |

Limitations:

- Per-node-varying values are NOT supported in v1 — every targeted
  node gets the same scalar per DOF. For different values per node,
  call `imposed_displacement` multiple times with disjoint
  `nodes=` lists, or build the `Plain` pattern manually via
  `ops.pattern.Plain(...)`.
- The pattern is registered globally; if used inside a staged deck
  it fires in every stage's analyze loop. Gate via the time series
  if that is not desired.

## Staged analysis (Phase SSI-2.A / SSI-2.B / SSI-2.C / SSI-2.D)

`ops.stage(name)` opens a context manager that frames one stage of a
multi-stage analysis. Each stage emits its own analysis chain, its
own ramped initial-stress records, optional stage-bound topology
activation, optional stage-bound BCs / recorders (`s.fix` / `s.mass`
/ `s.region` / `s.recorder` — Phase SSI-2.D), optional stage-bound
MP constraints (`s.embedded` / `s.equal_dof` / `s.rigid_link` /
`s.rigid_diaphragm` / `s.kinematic_coupling` / `s.tie` /
`s.distributing` / `s.node_to_surface` / `s.node_to_surface_spring`
— Phase SSI-2.D extension), an `analyze` loop, and an inter-stage
cleanup block (`loadConst -time 0.0` + `wipeAnalysis` + hook-list
clear). Combining stages with MP partitions is supported (Phase
SSI-2.C).

```python
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
# ... materials / elements / fixes / masses ...

# Construct primitives the stages will reuse.  Each stage holds a
# reference to its chain; the bridge emits each primitive once per
# stage in which it's referenced (so OpenSees gets a fresh
# ``constraints Plain`` / ``numberer RCM`` per stage).
test_norm   = ops.test.NormDispIncr(tol=1e-4, max_iter=150)
algo_newton = ops.algorithm.Newton()
constr      = ops.constraints.Plain()
numb        = ops.numberer.RCM()
sysolv      = ops.system.UmfPack()
analy       = ops.analysis.Static()

# Stage 1 — in-situ stress install (no topology activation).
insitu = ops.initial_stress(
    name="rock_insitu",
    pg="Rock",
    sigma_xx=-6300.0, sigma_yy=-6300.0, sigma_zz=-6300.0,
    ramp_steps=10,
)
with ops.stage(name="insitu") as s:
    s.add(insitu)                              # bind to this stage
    s.analysis(
        test=test_norm, algorithm=algo_newton,
        integrator=ops.integrator.LoadControl(dlam=0.1),
        constraints=constr, numberer=numb, system=sysolv,
        analysis=analy,
    )
    s.run(n_increments=10, dt=0.1)

# Stage 2 — excavate the soil + install the lining + put a recorder on it.
# Phase SSI-2.D: stage-bound `s.fix` / `s.mass` / `s.region` /
# `s.recorder` emit INSIDE the stage block (after the stage's topology
# and before `domain_change`; recorders emit after the chain and
# before `analyze`).
lining_recorder = ops.recorder.Element(
    file="lining_force.out", response=("globalForce",), pg="Lining",
)
# Phase SSI-2.D extension: name the MP constraint at apeGmsh time so
# the stage can claim it by name at bridge time.
g.constraints.embedded(
    host_label="Rock", embedded_label="Lining",
    name="lining_embed", stiffness=1e8,
)
with ops.stage(name="excavate") as s:
    s.activate(pgs=["Lining"])                 # element-PG activation
    s.fix(pg="LiningAnchor", dofs=(1, 1, 1))   # SSI-2.D: stage-bound fix
    s.mass(pg="Lining", values=(100.0, 100.0, 100.0))  # stage-bound mass
    s.region(name="lining_rayleigh", pg="Lining")      # for Rayleigh damping
    s.embedded(name="lining_embed")            # SSI-2.D ext: claim MP constraint
    s.initial_stress(                          # SSI-2.D ext: PUSH mirror of
        name="lining_install",                 # ops.initial_stress, no s.add step
        pg="Lining",
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=5,
    )
    s.recorder(lining_recorder)                # PULL: claim from global pool
    s.analysis(
        test=test_norm, algorithm=algo_newton,
        integrator=ops.integrator.LoadControl(dlam=0.05),
        constraints=constr, numberer=numb, system=sysolv,
        analysis=analy,
    )
    s.run(n_increments=20, dt=0.05)

# Only Tcl/Py text-emit is supported for staged decks today.  Live
# execution (``ops.analyze`` / ``ops.eigen``) refuses staged models
# with NotImplementedError; emit and run a subprocess instead.
ops.tcl("staged.tcl", run=True)
# verified: tests/opensees/subprocess/test_stages_subprocess.py
#           tests/opensees/subprocess/test_stage_activation_subprocess.py
```

`_StageBuilder` lifecycle:

| Step | Required | Behaviour |
|---|---|---|
| `with ops.stage(name) as s:` | yes | Opens the builder. Refuses nested `with` blocks (a second open while one is in-progress raises `RuntimeError` — post-merge hardening M4). |
| `s.add(initial_stress_record)` | optional | Binds an `InitialStressRecord` (returned by `ops.initial_stress(...)`) to this stage. The record is removed from the bridge's global pool. Double-adding a record to two stages raises `ValueError`. Other record types raise `TypeError`. |
| `s.activate(pgs=[...])` | optional | Marks element-PG names as activated by this stage. Elements + their referenced nodes emit inside the stage block, not in the global pre-stage emit. Same PG activated in two stages raises `BridgeError` at build time. |
| `s.fix(*, pg=None, nodes=None, dofs)` | optional | **PUSH model** (Phase SSI-2.D). Stage-bound SP constraint. Mirrors `apeSees.fix` signature verbatim; pg XOR nodes. Records emit inside the stage block alongside `s.mass` / `s.region`. Validated at build time by V1 + V2. |
| `s.mass(*, pg=None, nodes=None, values)` | optional | **PUSH model** (Phase SSI-2.D). Stage-bound nodal mass. Mirrors `apeSees.mass`. V2 refuses (node)-duplicate across tiers since OpenSees `setMass` silently overwrites. |
| `s.region(*, name, pg=None, nodes=None)` | optional | **PUSH model** (Phase SSI-2.D). Stage-bound named region. Per-stage tag cache under MP so all contributing ranks agree on one tag per (stage, name). V3 refuses same `name=` across scopes; mangle the label (`lining_r_stage2`) if you really mean a per-stage region with conceptual continuity. |
| `s.recorder(spec)` | optional | **PULL model** (Phase SSI-2.D). `spec` is a `Recorder` registered via `ops.recorder.Node` / `Element` / `MPCO`. The spec keeps its allocated tag and stays in `bridge._primitives`, but the bridge marks it claimed so the global post-element recorder emit loop skips it; the stage emit drives it AFTER the chain and BEFORE `analyze`. V4 refuses targets owned by a later stage. Same spec claimed by two stages raises `ValueError`. |
| `s.initial_stress(*, name, pg=None, elements=None, sigma_xx, sigma_yy, sigma_zz, ramp_steps, lambda_install=1.0)` | optional | **PUSH model** (Phase SSI-2.D extension). Stage-bound initial-stress mirror of `ops.initial_stress(...)`. Creates the `InitialStressRecord` directly in this stage's pool, no intermediate `s.add(record)` step. Coexists with the existing `s.add(InitialStressRecord)` PULL path; pick by style. Byte-identical decks. PUSH is safe because side effects (parameter tag allocation, ramp proc emission) fire at emit time, not at call time — ADR 0034 §5b. |
| `s.embedded(*, name)` / `s.tie(*, name)` / `s.distributing(*, name)` / `s.equal_dof(*, name)` / `s.rigid_link(*, name)` / `s.rigid_diaphragm(*, name)` / `s.kinematic_coupling(*, name)` / `s.node_to_surface(*, name)` / `s.node_to_surface_spring(*, name)` | optional | **CLAIM-by-name model** (Phase SSI-2.D extension). Each method claims resolved MP-constraint records previously named at apeGmsh time via `g.constraints.<kind>(..., name=...)`. The records stay on the FEMData broker but the bridge marks them claimed so the global MP-constraint emit pass skips them; the stage emit pass drives them AFTER stage regions and BEFORE the stage's `domain_change`. Missing name → `ValueError`. Double-claim across two stages → `ValueError`. Why CLAIM not PUSH: the kernel resolver needs a live `gmsh` model + parts registry that are typically gone by bridge time — ADR 0034 §5a. **`s.tied_contact` / `s.mortar` are deferred** (SurfaceCouplingRecord nesting + mortar NIY; see [_DEFERRED.md](_DEFERRED.md)). |
| `s.analysis(test=, algorithm=, integrator=, constraints=, numberer=, system=, analysis=)` | required | All seven kwargs. Each must be a primitive already registered with the bridge. Second call raises `ValueError`. |
| `s.run(n_increments=, dt=None)` | required | Sets analyze-loop length + step size. Second call raises `ValueError`; `n_increments < 1` raises `ValueError`. |
| Clean `__exit__` | — | Validates `analysis_set` + `run_set`; appends a frozen `StageRecord` to the bridge. Exception in the body propagates and the in-progress stage is discarded. |

Caveats:

- **Live execution unsupported.** Once `ops.stage(...)` has been
  used, `ops.analyze(...)` and `ops.eigen(...)` raise
  `NotImplementedError`. Emit via `ops.tcl(p)` / `ops.py(p)` and run
  the subprocess (`run=True` or external invocation).
- **Staged + MP partitioned is supported (Phase SSI-2.C).** Combine
  `ops.stage(...)` blocks freely with a partitioned FEM —
  `BuiltModel._emit_partitioned` dispatches to
  `_emit_stages_partitioned` and emits per-rank topology /
  initial-stress fan-outs inside each stage. See
  [staged-analysis.md](staged-analysis.md) §"MP partitioned +
  initial_stress + stages (Phase SSI-2.C)" for the layout.
- **Global `fix` / `mass` / `region` on stage-bound nodes is refused
  at build time** (H1 validator). Those directives emit in the pre-
  stage global block; a node owned by stage 2 doesn't exist yet at
  that point, so OpenSees would error at parse time. The validator
  raises `BridgeError` with the offender list naming each
  `(kind, target, node, stage)` tuple. **The workaround is now**
  `s.fix(...)` / `s.mass(...)` / `s.region(...)` inside the owning
  stage's `with` block (Phase SSI-2.D).
- **Stage-bound BCs / recorders are append-only across stages.** A
  stage cannot release a prior stage's fix or zero out a prior
  stage's mass via the SSI-2.D verbs. For excavation-style decks
  that genuinely need to release support during construction, users
  currently drop to raw Tcl for the release step. A future
  `s.remove_sp(...)` / `s.zero_mass(...)` verb would lift this —
  see [staged-analysis.md](staged-analysis.md) §"Deferred work".
- **PUSH vs PULL builder asymmetry is intentional.** `s.fix` /
  `s.mass` / `s.region` use PUSH (inert dataclasses created on the
  stage directly); `s.add(initial_stress)` and `s.recorder(spec)`
  use PULL (specs with registration side effects — parameter tags
  for `InitialStressRecord`, recorder tag for `Recorder` —
  registered globally first, then claimed by the stage). See
  [decisions/0034-stage-bound-bcs-and-recorders.md](decisions/0034-stage-bound-bcs-and-recorders.md)
  for the rationale.
- **H5 archival of staged structure is deferred and fail-loud.**
  `apeSees.h5(path)` on a staged model — or one carrying any global
  `initial_stress` record — raises `NotImplementedError` (#313)
  pointing the user at `ops.tcl(path)` / `ops.py(path)`. The
  `H5Emitter`-side methods (`stage_open` / `stage_close` /
  `addToParameter` / `step_hook_ramp` / `domain_change`) remain
  no-ops; the bridge-side guard is what refuses the round-trip.
  A future `opensees_schema_version` bump (`2.11.0` → `2.12.0`,
  per [ADR 0023](decisions/0023-per-zone-schema-versioning.md))
  would persist stages under `/opensees/stages/` and lift the
  guard. See [staged-analysis.md](staged-analysis.md) §"Deferred work".

See [staged-analysis.md](staged-analysis.md) for the internals —
node + element ownership computation, hook-dispatcher mechanics,
stage-close cleanup contract.
