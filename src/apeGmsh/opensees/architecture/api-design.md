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
ops.analyze(*, steps, dt=None) -> int
ops.tcl(path, *, run=False)
ops.py(path, *, run=False)
ops.run(*, wipe=True)
```

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
