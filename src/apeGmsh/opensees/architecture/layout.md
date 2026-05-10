# Folder layout

```
apeGmsh/opensees/
├── __init__.py             public exports: apeSees, plus convenience re-exports
├── apesees.py              the apeSees class (the bridge)
│
├── element/                element primitives — flattened across OpenSees subfolders
│     beam_column.py        ElasticBeamColumn (2d/3d/Warping/Timoshenko),
│                           ForceBeamColumn, DispBeamColumn, CatenaryCable
│     truss.py              Truss, CorotTruss, InertiaTruss
│     shell.py              ShellMITC3, ShellMITC4, ShellDKGQ, ASDShellQ4, ASDShellT3
│     solid.py              FourNodeTetrahedron, TenNodeTetrahedron, stdBrick,
│                           bbarBrick, SSPbrick, FourNodeQuad, Tri31, SSPquad
│     zero_length.py        ZeroLength, ZeroLengthSection, ZeroLengthContact
│     joint.py              Joint2D, Joint3D
│
├── material/
│     uniaxial.py           Steel01, Steel02, ASDSteel1D, Concrete01, Concrete02,
│                           ASDConcrete1D, Hysteretic, ElasticMaterial, ENT, Cable, ...
│     nd.py                 ElasticIsotropic, J2Plasticity, DruckerPrager,
│                           PressureIndepMultiYield, PM4Sand, ASDConcrete3D, ...
│     yield_surface.py      (rare; brought in only when a model needs it)
│
├── section/                separated from material (see ADR 0004)
│     fiber.py              Fiber, GeneralFiberSection
│     plate.py              ElasticMembranePlateSection, ElasticShellSection,
│                           LayeredShell, LayeredShellFiberSection
│     beam.py               ElasticSection (1-D scalar A,E,G,J,Iy,Iz wrapper)
│     aggregator.py         Aggregator, Parallel
│
├── transform.py            Linear, PDelta, Corotational
│                           (csys.Cartesian / Cylindrical / Spherical re-exported)
│
├── pattern/
│     pattern.py            Plain, UniformExcitation, MultiSupport, Earthquake
│
├── time_series/            separated from pattern (OpenSees lumps; we don't)
│     time_series.py        Linear, Constant, Path, Trig, Pulse,
│                           ASCE41Protocol, FEMA461Protocol, ATC24Protocol
│
├── load/                   mostly internal — concrete element-load shapes
│     beam_load.py          Beam2dPointLoad, Beam2dUniformLoad, Beam3dPointLoad, ...
│
├── recorder.py             Node, Element, MPCO — surfaces existing Recorders system
│
├── analysis/
│     constraint_handler.py Plain, Penalty, Transformation, Lagrange, Auto
│     numberer.py           Plain, RCM, AMD, ParallelPlain
│     system.py             BandGeneral, BandSPD, ProfileSPD, UmfPack, Mumps,
│                           SparseGeneral, FullGeneral
│     test.py               NormDispIncr, NormUnbalance, EnergyIncr, FixedNumIter,
│                           RelativeNormDispIncr
│     algorithm.py          Linear, Newton, ModifiedNewton, NewtonLineSearch,
│                           KrylovNewton, BFGS, Broyden
│     integrator.py         LoadControl, DisplacementControl, ArcLength,
│                           Newmark, HHT, CentralDifference, ExplicitDifference
│     analysis.py           Static, Transient, VariableTransient
│
├── recipes/                higher-level builders (off the core path)
│     section_recipes.py    RectangularConfinedColumn, IShape, RC_Beam, ...
│     element_recipes.py    (rare, but allowed)
│
├── emitter/
│     base.py               Emitter (Protocol) — frozen interface
│     live.py               LiveOpsEmitter
│     tcl.py                TclEmitter
│     py.py                 PyEmitter
│     recording.py          RecordingEmitter (test fixture)
│
├── _internal/              not part of the public API
│     tag_allocator.py      TagAllocator
│     build.py              run_build()  — replaces solvers/_opensees_build.py
│     ns.py                 namespace classes (_UniaxialMaterialNS, etc.)
│     types.py              shared Protocol / TypeVar definitions
│
└── architecture/           this folder
```

## Naming conventions

### Class names — match OpenSees exactly

Concrete element / material / etc. classes use the **exact** OpenSees
type token, case-sensitive:

| OpenSees command | Python class |
|---|---|
| `uniaxialMaterial Steel02` | `Steel02` |
| `uniaxialMaterial Concrete02` | `Concrete02` |
| `nDMaterial ElasticIsotropic` | `ElasticIsotropic` |
| `nDMaterial PressureIndepMultiYield` | `PressureIndepMultiYield` |
| `section ElasticMembranePlateSection` | `ElasticMembranePlateSection` |
| `section Fiber` | `Fiber` |
| `geomTransf Linear` | `Linear` |
| `element forceBeamColumn` | `forceBeamColumn` (lowerCamel — match!) |
| `element elasticBeamColumn` | `elasticBeamColumn` |
| `element FourNodeTetrahedron` | `FourNodeTetrahedron` |
| `element ShellMITC4` | `ShellMITC4` |
| `algorithm Newton` | `Newton` |
| `integrator LoadControl` | `LoadControl` |

This means the Python file may contain a mix of `CamelCase` and
`lowerCamelCase` class names — that's correct. We're not normalizing
OpenSees's inconsistencies; we're surfacing them so users can match
the manual one-for-one.

Where OpenSees writes a type token in two ways (e.g. `pdelta` vs
`PDelta`), pick the manual's canonical form.

### Module / file names — `snake_case`

Modules group related classes; their names are snake_case Python
convention regardless of the OpenSees source folder name:

| OpenSees source | Python module |
|---|---|
| `SRC/element/forceBeamColumn/` | `element/beam_column.py` |
| `SRC/material/uniaxial/` | `material/uniaxial.py` |
| `SRC/analysis/integrator/` | `analysis/integrator.py` |

### Sub-packages — match OpenSees mental model

Top-level subfolders mirror OpenSees mental categories
(`element`, `material`, `section`, `pattern`, `analysis`, `recorder`)
with two deliberate departures:

1. **`section/` is separated from `material/`** — see
   [ADR 0004](decisions/0004-section-separated-from-material.md).
2. **`time_series/` is separated from `pattern/`** — see
   [ADR 0007](decisions/0007-time-series-separated-from-pattern.md).

### Internal vs public

- Anything under `_internal/` is implementation detail and may move
  without notice.
- Methods prefixed with `_` (e.g. `_emit(emitter, tag)`) are internal
  to the primitive system. User code never calls them directly.
- The public API surface is exhausted by:
  - `apeSees` and its method namespaces
  - typed primitive classes (constructable standalone, see P11)
  - `recipes/`
  - `transform` (csys re-exports)

## Where to add a new thing

| Adding a new… | File | Notes |
|---|---|---|
| Uniaxial material | `material/uniaxial.py` | typed dataclass + `_emit` |
| nD material | `material/nd.py` | same |
| Section | `section/{fiber,plate,beam,aggregator}.py` | pick by family |
| Element | `element/{beam_column,truss,shell,solid,zero_length,joint}.py` | pick by family |
| Time series | `time_series/time_series.py` | typed dataclass + `_emit` |
| Pattern | `pattern/pattern.py` | typed context manager |
| Recorder | `recorder.py` | thin layer over existing `Recorders` system |
| Analysis component | `analysis/<kind>.py` | one file per kind |
| Emitter target | `emitter/<name>.py` | implement the Protocol |
| Higher-level recipe | `recipes/section_recipes.py` (or new file) | composes primitives |

The standard for each addition: see the per-kind sections in
[api-design.md](api-design.md).
