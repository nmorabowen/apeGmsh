> **Absorbed** into `docs/concepts/opensees-bridge.md` (ADR 0079 P2-B). This guide is retained as working memory; the concepts page is canonical.

# The OpenSees Bridge

apeGmsh ships with a first-class bridge to OpenSees. The bridge takes a
`FEMData` snapshot from an apeGmsh session, combines it with material
definitions, element assignments, boundary conditions, explicitly
re-declared masses, and explicitly imported or authored loads, and
produces ready-to-run Tcl or openseespy scripts. You never touch raw
node numbering or element connectivity -- the bridge resolves physical
groups against the snapshot.

Two things from the session are brought in differently (ADR 0051):
**MP constraints** (`g.constraints.*`) **auto-emit** — you do not
re-declare them (§4.4). **Loads** (`g.loads.*` / `g.displacements.*`)
are **opt-in**: they reach the deck only when you explicitly import a
load *case* into a bridge pattern with `p.from_model(case)`, or author
the load directly on a pattern with `p.load(...)` (§4). Masses and
support fixities are re-declared explicitly with `ops.mass` / `ops.fix`
(unchanged).

The legacy in-session `g.opensees.*` composite (and the `apeGmsh.solvers`
package) was **removed** in the Phase-8 teardown (ADR 0009 -- no
back-compat shim). The OpenSees surface is now a single class,
`apeSees`, constructed **after** the session from a `FEMData` snapshot:

```python
from apeGmsh.opensees import apeSees

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
```

`apeSees` exposes typed namespaces and explicit declaration verbs:

| Namespace / verb                                  | Purpose                                          |
|---------------------------------------------------|--------------------------------------------------|
| `ops.nDMaterial`, `ops.uniaxialMaterial`, `ops.section` | typed material / section primitives        |
| `ops.geomTransf`, `ops.beamIntegration`           | beam local frame + integration                   |
| `ops.element.<Type>(pg=…)`                        | element assignment by physical group             |
| `ops.fix`, `ops.mass`                             | homogeneous SP + lumped mass (explicit)          |
| `ops.timeSeries`, `ops.pattern`                   | load patterns + prescribed SP (explicit)         |
| `ops.recorder.<Type>`                             | recorder declarations                            |
| `ops.tcl / .py / .h5 / .run / .analyze`           | emit / run                                       |

Plus the lifecycle entry points: `ops.model(ndm=, ndf=)` (first) and
`ops.build()` (usually implicit -- each emit builds internally).

> **What the bridge brings in (ADR 0051).** `apeSees` does **not**
> auto-pull everything session-declared. The split is:
>
> - **MP constraints** (`g.constraints.*`) — **auto-emit** from the
>   snapshot (§4.4). Unchanged.
> - **Loads** (`g.loads.*`) and **prescribed displacements**
>   (`g.displacements.*`) — **opt-in**. A resolved load *case* reaches
>   the deck only when a bridge pattern imports it with
>   `p.from_model(case)`, or you author the load directly with
>   `p.load(...)` / `p.sp(...)`. There is **no** load auto-emit.
> - **Masses** (`ops.mass`) and **support fixities / homogeneous SPs**
>   (`ops.fix`) — **re-declared explicitly**. The bridge reads only its
>   own `ops.mass(...)` / `ops.fix(...)` records, not `g.masses` /
>   `fem.nodes.sp`.
>
> For everything else, `apeSees` reads the `fem` snapshot to resolve
> `pg=` / `label=` selectors to node/element tags and to get
> coordinates/connectivity. All session declarations also flow into the
> **`model.h5` neutral zone** for the **viewer / `Results`**, whether or
> not they were imported into the deck.
>
> **case vs pattern.** The geometry groups loads by **case**
> (`g.loads.case("dead")`) — a label with no temporal meaning. The
> OpenSees **pattern** (with its `timeSeries`, its owning stage, its
> `loadConst` freeze) is born on the **bridge**. `p.from_model("dead")`
> is the seam: it replays the resolved nodal records tagged case
> `"dead"` as `load` / `sp` lines inside the pattern you opened.
>
> **No double-counting; the deck is authoritative.** Because nothing
> auto-emits, importing a case once with `p.from_model(case)` is the
> single channel — the old "declared on the session *and* on the bridge
> → 2×" trap is gone. The bridge applies exactly the cases you import
> and does not audit the geometry's case list against the deck: which
> cases belong in *this* analysis is your call (one geometry can feed a
> gravity-only deck, a seismic deck, etc.). A case you don't import is
> simply not applied.

The overall pipeline is:

```
geometry --> mesh --> FEMData snapshot
    --> apeSees(fem) --> typed materials/elements
    --> explicit fix/mass/patterns --> emit (tcl/py/h5/run)
```

## Tasks on this page

- [Set model dimensions](#1-model-dimensions-opsmodel) · [Declare materials](#2-materials) · [Assign elements](#3-element-assignment)
- [Fix a support](#34-fix-a-support-opsfix) · [Add masses](#41-masses-opsmass) · [Apply loads / prescribed SP](#42-loads-and-prescribed-sp-patterns) · [Tie meshes / MP constraints](#44-multi-point-constraints) · [Damping](#46-damping-opsdamping)
- [Build the model](#5-building-the-model-build) · [Emit Tcl / py / h5 / run](#6-emit-run) · [Inspect the broker](#7-inspection)


## 1. Model Dimensions -- `ops.model`

Tell the bridge how many spatial dimensions and DOFs per node your model
has:

```python
ops.model(ndm=3, ndf=3)
```

Both arguments are keyword-only. Typical combinations:

| `ndm` | `ndf` | Use case                                   |
|-------|-------|--------------------------------------------|
| 2     | 2     | 2-D solid (ux, uy)                         |
| 2     | 3     | 2-D frame (ux, uy, rz)                     |
| 3     | 3     | 3-D solid (ux, uy, uz)                     |
| 3     | 6     | 3-D frame or shell (ux, uy, uz, rx, ry, rz)|

**`ops.model(...)` must be the first call** -- materials, elements,
fix, mass, and patterns all depend on it. `ndf` sets the required
length of the tuples you pass to `ops.fix(dofs=…)`,
`ops.mass(values=…)`, and `p.load(forces=…)`. `build()` raises early
if any assigned element type is incompatible with the declared
`ndm`/`ndf`.

`apeSees(fem, default_orientation=Cartesian())` sets a model-wide
`geomTransf` orientation default (Z-up). Pass `None` for 2-D models;
pass a custom `Orientation` for a Y-up CAD import.


## 2. Materials

Three typed namespaces mirror how OpenSees organises material-like
objects. Every method has an explicit, fully-typed signature (no
`**kwargs`) and **returns a handle** -- there are no string names and
no registry. Capture the handle in a variable and pass it by reference
to the element constructor; handles auto-register on the bridge.

### 2.1 nDMaterial -- continuum solids

Use `ops.nDMaterial.<Type>` for any element that takes an `nDMaterial`
in OpenSees: tetrahedra, bricks, quads, and triangles.

```python
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
```

The returned `conc` handle is passed later as
`ops.element.FourNodeTetrahedron(..., material=conc)`. The method name
*is* the OpenSees nDMaterial type -- `ElasticIsotropic`,
`J2Plasticity`, `DruckerPrager`, etc.

### 2.2 uniaxialMaterial -- trusses and springs

Use `ops.uniaxialMaterial.<Type>` for trusses, corotational trusses,
zeroLength springs, and fiber-section beams.

```python
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
```

Steel02-family uses `fy=` (the legacy `Fy=` is gone).

### 2.3 Section -- shells and fiber sections

Use `ops.section.<Type>` for shell elements and fiber sections for
beams. The most common shell section is `ElasticMembranePlateSection`,
combining membrane and bending behaviour.

```python
slab = ops.section.ElasticMembranePlateSection(
    E=30e9, nu=0.2, h=0.2, rho=2400,
)

from apeGmsh.opensees.section.fiber import FiberPoint
sec = ops.section.Fiber(
    fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
)
```

Which namespace: `ops.nDMaterial` -> solid elements;
`ops.uniaxialMaterial` -> truss / spring / fibre-section beams;
`ops.section` -> shells and fiber sections for beams. The constructors
are **not** chainable across namespaces -- write a separate statement
for each handle.


## 3. Element Assignment

`ops.element.<Type>(pg="PG", ...)` writes every mesh element in that
physical group as `<Type>`. PG resolution is "FEM-direct" against
`fem`. The call returns a typed `ElementGroup`.

### 3.1 Assigning element types -- `ops.element.<Type>`

```python
ops.element.<Type>(
    pg="PG",          # physical-group or label name
    *,
    material=…,       # nDMaterial / uniaxialMaterial handle
    section=…,        # section handle (shells / fiber beams)
    transf=…,         # geomTransf handle (beams only)
    integration=…,    # beamIntegration handle (force-based beams)
    body_force=…,     # element body/gravity force (solids)
    pressure=…,       # 2-D solids
    **scalars,        # element-specific scalar parameters
)
```

`pg=` resolves against both physical groups and apeGmsh labels (labels
resolve automatically -- no `promote_to_physical`). Keep PG names
dimension-unique; an ambiguous `pg=` that exists at multiple dimensions
is an error.

Material / section / transf / integration are passed as **handles**
(the variables the constructors returned), not string names:

- Solid elements need an `nDMaterial` handle (`ops.nDMaterial.*`)
- Truss elements need a `uniaxialMaterial` handle
  (`ops.uniaxialMaterial.*`)
- Shell elements need a `section` handle (`ops.section.*`)
- Beam elements take a `transf` handle plus either section properties
  as scalar kwargs (`elasticBeamColumn`) or a `beamIntegration` handle
  (`forceBeamColumn`)

Examples:

```python
# 3-D solid tetrahedron -- gravity is an ELEMENT body_force param
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc,
    body_force=(0.0, 0.0, -9.81 * 2400),
)

# 2-D plane-stress quad
ops.element.quad(
    pg="Plate", material=steel_2d,
    thick=0.01, eleType="PlaneStress",
)

# Truss with cross-section area
ops.element.corotTruss(pg="Diagonals", material=steel, A=3.14e-4)

# Elastic beam-column (3-D)
t = ops.geomTransf.PDelta(vecxz=(0, 0, 1))
ops.element.elasticBeamColumn(
    pg="Columns", transf=t,
    A=0.04, E=200e9, G=77e9, J=1e-4, Iy=2e-4, Iz=2e-4,
)

# Force-based beam-column with a fiber section
integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
ops.element.forceBeamColumn(pg="Cols", transf=t, integration=integ)

# Shell (section-based)
ops.element.ShellMITC4(pg="SlabSurface", section=slab)
```

There is **no `eleLoad` pattern verb** -- distributed/body loads are
element parameters (`body_force=`, `pressure=`), not loads. See the
recipes [Apply gravity](../how-to/gravity.md) and
[Apply a face pressure](../how-to/face-pressure.md).

### 3.2 Supported element types

| Family | Types |
|--------|-------|
| 3-D solid | `FourNodeTetrahedron`, `TenNodeTetrahedron`, `stdBrick`, `bbarBrick`, `SSPbrick` |
| 2-D solid | `quad`, `tri31`, `SSPquad` |
| Shell | `ShellMITC3`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4` |
| Truss | `truss`, `corotTruss` |
| Beam | `elasticBeamColumn`, `forceBeamColumn`, `ElasticTimoshenkoBeam` |

If your Gmsh mesh uses second-order elements but the assigned OpenSees
type only supports first-order, `build()` issues a `UserWarning` and
discards mid-side nodes automatically.

### 3.3 Geometric transformations -- `ops.geomTransf`

Beam elements require a geometric transformation defining their local
coordinate system. The constructor returns a handle you pass to
`ops.element.*` as `transf=`:

```python
t = ops.geomTransf.PDelta(vecxz=(0, 0, 1))   # or .Linear / .Corotational
ops.element.elasticBeamColumn(pg="Columns", transf=t, A=…, E=…, …)
```

- `vecxz` -- the local x-z plane vector (3-D only; omit for 2-D models)
- the namespace method name is the transform type: `.Linear`,
  `.PDelta`, `.Corotational`

### 3.4 Fix a support -- `ops.fix`

> See also the dedicated recipe: [Apply supports & BCs](../how-to/supports-bcs.md).

Apply homogeneous single-point constraints to every node in a physical
group:

```python
ops.fix(pg="BasePlate", dofs=(1, 1, 1))
ops.fix(pg="PinnedBase", dofs=(1, 1, 1, 0, 0, 0))   # ndf=6
ops.fix(nodes=[101, 102], dofs=(1, 1, 1))           # explicit-node form
```

The `dofs` tuple must have exactly `ndf` entries (`1` = fixed, `0` =
free). Homogeneous SPs are model-level here; non-zero prescribed
displacements go inside a pattern via `p.sp` (§4).

**Solid faces:** For solid meshes where you declared a face SP via
`g.displacements.surface(...)` on the session, that record resolves into
`fem.nodes.sp` but is **not** auto-emitted. Bring it in explicitly:
homogeneous -> `ops.fix(pg=…, dofs=…)`; prescribed -> import the case
with `p.from_model(case)` (or author `p.sp(...)`) inside a pattern. See
`guide_loads.md` §11.


## 4. Loads, Masses, and SP

Loads are **opt-in** (ADR 0051). Session loads (`g.loads.*`) and
prescribed displacements (`g.displacements.*`) resolve onto
`fem.nodes.loads` / `fem.nodes.sp` and persist into `model.h5`, but they
do **not** auto-emit. They reach the deck only when a bridge pattern
imports their *case* with `p.from_model(case)` (§4.2), or you author the
load directly with `p.load(...)` / `p.sp(...)`. **Masses** and **support
fixities / homogeneous SPs** are re-declared explicitly: the bridge
reads only its own `ops.mass(...)` / `ops.fix(...)` records, **not**
`g.masses` / `fem.nodes.sp`.

### 4.1 Masses -- `ops.mass`

```python
ops.mass(pg="Roof", values=(m, m, m, 0.0, 0.0, 0.0))
```

`values` is fitted to each target node's **per-node `ndf`** (inferred
from its elements, ADR 0048): a shorter tuple is zero-padded on the
trailing DOFs — so `values=(m, m, m)` is the natural translational-only
mass on a 6-DOF node — and a non-zero component beyond the node's `ndf`
(e.g. a rotational-inertia term on a 3-DOF solid node) fails loud rather
than being silently dropped by OpenSees. No pattern grouping, no ordering
concerns -- one lumped-mass declaration per physical group.

### 4.2 Loads and prescribed SP -- patterns

A bridge **pattern** carries the loads: open one with a `timeSeries`,
then either **import** a geometry-declared case with `p.from_model(case)`
or **author** loads directly with `p.load(...)` / `p.sp(...)`. Both can
be mixed in the same pattern.

```python
ts = ops.timeSeries.Linear()              # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:   # also UniformExcitation
    p.from_model("dead")                       # import the resolved "dead" case
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))  # + ad-hoc bridge-authored load
    p.sp(pg="LoadingPin", dof=3, value=0.01)   # prescribed displacement
```

`p.from_model(case)` replays the resolved nodal records that
`g.loads.case(case)` / `g.displacements.case(case)` produced — nodal
forces become `load` lines, non-homogeneous (prescribed) displacements
become `sp` lines, both scaled by the pattern's series. Homogeneous
fixes are model-level and are **not** imported (use `ops.fix`). The
DOF-agnostic 3-D records map onto each node's **per-node `ndf`** at emit
time — a force on a 3-DOF solid node emits 3 components even in a 6-DOF
model, and a component the node's DOFs can't carry (e.g. a moment on a
solid node) fails loud.

`p.load` / `p.sp` fan a `pg=` across the group's nodes at build time;
`node=` takes an explicit tag or a `Node` from `ops.nodes.get(...)`. The
`forces=` tuple is fitted to each target node's **per-node `ndf`**: a
shorter vector is zero-padded on the trailing DOFs, and a non-zero
component beyond the node's `ndf` fails loud (OpenSees would otherwise
drop the whole load silently).

Distributed/body loads (gravity, surface pressure) are **not** patterns
-- they are element parameters (`body_force=`, `pressure=`) on the
`ops.element.*` call (§3.1). See the recipe
[Apply a nodal point load](../how-to/point-load.md).

> **Two execution modes — no mixing (ADR 0051 §5).** A model is either
> **non-staged** (a global `ops.pattern.*` + the analysis chain +
> `ops.analyze` / `ops.eigen`) **or** **staged** (every pattern
> stage-scoped via `s.pattern(series=...)`, run through `ops.tcl` /
> `ops.py`). Registering a global pattern **and** opening a stage raises
> `BridgeError` at build — a global pattern would fire in every stage's
> analyze loop and double-apply across the staged `loadConst`
> boundaries. For a load in a staged deck, open the pattern inside the
> stage:
>
> ```python
> with ops.stage(name="push") as s:
>     with s.pattern(series=ops.timeSeries.Linear()) as p:
>         p.from_model("live")
>     s.analysis(...)
>     s.run(n_increments=10, dt=0.1)
> ```

### 4.3 Migration of the old ingest / auto-emit

| Old behavior | New |
|---|---|
| `g.loads.*` auto-emitted into the deck | **opt-in** — import the case with `p.from_model(case)` (or author with `p.load`) |
| `g.opensees.ingest.loads(fem)` | `p.from_model(case)` inside a pattern |
| `.masses(fem)` | `ops.mass(pg=…, values=…)` |
| `.sp(fem)` (homogeneous `g.constraints.bc` / `g.displacements`) | `ops.fix(pg=…, dofs=…)` |
| `.sp(fem)` (prescribed `g.displacements`) | `p.from_model(case)` or `p.sp(pg=…, dof=…, value=…)` |
| gravity via `g.loads.gravity(...)` | element `body_force=(b1,b2,b3)` param |
| `.constraints(fem, tie_penalty=)` | `g.constraints.X(...)` resolves into `FEMData` and emits automatically (§4.4); stage-bind via `s.X(name=...)` |

### 4.5 The deck is authoritative — no unconsumed-case audit

The bridge applies exactly the cases a pattern imports. It does **not**
check the geometry's declared cases against the deck: which cases belong
in *this* analysis is the modeller's decision, not something the bridge
second-guesses. A case you don't import is simply not applied (it still
persists into `model.h5` for the viewer / `Results`).

> An earlier revision warned at build for any declared-but-unimported
> case (`WarnUnconsumedModelLoads`) with an `ops.ignore_model_loads(case)`
> silencer. Both were **removed**: with loads opt-in, the explicit deck
> is the source of truth, and a completeness audit re-coupled the
> geometry case-list to the deck — exactly the link `case` vs `pattern`
> was designed to sever. If you import a case name that resolves to no
> records (a typo), that import is simply a no-op; double-check the case
> name against `fem.nodes.loads.patterns()`.

### 4.4 Multi-point constraints

> See also the recipe: [Tie non-matching meshes](../how-to/tie-meshes.md).

MP constraints declared on the session via `g.constraints.*` (`tie`,
`embedded`, `equal_dof`, `rigid_link`, `rigid_diaphragm`,
`kinematic_coupling`, `node_to_surface`, `distributing`,
`tied_contact`) resolve into `FEMData` and **emit automatically** as
part of the runnable Tcl / Py deck (Phase 7b, ADR 0022). The
`Emitter` protocol carries `equalDOF`, `rigidLink`, `rigidDiaphragm`,
and `embeddedNode` methods; the bridge orchestrates per-kind fan-out
via `emit_mp_constraints` between the element emission and pattern
emission passes.

```python
# Declare at apeGmsh time:
g.constraints.embedded(
    host_label="Rock", embedded_label="Rebar",
    name="rebar_embed", stiffness=1.0e8,
)

# Use as usual:
ops = apeSees(fem)
# ... materials / elements ...
ops.tcl("model.tcl")  # the embed lines appear automatically
```

The `Transformation` constraint handler is auto-emitted when any MP
constraint is present (warning visible on stdout); pass an explicit
`ops.constraints.X()` to override.

**Stage-bound routing (Phase SSI-2.D extension).** For staged decks
(`with ops.stage(...) as s:` blocks), an MP constraint can be
**claimed by name** for a specific stage so it emits inside that
stage's block instead of the global pre-stage pass — required when
the constraint references nodes that only come online via
`s.activate(pgs=[...])` in a later stage. Name the constraint at
apeGmsh time, then claim it inside the stage block:

```python
g.constraints.embedded(
    host_label="Rock", embedded_label="Lining",
    name="lining_embed", stiffness=1.0e8,
)
# ... mesh, FEMData, apeSees(fem) ...

with ops.stage(name="install_lining") as s:
    s.activate(pgs=["Lining"])
    s.embedded(name="lining_embed")  # claim — emits inside this stage
    s.analysis(...)
    s.run(n_increments=10, dt=0.1)
```

Available CLAIM verbs: `s.embedded`, `s.tie`, `s.distributing`,
`s.equal_dof`, `s.rigid_link`, `s.rigid_diaphragm`,
`s.kinematic_coupling`, `s.node_to_surface`,
`s.node_to_surface_spring`. `s.tied_contact` / `s.mortar` are
deferred (see [_DEFERRED.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/_DEFERRED.md)).
Forgetting to claim is caught by the V1 ownership-tier validator
with an actionable error pointing at the offending node and stage.

Deck consequences:

- Constraints emit automatically into `ops.tcl(path)` / `ops.py(path)`
  output. The model runs end-to-end without hand-editing.
- The `model.h5` neutral zone persists the resolved records under
  `/opensees/constraints/*` for the viewer / `Results`.
- For multi-stage decks, name your constraints up front
  (`name="..."` on every `g.constraints.X(...)` call you intend to
  stage-bind) — claim-by-name requires it.


### 4.6 Damping -- `ops.damping`

Damping is a **domain-level** concern (a sibling of `fix` / `mass` /
`region`, *not* part of the analysis chain). The `ops.damping` namespace
owns four of OpenSees' damping channels — global Rayleigh, region-scoped
Rayleigh, modal damping, and the tagged `damping` objects. Material
dashpots (`Viscous`, `ViscousDamper`, …) stay in `ops.uniaxialMaterial.*`
and numerical/algorithmic damping (HHT-α, …) stays in `ops.integrator.*`.
Every member is a **declaration resolved at emit** — there is no `assign`
step and you never hold the object's tag.

**Rayleigh — raw or ratio, global or scoped.**

```python
# Raw: the four OpenSees coefficients straight through.
ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01)        # global

# Ratio helper: a target ξ at two frequencies (Hz). α lands in alpha_m;
# β lands in the slot named by stiffness= (default "initial" = βK0, the
# nonlinear-safe choice — current/committed are explicit opt-ins).
ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0)

# Region-scoped: on= a physical group (or a list). Uses -ele membership
# because βK is stiffness-proportional. Region refines global — OpenSees
# OVERWRITES element Rayleigh per element (not additive), so the emit pass
# warns (RayleighOverwriteWarning) when a global and an on= form overlap.
ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0, on="Soil")
```

Setting `ops.damping.rayleigh(...)` is what finally gives the per-element
`do_rayleigh=True` flags something to opt into.

**Modal damping** — bundles its own `eigen` (computes the modes, then sets
the per-mode factors). Domain-wide, so no `on=`.

```python
ops.damping.modal(0.05, modes=4)              # uniform ζ across 4 modes
ops.damping.modal([0.02, 0.03, 0.05], modes=3)  # per-mode (len == modes)
```

There is intentionally **no `modal_q`** — OpenSees `modalDampingQ` is a
verified upstream anti-damping bug (wrong sign). Modal damping only damps
the modes you computed and is *additive* with any Rayleigh — easy to
over-damp.

**Tagged `damping` objects** — frequency-band viscous dissipators. `on=`
attaches them to a physical group's elements via `region -damp`:

```python
ops.damping.uniform(ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Soil")
ops.damping.sec_stif(beta=0.002, on="Soil")
ops.damping.urd(points=[(0.5, 0.02), (5.0, 0.03), (20.0, 0.05)], on="Soil")
ops.damping.urd_beta(points=[(0.5, 1e-3), (10.0, 2e-3)], on="Soil")
```

`uniform`'s `ratio=` is the **physical** ζ (OpenSees applies the internal
factor of two — don't pre-divide). All four take time-window kwargs
`activate_time` / `deactivate_time` (the "no damping during the gravity
stage" lever) and `factor=` (an `ops.timeSeries.*` object → `-factor`).

Instead of a region, you can attach an object **directly to a supported
element** via its `damp=` kwarg (omit `on=`):

```python
damp = ops.damping.uniform(ratio=0.03, freq_lower=0.5, freq_upper=10.0)
ops.element.elasticBeamColumn(pg="Cols", transf=t, A=.01, E=2e11, Iz=1e-4,
                              damp=damp)
```

`damp=` is accepted only on the elements whose OpenSees parser supports
`-damp` (`elasticBeamColumn` / `forceBeamColumn` / `dispBeamColumn` /
`stdBrick` / `FourNodeQuad` / the Shell family / `ZeroLength`); passing it
elsewhere fails loud. A damping object that attaches to **nothing** (no
`on=`, never handed to an element) raises at `build()` — there is no
global `-damp`.

**Staged damping** — inside a `with ops.stage(...) as s:` block the same
verbs live on `s.damping` and resolve *inside* that stage (after the
stage's `domainChange`, so they bind the stage's elements):

```python
with ops.stage(name="dynamic") as s:
    s.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0)
    s.damping.uniform(ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Soil")
    ...  # s.analysis(...) + s.run(...)
```

`s.damping.modal` is not available (per-stage modal damping interacts with
`eigen` / `wipeAnalysis` and is deferred); declare modal damping on the
flat bridge instead.


## 5. Building the Model -- `build()`

```python
ops.build()   # -> immutable BuiltModel
```

`ops.build()` resolves every typed declaration and `pg=` selector
against the bound `fem` snapshot and returns an **immutable
`BuiltModel`** that the emitters consume. You rarely call it directly
-- `ops.tcl / py / h5 / run` each build internally.

`build()` raises early with a pointed error if the model is
inconsistent: a missing geomTransf on a beam, an `ndm`/`ndf` mismatch,
an ambiguous `pg=`, or a `dofs` mask whose length is not `ndf`.


## 6. Emit / Run

`apeSees` writes the built model to disk or runs it in process. These
are **separate statements -- not a fluent chain.** Each `tcl / py / h5
/ run` calls `build()` internally. See the recipe
[Export a solver script](../how-to/export-script.md).

### 6.1 Tcl script -- `ops.tcl(path)`

```python
ops.tcl("model.tcl")
```

Produces a complete OpenSees Tcl input file: model builder, nodes,
materials/sections/transforms, element connectivity with
physical-group comments, fix commands, nodal masses, load patterns,
and MP constraints (`equalDOF`, `rigidLink`, `rigidDiaphragm`,
`ASDEmbeddedNodeElement` — §4.4).

### 6.2 openseespy script -- `ops.py(path)`

```python
ops.py("model.py", run=False)   # run=True subprocesses the script
```

Produces an equivalent Python script using
`openseespy.opensees as ops`. The structure mirrors the Tcl output.

`ops.tcl(...)` and `ops.py(...)` are independent statements -- write
them on separate lines, not as a chain:

```python
ops.tcl("model.tcl")
ops.py("model.py")
```

### 6.3 Native HDF5 + live run

```python
ops.h5("model.h5")              # bridge /opensees/ zone + broker neutral zone
ops.run()                       # in-process openseespy (LiveOpsEmitter)
ops.analyze(steps=10, dt=0.01)  # drive the analysis chain
```

`ops.h5(path)` writes the canonical model.h5: the bridge's emitted
`/opensees/` zone **plus** the broker neutral zone (which carries the
session's loads/masses/constraints for the viewer / `Results`).

> **Two zones vs. neutral-only.** `apeSees(fem).h5(path)` is the only
> writer that emits **both** zones. The session-side persistence verbs
> write the **neutral zone only** (no `/opensees/`): `g.save(path=None)`
> (or `apeGmsh(..., save_to="model.h5")`, which autosaves on
> context-exit) and the broker's own `fem.to_h5(path)`. Round-trip with
> `FEMData.from_h5(path)` or rebuild a chain-phase session for
> composition with `apeGmsh.from_h5(path)` (no gmsh — only
> `compose`/`save`). A neutral-only file feeds the viewer / `Results`
> but is **not** a runnable deck; emit Tcl/Py or call `apeSees(fem).h5`
> for the solver zone. See the recipe
> [Save & reload a model](../how-to/save-reload.md).

### 6.4 Recorders

```python
rec = ops.recorder.Node(...)    # ops.recorder.<Type>
```

Recorder declarations live on `ops.recorder.<Type>`. They are emitted
alongside the model in `tcl / py / h5`. See
`guide_recorders_reference.md`.


## 7. Inspection

Inspection is broker-side or post-emit -- there is no `inspect`
sub-composite on the bridge.

### 7.1 Summary

```python
fem.inspect.summary()
```

Returns a multi-line string from the broker: registered record sets,
node/element counts by type and physical group. The quickest sanity
check before emitting.

### 7.2 Broker tables

```python
fem.inspect.node_table()
fem.inspect.element_table()
```

DataFrames of the broker's nodes and elements -- coordinates,
connectivity, physical-group membership. Useful for verifying
placement before emitting.

### 7.3 Post-emit reader

```python
from apeGmsh.opensees.emitter.h5_reader import open as open_h5

open_h5("model.h5")
```

The reference reader inspects an emitted `model.h5` -- the
ground-truth check of what actually went into the deck.


## 8. Complete Example

End-to-end: create a concrete block, mesh, take the snapshot, declare
materials/BCs/loads explicitly on the bridge, and emit.

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# ── Session ───────────────────────────────────────────────
with apeGmsh("concrete_block") as g:

    # ── Geometry ──────────────────────────────────────────
    # A simple 2m x 1m x 1m block
    g.model.geometry.add_box(0, 0, 0,  2.0, 1.0, 1.0, label="Block")

    # Label the base surface for boundary conditions
    g.model.selection.select_surfaces(on_plane=("z", 0.0)).label("Base")

    # Label the top surface for loading
    g.model.selection.select_surfaces(on_plane=("z", 1.0)).label("Top")

    g.model.geometry.synchronize()

    # Promote the labels the solver needs into physical groups
    g.physical.from_label("Block", name="Block")
    g.physical.from_label("Base",  name="Base")
    g.physical.from_label("Top",   name="Top")

    # ── Mesh ──────────────────────────────────────────────
    g.mesh.generation.set_size_global(0.15)
    g.mesh.generation.generate(3)

    # ── FEMData snapshot ──────────────────────────────────
    fem = g.mesh.queries.get_fem_data(dim=3)

# ── OpenSees -- post-session, explicit declarations ──────
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)

# ── Material ──────────────────────────────────────────────
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)

# ── Element assignment (gravity is an element body_force) ─
ops.element.FourNodeTetrahedron(
    pg="Block", material=conc,
    body_force=(0.0, 0.0, -9.81 * 2400),
)

# ── Boundary conditions ──────────────────────────────────
ops.fix(pg="Base", dofs=(1, 1, 1))

# ── Loads -- declared on the bridge (no g.loads on this session) ──
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="Top", forces=(0.0, 0.0, -50e3))

# ── Sanity check + emit ──────────────────────────────────
fem.inspect.summary()
ops.tcl("block_model.tcl")
ops.py("block_model.py")
```

The emitted files are self-contained model definitions. Run them with
`opensees block_model.tcl` or `python block_model.py` after appending
your analysis commands (the bridge intentionally omits analysis setup).


## 9. Practical Advice

### Declaration order

`ops.model(...)` **must** come first. After that, materials, elements,
fix, mass, and patterns can be declared in the order that reads best;
they are resolved at `build()` (implicit in emit). Capture every
material/section/transf/integration handle in a variable before the
element call that consumes it.

### Physical groups vs. labels

`pg=` resolves either a Gmsh physical-group name or an apeGmsh label --
labels resolve automatically (FEM-direct), so `promote_to_physical` is
not required for the OpenSees workflow. Keep PG names dimension-unique
so `pg=` is never ambiguous.

### Beam elements need geomTransf

Beam elements do not use a material registry. Their section properties
are scalar kwargs (`A=0.04, E=200e9`) for `elasticBeamColumn`, or a
`beamIntegration` handle for `forceBeamColumn`, but they always require
a `geomTransf` handle. Forgetting it is the most common beam-model
error -- `build()` catches it.

### Per-node ndf is inferred from the elements

Shell elements require six DOFs per node, solids three, a 3-D beam six,
a 3-D truss three. You do **not** declare any of this: apeGmsh **infers
each node's `ndf` from the element classes you declared** on it
(ADR 0048). `ops.model(ndm=K, ndf=N)` still takes `ndf`, but it is only
the **envelope** — the OpenSees builder default and the fallback for
nodes inference can't see (element-less / decoupled nodes). Set it to the
largest DOF count in the model (e.g. `6` when shells are present); the
emitter writes a per-node `-ndf K` token wherever a node's inferred value
differs from the envelope, and elides it where they match (so a uniform
model's deck is token-free).

```python
ops.model(ndm=3, ndf=6)                 # envelope = 6 (shells present)
ops.element.ShellMITC4(pg="Deck", ...)  # Deck nodes infer 6 → -ndf elided
ops.element.FourNodeTetrahedron(pg="Soil", ...)  # Soil nodes infer 3 → "-ndf 3"
```

A node **shared** by two elements whose DOF counts disagree (a shell's
`{6}` meeting a solid's `{3}`) cannot exist — OpenSees would mis-assemble
it (`FE_Element::setID` truncation). apeGmsh **fails loud** at build; the
fix is separate coincident nodes joined by `equal_dof` / `tie` on the
shared (translational) DOFs, never a shared node. See
[ADR 0046](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0046-shell-on-solid-node-sharing-guard.md)
for the shell-on-solid idiom and
[ADR 0048](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0048-infer-per-node-ndf-from-elements.md)
for the inference rule. The one explicit per-node `ndf` channel is
`ops.ndf(target, ndf=K)`, restricted to **element-less decoupled nodes**
(a stand-alone constraint master / control node, created via
`g.decouple_node`) — it never competes with inference because those
nodes have no element to infer from. See
[ADR 0049](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0049-decoupled-nodes.md).

### The deck stops at model definition

The emitted scripts omit analysis commands (integrator, algorithm,
system). Append your analysis block or source/import the generated
file from a driver script. Recorders, if you declared them on
`ops.recorder.*`, *are* emitted.

### Tie / non-conformal interfaces emit automatically

You can declare ties for non-matching meshes on the session:

```python
g.constraints.tie("ColumnTop", "SlabBottom", dofs=[1, 2, 3],
                  name="col_slab_tie")
```

The tie resolves into `FEMData`, persists into `model.h5` for the
viewer / `Results`, **and** emits into the runnable Tcl / Py deck as
`ASDEmbeddedNodeElement` lines (Phase 7b, ADR 0022). Pass `name=` if
you'll need to stage-bind the constraint via
`s.tie(name="col_slab_tie")` inside a stage block (see §4.4).
`s.tied_contact` / `s.mortar` remain deferred for stage-binding — see
[_DEFERRED.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/_DEFERRED.md).

### What the bridge brings in (ADR 0051)

The bridge is not all-or-nothing. **Auto-emitted** (do not re-declare):
**MP constraints** (`g.constraints.*`, §4.4). **Opt-in** (§4.2): session
**loads** (`g.loads.*`) and **prescribed displacements**
(`g.displacements.*`) reach the deck only via `p.from_model(case)` (or
an ad-hoc `p.load` / `p.sp`) — there is no load auto-emit, and the bridge
applies exactly the cases you import without auditing the geometry's
case-list (§4.5). **Re-declared explicitly on `ops`** (§4): **masses**
(`ops.mass`) and **support fixities / homogeneous SPs** (`ops.fix`) — the
bridge reads only its own `ops.mass` / `ops.fix` records, not `g.masses` /
`fem.nodes.sp`. Every session declaration is also preserved for the
viewer / `Results` via the `model.h5` neutral zone, imported or not.

### Emit calls are separate statements

`ops.tcl(...)`, `ops.py(...)`, `ops.h5(...)`, `ops.run()` are **not**
fluent. Write each on its own line; each builds internally.
