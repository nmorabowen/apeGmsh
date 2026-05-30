# The OpenSees Bridge

apeGmsh ships with a first-class bridge to OpenSees. The bridge takes a
`FEMData` snapshot from an apeGmsh session, combines it with material
definitions, element assignments, boundary conditions, and explicitly
re-declared masses, and produces ready-to-run Tcl or openseespy
scripts. Session-declared loads and MP constraints are auto-emitted --
you do not re-declare those. You never touch raw node numbering or
element connectivity -- the bridge resolves physical groups against the
snapshot.

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

> **The behavioural change: selective ingest.** The old bridge had a
> single `ingest` step that pulled *everything* session-declared
> (`g.loads` / `g.masses` / `g.constraints`) into the deck. **`apeSees`
> ingests selectively.** Two categories auto-emit from the snapshot:
> session **loads** (`g.loads.*`, resolved onto `fem.nodes.loads`,
> emitted as synthesized Plain patterns) and **MP constraints**
> (`g.constraints.*`, §4.4). The other two categories are **re-declared
> explicitly** on `ops`: **masses** (`ops.mass`) and **support
> fixities / homogeneous SPs** (`ops.fix`) — the bridge reads only its
> own `ops.mass(...)` / `ops.fix(...)` records for these, not
> `g.masses` / `fem.nodes.sp` (§4). For everything else, `apeSees` reads
> the `fem` snapshot to resolve `pg=` / `label=` selectors to
> node/element tags and to get coordinates/connectivity. All session
> declarations also flow into the **`model.h5` neutral zone** for the
> **viewer / `Results`**.
>
> **Don't double-declare a load.** Because session loads auto-emit, if
> you *also* re-declare the same load via a bridge `p.load(...)`, it
> lands in the deck **twice** — verified: reactions come out at 2×.
> Pick one channel per load: declare it on the session via `g.loads.*`
> (auto-emitted) **or** on a bridge pattern via `p.load(...)`, not both.

The overall pipeline is:

```
geometry --> mesh --> FEMData snapshot
    --> apeSees(fem) --> typed materials/elements
    --> explicit fix/mass/patterns --> emit (tcl/py/h5/run)
```


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
element parameters (`body_force=`, `pressure=`), not loads.

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

### 3.4 Boundary conditions -- `ops.fix`

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
`g.loads.face_sp(...)` on the session, that record resolves into
`fem.nodes.sp` but is **not** ingested. Re-declare it explicitly:
homogeneous -> `ops.fix(pg=…, dofs=…)`; prescribed -> `p.sp(...)`
inside a pattern. See `guide_loads.md` §11.


## 4. Loads, Masses, and SP

Ingest is **selective.** Session **loads** (`g.loads.*`, resolved onto
`fem.nodes.loads`) auto-emit as synthesized Plain patterns — you do
**not** re-declare them on the bridge. **Masses** and **support
fixities / homogeneous SPs** are different: the bridge reads only its
own `ops.mass(...)` / `ops.fix(...)` records, **not** `g.masses` /
`fem.nodes.sp`, so you re-declare those explicitly on `ops`. (Every
session declaration also resolves into `FEMData` and persists into the
`model.h5` neutral zone for the viewer / `Results`.)

> **Don't double-declare a load.** A load declared via `g.loads.*`
> already auto-emits; re-declaring the same load on a bridge
> `p.load(...)` doubles it in the deck (reactions come out at 2×). Use
> one channel per load.

### 4.1 Masses -- `ops.mass`

```python
ops.mass(pg="Roof", values=(m, m, m, 0.0, 0.0, 0.0))
```

`values` is an `ndf`-length tuple. No pattern grouping, no ordering
concerns -- one lumped-mass declaration per physical group.

### 4.2 Loads and prescribed SP -- patterns

Session loads (`g.loads.*`) auto-emit as synthesized Plain patterns, so
you usually declare **no** load patterns on the bridge. The bridge
pattern verbs below are for loads you did **not** declare on the session
and for non-zero prescribed displacements (`p.sp`), which are not
auto-emitted. Both are pattern-scoped:

```python
ts = ops.timeSeries.Linear()              # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:   # also UniformExcitation
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))  # only for loads NOT in g.loads
    p.sp(pg="LoadingPin", dof=3, value=0.01)   # prescribed displacement
```

`p.load` / `p.sp` fan a `pg=` across the group's nodes at build time;
`node=` takes an explicit tag or a `Node` from `ops.nodes.get(...)`. The
`forces=` tuple length must match the model `ndf`. **Don't** re-declare
a `g.loads.*` load here — it auto-emits already, and declaring it on
both channels doubles it in the deck (reactions at 2×).

Distributed/body loads (gravity, surface pressure) are **not** patterns
-- they are element parameters (`body_force=`, `pressure=`) on the
`ops.element.*` call (§3.1).

### 4.3 Migration of the old ingest call

| Old `g.opensees.ingest.X(fem)` | New |
|---|---|
| `.loads(fem)` | nothing — `g.loads.*` auto-emits as a synthesized Plain pattern (re-declare on `p.load` only for loads not on the session) |
| `.masses(fem)` | `ops.mass(pg=…, values=…)` |
| `.sp(fem)` (homogeneous `face_sp`) | `ops.fix(pg=…, dofs=…)` |
| `.sp(fem)` (prescribed `face_sp`) | `p.sp(pg=…, dof=…, value=…)` |
| gravity via `g.loads.gravity(...)` | element `body_force=(b1,b2,b3)` param |
| `.constraints(fem, tie_penalty=)` | `g.constraints.X(...)` resolves into `FEMData` and emits automatically (§4.4); stage-bind via `s.X(name=...)` |

### 4.4 Multi-point constraints

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
/ run` calls `build()` internally.

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
> for the solver zone.

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

### Shell elements need ndf=6

Shell elements require six DOFs per node. If your model mixes shells
with solids, set `ndf=6` for the entire model.

For shell-on-solid mixed-ndf models where the model-wide bump is
wasteful, the top-level `g.node_ndf` composite (sibling to
`g.constraints` / `g.loads` / `g.masses`) declares per-region DOF
counts. Pair a default with a targeted override:

```python
g.node_ndf.set_default(ndf=3)          # solids stay at 3
g.node_ndf.set("ShellRegion", ndf=6)   # shell nodes get rotational DOFs
```

apeGmsh resolves per-node `ndf` at FEM-build time; nodes not covered
by any declaration raise `LookupError` from `fem.nodes.ndf_for(nid)`.
See [ADR 0032](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0032-explicit-only-per-node-ndf.md)
for the explicit-only doctrine and the dimensional-resolution-contract
alignment. See [ADR 0033](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0033-s2-emit-wiring-per-node-ndf.md)
for how the broker's per-node `ndf` flows into the OpenSees emit paths:
`-ndf K` is passed per-node when an override exists (`g.node_ndf.set`);
sentinel slots elide `-ndf` and the envelope (`apeSees(fem).model(ndm,
ndf=K)`) wins — matching OpenSees-native `model -ndf K` + per-node
override semantics.

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

### Ingest is selective

Ingest from the session is **selective**, not all-or-nothing.
**Auto-emitted** (do not re-declare): session **loads** (`g.loads.*`,
emitted as synthesized Plain patterns) and **MP constraints**
(`g.constraints.*`, §4.4). **Re-declared explicitly on `ops`** (§4):
**masses** (`ops.mass`) and **support fixities / homogeneous SPs**
(`ops.fix`) — the bridge reads only its own `ops.mass` / `ops.fix`
records, not `g.masses` / `fem.nodes.sp`. Because loads auto-emit,
re-declaring a `g.loads.*` load on a bridge `p.load(...)` doubles it
(reactions at 2×) — pick one channel per load. Every session
declaration is also preserved for the viewer / `Results` via the
`model.h5` neutral zone.

### Emit calls are separate statements

`ops.tcl(...)`, `ops.py(...)`, `ops.h5(...)`, `ops.run()` are **not**
fluent. Write each on its own line; each builds internally.
