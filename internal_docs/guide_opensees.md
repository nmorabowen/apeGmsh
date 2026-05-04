# The OpenSees Bridge

apeGmsh ships with a first-class bridge to OpenSees. The bridge takes geometry
and mesh from an apeGmsh session, combines them with material definitions,
element assignments, loads, and boundary conditions, and produces ready-to-run
Tcl or openseespy scripts. You never touch raw node numbering or element
connectivity -- the bridge builds the entire model from physical groups and
FEMData snapshots you have already defined.

The bridge lives on every session as `g.opensees`, a composition container with
six sub-composites and two lifecycle entry points:

| Sub-composite         | Purpose                                      |
|-----------------------|----------------------------------------------|
| `g.opensees.materials`| nDMaterial, uniaxialMaterial, section         |
| `g.opensees.elements` | element assignment, geomTransf, fix           |
| `g.opensees.ingest`   | pull resolved loads/masses/constraints from FEMData |
| `g.opensees.inspect`  | post-build node/element tables and summary    |
| `g.opensees.export`   | Tcl and openseespy script emission            |
| `g.opensees.recorders`| recorder declarations (mounted from `Recorders`); see `guide_recorders_reference.md` |

Plus two lifecycle methods on the bridge itself: `g.opensees.set_model(ndm, ndf)`
and `g.opensees.build()`.

Every declaration method returns `self` for fluent chaining. The bridge validates
nothing until `build()`, so you can declare materials, element types, and boundary
conditions in any order. The overall pipeline is:

```
geometry --> mesh --> materials & assignments --> FEMData snapshot
    --> ingest loads/masses/constraints --> build() --> export
```


## 1. Model Dimensions -- `set_model`

Tell the bridge how many spatial dimensions and DOFs per node your model has:

```python
g.opensees.set_model(ndm=3, ndf=3)
```

Both arguments are keyword-only. Typical combinations:

| `ndm` | `ndf` | Use case                                   |
|-------|-------|--------------------------------------------|
| 2     | 2     | 2-D solid (ux, uy)                         |
| 2     | 3     | 2-D frame (ux, uy, rz)                     |
| 3     | 3     | 3-D solid (ux, uy, uz)                     |
| 3     | 6     | 3-D frame or shell (ux, uy, uz, rx, ry, rz)|

The default is `ndm=3, ndf=3`. `build()` raises `ValueError` if any assigned
element type is incompatible with the declared `ndm`/`ndf`.


## 2. Materials

The materials sub-composite (`g.opensees.materials`) holds three registries
mirroring how OpenSees organises material-like objects. Each method takes a
user-chosen name, the OpenSees type string, and keyword arguments forwarded
verbatim in declaration order.

### 2.1 nDMaterial -- continuum solids

Use `add_nd_material` for any element that takes an `nDMaterial` in OpenSees:
tetrahedra, bricks, quads, and triangles.

```python
g.opensees.materials.add_nd_material(
    "Concrete", "ElasticIsotropic",
    E=30e9, nu=0.2, rho=2400,
)
```

The `name` is referenced later in `elements.assign(... material="Concrete")`.
The `ops_type` must match an OpenSees nDMaterial type exactly --
`"ElasticIsotropic"`, `"J2Plasticity"`, `"DruckerPrager"`, etc. Keyword
arguments are written to the exported script in the order you pass them.

### 2.2 uniaxialMaterial -- trusses and springs

Use `add_uni_material` for trusses, corotational trusses, and zeroLength springs.

```python
g.opensees.materials.add_uni_material(
    "Steel", "Steel01",
    Fy=250e6, E=200e9, b=0.01,
)
```

### 2.3 Section -- shells

Use `add_section` for shell elements. The most common is
`ElasticMembranePlateSection`, combining membrane and bending behaviour.

```python
g.opensees.materials.add_section(
    "Slab", "ElasticMembranePlateSection",
    E=30e9, nu=0.2, h=0.2, rho=2400,
)
```

Note the parameter name is `section_type` rather than `ops_type`, matching the
OpenSees `section` command syntax. All three methods return the `_Materials`
instance for chaining:

```python
(g.opensees.materials
    .add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2, rho=2400)
    .add_uni_material("Steel", "Steel01", Fy=250e6, E=200e9, b=0.01)
    .add_section("Slab", "ElasticMembranePlateSection", E=30e9, nu=0.2, h=0.2, rho=2400))
```


## 3. Element Assignment

The elements sub-composite (`g.opensees.elements`) binds OpenSees element types
to physical groups. It has three methods: `assign`, `fix`, and `add_geom_transf`.

### 3.1 Assigning element types -- `assign`

```python
g.opensees.elements.assign(
    pg_name,         # physical-group or label name
    ops_type,        # OpenSees element type string
    *,
    material=...,    # name from the matching material registry
    geom_transf=..., # name from add_geom_transf (beams only)
    dim=...,         # disambiguation hint when a name exists in multiple dims
    **extra,         # element-specific scalar parameters
)
```

The bridge resolves `pg_name` against both physical groups and apeGmsh labels,
so `promote_to_physical` is not needed. If a name exists in multiple dimensions,
pass `dim=3` or `dim=2` to disambiguate.

The `material` argument must match a name in the correct registry:

- Solid elements need an `nDMaterial` (`materials.add_nd_material`)
- Truss elements need a `uniaxialMaterial` (`materials.add_uni_material`)
- Shell elements need a `section` (`materials.add_section`)
- Beam elements take no material -- section properties go in `**extra`

The `**extra` keyword arguments fill element-specific slots: `bodyForce` for
solids, `thick`/`eleType` for 2-D, `A` for trusses, `A`/`E`/`G`/`Jx`/`Iy`/`Iz`
for beams. Examples:

```python
# 3-D solid tetrahedron
g.opensees.elements.assign(
    "Body", "FourNodeTetrahedron",
    material="Concrete",
    bodyForce=[0, 0, -9.81 * 2400],
)

# 2-D plane-stress quad
g.opensees.elements.assign(
    "Plate", "quad",
    material="Steel_2D",
    thick=0.01, eleType="PlaneStress",
)

# Truss with cross-section area
g.opensees.elements.assign(
    "Diagonals", "corotTruss",
    material="Steel", A=3.14e-4,
)

# Elastic beam-column (3-D)
g.opensees.elements.assign(
    "Columns", "elasticBeamColumn",
    geom_transf="ColTransf",
    A=0.04, E=200e9, G=77e9,
    Jx=1e-4, Iy=2e-4, Iz=2e-4,
)

# Shell (section-based)
g.opensees.elements.assign(
    "SlabSurface", "ShellMITC4",
    material="Slab",
)
```

### 3.2 Supported element types

| Family | Types |
|--------|-------|
| 3-D solid | `FourNodeTetrahedron`, `TenNodeTetrahedron`, `stdBrick`, `bbarBrick`, `SSPbrick` |
| 2-D solid | `quad`, `tri31`, `SSPquad` |
| Shell | `ShellMITC3`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4` |
| Truss | `truss`, `corotTruss` |
| Beam | `elasticBeamColumn`, `ElasticTimoshenkoBeam` |

Passing an unrecognised `ops_type` raises `ValueError` listing all supported
types. If your Gmsh mesh uses second-order elements but the assigned OpenSees
type only supports first-order, `build()` issues a `UserWarning` and discards
mid-side nodes automatically.

### 3.3 Geometric transformations -- `add_geom_transf`

Beam elements require a geometric transformation defining their local coordinate
system. Register one before calling `assign`:

```python
g.opensees.elements.add_geom_transf(
    "ColTransf", "PDelta",
    vecxz=[0, 0, 1],
)
```

- `name` -- referenced in `assign(..., geom_transf="ColTransf")`
- `transf_type` -- `"Linear"`, `"PDelta"`, or `"Corotational"`
- `vecxz` -- the local x-z plane vector (3-D only; ignored for 2-D models)

```python
# 2-D frame -- no vecxz needed
g.opensees.elements.add_geom_transf("Beams", "Linear")

# 3-D frame -- vecxz required
g.opensees.elements.add_geom_transf("Beams3D", "Corotational", vecxz=[0, 1, 0])
```

For non-axis-aligned beams, the static helper
`g.opensees.elements.vecxz(axis, local_z=(0, 0, 1), roll_deg=0.0)`
computes a valid `vecxz` from the beam's axis vector and a reference
local-z direction, with optional section rotation about the beam axis
via `roll_deg`. See `_opensees_elements.py:44-116`.

```python
# Inclined beam, default local-z toward global +Z
vxz = g.opensees.elements.vecxz(axis=(1, 0, 1))
g.opensees.elements.add_geom_transf("Diag", "Linear", vecxz=vxz)
```

### 3.4 Boundary conditions -- `fix`

Apply homogeneous single-point constraints to every node in a physical group:

```python
g.opensees.elements.fix("BasePlate", dofs=[1, 1, 1])
g.opensees.elements.fix("PinnedBase", dofs=[1, 1, 1, 0, 0, 0])  # ndf=6
```

The `dofs` list must have exactly `ndf` entries (`1` = fixed, `0` = free). The
`dim` keyword disambiguates when a name exists at multiple dimensions. Unlike
other declarations, `fix` validates the `dofs` length immediately.

**Alternative for solid faces:** For solid meshes where you need to prescribe
displacements or rotations at a face, use `g.loads.face_sp(...)` instead. It
maps a rigid-body motion at the face centroid to per-node displacements via
`u_i = disp_xyz + rot_xyz × r_i` and emits `ops.fix()` (homogeneous) or
`ops.sp()` (non-zero) at each face node. See `guide_loads.md` §11.


## 4. Ingesting Loads, Masses, and Constraints from FEMData

The bridge does not define loads or masses itself. Those are defined on the
apeGmsh session (`g.loads`, `g.masses`), resolved into per-node and per-element
records by the FEMData snapshot, and injected into the bridge through
`g.opensees.ingest`. This "define on the session, resolve on the snapshot"
architecture means the same definitions work for any solver bridge.

### 4.1 Pulling loads -- `ingest.loads(fem)`

```python
fem = g.mesh.queries.get_fem_data(dim=3)
g.opensees.ingest.loads(fem)
```

This translates `fem.nodes.loads` and `fem.elements.loads` into internal
load-pattern dicts. Each pattern you defined with `g.loads.pattern(...)` becomes
its own `pattern Plain` block in the output. Supported element load types are
`beamUniform` (`wy`, `wz`, `wx`) and `surfacePressure` (`p`).

### 4.2 Pulling masses -- `ingest.masses(fem)`

```python
g.opensees.ingest.masses(fem)
```

Translates `fem.nodes.masses` into `ops.mass(node, mx, my, mz, ...)` commands.
If you defined volume masses with `g.masses.volume(...)`, the FEMData snapshot
has already distributed them to nodes -- the bridge passes them through.

### 4.3 Pulling SP records -- `ingest.sp(fem)`

```python
g.opensees.ingest.sp(fem)
```

Translates `fem.nodes.sp` (populated by `g.loads.face_sp(...)`) into internal
SP records. At export time, homogeneous records become `ops.fix()` calls and
non-zero prescribed displacements become `ops.sp(node, dof, value)` calls.

### 4.4 Pulling constraints -- `ingest.constraints(fem)`

```python
g.opensees.ingest.constraints(fem, tie_penalty=1e12)
```

Ingests constraint records from `fem.nodes.constraints` and
`fem.elements.constraints`. Currently the bridge emits **tie** constraints as
`element ASDEmbeddedNodeElement` commands -- a penalty formulation that
constrains a slave node to interpolate within three retained master nodes. The
optional `tie_penalty` sets the `-K` flag; when omitted, OpenSees defaults to
`1.0e18`. Drop to `1e10`--`1e12` if you see conditioning issues.

Quad-face master topologies are split at the diagonal into two triangles, and
higher-order faces (Tri6, Quad8) are downgraded to corners. Node-pair
constraints (`equalDOF`, `rigidLink`) and rigid diaphragms are ingested but
emission is deferred to future phases.

### 4.5 Chaining ingest calls

All methods return `self`, so a common pattern is:

```python
fem = g.mesh.queries.get_fem_data(dim=3)
(g.opensees.ingest
    .loads(fem)
    .sp(fem)
    .masses(fem)
    .constraints(fem, tie_penalty=1e12))
```


## 5. Building the Model -- `build()`

```python
g.opensees.build()
```

This extracts the active Gmsh mesh and constructs all internal tables. It must
be called after `g.mesh.generation.generate()` and after all declarations.

`build()` performs five steps: (1) reads every unique mesh node and builds a
sequential ID mapping, pruning disconnected nodes; (2) assigns sequential tags
to all registered materials, sections, and transforms; (3) extracts elements
per assigned physical group, validates topology compatibility, and maps node
connectivity; (4) validates all material references, geomTransf requirements,
and ndm/ndf compatibility -- raising `ValueError` with detailed messages on
mismatch; (5) emits tie constraint elements if any were ingested.

After `build()` returns, the model is compiled and ready for inspection or
export.


## 6. Export

The export sub-composite writes the built model to disk. Both exporters read
exclusively from internal tables -- no Gmsh I/O.

### 6.1 Tcl script -- `export.tcl(path)`

```python
g.opensees.export.tcl("model.tcl")
```

Produces a complete OpenSees Tcl input file containing (in order): model
builder, nodes, materials/sections/transforms, element connectivity with
physical-group comments, tied-interface elements, fix commands, nodal masses,
and load patterns with `eleLoad` commands for beam-uniform and surface loads.

### 6.2 openseespy script -- `export.py(path)`

```python
g.opensees.export.py("model.py")
```

Produces an equivalent Python script using `openseespy.opensees as ops`. The
structure mirrors the Tcl output. Both methods return `_Export` for chaining:

```python
g.opensees.export.tcl("model.tcl").py("model.py")
```

### 6.3 Embedding recorders in the exported script

Both `export.tcl` and `export.py` accept four recorder-related kwargs
(`_opensees_export.py:30-38, 260-268`):

| Kwarg                    | Purpose |
|--------------------------|---------|
| `recorders=`             | A `ResolvedRecorderSpec` to emit alongside the model. One `recorder ...` line (or `ops.recorder(...)` call) per resolved record, after the model definition. |
| `recorders_output_dir=`  | Directory prefix for recorder output files. Default `""` = same dir as the script. Trailing `/` optional. |
| `recorders_file_format=` | `"out"` (text, default) or `"xml"`, or `"mpco"`. The `"mpco"` value switches the emitter to `to_mpco_tcl_command` / `to_mpco_python_command` and writes a single MPCO recorder block instead of per-record `.out` lines. |
| `manifest_path=`         | Where to write the recorder manifest sidecar (HDF5). Required for the `.out` transcoder to decode the emitted files. Defaults to `<path>.manifest.h5` when `recorders` is given. |

The user is responsible for resolving `recorders` against the same
`FEMData` snapshot the bridge was built from — there is no enforcement
in the exporter.


## 7. Inspection

The inspect sub-composite provides three methods. `summary()` works any time;
`node_table()` and `element_table()` require `build()` (raise `RuntimeError`
otherwise).

### 7.1 Summary

```python
print(g.opensees.inspect.summary())
```

Returns a multi-line string listing all registered materials, element
assignments, boundary conditions, load patterns, and (post-build) node/element
counts by type and physical group. The quickest sanity check before exporting.

### 7.2 Node table

```python
df = g.opensees.inspect.node_table()
```

DataFrame indexed by OpenSees node ID with `x`, `y`, `z` coordinates,
`fix_1`...`fix_{ndf}` boolean flags, and `load_1`...`load_{ndf}` cumulative
force values. Useful for verifying boundary conditions and load placement.

### 7.3 Element table

```python
df = g.opensees.inspect.element_table()
```

DataFrame indexed by element ID with `gmsh_id`, `ops_type`, `pg_name`,
`mat_name`, `mat_tag`, `sec_tag`, `transf_tag`, `n_nodes`, `nodes` (tuple),
`slots`, and `extra`. Use it to verify element counts and connectivity.


## 8. Complete Example

End-to-end: create a concrete block, mesh, define materials and BCs, apply
loads, build, and export.

```python
from apeGmsh import apeGmsh

# ── Session ───────────────────────────────────────────────
g = apeGmsh("concrete_block")
g.begin()

# ── Geometry ──────────────────────────────────────────────
# A simple 2m x 1m x 1m block
g.model.geometry.add_box(0, 0, 0,  2.0, 1.0, 1.0, label="Block")

# Label the base surface for boundary conditions
g.model.selection.select_surfaces(on_plane=("z", 0.0)).label("Base")

# Label the top surface for loading
g.model.selection.select_surfaces(on_plane=("z", 1.0)).label("Top")

g.model.geometry.synchronize()

# ── Mesh ──────────────────────────────────────────────────
g.mesh.generation.set_size_global(0.15)
g.mesh.generation.generate(3)

# ── OpenSees material ────────────────────────────────────
g.opensees.materials.add_nd_material(
    "Concrete", "ElasticIsotropic",
    E=30e9, nu=0.2, rho=2400,
)

# ── Element assignment ───────────────────────────────────
g.opensees.elements.assign(
    "Block", "FourNodeTetrahedron",
    material="Concrete",
    bodyForce=[0, 0, -9.81 * 2400],
)

# ── Boundary conditions ──────────────────────────────────
g.opensees.elements.fix("Base", dofs=[1, 1, 1])

# ── Loads ─────────────────────────────────────────────────
with g.loads.pattern("Gravity"):
    g.loads.point("Top", force_xyz=(0, 0, -50e3))

# ── Masses ────────────────────────────────────────────────
g.masses.volume("Block", density=2400)

# ── Model dimensions ─────────────────────────────────────
g.opensees.set_model(ndm=3, ndf=3)

# ── FEMData snapshot and ingest ──────────────────────────
fem = g.mesh.queries.get_fem_data(dim=3)
(g.opensees.ingest
    .loads(fem)
    .masses(fem))

# ── Build and inspect ────────────────────────────────────
g.opensees.build()
print(g.opensees.inspect.summary())

# ── Export ────────────────────────────────────────────────
g.opensees.export.tcl("block_model.tcl").py("block_model.py")

g.end()
```

The exported files are self-contained model definitions. Run them with
`opensees block_model.tcl` or `python block_model.py` after appending your
analysis commands (the bridge intentionally omits analysis setup).


## 9. Practical Advice

### Declaration order does not matter

Materials, assignments, BCs, and transforms can be declared in any order. The
bridge resolves everything at `build()` time. The only hard constraint is that
`build()` comes after `generate()` and after `ingest.*()` calls.

### Physical groups vs. labels

You can pass either a Gmsh physical-group name or an apeGmsh label to `assign`,
`fix`, or any target argument. The bridge resolves labels automatically, so
`promote_to_physical` is not needed for the OpenSees workflow.

### Beam elements need geomTransf

Beam elements do not use a material registry. Their section properties are
scalar kwargs in `assign` (`A=0.04, E=200e9`), but they require a
`geomTransf`. Forgetting it is the most common beam-model error -- `build()`
catches it.

### Shell elements need ndf=6

Shell elements require six DOFs per node. If your model mixes shells with
solids, set `ndf=6` for the entire model.

### The export stops at model definition

The scripts omit analysis commands (integrator, algorithm, system, recorder).
Append your analysis block or source/import the generated file from a driver
script.

### Tie constraints for non-conformal interfaces

Connect parts with non-matching meshes using apeGmsh ties:

```python
g.constraints.tie("ColumnTop", "SlabBottom", dofs=[1, 2, 3])
fem = g.mesh.queries.get_fem_data(dim=3)
g.opensees.ingest.constraints(fem, tie_penalty=1e12)
g.opensees.build()
```

The bridge emits `ASDEmbeddedNodeElement` commands with penalty formulation.
Set `tie_penalty` to `1e10`--`1e12` for structural models; the OpenSees default
of `1e18` can cause ill-conditioning with iterative solvers.

### Chaining within sub-composites

Every method returns `self`, so you can chain within a sub-composite:

```python
(g.opensees.elements
    .add_geom_transf("ColTransf", "PDelta", vecxz=[0, 0, 1])
    .assign("Columns", "elasticBeamColumn", geom_transf="ColTransf",
            A=0.04, E=200e9, G=77e9, Jx=1e-4, Iy=2e-4, Iz=2e-4)
    .assign("SlabSurface", "ShellMITC4", material="Slab")
    .fix("Base", dofs=[1, 1, 1, 0, 0, 0]))
```

Chaining across sub-composites does not work -- `materials.add_*` returns the
`_Materials` instance, not the bridge. Start a new expression for each
sub-composite.
