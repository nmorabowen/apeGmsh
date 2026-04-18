# OpenSees bridge — `g.opensees.*`

The OpenSees bridge is a five-composite pipeline that turns a
session's mesh + declarations into an OpenSees script:

```
g.opensees.materials   → nDMaterial / uniaxialMaterial / section
g.opensees.elements    → geomTransf / element assignment / fix
g.opensees.ingest      → pull resolved records from FEMData
g.opensees.inspect     → post-build tables and summary
g.opensees.export      → write .tcl or .py script
```

Plus two lifecycle entry points that stay flat on `g.opensees`:

```python
g.opensees.set_model(ndm=3, ndf=3)    # must come first
g.opensees.build()                      # must come after all declarations
```

All signatures are read from `src/apeGmsh/solvers/OpenSees.py`,
`_opensees_materials.py`, `_opensees_elements.py`, `_opensees_ingest.py`,
`_opensees_inspect.py`, `_opensees_export.py`.

## Lifecycle

```python
g.opensees.set_model(ndm=3, ndf=3)       # 3-D solid
# … declarations …
g.opensees.build()
# … inspect / export …
```

`ndm` / `ndf` combinations the emitter understands:

| ndm | ndf | typical use |
|-----|-----|-------------|
| 2   | 2   | 2-D solid (ux, uy) |
| 2   | 3   | 2-D frame (ux, uy, θz) |
| 3   | 3   | 3-D solid (ux, uy, uz) |
| 3   | 6   | 3-D frame / shell (ux, uy, uz, θx, θy, θz) |

`build()` extracts the active mesh, allocates tags for every named
registry entry, and validates the whole model (element specs,
material / section / transf references, ndm/ndf compatibility).
It raises early with a pointed error if anything is inconsistent —
don't skip the error and try to export anyway.

Everything that follows must be **before** `build()`, except
`inspect.*` and `export.*` which are post-build only.

## `g.opensees.materials`

Three separate registries, one for each OpenSees material-like
concept.  All three are fluent (return self) so you can chain.

```python
(g.opensees.materials
    .add_nd_material("Conc", "ElasticIsotropic", E=30e9, nu=0.2, rho=2400)
    .add_nd_material("Soil", "DruckerPrager",
                     K=80e6, G=60e6, sigmaY=20e3, rho=0.0,
                     rhoBar=0.0, Kinf=0.0, Ko=0.0,
                     delta1=0.0, delta2=0.0, H=0.0, theta=0.0)
    .add_uni_material("Steel", "Steel01", Fy=250e6, E=200e9, b=0.01)
    .add_section("Slab", "ElasticMembranePlateSection",
                 E=30e9, nu=0.2, h=0.2, rho=2400))
```

Rules of thumb for which registry:

- **`add_nd_material`** → solid elements (`FourNodeTetrahedron`,
  `stdBrick`, `SSPbrick`, `bbarBrick`, `quad`, `tri31`, `SSPquad`).
- **`add_uni_material`** → `truss`, `corotTruss`, `zeroLength`
  springs, and nonlinear beam fibers.
- **`add_section`** → shell elements (`ShellMITC4`, `ShellDKGQ`,
  `ASDShellQ4`).  Most-used section type is
  `ElasticMembranePlateSection`.

`**params` is forwarded verbatim to the OpenSees command in
declaration order — mind the OpenSees argument order for each
material type.

## `g.opensees.elements`

Three concerns, all fluent:

### Geometric transformations (beam elements only)

```python
# 2-D: no vecxz
g.opensees.elements.add_geom_transf("Cols", "PDelta")

# 3-D: vecxz is the vector in the local x-z plane (NOT the z-axis)
g.opensees.elements.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
```

`transf_type` ∈ `{"Linear", "PDelta", "Corotational"}`.

### Element assignment

`assign(pg_name, ops_type, *, material=None, geom_transf=None, dim=None, **extra)`
says "write every mesh element in this physical group as
*ops_type*".  Validation is deferred to `build()` so you can chain
many assignments without immediately querying Gmsh.

```python
# Solid — material from the nd registry
g.opensees.elements.assign(
    "Body", "FourNodeTetrahedron", material="Conc",
    bodyForce=[0, 0, -9.81 * 2400],
)

# Truss / diagonals — material from the uni registry
g.opensees.elements.assign(
    "Diags", "corotTruss", material="Steel", A=3.14e-4,
)

# Elastic beam — scalar props via **extra, transf by name
g.opensees.elements.assign(
    "Cols", "elasticBeamColumn", geom_transf="ColTransf",
    A=0.04, E=200e9, G=77e9, Jx=1e-4, Iy=2e-4, Iz=2e-4,
)

# Shell — section from the section registry
g.opensees.elements.assign("Deck", "ShellMITC4", material="Slab")
```

`material=` resolves against the right registry for the chosen
`ops_type` (enforced by the element spec registry in
`_element_specs.py`).  `geom_transf=` is **required** for beam
element types.  If a PG name exists at multiple dimensions (e.g.
because you created both a volume and its boundary with the same
name), pass `dim=` to disambiguate.

### Fix (single-point constraints)

```python
g.opensees.elements.fix("Base", dofs=[1, 1, 1])                # 3-D solid
g.opensees.elements.fix("PinnedEnd", dofs=[1, 1, 1, 0, 0, 0])  # 3-D frame/shell
```

`len(dofs)` must equal `ndf`; it's a hard check at declare-time,
not build-time.  `dim=` disambiguates like `assign`.

## `g.opensees.ingest`

The ingest composite pulls *resolved* records out of a FEMData
snapshot into the bridge's internal tables.  All four methods are
fluent; the snapshot passed in is the one returned by
`g.mesh.queries.get_fem_data(...)`.

```python
(g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .sp(fem)
    .constraints(fem, tie_penalty=1e12))
```

- `loads(fem)` → consumes `fem.nodes.loads` and `fem.elements.loads`.
  Nodal records become `pattern Plain` block entries; element
  records become `eleLoad -beamUniform` or `eleLoad -surfaceLoad`
  depending on `load_type`.
- `masses(fem)` → consumes `fem.nodes.masses`.  Each record
  becomes one `mass` command, padded / sliced to `ndf`.
- `sp(fem)` → consumes `fem.nodes.sp` (populated by
  `g.loads.face_sp(...)`).  Homogeneous records group into
  `fix` with a DOF mask; prescribed displacements emit `sp`.
- `constraints(fem, tie_penalty=None)` → stores the surface
  constraint set (tie interpolations) and node constraint set
  on the bridge.  Tie records become `element ASDEmbeddedNodeElement`
  with the given penalty stiffness.  `tie_penalty=None` uses
  OpenSees' built-in default (1e18); 1e10 – 1e12 is safer for
  conditioning.  Other node-pair kinds (equal_dof, rigid_beam,
  rigid_rod, rigid_diaphragm) are ingested but emission is in
  later phases — until then, emit them yourself via
  `fem.nodes.constraints.pairs()`.

Any `ingest.*` call on an empty set is a no-op.

## `g.opensees.build()`

Single entry point to:

1. Pull the mesh into the node / element DataFrames
   (`ops._nodes_df`, `ops._elements_df`).
2. Build the node map (Gmsh tag → solver ID) — after
   `g.mesh.partitioning.renumber()` this is the identity, so
   the solver IDs are the Gmsh tags.
3. Allocate sequential integer tags for every material, section,
   and transf name.
4. Validate every element assignment:
   - element type is known
   - ndm / ndf match the element's spec
   - material name resolves to the correct registry
   - geom_transf is declared for beam types
   - Gmsh element order is compatible (warns on higher-order
     downgrade)
5. Emit any tie elements from the ingested constraint set.

If anything is wrong, `build()` raises a clear error naming the
offending PG / name.  **Always** call `build()` before any
`inspect.*` or `export.*` — they raise `RuntimeError` otherwise.

## `g.opensees.inspect` (post-build)

```python
g.opensees.inspect.node_table()      # DataFrame indexed by ops node ID,
                                      # columns: x, y, z, fix_i, load_i
g.opensees.inspect.element_table()   # connectivity, material, transf
g.opensees.inspect.summary()         # multi-line text summary
```

Use `node_table` / `element_table` when you want to sanity-check
the model (e.g. confirm every support is actually fixed, or that
a load shows up on the right DOF) without opening the emitted
script.

## `g.opensees.export`

Two emitters, both fluent so you can chain them:

```python
(g.opensees.export
    .tcl("out/model.tcl")
    .py("out/model.py"))
```

Both emit the same section order:

1. `model BasicBuilder -ndm N -ndf M`
2. `node` lines
3. `nDMaterial`, `uniaxialMaterial`, `section`, `geomTransf`
4. `element` lines
5. Tied-interface `ASDEmbeddedNodeElement` lines (if any)
6. `fix` lines (from `elements.fix`)
7. Face-SP `fix` (homogeneous) and `sp` (prescribed displacements)
8. `mass` lines
9. `pattern Plain <i> Linear { ... }` blocks with nodal loads
   and `eleLoad -beamUniform` / `-surfaceLoad` entries

The Tcl and Python outputs are structurally identical — pick Tcl
if you're running classic OpenSees, `.py` if you want openseespy.

## Canonical skeleton

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="solid") as g:
    # 1. Geometry
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    g.physical.add(3, [box], name="Body")
    g.physical.add_surface([1], name="Base")   # face tag

    # 2. Pre-mesh definitions
    with g.loads.pattern("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)

    # 3. Mesh + renumber for small bandwidth
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)

    # 4. Snapshot
    fem = g.mesh.queries.get_fem_data(dim=3)

    # 5. OpenSees
    (g.opensees
        .set_model(ndm=3, ndf=3))
    (g.opensees.materials
        .add_nd_material("Conc", "ElasticIsotropic",
                         E=30e9, nu=0.2, rho=2400))
    (g.opensees.elements
        .assign("Body", "FourNodeTetrahedron", material="Conc")
        .fix("Base", dofs=[1, 1, 1]))
    (g.opensees.ingest
        .loads(fem)
        .masses(fem))
    g.opensees.build()

    print(g.opensees.inspect.summary())
    (g.opensees.export
        .tcl("out/model.tcl")
        .py("out/model.py"))
```

## When things go wrong

- **`build() must be called after …`** — you called an `inspect` or
  `export` method without `build()`.  Don't skip it.
- **`Ambiguous physical-group name X`** — the same name was used at
  multiple dimensions.  Pass `dim=` to `assign(...)` / `fix(...)`.
- **`Physical group or label X not found`** — the bridge also falls
  back to labels (via the internal `_label:` prefix), so the name
  doesn't need to be a PG.  If it's not found, confirm with
  `g.physical.summary()` / `g.labels.get_all()` that the name
  exists at all.
- **`unknown ops_type X`** — the element registry doesn't know the
  name.  Check `_element_specs._ELEM_REGISTRY` — the set is
  deliberately small and well-tested; submit a PR to add a new one
  rather than working around it.
- **`len(dofs) != ndf`** — classic off-by-three.  `fix` always
  requires a mask of length `ndf`, not the element's node count.
- **Ingested element loads with `load_type` other than
  `beamUniform` or `surfacePressure`** — emission falls through
  to a commented-out line in the script.  Either extend the
  emitter or emit the load manually via `fem.elements.loads`.
