# apeGmsh workflows — end-to-end patterns

Concrete recipes for the four workflows that come up most often.
Each one is a working skeleton — fill in the geometry details and
it should run against the v1.0 API without modification.

If a workflow you need isn't here, the `examples/` folder in the
project is the authoritative gallery.  These are the patterns worth
memorizing.

## 1. Single-session solid

One session, one kernel, one mesh → OpenSees.  Everything happens
inside a single `with` block.

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="block", verbose=True) as g:
    # Geometry (add_box auto-synchronizes; sync=True is default)
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")

    # Pull the base face tags out of the model via a thin-slab
    # bounding box at z = 0.  Returns a list of (dim, tag) DimTags.
    eps = 1e-6
    base_dts = g.model.queries.entities_in_bounding_box(
        -eps, -eps, -eps,
        10 + eps, 5 + eps, eps,
        dim=2,
    )
    base_tags = [t for _, t in base_dts]

    # Physical groups — the solver contract
    g.physical.add(3, [box], name="Body")
    g.physical.add_surface(base_tags, name="Base")

    # Loads / masses — declared pre-mesh, resolved by get_fem_data
    with g.loads.pattern("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)

    # Mesh
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)

    # Snapshot
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info.summary())

    # OpenSees
    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material(
        "Conc", "ElasticIsotropic",
        E=30e9, nu=0.2, rho=2400,
    )
    (g.opensees.elements
        .assign("Body", "FourNodeTetrahedron", material="Conc")
        .fix("Base", dofs=[1, 1, 1]))
    (g.opensees.ingest
        .loads(fem)
        .masses(fem))
    g.opensees.build()
    g.opensees.export.py("out/block.py")
```

## 2. Multi-part assembly via `Part`

When parts come from separate CAD files or need to be instanced,
build each part in its own session, then import into an assembly
session via `g.parts`.

```python
from apeGmsh import apeGmsh, Part

# ── Build parts in isolation ───────────────────────────────────
with Part("girder") as girder:
    girder.model.geometry.add_box(0, 0, 0, 20, 0.6, 1.5, label="girder")
    # No save() → auto-persists to a tempfile on __exit__

with Part("deck") as deck:
    deck.model.geometry.add_box(-0.5, -2, 1.5, 21, 4, 0.25, label="deck")
    deck.save("out/deck.step")    # explicit save = caller owns the file

# ── Assemble ───────────────────────────────────────────────────
with apeGmsh(model_name="bridge") as g:
    g.parts.add(girder, label="girder")
    g.parts.add(deck,   label="deck")

    # Fragment so shared interfaces become conformal.  fragment_all
    # synchronizes internally, so no explicit synchronize call here.
    g.parts.fragment_all(dim=3)

    # Physical groups can be created by entity tags from each instance
    for label in g.parts.labels():
        inst = g.parts.get(label)
        for tag in inst.entities.get(3, []):
            g.physical.add(3, [tag], name=label.capitalize())

    # Loads / masses reference part labels directly
    with g.loads.pattern("dead"):
        g.loads.gravity("girder", g=(0, 0, -9.81), density=7850)
        g.loads.gravity("deck",   g=(0, 0, -9.81), density=2400)
    g.masses.volume("girder", density=7850)
    g.masses.volume("deck",   density=2400)

    # Mesh + snapshot
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Hand off to OpenSees
    (g.opensees
        .set_model(ndm=3, ndf=3))
    (g.opensees.materials
        .add_nd_material("Steel", "ElasticIsotropic",
                         E=200e9, nu=0.3, rho=7850)
        .add_nd_material("Conc",  "ElasticIsotropic",
                         E=30e9, nu=0.2, rho=2400))
    (g.opensees.elements
        .assign("Girder", "FourNodeTetrahedron", material="Steel")
        .assign("Deck",   "FourNodeTetrahedron", material="Conc"))
    (g.opensees.ingest
        .loads(fem)
        .masses(fem))
    g.opensees.build()
    g.opensees.export.py("out/bridge.py")
```

Key points:

- Each `Part` owns its own Gmsh session; `with Part(...)` auto-
  persists the geometry to a tempfile on exit so `g.parts.add(part)`
  finds something on disk.
- `g.parts.fragment_all(dim=3)` makes shared faces / edges conformal
  across instances.  Do it *before* creating physical groups at the
  fragmented dimension.
- String selectors in `g.loads.*`, `g.masses.*`, `g.constraints.*`,
  and `fem.nodes.get(target=...)` accept part labels directly — you
  don't have to promote them into physical groups.

## 3. Solid ↔ frame coupling via constraints

A common hybrid: a solid soil block coupled to a frame
superstructure.  `g.constraints.*` declares the coupling pre-mesh
and the FEMData snapshot carries the resolved constraint records
into the OpenSees bridge.

```python
with apeGmsh(model_name="hybrid") as g:
    # Soil volume
    soil = g.model.geometry.add_box(-10, -10, -20, 20, 20, 20,
                                     label="soil")

    # Frame: column points + beam line (kept in a separate PG so we
    # can mesh them as 1-D beams and couple to the soil's top face)
    p_base = g.model.geometry.add_point(0, 0, 0, label="col_base")
    p_top  = g.model.geometry.add_point(0, 0, 6, label="col_top")
    col    = g.model.geometry.add_line(p_base, p_top, label="col")
    # add_point / add_line default to sync=True — no explicit call needed.

    # Physical groups
    g.physical.add(3, [soil], name="Soil")
    g.physical.add(1, [col],  name="Column")
    # Top face of the soil block — thin slab at z = 0
    eps = 1e-6
    top_faces = g.model.queries.entities_in_bounding_box(
        -10 - eps, -10 - eps, -eps,
         10 + eps,  10 + eps,  eps,
        dim=2,
    )
    g.physical.add_surface([t for _, t in top_faces], name="SoilTop")
    g.physical.add(0, [p_base], name="ColBase")
    g.physical.add(0, [p_top],  name="ColTop")

    # Coupling: embed the column's base node into the soil's top
    # face.  tie() creates a surface-constraint record that emits as
    # ASDEmbeddedNodeElement on the OpenSees side.
    g.constraints.tie("SoilTop", "ColBase")

    # Loads + masses
    with g.loads.pattern("dead"):
        g.loads.gravity("Soil", g=(0, 0, -9.81), density=2000)
    g.masses.volume("Soil", density=2000)

    # Mesh
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=None)   # all dims

    # OpenSees (3-D frame/shell: ndm=3, ndf=6 for rotations on the column)
    g.opensees.set_model(ndm=3, ndf=6)
    (g.opensees.materials
        .add_nd_material("Soil", "ElasticIsotropic",
                         E=50e6, nu=0.3, rho=2000))
    (g.opensees.elements
        .add_geom_transf("Cols", "Linear", vecxz=[1, 0, 0])
        .assign("Soil", "stdBrick", material="Soil")
        .assign("Column", "elasticBeamColumn",
                geom_transf="Cols",
                A=0.09, E=30e9, G=12.5e9,
                Jx=1e-3, Iy=6.75e-4, Iz=6.75e-4)
        .fix("ColTop", dofs=[0, 0, 0, 0, 0, 0]))   # free top
    (g.opensees.ingest
        .loads(fem)
        .masses(fem)
        .constraints(fem, tie_penalty=1e12))
    g.opensees.build()
    g.opensees.export.py("out/hybrid.py")
```

## 4. Pushover-style second pattern

Once the gravity pattern is built, add a lateral pattern that
references an already-defined control PG.  The pushover loading is
just another `with g.loads.pattern(...)` block; the OpenSees bridge
emits every pattern in declaration order as `pattern Plain <i>
Linear { ... }`, indexed from 1.

```python
with apeGmsh(model_name="pushover") as g:
    # ... geometry / mesh setup ...

    # Gravity — pattern 1
    with g.loads.pattern("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)

    # Pushover — pattern 2 (unit lateral point load at control node)
    with g.loads.pattern("pushover"):
        g.loads.point("ControlNode", force_xyz=(1.0, 0, 0))

    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # ... declare materials + elements ...
    g.opensees.ingest.loads(fem).masses(fem)
    g.opensees.build()
    g.opensees.export.py("out/pushover.py")
```

Then on the openseespy side you can integrate the two patterns with
a `LoadControl` analysis for gravity and a `DisplacementControl`
analysis for the pushover step — the export writes both pattern
blocks in order, so pattern `1` is gravity and pattern `2` is the
lateral.

## Patterns worth knowing (not full workflows)

### Label → physical group promotion

Labels (Tier 1) are great during geometry because you don't have
to commit to a dimension.  When you want a label to be visible to
the OpenSees bridge (which uses the PG system), promote it:

```python
g.labels.promote_to_physical("col.web")
```

The bridge also accepts label names directly (via the `_label:`
prefix) — so promotion is only needed when an external consumer
reads the raw `.msh` file.

### Selection sets for post-mesh queries

After meshing, you often want to tag a node / element set for later
retrieval (e.g. "all nodes on a plane").  The pattern is: pick the
entities with the geometric `g.model.selection.*` API, then bridge
into mesh-space with `g.mesh_selection.from_geometric(...)`:

```python
# Pre-mesh geometric selection (works on any dim)
top = g.model.selection.select_surfaces(on_plane=("z", 10))

g.mesh.generation.generate(dim=3)

# After meshing, extract the mesh nodes on those surfaces
g.mesh_selection.from_geometric(top, kind="nodes", name="top_nodes")

# Later, when you have a FEMData snapshot:
fem = g.mesh.queries.get_fem_data(dim=3)
tag = fem.mesh_selection.get_tag(dim=0, name="top_nodes")
data = fem.mesh_selection.get_nodes(dim=0, tag=tag)
```

### Ramp to STKO-style post-processing

`g.mesh.queries.get_fem_data()` is also the entry point for
post-processing — combine with `Results.from_fem(fem)` (or use
`fem.viewer(blocking=False)`) to visualize without re-running the
analysis.  For MPCO / STKO outputs see the `stko-to-python` skill.

## Workflow-level pitfalls

- **Assuming you must call synchronize by hand.**  apeGmsh's
  `g.model.geometry.add_*` and `g.model.boolean.*` ops sync
  internally (each has `sync=True` by default).  If you pass
  `sync=False` to batch many ops, the next public call that queries
  the model will synchronize — you almost never need to call
  `gmsh.model.occ.synchronize()` directly.
- **Creating labels on the assembly session and expecting them to
  be solver-visible without `promote_to_physical`.**  Part sessions
  auto-promote labels; assembly sessions do not.
- **Fragmenting before PGs are attached.**  `fragment_all` rewrites
  entity tags; any PG you added by tag (not by label) before
  fragmenting gets stale.  Create PGs after fragmentation, or use
  labels which survive fragmentation.
- **Asking for 3-D FEMData when the mesh has tets + shell bodies.**
  `get_fem_data(dim=3)` drops the 2-D surface mesh — use
  `dim=None` (all dims) when shells or tied interfaces need to
  reach the bridge.
- **Calling `set_model(ndm=3, ndf=3)` then assigning a beam
  element.**  Beams need rotational DOFs — use `ndf=6` for 3-D
  frame/shell models.  The build validator will catch this; it
  means the model isn't built yet, not that you can keep going.
