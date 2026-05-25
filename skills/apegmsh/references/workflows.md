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

    # OpenSees — post-session bridge, typed primitives. Loads/masses
    # are re-declared explicitly (the bridge does NOT ingest g.loads /
    # g.masses); see opensees-bridge.md.
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(
        pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
    )
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tip", forces=(0.0, 0.0, -1e4))
    ops.py("out/block.py")
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

    # Hand off to OpenSees — post-session bridge. Materials are typed
    # handles; loads/masses re-declared explicitly. See opensees-bridge.md.
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3, rho=7850)
    conc  = ops.nDMaterial.ElasticIsotropic(E=30e9,  nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(
        pg="Girder", material=steel, body_force=(0.0, 0.0, -9.81 * 7850),
    )
    ops.element.FourNodeTetrahedron(
        pg="Deck", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
    )
    ops.py("out/bridge.py")
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
    # face.  tie() creates a surface-constraint record in the FEMData
    # snapshot.  ⚠️ apeSees does NOT emit it — MP-constraint emission
    # is deferred (see opensees-bridge.md "constraints are DEFERRED").
    # The record persists into model.h5 for the viewer/Results only;
    # to actually run this coupled model you must hand-emit the
    # constraint into raw openseespy yourself.
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
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    soil = ops.nDMaterial.ElasticIsotropic(E=50e6, nu=0.3, rho=2000)
    cols_t = ops.geomTransf.Linear(vecxz=(1, 0, 0))
    ops.element.stdBrick(pg="Soil", material=soil)
    ops.element.elasticBeamColumn(
        pg="Column", transf=cols_t,
        A=0.09, E=30e9, G=12.5e9, J=1e-3, Iy=6.75e-4, Iz=6.75e-4,
    )
    # loads/masses re-declared explicitly; the g.constraints.tie above
    # is NOT emitted (deferred — see the note at the tie() call).
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="ColTop", forces=(1e4, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.py("out/hybrid.py")
```

## 4. Pushover-style second pattern

In the new bridge, **patterns are explicit** — each is its own
`with ops.pattern.Plain(series=...) as p:` block, opened directly
on `ops`. The session may still declare loads for the
viewer/`Results`, but the runnable deck only contains patterns you
open on the bridge. Gravity is best expressed as an element
`body_force=`; the lateral pushover is a nodal `p.load`.

```python
with apeGmsh(model_name="pushover") as g:
    # ... geometry / mesh setup ...
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)

# Gravity — body force on the elements (pattern 1 equivalent)
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))

# Pushover — explicit pattern 2: unit lateral load at the control PG
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="ControlNode", forces=(1.0, 0.0, 0.0))

ops.py("out/pushover.py")
```

Patterns appear in the deck in the order you open them on `ops`
(`pattern Plain <i> Linear { ... }`, indexed from 1). On the
openseespy side, drive gravity with `LoadControl` and the lateral
step with `DisplacementControl`.

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

### Diagnosing disjoint topology (arc-line wires, IGES imports)

When a wire is built from a partial arc plus straight lines —
`add_ellipse(angle1, angle2)` + `add_line(...)`, or an IGES import
with un-welded joints — OCC frequently fails to weld the arc/line
endpoints into a single point.  The mesh then carries two distinct
nodes at every corner with no element or constraint bridging them,
and moments at the corner read wrong (no continuity).

The canonical fix is at the **geometry layer**:

```python
# Build the wire
g.model.geometry.add_ellipse(0, 0, 0, 2.55, 2.75,
                             angle1=0, angle2=math.pi, label="arch")
g.model.geometry.add_line("p_left",   "arch_start")
g.model.geometry.add_line("arch_end", "p_right")

# Weld arc-line junctions BEFORE constructing Parts or meshing
g.model.queries.make_conformal(dims=[1])

# Now build PGs / Parts and mesh
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data(dim=1)

# Verify there are no unbridged coincident pairs left
pairs = fem.inspect.find_coincident_node_pairs(pg="cimbra", tol=1e-6)
unbridged = {k: v for k, v in pairs.items() if not v}
assert not unbridged, f"unbridged corners: {sorted(unbridged)}"
```

Reading the diagnostic output:

* `pairs == {}` — no coincident pairs anywhere; topology is clean.
* `pairs[(a, b)] == []` — **bug**: two nodes share XYZ but nothing
  ties them. Either re-fragment (`make_conformal`) or add an
  explicit `equal_dof` / `rigid_link` constraint.
* `pairs[(a, b)] == ["element zeroLength#7"]` — legitimate; this is
  what `zeroLength` is for.
* `pairs[(a, b)] == ["constraint equal_dof"]` — legitimate; user
  has explicitly tied the pair.

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
- **Calling `make_conformal()` after building `Part` instances.**
  Fragment renumbers entities and the best-effort remap covers only
  parts registered on the session at call time. A `Part` built before
  fragmenting holds stale tag dicts in its `Instance.entities` and
  will silently misresolve. Run `make_conformal()` *before*
  constructing Parts, or rebuild Parts after fragmenting.
- **Joining partial arcs to straight lines without `make_conformal`.**
  `add_ellipse(angle1, angle2)` / `add_circle(angle1, angle2)` /
  `add_arc(...)` joined to lines produces a wire that OCC treats as
  three disjoint pieces — the mesh carries two nodes at every corner
  with no moment continuity. Call
  `g.model.queries.make_conformal(dims=[1])` after assembly, or
  diagnose with `fem.inspect.find_coincident_node_pairs(pg=, tol=)`
  (entries with empty refs lists are the unbridged duplicates).
- **Asking for 3-D FEMData when the mesh has tets + shell bodies.**
  `get_fem_data(dim=3)` drops the 2-D surface mesh — use
  `dim=None` (all dims) when shells or tied interfaces need to
  reach the bridge.
- **Calling `ops.model(ndm=3, ndf=3)` then declaring a beam
  element.**  Beams need rotational DOFs — use `ndf=6` for 3-D
  frame/shell models.  `ops.build()` (called by any emit) catches
  this; it means the model isn't built yet, not that you can keep
  going.
