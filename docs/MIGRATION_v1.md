# Migrating to pyGmsh → apeGmsh v1.0

v1.0 bundles four breaking changes into a single release:

1. **Package rename**: `pyGmsh` → `apeGmsh`
2. **Model API restructure**: `g.model.*` methods split into five focused
   sub-composites (`geometry`, `boolean`, `transforms`, `io`, `queries`).
3. **Mesh API restructure**: `g.mesh.*` methods split into seven focused
   sub-composites (`generation`, `sizing`, `field`, `structured`,
   `editing`, `queries`, `partitioning`).
4. **OpenSees API restructure**: `g.opensees.*` methods split into five
   focused sub-composites (`materials`, `elements`, `ingest`,
   `inspect`, `export`), with `set_model` and `build` staying flat.

Plus several smaller cleanups (`g.mass` → `g.masses`, lifecycle aliases,
legacy deprecation removals).

This guide is a drop-in find/replace checklist. Most codebases can
migrate in one pass with a single search across the project.

**If you were on v0.3.0 (last `pyGmsh` release)**, apply all sections below.
**If you were on v0.3.1 (the transitional `apeGmsh` rename)**, skip section 1.

---

## 1. Package rename: `pyGmsh` → `apeGmsh`

```diff
-from pyGmsh import pyGmsh
+from apeGmsh import apeGmsh

-from pyGmsh import Part, PartsRegistry, Instance
+from apeGmsh import Part, PartsRegistry, Instance

-from pyGmsh.viewers.ui.theme import STYLESHEET
+from apeGmsh.viewers.ui.theme import STYLESHEET

-g = pyGmsh(model_name="bridge")
+g = apeGmsh(model_name="bridge")
```

The class name stays lowercase to match the package (`from apeGmsh import apeGmsh`).

**Install:** uninstall the old wheel (`pip uninstall pyGmsh`) and install the
new one (`pip install -e .` from the repo root).

**What's NOT renamed:** the companion app `pyGmshViewer` keeps its own name.
Its internal imports now read `from apeGmsh.viewers.ui.theme import ...` but
the entry point `pygmsh-viewer` and the package `pyGmshViewer` are unchanged.

---

## 2. Model methods → sub-composites

`Model` is now a composition of five focused sub-composites. Every geometry
method moved behind a categorical namespace.

### Geometry primitives (`g.model.geometry.*`)

```diff
-p = g.model.add_point(0, 0, 0)
+p = g.model.geometry.add_point(0, 0, 0)

-l = g.model.add_line(p1, p2)
+l = g.model.geometry.add_line(p1, p2)

-box = g.model.add_box(0, 0, 0, 1, 1, 1)
+box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

-cyl = g.model.add_cylinder(0, 0, 0, 0, 0, 10, 0.5)
+cyl = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 10, 0.5)
```

All 19 methods move: `add_point`, `add_line`, `add_arc`, `add_circle`,
`add_ellipse`, `add_spline`, `add_bspline`, `add_bezier`, `add_wire`,
`add_curve_loop`, `add_plane_surface`, `add_surface_filling`,
`add_rectangle`, `add_box`, `add_sphere`, `add_cylinder`, `add_cone`,
`add_torus`, `add_wedge`.

### Boolean operations (`g.model.boolean.*`)

```diff
-result = g.model.fuse(a, b)
+result = g.model.boolean.fuse(a, b)

-part = g.model.cut(box, hole)
+part = g.model.boolean.cut(box, hole)

-common = g.model.intersect(a, b)
+common = g.model.boolean.intersect(a, b)

-pieces = g.model.fragment(objects=[1], tools=[2])
+pieces = g.model.boolean.fragment(objects=[1], tools=[2])
```

### Transforms & sweeps (`g.model.transforms.*`)

```diff
-g.model.translate(v, 0, 0, 5)
+g.model.transforms.translate(v, 0, 0, 5)

-g.model.rotate(v, angle=np.pi/4, ax=0, ay=0, az=1)
+g.model.transforms.rotate(v, angle=np.pi/4, ax=0, ay=0, az=1)

-g.model.scale(v, 2, 2, 2)
+g.model.transforms.scale(v, 2, 2, 2)

-g.model.mirror(v, 1, 0, 0, 0)
+g.model.transforms.mirror(v, 1, 0, 0, 0)

-copies = g.model.copy(v)
+copies = g.model.transforms.copy(v)

-ext = g.model.extrude(surf, 0, 0, 10)
+ext = g.model.transforms.extrude(surf, 0, 0, 10)

-rev = g.model.revolve(surf, np.pi, ax=0, ay=1, az=0)
+rev = g.model.transforms.revolve(surf, np.pi, ax=0, ay=1, az=0)

-swept = g.model.sweep(profile, path)
+swept = g.model.transforms.sweep(profile, path)

-loft = g.model.thru_sections([w1, w2])
+loft = g.model.transforms.thru_sections([w1, w2])
```

**Fluent chaining still works** on sub-composites:

```python
(g.model.transforms
    .translate(v, 5, 0, 0)
    .rotate(v, angle=np.pi/2, ax=0, ay=0, az=1))
```

### Import / export (`g.model.io.*`)

```diff
-g.model.load_step("bracket.step")
+g.model.io.load_step("bracket.step")

-g.model.save_step("output.step")
+g.model.io.save_step("output.step")

-g.model.load_iges("file.iges")
+g.model.io.load_iges("file.iges")

-g.model.load_dxf("plan.dxf")
+g.model.io.load_dxf("plan.dxf")

-g.model.save_msh("mesh.msh")
+g.model.io.save_msh("mesh.msh")

-g.model.heal_shapes(tolerance=1e-6)
+g.model.io.heal_shapes(tolerance=1e-6)
```

### Queries & cleanup (`g.model.queries.*`)

```diff
-bb = g.model.bounding_box(tag)
+bb = g.model.queries.bounding_box(tag)

-cm = g.model.center_of_mass(tag)
+cm = g.model.queries.center_of_mass(tag)

-vol = g.model.mass(tag)
+vol = g.model.queries.mass(tag)

-bdry = g.model.boundary(tags)
+bdry = g.model.queries.boundary(tags)

-up, down = g.model.adjacencies(tag)
+up, down = g.model.queries.adjacencies(tag)

-ents = g.model.entities_in_bounding_box(0, 0, 0, 1, 1, 1)
+ents = g.model.queries.entities_in_bounding_box(0, 0, 0, 1, 1, 1)

-g.model.remove([(3, 1)])
+g.model.queries.remove([(3, 1)])

-g.model.remove_duplicates()
+g.model.queries.remove_duplicates()

-g.model.make_conformal()
+g.model.queries.make_conformal()

-df = g.model.registry()
+df = g.model.queries.registry()
```

### What stays on `Model` directly

These methods remain flat on `g.model`:

- `g.model.sync()`
- `g.model.viewer(fem=...)`
- `g.model.gui()`
- `g.model.launch_picker()`
- `g.model.selection` (the entity selection sub-composite)

And the existing `g.parts`, `g.physical`, `g.constraints`, `g.loads`,
`g.mesh_selection` composites are unchanged in location.

---

## 3. Mesh methods → sub-composites

`Mesh` is now a thin composition container.  Every action method moved
behind one of seven focused sub-composites.  The old flat methods
(`g.mesh.generate(3)`, `g.mesh.set_global_size(0.5)`, ...) are **gone**
— there are no shortcuts, and no backwards-compatible aliases.

The mapping is mechanical — every old method went to exactly one
sub-composite.  Pick the right namespace, prefix the method, done.

### Generation & algorithm (`g.mesh.generation.*`)

```diff
-g.mesh.generate(3)
+g.mesh.generation.generate(3)

-g.mesh.set_order(2)
+g.mesh.generation.set_order(2)

-g.mesh.refine()
+g.mesh.generation.refine()

-g.mesh.optimize("Netgen", niter=3)
+g.mesh.generation.optimize("Netgen", niter=3)

-g.mesh.set_algorithm(surf_tag, "frontal_delaunay_quads")
+g.mesh.generation.set_algorithm(surf_tag, "frontal_delaunay_quads")

-g.mesh.set_algorithm_by_physical("Flanges", "quads")
+g.mesh.generation.set_algorithm_by_physical("Flanges", "quads")
```

### Size control (`g.mesh.sizing.*`)

```diff
-g.mesh.set_global_size(6000)
+g.mesh.sizing.set_global_size(6000)

-g.mesh.set_size_global(min_size=15, max_size=25)
+g.mesh.sizing.set_size_global(min_size=15, max_size=25)

-g.mesh.set_size_sources(from_points=False)
+g.mesh.sizing.set_size_sources(from_points=False)

-g.mesh.set_size([p1, p2], 0.05)
+g.mesh.sizing.set_size([p1, p2], 0.05)

-g.mesh.set_size_all_points(6000)
+g.mesh.sizing.set_size_all_points(6000)

-g.mesh.set_size_callback(my_size_fn)
+g.mesh.sizing.set_size_callback(my_size_fn)

-g.mesh.set_size_by_physical("Corners", 0.05)
+g.mesh.sizing.set_size_by_physical("Corners", 0.05)
```

`g.mesh.field` (the `FieldHelper`) is **unchanged** — it was already a
sub-composite and lives alongside `g.mesh.sizing`.

### Structured meshing (`g.mesh.structured.*`)

```diff
-g.mesh.set_transfinite_curve(c, 20, coef=1.3)
+g.mesh.structured.set_transfinite_curve(c, 20, coef=1.3)

-g.mesh.set_transfinite_surface(s, arrangement="Left")
+g.mesh.structured.set_transfinite_surface(s, arrangement="Left")

-g.mesh.set_transfinite_volume(v, corners=[...])
+g.mesh.structured.set_transfinite_volume(v, corners=[...])

-g.mesh.set_transfinite_automatic()
+g.mesh.structured.set_transfinite_automatic()

-g.mesh.set_transfinite_by_physical("Web", dim=1, n_nodes=40)
+g.mesh.structured.set_transfinite_by_physical("Web", dim=1, n_nodes=40)

-g.mesh.set_recombine(s)
+g.mesh.structured.set_recombine(s)

-g.mesh.recombine()
+g.mesh.structured.recombine()

-g.mesh.set_recombine_by_physical("Flanges")
+g.mesh.structured.set_recombine_by_physical("Flanges")

-g.mesh.set_smoothing(s, 5)
+g.mesh.structured.set_smoothing(s, 5)

-g.mesh.set_smoothing_by_physical("Web", 5)
+g.mesh.structured.set_smoothing_by_physical("Web", 5)

-g.mesh.set_compound(2, [s1, s2, s3])
+g.mesh.structured.set_compound(2, [s1, s2, s3])

-g.mesh.remove_constraints()
+g.mesh.structured.remove_constraints()
```

### Editing (`g.mesh.editing.*`)

Topology editing, periodicity, STL → discrete pipeline, and embed.

```diff
-g.mesh.embed(crack_surf, body_tag, dim=2, in_dim=3)
+g.mesh.editing.embed(crack_surf, body_tag, dim=2, in_dim=3)

-g.mesh.set_periodic([2], [1], transform)
+g.mesh.editing.set_periodic([2], [1], transform)

-g.mesh.clear()
+g.mesh.editing.clear()

-g.mesh.reverse()
+g.mesh.editing.reverse()

-g.mesh.relocate_nodes()
+g.mesh.editing.relocate_nodes()

-g.mesh.remove_duplicate_nodes()
+g.mesh.editing.remove_duplicate_nodes()

-g.mesh.remove_duplicate_elements()
+g.mesh.editing.remove_duplicate_elements()

-g.mesh.affine_transform(matrix)
+g.mesh.editing.affine_transform(matrix)

-g.mesh.import_stl()
+g.mesh.editing.import_stl()

-g.mesh.classify_surfaces(math.radians(30))
+g.mesh.editing.classify_surfaces(math.radians(30))

-g.mesh.create_geometry()
+g.mesh.editing.create_geometry()
```

### Queries (`g.mesh.queries.*`)

Every read-only extractor.  `get_fem_data` lives here — this is the
single most-touched rewrite in most projects.

```diff
-nodes = g.mesh.get_nodes(dim=2)
+nodes = g.mesh.queries.get_nodes(dim=2)

-elems = g.mesh.get_elements(dim=3)
+elems = g.mesh.queries.get_elements(dim=3)

-props = g.mesh.get_element_properties(4)
+props = g.mesh.queries.get_element_properties(4)

-fem = g.mesh.get_fem_data(dim=3)
+fem = g.mesh.queries.get_fem_data(dim=3)

-q = g.mesh.get_element_qualities(tags, "minSICN")
+q = g.mesh.queries.get_element_qualities(tags, "minSICN")

-df = g.mesh.quality_report()
+df = g.mesh.queries.quality_report()
```

### Partitioning & renumbering (`g.mesh.partitioning.*`)

MPI-style partitioning and node/element renumbering now live in the
same namespace — they are the "reorganise DOFs" surface.

```diff
-g.mesh.partition(4)
+g.mesh.partitioning.partition(4)

-g.mesh.unpartition()
+g.mesh.partitioning.unpartition()

-old, new = g.mesh.compute_renumbering("RCMK")
+old, new = g.mesh.partitioning.compute_renumbering("RCMK")

-g.mesh.renumber_nodes(old, new)
+g.mesh.partitioning.renumber_nodes(old, new)

-g.mesh.renumber_elements(old, new)
+g.mesh.partitioning.renumber_elements(old, new)

-g.mesh.renumber_mesh(method="rcm", base=1)
+g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
```

### What stays on `Mesh` directly

Only two entry points remain flat on `g.mesh` — both open interactive
windows and neither makes sense inside a sub-composite:

- `g.mesh.viewer(**kwargs)`        — open the interactive mesh viewer
- `g.mesh.results_viewer(...)`     — open the results viewer

Plus the `FieldHelper` sub-composite which was already in place:

- `g.mesh.field.*`                 — unchanged from v0.x

### Chaining inside a composite

Every non-query method on a sub-composite returns `self` (the
sub-composite) so chaining works the same way it used to — just inside
one composite at a time:

```python
(g.mesh.sizing
    .set_size_sources(from_points=False)
    .set_global_size(6000))

(g.mesh.generation
    .set_algorithm(0, "hxt", dim=3)
    .generate(3)
    .set_order(2))
```

To chain across composites, break the chain — the cross-composite
`g.mesh.sizing.set_global_size(...).generate(...)` form does **not**
work, because `generate` is on `generation`, not `sizing`.

---

## 4. OpenSees methods → sub-composites

`OpenSees` is now a thin composition container.  Every declaration
method moved behind one of five focused sub-composites, and several
long-winded names were shortened in the process.

Two methods stay flat on `g.opensees` — the lifecycle verbs that
every workflow calls exactly once or twice:

- `g.opensees.set_model(ndm, ndf)` — unchanged
- `g.opensees.build()`            — unchanged

### Materials (`g.opensees.materials.*`)

Three material-like registries live here.  Names kept as-is — the
composite prefix already provides the disambiguation you used to get
from the `add_*_material` suffix.

```diff
-g.opensees.add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2)
+g.opensees.materials.add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2)

-g.opensees.add_uni_material("Steel", "Steel01", Fy=250e6, E=200e9, b=0.01)
+g.opensees.materials.add_uni_material("Steel", "Steel01", Fy=250e6, E=200e9, b=0.01)

-g.opensees.add_section("Slab", "ElasticMembranePlateSection",
-                        E=30e9, nu=0.2, h=0.2, rho=2400)
+g.opensees.materials.add_section("Slab", "ElasticMembranePlateSection",
+                                  E=30e9, nu=0.2, h=0.2, rho=2400)
```

### Elements (`g.opensees.elements.*`)

Geometric transformations, element assignments, and single-point
constraints — everything that attaches to a physical group during
model declaration.

**Renamed**: `assign_element` → `assign` (the composite name already
says "elements", so `elements.assign_element` would stutter).

```diff
-g.opensees.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
+g.opensees.elements.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])

-g.opensees.assign_element("Body", "FourNodeTetrahedron", material="Concrete")
+g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="Concrete")

-g.opensees.assign_element("Cols", "elasticBeamColumn",
-                           geom_transf="Cols", A=0.04, E=200e9, ...)
+g.opensees.elements.assign("Cols", "elasticBeamColumn",
+                           geom_transf="Cols", A=0.04, E=200e9, ...)

-g.opensees.fix("Base", dofs=[1, 1, 1])
+g.opensees.elements.fix("Base", dofs=[1, 1, 1])
```

### Ingest from FEMData (`g.opensees.ingest.*`)

The two `consume_*_from_fem` methods are renamed — the composite
name "ingest" already says what they do, and `fem` is the argument,
so the method names drop everything but the noun.

```diff
-g.opensees.consume_loads_from_fem(fem)
+g.opensees.ingest.loads(fem)

-g.opensees.consume_masses_from_fem(fem)
+g.opensees.ingest.masses(fem)
```

### Inspect (`g.opensees.inspect.*`)

Post-build tables and the human-readable summary.  Names unchanged.

```diff
-df = g.opensees.node_table()
+df = g.opensees.inspect.node_table()

-df = g.opensees.element_table()
+df = g.opensees.inspect.element_table()

-print(g.opensees.summary())
+print(g.opensees.inspect.summary())
```

### Export (`g.opensees.export.*`)

Script emission.  **Renamed**: `export_tcl` → `tcl`, `export_py` →
`py` (same reasoning as `elements.assign`).

```diff
-g.opensees.export_tcl("model.tcl")
+g.opensees.export.tcl("model.tcl")

-g.opensees.export_py("model.py")
+g.opensees.export.py("model.py")
```

Both return the composite so they still chain:

```python
g.opensees.export.tcl("model.tcl").py("model.py")
```

### Full before/after

```diff
 # Declaration
-g.opensees.set_model(ndm=3, ndf=6)
-g.opensees.add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2)
-g.opensees.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
-g.opensees.assign_element("Body", "FourNodeTetrahedron", material="Concrete")
-g.opensees.assign_element("Cols", "elasticBeamColumn", geom_transf="Cols",
-                           A=0.04, E=200e9, G=77e9, Jx=1e-4, Iy=2e-4, Iz=2e-4)
-g.opensees.fix("Base", dofs=[1, 1, 1, 1, 1, 1])
+g.opensees.set_model(ndm=3, ndf=6)
+g.opensees.materials.add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2)
+g.opensees.elements.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
+g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="Concrete")
+g.opensees.elements.assign("Cols", "elasticBeamColumn", geom_transf="Cols",
+                           A=0.04, E=200e9, G=77e9, Jx=1e-4, Iy=2e-4, Iz=2e-4)
+g.opensees.elements.fix("Base", dofs=[1, 1, 1, 1, 1, 1])

 # Build + inspect + export
 fem = g.mesh.queries.get_fem_data(dim=3)
-g.opensees.consume_loads_from_fem(fem)
-g.opensees.consume_masses_from_fem(fem)
+g.opensees.ingest.loads(fem).masses(fem)
 g.opensees.build()
-print(g.opensees.summary())
-g.opensees.export_tcl("model.tcl").export_py("model.py")
+print(g.opensees.inspect.summary())
+g.opensees.export.tcl("model.tcl").py("model.py")
```

### Chaining inside a composite

Like Mesh, every non-query method on an OpenSees sub-composite
returns `self` (the sub-composite) so chaining works inside one
namespace at a time:

```python
(g.opensees.materials
    .add_nd_material("Concrete", "ElasticIsotropic", E=30e9, nu=0.2)
    .add_nd_material("Soil", "DruckerPrager", K=80e6, G=60e6, sigmaY=20e3))

(g.opensees.elements
    .add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
    .assign("Cols", "elasticBeamColumn", geom_transf="Cols", ...)
    .fix("Base", dofs=[1, 1, 1, 1, 1, 1]))
```

To chain across composites, break the chain — the cross-composite
form does not work.

---

## 5. Rename: `g.mass` → `g.masses`

Every other composite is plural. The last outlier is fixed.

```diff
-g.mass.volume("concrete", density=2400)
+g.masses.volume("concrete", density=2400)

-g.mass.point("tip", mass=500)
+g.masses.point("tip", mass=500)

-g.mass.line("beam", linear_density=80)
+g.masses.line("beam", linear_density=80)

-g.mass.surface("slab", areal_density=300)
+g.masses.surface("slab", areal_density=300)
```

The FEMData field is also renamed:

```diff
 fem = g.mesh.queries.get_fem_data(dim=3)
-print(fem.mass.total_mass())
+print(fem.masses.total_mass())
```

The class names (`MassesComposite`, `MassDef`, `MassRecord`, `MassSet`)
are unchanged — only the session attribute and FEMData field renamed.

**Not renamed** (intentionally): `r.mass` on `MassRecord` is the value field,
`d.mass` on `PointMassDef`, `g.model.queries.mass(tag)` (geometric mass),
and `ops.mass(...)` (OpenSees solver command).

---

## 6. Removed legacy aliases and deprecated methods

These had lived as shims or `DeprecationWarning` for one or more releases.
v1.0 deletes them entirely.

### Session lifecycle

```diff
-g.initialize()   # old alias
+g.begin()

-g.finalize()     # old alias
+g.end()

-g._initialized   # old property
+g.is_active

-g.model_name     # old property alias for g.name
+g.name
```

### Fast/slow viewer split

```diff
-g.model.viewer_fast()   # always fast now
+g.model.viewer()

-g.mesh.viewer_fast()
+g.mesh.viewer()
```

### Parts / PhysicalGroups separation

```diff
-# Old: auto 1:1 mapping from parts to physical groups
-g.parts.add_physical_groups()

+# New: be explicit about which entities join which group
+inst = g.parts.get("i_beam")
+g.physical.add_volume(inst.entities[3], name="steel")
```

Parts track assembly identity ("which geometry belongs to which part").
Physical groups tag regions for solver purposes ("which entities get
this material / BC"). A part can have many groups; a group can span
parts. The auto-mapping collapsed this richness.

### OpenSees bridge

```diff
-g.opensees.add_nodal_load(
-    "Wind", "WindwardFace", force=[1e4, 0, 0]
-)

+with g.loads.pattern("Wind"):
+    g.loads.point("WindwardFace", force_xyz=(1e4, 0, 0))

 # Then `fem.loads` is auto-populated by get_fem_data(),
 # and the OpenSees bridge consumes it via consume_loads_from_fem()
 # (auto-called from build_from_fem()).
```

### Mesh selection parameter alias

```diff
 g.mesh_selection.add_nodes(
-    nearest_to=(1.0, 0.5, 0.0),     # deprecated alias
+    closest_to=(1.0, 0.5, 0.0),
     count=3,
 )
```

### Convenience delegates on the session

```diff
-g.remove_duplicates()        # apeGmsh convenience delegate
+g.model.queries.remove_duplicates()

-g.make_conformal()
+g.model.queries.make_conformal()
```

---

## 7. Automated migration

For a mechanical find-replace across your own codebase, this Python
script handles every transform in this guide — including the Model,
Mesh, and OpenSees sub-composite splits:

```python
#!/usr/bin/env python3
"""Migrate pyGmsh v0.x → apeGmsh v1.0 across a project."""
import re, os, sys

# --- Model methods → sub-composites ---
MODEL_GEOMETRY = {
    'add_point', 'add_line', 'add_arc', 'add_circle', 'add_ellipse',
    'add_spline', 'add_bspline', 'add_bezier', 'add_wire',
    'add_curve_loop', 'add_plane_surface', 'add_surface_filling',
    'add_rectangle', 'add_box', 'add_sphere', 'add_cylinder',
    'add_cone', 'add_torus', 'add_wedge',
}
MODEL_BOOLEAN = {'fuse', 'cut', 'intersect', 'fragment'}
MODEL_TRANSFORMS = {
    'translate', 'rotate', 'scale', 'mirror', 'copy',
    'extrude', 'revolve', 'sweep', 'thru_sections',
}
MODEL_IO = {
    'load_iges', 'load_step', 'load_dxf', 'load_msh',
    'save_iges', 'save_step', 'save_dxf', 'save_msh', 'heal_shapes',
}
MODEL_QUERIES = {
    'remove', 'remove_duplicates', 'make_conformal', 'fragment_all',
    'bounding_box', 'center_of_mass', 'mass', 'boundary',
    'adjacencies', 'entities_in_bounding_box', 'registry',
}

# --- Mesh methods → sub-composites ---
MESH_MAPPING = {
    # generation
    'generate': 'generation', 'set_order': 'generation',
    'refine': 'generation', 'optimize': 'generation',
    'set_algorithm': 'generation',
    'set_algorithm_by_physical': 'generation',
    # sizing
    'set_global_size': 'sizing', 'set_size_sources': 'sizing',
    'set_size_global': 'sizing', 'set_size': 'sizing',
    'set_size_all_points': 'sizing', 'set_size_callback': 'sizing',
    'set_size_by_physical': 'sizing',
    # structured
    'set_transfinite_curve': 'structured',
    'set_transfinite_surface': 'structured',
    'set_transfinite_volume': 'structured',
    'set_transfinite_automatic': 'structured',
    'set_transfinite_by_physical': 'structured',
    'set_recombine': 'structured',
    'set_recombine_by_physical': 'structured',
    'recombine': 'structured', 'set_smoothing': 'structured',
    'set_smoothing_by_physical': 'structured',
    'set_compound': 'structured',
    'remove_constraints': 'structured',
    # editing
    'embed': 'editing', 'set_periodic': 'editing',
    'import_stl': 'editing', 'classify_surfaces': 'editing',
    'create_geometry': 'editing', 'clear': 'editing',
    'reverse': 'editing', 'relocate_nodes': 'editing',
    'remove_duplicate_nodes': 'editing',
    'remove_duplicate_elements': 'editing',
    'affine_transform': 'editing',
    # queries
    'get_nodes': 'queries', 'get_elements': 'queries',
    'get_element_properties': 'queries',
    'get_fem_data': 'queries',
    'get_element_qualities': 'queries',
    'quality_report': 'queries',
    # partitioning
    'partition': 'partitioning', 'unpartition': 'partitioning',
    'compute_renumbering': 'partitioning',
    'renumber_nodes': 'partitioning',
    'renumber_elements': 'partitioning',
    'renumber_mesh': 'partitioning',
}

# --- OpenSees methods → sub-composites ---
# Value format: (composite, new_method_name_or_None_to_keep)
OPENSEES_MAPPING = {
    # materials
    'add_nd_material':          ('materials', None),
    'add_uni_material':         ('materials', None),
    'add_section':              ('materials', None),
    # elements (assign_element renamed to assign)
    'add_geom_transf':          ('elements',  None),
    'assign_element':           ('elements',  'assign'),
    'fix':                      ('elements',  None),
    # ingest (consume_*_from_fem renamed)
    'consume_loads_from_fem':   ('ingest',    'loads'),
    'consume_masses_from_fem':  ('ingest',    'masses'),
    # inspect
    'node_table':               ('inspect',   None),
    'element_table':            ('inspect',   None),
    'summary':                  ('inspect',   None),
    # export (export_tcl → tcl, export_py → py)
    'export_tcl':               ('export',    'tcl'),
    'export_py':                ('export',    'py'),
}

def model_sub_for(m):
    if m in MODEL_GEOMETRY: return 'geometry'
    if m in MODEL_BOOLEAN: return 'boolean'
    if m in MODEL_TRANSFORMS: return 'transforms'
    if m in MODEL_IO: return 'io'
    if m in MODEL_QUERIES: return 'queries'
    return None

pat_model = re.compile(r'(\.model\.)([a-zA-Z_][a-zA-Z_0-9]*)')
# NB: (?<!model\.) skips `gmsh.model.mesh.<method>(`
pat_mesh = re.compile(
    r'(?<!model\.)(?<!_)(\bmesh)\.(' +
    '|'.join(sorted(MESH_MAPPING, key=len, reverse=True)) +
    r')(?=\()'
)
pat_opensees = re.compile(
    r'(\bopensees)\.(' +
    '|'.join(sorted(OPENSEES_MAPPING, key=len, reverse=True)) +
    r')(?=\()'
)

def transform(src):
    def model_repl(m):
        method = m.group(2)
        sub = model_sub_for(method)
        return f'.model.{sub}.{method}' if sub else m.group(0)

    def mesh_repl(m):
        return f'mesh.{MESH_MAPPING[m.group(2)]}.{m.group(2)}'

    def opensees_repl(m):
        old = m.group(2)
        comp, new_name = OPENSEES_MAPPING[old]
        name = new_name if new_name else old
        return f'opensees.{comp}.{name}'

    out = pat_model.sub(model_repl, src)
    out = pat_mesh.sub(mesh_repl, out)
    out = pat_opensees.sub(opensees_repl, out)
    out = out.replace('from pyGmsh', 'from apeGmsh')
    out = out.replace('import pyGmsh', 'import apeGmsh')
    out = re.sub(r'\bpyGmsh\(', 'apeGmsh(', out)
    out = re.sub(r'\bg\.mass\.', 'g.masses.', out)
    out = re.sub(r'\bfem\.mass\b', 'fem.masses', out)
    out = re.sub(r'\bg\.initialize\(\)', 'g.begin()', out)
    out = re.sub(r'\bg\.finalize\(\)', 'g.end()', out)
    out = re.sub(r'\bg\.model_name\b', 'g.name', out)
    return out

SKIP = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}

for dirpath, dirnames, filenames in os.walk(sys.argv[1] if len(sys.argv) > 1 else '.'):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for fn in filenames:
        if not (fn.endswith('.py') or fn.endswith('.ipynb') or fn.endswith('.md')):
            continue
        path = os.path.join(dirpath, fn)
        with open(path, 'r', encoding='utf-8') as fh:
            src = fh.read()
        new = transform(src)
        if new != src:
            with open(path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(new)
            print(f'  updated: {path}')
```

Save as `migrate_v1.py` and run: `python migrate_v1.py /path/to/your/project`

**Caveats:**
- Backup first. Run it on a clean git branch.
- The `(?<!model\.)` lookbehind on `pat_mesh` intentionally skips
  raw `gmsh.model.mesh.<method>(` calls — those are the real Gmsh API
  and must stay untouched.
- The regex `\bg\.mass\.` requires a dot after `mass` — it will catch
  `g.mass.point(...)` but not a bare `g.mass` reference. Grep manually
  for any leftover bare references.
- `gmsh.initialize()` / `gmsh.finalize()` at the raw Gmsh level are
  preserved — the script only rewrites `g.initialize` (our wrapper).
- If your project has custom subclasses of `Model` or `Mesh` that
  referenced the old mixin classes directly, those need manual updates
  (the mixins are gone — replaced with `_Geometry`, `_Generation`,
  etc. that take a parent reference in `__init__`).

---

## 8. What didn't change

- The session lifecycle contract (`g.begin()` / `g.end()` / context manager)
- `g.parts`, `g.physical`, `g.constraints`, `g.loads`, `g.mesh_selection`
  composite APIs
- `fem.node_ids`, `fem.node_coords`, `fem.element_ids`, `fem.connectivity`,
  `fem.physical`, `fem.constraints`, `fem.loads` — all unchanged
- `Part` and `PartsRegistry` APIs (only docstring examples were swept)
- `g.mesh.field.*` — the FieldHelper sub-composite was already in place
- `g.mesh.viewer()` and `g.mesh.results_viewer()` — the two interactive
  entry points stay flat on `g.mesh`
- `g.opensees.set_model()` and `g.opensees.build()` — the two
  OpenSees lifecycle verbs stay flat

---

## 9. Full example — old vs new

### Old (v0.x pyGmsh)

```python
from pyGmsh import pyGmsh

g = pyGmsh(model_name="cantilever")
g.initialize()

v = g.model.add_box(0, 0, 0, 1, 0.2, 5)
g.model.translate(v, 0.5, 0, 0)
bb = g.model.bounding_box(v)

g.physical.add_volume([v], name="concrete")

with g.loads.pattern("dead"):
    g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
g.mass.volume("concrete", density=2400)

g.mesh.set_global_size(0.5)
g.mesh.generate(3)
fem = g.mesh.get_fem_data(3)

print(f"total mass: {fem.mass.total_mass():.0f} kg")
g.finalize()
```

### New (v1.0 apeGmsh)

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="cantilever") as g:
    v = g.model.geometry.add_box(0, 0, 0, 1, 0.2, 5)
    g.model.transforms.translate(v, 0.5, 0, 0)
    bb = g.model.queries.bounding_box(v)

    g.physical.add_volume([v], name="concrete")

    with g.loads.pattern("dead"):
        g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
    g.masses.volume("concrete", density=2400)

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

    print(f"total mass: {fem.masses.total_mass():.0f} kg")
```

Compared to v0.x:

- `with apeGmsh(...) as g:` replaces `g.initialize()` / `g.finalize()`.
- Model methods now live under `geometry`, `transforms`, `queries`.
- Mesh methods now live under `sizing`, `generation`, `partitioning`,
  `queries`.
- `g.mass` is now `g.masses`; `fem.mass` is now `fem.masses`.
