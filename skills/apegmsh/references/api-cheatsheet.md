# apeGmsh API cheatsheet

One-page map of the public apeGmsh surface. Every entry below is a
concrete composite attribute on a live session `g = apeGmsh(...)`
(after `g.begin()` or inside a `with` block). Signatures reflect
v1.0; if in doubt, grep the source.

## Session object

```python
from apeGmsh import apeGmsh, Part

g = apeGmsh(model_name="...", verbose=False)
g.begin()           # opens gmsh, wires composites
g.is_active         # True while session is open
g.name              # the model name
g.end()             # closes gmsh

# Preferred form
with apeGmsh(model_name="...") as g:
    ...             # begin/end run automatically
```

Optional extras (pip): `matplotlib` (plots), `openseespy` (analysis),
`gmsh2opensees` (legacy `g.g2o`), `pyvista` + `PySide6` + `vtk` + `qtpy`
(viewers), `ezdxf` (DXF).

## Top-level composites on `g`

| Attribute          | Class               | Purpose |
|--------------------|---------------------|---------|
| `g.inspect`        | `Inspect`           | Session-level diagnostics |
| `g.model`          | `Model`             | OCC geometry (5 sub-composites) |
| `g.labels`         | `Labels`            | Tier-1 naming (geometry-time) |
| `g.sections`       | `SectionsBuilder`   | Section primitives (profiles, shells, solids) |
| `g.parts`          | `PartsRegistry`     | Multi-part assembly bookkeeping |
| `g.constraints`    | `ConstraintsComposite` | Solver-agnostic constraint defs |
| `g.loads`          | `LoadsComposite`    | Load patterns & defs |
| `g.masses`         | `MassesComposite`   | Mass defs |
| `g.mesh`           | `Mesh`              | Meshing (7 sub-composites) |
| `g.loader`         | `MshLoader`         | Load `.msh` files |
| `g.physical`       | `PhysicalGroups`    | Solver-facing named groups |
| `g.mesh_selection` | `MeshSelectionSet`  | Post-mesh node/element selection sets |
| `g.view`           | `View`              | Gmsh post-processing scalar/vector views |
| `g.g2o`            | `Gmsh2OpenSees`     | Legacy one-liner transfer (keep for back-compat) |
| `g.opensees`       | `OpenSees`          | OpenSees bridge (5 sub-composites) |
| `g.plot`           | `Plot`              | Matplotlib visualisations (optional extra) |

---

## `g.model` — geometry (OCC kernel)

`g.model.sync()` — flush the OCC kernel (rarely needed manually; most
methods auto-sync). `g.model.viewer()` opens the Qt viewer;
`g.model.gui()` opens the native Gmsh FLTK window;
`g.model.launch_picker()` opens FLTK with entity labels on.

### `g.model.geometry` — primitives (`_Geometry`)

Points & lines (each accepts `label=` for auto-PG inside `Part`):

```
add_point(x, y, z, *, lc=0.0, label=None)
add_line(p1, p2, *, label=None)
add_imperfect_line(...)
add_arc(p1, p_center, p2, *, label=None)
add_circle(xc, yc, zc, r, *, label=None)
add_ellipse(xc, yc, zc, r1, r2, *, label=None)
add_spline(points, *, label=None)
add_bspline(points, *, label=None)
add_bezier(points, *, label=None)
add_wire(curves, *, label=None)
add_curve_loop(curves, *, label=None)
add_plane_surface(loop, *, holes=None, label=None)
add_surface_filling(loop, *, label=None)
add_rectangle(x, y, z, dx, dy, *, label=None)
```

Solid primitives:

```
add_box(x, y, z, dx, dy, dz, *, label=None)
add_sphere(xc, yc, zc, r, *, label=None)
add_cylinder(x, y, z, dx, dy, dz, r, *, label=None)
add_cone(x, y, z, dx, dy, dz, r1, r2, *, label=None)
add_torus(xc, yc, zc, r1, r2, *, label=None)
add_wedge(x, y, z, dx, dy, dz, *, ltx=0.0, label=None)
```

Extrusion helpers (`sweep`, `add_cutting_plane`, `add_axis_cutting_plane`,
`cut_by_surface`, `cut_by_plane`, `slice`, `replace_line`) — read the
source when you need them; signatures are stable but the argument
lists are long.

### `g.model.boolean` — boolean ops (`_Boolean`)

```
fuse(objects, tools=None, *, remove_objects=True, remove_tools=True)
cut(objects, tools, *, remove_objects=True, remove_tools=True)
intersect(objects, tools, *, remove_objects=True, remove_tools=True)
fragment(objects, tools=None, *, remove_objects=True, remove_tools=True)
```

### `g.model.transforms` — rigid-body & generative ops (`_Transforms`)

```
translate(dimtags, dx, dy, dz)
rotate(dimtags, x, y, z, ax, ay, az, angle)
scale(dimtags, x, y, z, sx, sy=None, sz=None)
mirror(dimtags, a, b, c, d)        # a x + b y + c z + d = 0
copy(dimtags)
extrude(dimtags, dx, dy, dz, *, num_elems=None, recombine=False)
revolve(dimtags, x, y, z, ax, ay, az, angle, *, num_elems=None)
sweep(dimtags, curves, *, num_elems=None)
thru_sections(wires, *, make_solid=True)
```

### `g.model.io` — CAD I/O (`_IO`)

```
load_step(path)       load_iges(path)      load_dxf(path, *, z_extrude=None)
save_step(path, dimtags=None)
save_msh(path)
heal_shapes(dimtags=None, *, tolerance=1e-3)
```

### `g.model.queries` — introspection (`_Queries`)

```
bounding_box(dim=-1, tag=-1)
center_of_mass(dim, tag)
mass(dim, tag)
boundary(dimtags, *, oriented=True, recursive=False)
adjacencies(dim, tag)
entities_in_bounding_box(xmin, ymin, zmin, xmax, ymax, zmax, *, dim=-1)
registry                     # dict[(dim, tag)] -> kind
remove(dimtags, *, recursive=False)
remove_duplicates()
make_conformal()
fragment_all()               # fragments every entity with every other
```

### `g.model.selection` — spatial entity picker

Passed a `SelectionComposite` — use `g.model.selection.select_points(...)`
(and friends) to query entities by coordinate/bounding box/tolerance.
Read `src/apeGmsh/viz/Selection.py` for the full menu.

---

## `g.mesh` — meshing

`g.mesh.viewer(**kw)` and `g.mesh.results_viewer(...)` are flat entry
points. Everything else lives in sub-composites.

### `g.mesh.generation` — (`_Generation`)

```
generate(dim: int = 3)
set_order(order: int)            # 1 or 2
refine()                          # uniform refine
optimize(method="Netgen", dim=-1)
set_algorithm(algo)              # Algorithm2D/Algorithm3D enum or int
set_algorithm_by_physical(pg_name, algo, *, dim=None)
```

`Algorithm2D` / `Algorithm3D` enums live on `apeGmsh.mesh.Mesh` (or
`from apeGmsh import Algorithm2D, Algorithm3D`).

### `g.mesh.sizing` — (`_Sizing`)

```
set_global_size(size)                    # sets MeshSizeMax (+Min)
set_size_sources(*, from_points=True, from_curvature=False, ...)
set_size_global(min=None, max=None)
set_size(dim, tags, size)
set_size_all_points(size)
set_size_callback(func)
set_size_by_physical(pg_name, size, *, dim=None)
```

### `g.mesh.field` — (`FieldHelper`)

Fluent chain for mesh size fields:

```
f_d = g.mesh.field.distance(edges=[l1, l2, l3])
f_t = g.mesh.field.threshold(input_field=f_d,
                              size_min=0.1, size_max=2.0,
                              dist_min=1.0, dist_max=10.0)
g.mesh.field.set_background(f_t)

# Other fields:
g.mesh.field.box(vin=0.1, vout=2.0, xmin=..., xmax=..., ...)
g.mesh.field.math_eval("0.1 + 0.01 * F1")
g.mesh.field.boundary_layer(edges_list=..., hwall_n=..., ratio=...)
g.mesh.field.minimum([f1, f2, f3])
```

### `g.mesh.structured` — (`_Structured`)

```
set_transfinite_curve(tag, num_nodes, *, mesh_type="Progression", coef=1.0)
set_transfinite_surface(tag, *, arrangement="Left", corners=None)
set_transfinite_volume(tag, *, corners=None)
set_transfinite_automatic(dimtags=None, corner_angle=2.35, recombine=False)
set_recombine(dim, tag, *, angle=45)
recombine()
set_smoothing(dim, tag, num_steps)
set_compound(dim, tags)
```

### `g.mesh.editing` — (`_Editing`)

```
embed(source_dimtags, into_dim, into_tag)
set_periodic(dim, dst_tags, src_tags, affine)
clear()
reverse(dim, tags)
relocate_nodes()
remove_duplicate_nodes()
remove_duplicate_elements()
affine_transform(matrix4x4, dim=-1, tag=-1)
import_stl(path, *, classify=True, curve_angle=180, surf_angle=180)
classify_surfaces(angle_rad, *, boundary=True, for_reparametrization=True)
create_geometry()                # build BRep from discrete mesh
```

### `g.mesh.queries` — (`_Queries`)

```
get_nodes(dim=-1, tag=-1, *, includeBoundary=True) -> dict
get_elements(dim=-1, tag=-1) -> dict
get_element_properties(element_type: int) -> dict
get_fem_data(dim=None, *, remove_orphans=False) -> FEMData
get_element_qualities(element_tags, quality_name="minSICN") -> ndarray
quality_report(element_tags=None, *, quality_name="minSICN", ...) -> DataFrame
```

### `g.mesh.partitioning` — (`_Partitioning`)

```
partition(num_parts, *, algorithm="metis")
unpartition()
renumber(dim=2, *, method="rcm", base=1) -> RenumberResult
    method ∈ {"simple", "rcm", "hilbert", "metis"}
```

After `renumber(...)`, the Gmsh node tags **are** solver-ready contiguous
integers starting from `base`. Prefer this over building your own mapping.

---

## `g.physical` — solver-facing named groups

```
add(dim, tags, *, name="", tag=-1)
add_point(tags, *, name="", tag=-1)
add_curve(tags, *, name="", tag=-1)
add_surface(tags, *, name="", tag=-1)
add_volume(tags, *, name="", tag=-1)
from_label(name, *, dim=None, tag=-1)           # promote one label
from_labels(*names, dim=None)                    # promote many labels
set_name(dim, tag, name)     remove_name(name)
remove(dim_tags)             remove_all()
get_all(dim=-1) -> list[DimTag]
get_entities(dim, tag) -> list[Tag]
entities(name, *, dim=None) -> list[Tag]         # name -> entity tags
get_groups_for_entity(dim, tag) -> list[Tag]
get_name(dim, tag)           get_tag(dim, name)
summary() -> pd.DataFrame
get_nodes(dim, tag)                               # PG nodes
```

## `g.labels` — Tier-1 naming (geometry-time)

Labels are prefixed with `_label:` internally so they don't clash with
user-authored physical groups.

```
add(dim, tags, name)                         -> internal PG tag
entities(name, *, dim=None) -> list[Tag]
get_all(*, dim=-1) -> list[str]
has(name, *, dim=None) -> bool
remove(name, *, dim=None)    rename(old, new, *, dim=None)
promote_to_physical(name, *, dim=None, ...)  # label -> solver-visible PG
reverse_map(*, dim=-1) -> dict[DimTag, str]
labels_for_entity(dim, tag) -> list[str]
set_result(...)              # advanced: attach a post-processing view
```

## `g.parts` — multi-part assembly

`Part("name")` is a lightweight standalone session (`begin()` /
`save("...step")` / `end()`). `g.parts` is the registry that imports
those STEPs back and keeps track of which entities belong to which
label even through fragmentation.

```
parts: dict                              # read-only mapping
instances() -> dict[str, Instance]
part(label) -> Part                      # lookup
register(label, dimtags) -> Instance
from_model(label, dimtags_or_model, ...) -> Instance
add(part: Part, *, label=None, translate=None, rotate=None, ...)
import_step(path, *, label, translate=None, rotate=None, ...)
build_node_map(*, conformal=True)        -> dict[str, set[int]]
build_face_map(...)                       -> dict[str, dict]
get(label) -> Instance
labels() -> list[str]
rename(old, new)      delete(label)
fragment_all()                            # fragments everything conformal
fuse_group(label, ...)                    # merges a group of entities
```

## `g.constraints` — solver-agnostic MP constraints

Every method returns a `ConstraintDef`; resolution happens automatically
when `get_fem_data(...)` is called. Master / slave are label names.

```
equal_dof(master_label, slave_label, *, master_entities=None, slave_entities=None, dofs=None)
rigid_link(master_label, slave_label, *, link_type="beam" | "bar" | "rotBeam")
penalty(master_label, slave_label, *, stiffness=1e10, dofs=None)
rigid_diaphragm(master_label, slave_label, *, perp_dirn=3)
rigid_body(master_label, slave_label, *, dofs=None)
kinematic_coupling(master_label, slave_label, *, dofs=None)
tie(master_label, slave_label, *, master_entities=None, slave_entities=None, tolerance=1e-6)
distributing_coupling(master_label, slave_label, *, ...)
embedded(host_label, embedded_label, *, tolerance=1.0)
node_to_surface(master, slave, *, ...)                 # creates phantom nodes
node_to_surface_spring(master, slave, *, stiffness=..., ...)
tied_contact(master_label, slave_label, *, ...)
mortar(master_label, slave_label, *, ...)

# Housekeeping
list_defs() -> list[dict]
list_records() -> list[dict]
clear()
```

## `g.loads` — load patterns & definitions

```
with g.loads.pattern("dead"):            # context manager groups defs
    g.loads.gravity(...)
    g.loads.line(...)

point(target, *, pg=None, label=None, tag=None,
      force_xyz=None, moment_xyz=None, name=None)
line(target, *, pg=None, label=None, tag=None,
     magnitude=None, direction=(0,0,-1), q_xyz=None, ...)
surface(target, *, pg=None, label=None, tag=None,
        magnitude=0.0, normal=True, direction=(0,0,-1), ...)
gravity(target, *, pg=None, label=None, tag=None,
        g=(0,0,-9.81), density=None, ...)
body(target, *, pg=None, label=None, tag=None,
     force_per_volume=(0,0,0), ...)
face_load(target, *, force_xyz=None, moment_xyz=None, name=None)
face_sp(target, *, dofs=None, disp_xyz=None, rot_xyz=None, name=None)

by_pattern(name) -> list[LoadDef]
patterns() -> list[str]
```

Target resolution: `label → physical group → part label`. `pg=` /
`label=` / `tag=` short-circuit to a single source.

## `g.masses` — mass definitions

```
point(target, *, mass, pg=None, label=None, tag=None, name=None)
line(target, *, mass_per_length, density, ...)
surface(target, *, density, thickness=None, ...)
volume(target, *, density, ...)
```

Resolved into `fem.nodes.masses` when `get_fem_data(...)` is called.

## `g.mesh_selection` — post-mesh selection sets

Named node/element subsets built *after* meshing. Useful for recorders,
post-processing, and any situation where "these specific nodes" matters.

```
add(dim, tag, *, name)
add_nodes(*, name, closest_to=None, in_box=..., on_entities=..., ...)
add_elements(*, name, in_box=..., on_entities=..., ...)
set_name(dim, tag, name)     remove_name(name)
remove(dim_tags)             remove_all()
get_all(dim=-1) -> list[DimTag]
get_entities(dim, tag) -> list[int]
get_name(dim, tag)           get_tag(dim, name)
get_nodes(dim, tag) -> dict   get_elements(dim, tag) -> dict

# Set algebra on existing sets
union(sets, *, name)         intersection(sets, *, name)    difference(sets, *, name)
from_physical(pg_name, *, name, ...)
from_geometric(*, name, ...)
filter_set(source_name, *, name, predicate)
sort_set(source_name, *, name, key)
summary() -> pd.DataFrame
to_dataframe(dim, tag) -> pd.DataFrame
```

## `g.opensees` — OpenSees bridge (see `opensees-bridge.md`)

```
g.opensees.set_model(*, ndm=3, ndf=3)
g.opensees.build()

g.opensees.materials
    .add_nd_material(name, ops_type, **params)
    .add_uni_material(name, ops_type, **params)
    .add_section(name, section_type, **params)

g.opensees.elements
    .add_geom_transf(name, transf_type, *, vecxz=None, **extra)
    .assign(pg_name, ops_type, *, material=None, geom_transf=None, dim=None, **extra)
    .fix(pg_name, *, dofs, dim=None)

g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .sp(fem)
    .constraints(fem)

g.opensees.inspect
    .node_table() -> DataFrame
    .element_table() -> DataFrame
    .summary() -> str

g.opensees.export
    .tcl(path) -> self
    .py(path)  -> self
```

Every declaration method returns `self` — fluent chaining everywhere.

## FEMData (see `fem-broker.md` for details)

```
fem = g.mesh.queries.get_fem_data(dim=3, *, remove_orphans=False)

fem.info       # MeshInfo (n_nodes, n_elems, bandwidth, types)
fem.nodes      # NodeComposite (ids, coords, get(...), sub-composites)
fem.elements   # ElementComposite (iter, get(...), resolve(), type_table())
fem.inspect    # summaries + DataFrames
fem.mesh_selection  # MeshSelectionStore (if sets were defined)

# Factories
FEMData.from_gmsh(dim=3, session=g, ndf=6, remove_orphans=False)
FEMData.from_msh("bridge.msh", dim=2)
```

## Quick reference: element type codes

Needed mostly when filtering in `fem.elements.get(element_type=...)`.
Accepted forms: alias string (`"tet4"`), Gmsh code (`4`), or Gmsh name
(`"Tetrahedron 4"`).

| Code | Gmsh name        | Alias   | OpenSees typical mapping |
|------|------------------|---------|--------------------------|
| 1    | Line 2           | `line2` | `truss`, `elasticBeamColumn` |
| 2    | Triangle 3       | `tri3`  | `tri31` |
| 3    | Quad 4           | `quad4` | `quad`, `SSPquad`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4` |
| 4    | Tetrahedron 4    | `tet4`  | `FourNodeTetrahedron` |
| 5    | Hexahedron 8     | `hex8`  | `stdBrick`, `SSPbrick`, `bbarBrick` |
| 6    | Prism 6          | —       | (not directly mapped) |
| 8    | Line 3 (quad)    | —       | `dispBeamColumn` w/ mid-node |
| 9    | Triangle 6       | `tri6`  | — |
| 10   | Quad 9           | `quad9` | — |
| 11   | Tetrahedron 10   | `tet10` | `TenNodeTetrahedron` |

Confirm element type availability in the mesh with `fem.info.types` or
`fem.elements.type_table()` before filtering.
