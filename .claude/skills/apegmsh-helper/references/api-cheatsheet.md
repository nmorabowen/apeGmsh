# apeGmsh API cheatsheet

One-page map of the public apeGmsh surface. Every entry is a concrete
composite attribute on a live session `g = apeGmsh(...)` (after
`g.begin()` or inside a `with` block). Signatures reflect **v2.0.0**
(`pyproject.toml` + the latest tagged `CHANGELOG.md` section agree; a
stale editable install may still print `v1.6.0` in the banner). When in
doubt, grep `src/apeGmsh/`.

Idiomatic apeGmsh is **verbose-by-name**: target geometry by label /
physical-group name / part label, never by raw tags or `[1]`-style edge
lists. The `.select()` fluent chain (below) is how you turn coordinates
into labels.

## Session object

```python
from apeGmsh import apeGmsh, Part

g = apeGmsh(model_name="...", verbose=False,
            save_to=None, overwrite=True)   # save_to= autosaves neutral zone on end()
g.begin()           # opens gmsh, wires composites
g.is_active         # True while session is open
g.name              # the model name
g.save(path=None)   # explicit neutral-zone checkpoint (uses save_to if path None)
g.end()             # closes gmsh (+ autosaves if save_to set)

# Preferred form
with apeGmsh(model_name="...", save_to="m.h5") as g:
    ...             # begin/end run automatically; m.h5 written on exit
```

Rebuild a chain-phase session from disk (NO gmsh state — only
`compose*`/`save` work; `g.model.*`/`g.mesh.*` raise):

```python
g = apeGmsh.from_h5("host.h5", model_name=None, verbose=False)
g.compose("module_a.h5", label="A")        # see compose.md
```

Optional extras (pip): `matplotlib` (plots), `openseespy` (analysis),
`[viewer]` extra = `trame`+`ipywidgets` (web viewers), `pyvista` +
`PySide6` + `vtk` + `qtpy` (Qt viewers), `ezdxf` (DXF).

## Top-level composites on `g`

| Attribute          | Class                  | Purpose |
|--------------------|------------------------|---------|
| `g.inspect`        | `Inspect`              | Session-level diagnostics |
| `g.model`          | `Model`                | OCC geometry (5 sub-composites + `.select()`) |
| `g.labels`         | `Labels`               | Tier-1 naming (geometry-time) |
| `g.sections`       | `SectionsBuilder`      | Section primitives (profiles, shells, solids) |
| `g.parts`          | `PartsRegistry`        | Multi-part assembly bookkeeping |
| `g.constraints`    | `ConstraintsComposite` | Solver-agnostic MP-constraint defs |
| `g.loads`          | `LoadsComposite`       | Load patterns & defs |
| `g.masses`         | `MassesComposite`      | Mass defs (NOTE: `g.masses`, not `g.mass`) |
| `g.decouple_node`  | `DecoupledNodesComposite` | Element-less analysis nodes (spring ground / master); ndf via `ops.ndf` (ADR 0048/0049) |
| `g.mesh`           | `Mesh`                 | Meshing (7 sub-composites) |
| `g.loader`         | `MshLoader`            | Load `.msh` files |
| `g.physical`       | `PhysicalGroups`       | Solver-facing named groups |
| `g.mesh_selection` | `MeshSelectionSet`     | Post-mesh node/element selection sets |
| `g.view`           | `View`                 | Gmsh post-processing scalar/vector views |
| `g.plot`           | `Plot`                 | Matplotlib visualisations (optional extra) |

Composite list verified at `src/apeGmsh/_core.py:40-82`. OpenSees is
**not** a session composite — `g.opensees` was removed. The OpenSees
entry point is the post-session bridge `apeSees(fem)`
(`from apeGmsh.opensees import apeSees`); see `opensees-bridge.md` and
the bridge section below.

Sub-composite-only forms (no top-level shortcuts):
`g.model.geometry.add_point(...)`, `g.model.boolean.fuse(...)`,
`g.mesh.generation.generate(...)` — never `g.add_point` / `g.fuse`.

---

## `g.model` — geometry (OCC kernel)

`g.model.sync()` — flush OCC (rarely needed; most methods auto-sync).
`g.model.viewer()` opens the Qt viewer; `g.model.gui()` opens the
native Gmsh FLTK window; `g.model.launch_picker()` opens FLTK with
labels on. Sub-composites: `geometry`, `boolean`, `transforms`, `io`,
`queries` (`src/apeGmsh/core/Model.py:78-82`), plus `g.model.select()`.

### `g.model.geometry` — primitives (`_Geometry`)

Each `add_*` accepts `label=` for auto-PG (especially inside a `Part`):

```
add_point(x, y, z, *, lc=0.0, label=None)      add_line(p1, p2, *, label=None)
add_arc(p1, p_center, p2, *, label=None)        add_circle(xc, yc, zc, r, *, label=None)
add_ellipse(xc, yc, zc, r1, r2, *, label=None)  add_spline / add_bspline / add_bezier(points, *, label=None)
add_wire(curves, *, label=None)                 add_curve_loop(curves, *, label=None)
add_plane_surface(loop, *, holes=None, label=None)   add_rectangle(x, y, z, dx, dy, *, label=None)
add_box(x, y, z, dx, dy, dz, *, label=None)     add_sphere / add_cylinder / add_cone / add_torus / add_wedge(...)
```

Cutting / slicing (auto-sweep dangling orphans internally — see sweep below):

```
slice(target=None, *, axis: 'x'|'y'|'z', offset=0.0, point=None, dim=1|2|3|'all', classify=False, label=None, tolerance=None)
#   target renamed from solid (now any-dim); offset XOR point; dim='all' = every maximal entity (volumes / shells / curves)
cut_by_surface(solid, surface, *, keep_surface=True, remove_original=True, label=None, tolerance=None)
cut_by_plane(...)   add_axis_cutting_plane(...)   sweep / replace_line(...)
```
`# src/apeGmsh/core/_model_geometry.py:2094 (slice), :1701 (cut_by_surface)`

**Orphan-geometry sweep** (PR #378). Slice/cut/fragment leave dim≤2
fragments bounding no volume; these methods inspect/clean them. An
entity in ANY physical group, apeGmsh label, or `_metadata` is
PROTECTED (3-channel `_user_intentional`). Distinct from the unrelated
`get_fem_data(remove_orphans=...)` kwarg (which drops mesh NODES):

```
g.model.geometry.find_orphans() -> dict[int, list[int]]      # {0:[],1:[],2:[]}; non-mutating
g.model.geometry.remove_orphans(*, dry_run=False) -> dict    # dry_run=True == find_orphans()
g.model.geometry.find_stale_metadata() -> list[(dim,tag)]    # CLOSED-world (only _metadata keys)
g.model.geometry.validate_pre_mesh(*, strict=False) -> None  # raises GeometryValidationError
```
`# src/apeGmsh/core/_model_geometry.py:2209/2228/2254/2285`
`# verified: tests/test_geometry_topology.py::TestFindOrphansDryRun::test_sweep_dangling_dry_run_does_not_modify`

`validate_pre_mesh()` default `strict=False` is **closed-world**
(stale-metadata only) and is **auto-invoked by `g.mesh.generation.generate()`**.
`strict=True` is **open-world** (`find_orphans`) — opt-in only; never
auto-wired (it broke 63 raw-gmsh tests). Slivers in raw `gmsh.model.*`
geometry are legitimate, so open-world checks would false-positive.
Typed warnings/errors:
`from apeGmsh.core._geometry_errors import GeometryValidationError, WarnGeomCoincidentFace, WarnGeomOneSidedCut`
(both warnings subclass `UserWarning`; silence with `warnings.simplefilter('ignore', ...)`).
`# verified: tests/test_geometry_topology.py::TestValidatePreMesh::test_validate_pre_mesh_default_passes_on_clean_model`
`# verified: tests/test_geometry_topology.py::TestFindStaleMetadata::test_clean_model_has_no_stale_metadata`

### `g.model.boolean` — boolean ops (`_Boolean`)

```
fuse(objects, tools=None, *, remove_objects=True, remove_tools=True)
cut(objects, tools, *, remove_objects=True, remove_tools=True)
intersect(objects, tools, *, remove_objects=True, remove_tools=True)
fragment(objects, tools, *, dim=3, remove_object=True, remove_tool=True, cleanup_free=True, sync=True, tolerance=None)
conformal(*, dims=None, tolerance=None, sync=True)
```
`fragment`'s `cleanup_free=True` runs the topology sweep when volumes
exist (skipped in 2D-only models). `# src/apeGmsh/core/_model_boolean.py:323`
`conformal` is a convenience alias delegating to
`g.model.queries.make_conformal` — fragments *all* entities against each
other (whole-model weld) rather than one object/tool pair.

### `g.model.transforms` — rigid-body & generative ops (`_Transforms`)

```
translate(dimtags, dx, dy, dz)        rotate(dimtags, x, y, z, ax, ay, az, angle)
scale(dimtags, x, y, z, sx, sy=None, sz=None)   mirror(dimtags, a, b, c, d)   copy(dimtags)
extrude(dimtags, dx, dy, dz, *, num_elems=None, recombine=False)
revolve(dimtags, x, y, z, ax, ay, az, angle, *, num_elems=None)
sweep(dimtags, curves, *, num_elems=None)        thru_sections(wires, *, make_solid=True)
```

### `g.model.io` — CAD I/O + health diagnostics (`_IO`)

```
diagnose(*, warn=False) -> ImportHealth          # NON-mutating health scan; never heals
load_step(path, *, highest_dim_only=True, heal=False, dedupe=False, fuse=False, label=None, sync=True)
load_iges(path, *, highest_dim_only=True, heal=False, dedupe=False, fuse=False, label=None, sync=True)
load_dxf(path, *, point_tolerance=1e-6, create_physical_groups=True, sync=True)
heal_shapes(tags=None, *, dim=3, tolerance=1e-8, fix_degenerated=True, fix_small_edges=True,
            fix_small_faces=True, sew_faces=True, make_solids=True, sync=True) -> _IO
save_step(path, dimtags=None)        save_msh(path)
```
`# src/apeGmsh/core/_model_io.py:701 (diagnose), :556 (load_step), :627 (heal_shapes)`

`ImportHealth` (frozen dataclass): `.n_solids`, `.is_suspect` (True iff
short_edges or tiny_faces — slivers; a surface-only import does NOT
trip), `.suggested_tolerance`, `.advisory()`.

**`heal=True` / `heal='auto'` on import is scale-aware** (~1e-6·bbox
diagonal) — a behaviour change: the old absolute 1e-8 was a no-op on
mm/m models, so `heal=True` now actually heals (and renumbers).
`heal_shapes(...)` called *directly* still defaults to the legacy 1e-8.
Raw imports auto-fire `WarnGeomImportHealth` (a `UserWarning`,
`from apeGmsh.core._geometry_errors import WarnGeomImportHealth`) when
slivers appear.
`# verified: tests/test_import_health.py::test_diagnose_clean_box`
`# verified: tests/test_import_health.py::test_load_step_heal_auto_uses_scale_aware_tolerance`

### `g.model.queries` — introspection (`_Queries`)

```
bounding_box(dim=-1, tag=-1)         center_of_mass(dim, tag)        mass(dim, tag)
boundary(dimtags, *, oriented=True, recursive=False)
adjacencies(dim, tag)                entities_in_bounding_box(xmin, ..., zmax, *, dim=-1)
registry                              # dict[(dim, tag)] -> kind
remove(dimtags, *, recursive=False)  remove_duplicates()   make_conformal()   fragment_all()
```

### `g.model.select(...)` — fluent CAD-entity selection

The single entity-selection surface (the former
`g.model.selection.select_*` composite was removed). Resolves
`target` label → PG → part, then chains spatial verbs; terminals turn
the selection into a label/PG without raw tags:

```python
(g.model.select("block", dim=3)        # or select(None, dim=2) for "all surfaces"
   .in_box(xmin, ymin, zmin, xmax, ymax, zmax)   # .in_sphere/.on_plane/.crossing_plane/.nearest_to/.where
   .to_label("loaded_faces"))           # or .to_physical("BC") / .to_dataframe() / .result().tags()
```
`# src/apeGmsh/core/Model.py:152` — verbs compose with `|` `&` `-` `^`.

---

## `g.mesh` — meshing

`g.mesh.viewer(**kw)` and `g.mesh.results_viewer(...)` are flat entry
points. Everything else lives in sub-composites: `generation`,
`sizing`, `field`, `structured`, `editing`, `queries`, `partitioning`,
`recipe`.

### `g.mesh.generation` — (`_Generation`)

```
generate(dim=3)        set_order(order)   # 1 or 2
refine()               optimize(method="Netgen", dim=-1)
set_algorithm(algo)    set_algorithm_by_physical(pg_name, algo, *, dim=None)
```
`Algorithm2D` / `Algorithm3D`: `from apeGmsh import Algorithm2D, Algorithm3D`.

### `g.mesh.sizing` — (`_Sizing`)

```
set_global_size(size)           set_size_global(min=None, max=None)
set_size(dim, tags, size)       set_size_all_points(size)
set_size_callback(func)         set_size_by_physical(pg_name, size, *, dim=None)
set_size_sources(*, from_points=True, from_curvature=False, ...)
```

### `g.mesh.field` — (`FieldHelper`) — fluent size fields

```python
f_d = g.mesh.field.distance(edges=[l1, l2, l3])
f_t = g.mesh.field.threshold(input_field=f_d, size_min=0.1, size_max=2.0, dist_min=1.0, dist_max=10.0)
g.mesh.field.set_background(f_t)
# Also: box(...) / math_eval("0.1 + 0.01 * F1") / boundary_layer(...) / minimum([f1, f2])
```

### `g.mesh.structured` — (`_Structured`)

```
set_transfinite_curve(tag, num_nodes, *, mesh_type="Progression", coef=1.0)
set_transfinite_surface(tag, *, arrangement="Left", corners=None)
set_transfinite_volume(tag, *, corners=None)
set_transfinite_automatic(dimtags=None, corner_angle=2.35, recombine=False)
set_recombine(dim, tag, *, angle=45)   recombine()   set_smoothing(dim, tag, num_steps)   set_compound(dim, tags)
```

### `g.mesh.editing` — (`_Editing`)

```
embed(source_dimtags, into_dim, into_tag)        set_periodic(dim, dst_tags, src_tags, affine)
reverse(dim, tags)   relocate_nodes()   remove_duplicate_nodes()   remove_duplicate_elements()
affine_transform(matrix4x4, dim=-1, tag=-1)      import_stl(...)   classify_surfaces(...)   create_geometry()
crack(...)           clear()

split_higher_order_lines(physical_group, *, policy: 'forbid'|'split'|'constrain', dim=1) -> _Editing
```
`# src/apeGmsh/mesh/_mesh_editing.py:593`

`split_higher_order_lines` demotes 2nd-order Line3 → 2× Line2 in place
on named PG(s) so the beam-column bridge (which hard-rejects 3-node
beams) accepts a 2nd-order continuum mesh. `policy` is **required**
(no default — destructive). Call **AFTER `generate()`, BEFORE
`get_fem_data()`/partition**, never in a stage block. `policy='split'`
mutates live gmsh and invalidates any prior FEMData snapshot.
`policy='forbid'` raises `RuntimeError` if any Line3 present;
`policy='constrain'` raises `NotImplementedError` (reserved).
`# verified: tests/test_mesh_editing_split_higher_order_lines.py::TestSplit::test_split_replaces_line3_with_line2_pair`
`# verified: tests/test_mesh_editing_split_higher_order_lines.py::TestValidation::test_invalid_policy_raises_value_error`

### `g.mesh.queries` — (`_Queries`)

```
get_nodes(dim=-1, tag=-1, *, includeBoundary=True) -> dict
get_elements(dim=-1, tag=-1) -> dict     get_element_properties(element_type) -> dict
get_fem_data(dim=None, *, remove_orphans=False) -> FEMData    # the broker snapshot — see fem-broker.md
get_element_qualities(element_tags, quality_name="minSICN") -> ndarray
quality_report(element_tags=None, *, quality_name="minSICN", ...) -> DataFrame
```
`# src/apeGmsh/mesh/_mesh_queries.py:122`. `get_fem_data` is the
**two-stage pipeline rendezvous**: `g.loads`/`g.masses`/`g.constraints`
are *declared* pre-mesh, then *resolved* here into the returned
`FEMData`. Call after `generate()`. (Per-node ndf is not declared here —
the bridge infers it; see the ndf row below.)

### `g.mesh.partitioning` — (`_Partitioning`)

```
partition(num_parts, *, algorithm="metis")    unpartition()
renumber(dim=2, *, method="rcm", base=1) -> RenumberResult     # method: simple|rcm|hilbert|metis
```
After `renumber(...)` Gmsh node tags ARE dense solver-ready ints from
`base`. (Renumber is for dense 1-based tags, NOT bandwidth — OpenSees'
numberer handles bandwidth.)

### `g.mesh.recipe` — one-call meshing (`_Recipe`, ADR 0059)

Whole-model or targeted meshing in one call — sets up size fields + (for
structured) transfinite + recombine, then generates. Each returns the recipe
(chainable). Targeted calls **compose** (size field, no generate); whole-model
calls **generate** immediately (`generate=None` default keys off scope).

```
unstructured(target=None, *, max_size=None, min_size=0.0, dim=None, generate=None)
structured(target=None, *, size=None, n=None, recombine=True,
           fallback="unstructured"|"warn"|"strict", dim=None, generate=None)
check() -> _Recipe       # run the mixed quad/tri interface guard without generating
```
`# src/apeGmsh/mesh/_mesh_recipe.py:79 (unstructured), :167 (structured), :275 (check)`

```python
g.mesh.recipe.unstructured(max_size=0.3)        # whole model → generates
g.mesh.recipe.unstructured("region", max_size=0.12)   # targeted → composes
g.mesh.recipe.unstructured()                    # zero-arg heuristic: max_size = bbox_diag/20
g.mesh.recipe.structured(n=4)                   # whole model, hex, recombined
g.mesh.recipe.structured("soil", n=8, recombine=False)   # structured tris/tets
```

Targeted region sizing uses **Constant + Min fields** (not per-point
characteristic lengths, which bleed across shared corners). The mixed quad/tri
interface guard (`MeshRecipeError` — recombined-structured volume meeting an
unstructured neighbour, no gmsh pyramid transition) fires **only on the recipe
path**, never in a raw `g.mesh.generation.generate()`. `fallback="strict"`
raises on non-decomposable transfinite entities; `"warn"` skips them;
`"unstructured"` (default) gives skipped entities a region size field so a mesh
always results. `from apeGmsh.mesh._mesh_recipe import MeshRecipeError`.

---

## `g.physical` — solver-facing named groups

```
add(dim, tags, *, name="", tag=-1)
add_point / add_curve / add_surface / add_volume(tags, *, name="", tag=-1)
from_label(name, *, dim=None, tag=-1)        from_labels(*names, dim=None)      # promote labels -> PG
entities(name, *, dim=None) -> list[Tag]     get_tag(dim, name)   get_name(dim, tag)
get_all(dim=-1) -> list[DimTag]              get_entities(dim, tag) -> list[Tag]
get_groups_for_entity(dim, tag)              get_nodes(dim, tag)
set_name / remove_name / remove / remove_all   summary() -> pd.DataFrame
```

## `g.labels` — Tier-1 naming (geometry-time)

Labels are stored prefixed `_label:` so they never clash with
user-authored PGs.

```
add(dim, tags, name) -> int                  entities(name, *, dim=None) -> list[Tag]
get_all(*, dim=-1) -> list[str]              has(name, *, dim=None) -> bool
labels_for_entity(dim, tag) -> list[str]     reverse_map(*, dim=-1) -> dict[DimTag, str]
remove(name, *, dim=None)   rename(old, new, *, dim=None)
promote_to_physical(name, *, dim=None, ...)  # label -> solver-visible PG
```

## Per-node DOF count — inferred + `ops.ndf` (ADR 0048/0049)

Per-node `ndf` is **inferred** from the declared element classes on the bridge
(the old `g.node_ndf` session composite was removed). Declare elements + `ndm`;
each node's `ndf` resolves from its incident elements and emits a `-ndf K`
token (elided when it equals the `ops.model` envelope). For an **element-less**
decoupled node only, state it explicitly:

```
ops.ndf(target, *, ndf: int)   # target = g.decouple_node handle OR int tag
                               # fails loud on a mesh / element-touched node
```
`# src/apeGmsh/opensees/apesees.py (apeSees.ndf); build.py (resolve_ndf_overlay, G1–G3)`

## `g.sections` — section-geometry builder (`SectionsBuilder`)

Builds section geometry directly in the session, returning an
`Instance`. Profile/solid/shell verbs:

```
W_solid / W_shell(bf, tf, h, tw, length, *, anchor="start", align="z", label=..., lc=..., translate=..., rotate=...)
rect_solid / rect_hollow / pipe_solid / pipe_hollow / angle_solid / channel_solid / tee_solid(...)
```
`# src/apeGmsh/sections/_builder.py:250 (W_solid), :739 (W_shell), :362+ (others)`

## `g.parts` — multi-part assembly

`Part("name")` is a lightweight standalone session
(`begin()` / `save("x.step")` / `end()`). `g.parts` imports those STEPs
back and tracks label↔entity membership through fragmentation.

```
instances() -> dict[str, Instance]    part(label) -> Part    get(label) -> Instance   labels() -> list[str]
register(label, dimtags) -> Instance  from_model(label, dimtags_or_model, ...) -> Instance
add(part, *, label=None, translate=None, rotate=None, ...)
import_step(path, *, label=None, translate=(0,0,0), rotate=None, highest_dim_only=True,
            heal=False, dedupe=False, properties=None) -> Instance
build_node_map(*, conformal=True) -> dict[str, set[int]]    build_face_map(...) -> dict
rename(old, new)   delete(label)   fragment_all()   fuse_group(label, ...)
```
`import_step` gained scale-aware `heal=`/`dedupe=` (same semantics as
`g.model.io.load_step`). `# src/apeGmsh/core/_parts_registry.py:793`

## `g.constraints` — solver-agnostic MP constraints

Every method returns a `ConstraintDef`; resolution happens in
`get_fem_data(...)`. The first two args are **label names**; their kwarg
names are `master_label`/`slave_label` (and `host_label`/`embedded_label`
for `embedded`) — **except** `node_to_surface*`, which take bare
`master`/`slave`. These now EMIT (see note below), so declare them once —
don't hand-write the deck.

```
equal_dof(master_label, slave_label, *, master_entities=None, slave_entities=None, dofs=None)
rigid_link(master_label, slave_label, *, link_type="beam"|"bar"|"rotBeam")
rigid_diaphragm(master_label, slave_label, *, perp_dirn=3)   penalty(master_label, slave_label, *, stiffness=1e10, dofs=None)
rigid_body(master_label, slave_label, *, dofs=None)
tie(master_label, slave_label, *, master_entities=None, slave_entities=None, tolerance=1.0)
embedded(host_label, embedded_label, *, tolerance=1.0)       tied_contact / mortar(master_label, slave_label, ...)
node_to_surface / node_to_surface_spring(master, slave, *, ...)   # phantom nodes — bare master/slave

# Fork coupling elements (RBE2 / RBE3) — shared control knobs:
kinematic_coupling(master_label, slave_label, *, master_point=(0,0,0), dofs=None,   # RBE2 → LadrunoKinematicCoupling (tag 33012)
    k=None|"auto", k_alpha=None, host=None, kr=None,
    enforce="penalty"|"al", bipenalty_dtcr=None, bipenalty_wcap=None, absolute=False, name=None)
distributing_coupling(master_label, slave_label, *, master_point=(0,0,0),  # RBE3 → LadrunoDistributingCoupling (tag 33011)
    weighting="uniform"|"area", k=None|"auto", k_alpha=None, host=None, kr=None,
    enforce="penalty"|"al", bipenalty_dtcr=None, bipenalty_wcap=None, absolute=False, name=None)
list_defs() / list_records() / clear()
```
`# src/apeGmsh/core/ConstraintsComposite.py:771 (kinematic_coupling), :971 (distributing_coupling)`

**RBE2 vs RBE3.** `kinematic_coupling` (RBE2) rigidly drives the slave set from
the reference node with correct moment-arm transport (`u_i = u_R + θ_R × d_i`).
`distributing_coupling` (RBE3) makes the reference a weighted-average rigid-body
fit of the independents and **adds no stiffness** (set stays flexible).
`weighting="area"` is tributary-area over the slave surface (requires meshed
faces). Coupling knobs (commit `818b91eb`): `k="auto"` scales the penalty off
the host element diagonal (`K_t = k_alpha·max|K_host|`, needs `host=`); `host=`
is the **FEM element id** (the bridge translates to the ops tag at emit);
`bipenalty_wcap` sets bipenalty via host frequency (needs `host=`, exclusive
with `bipenalty_dtcr`). Both elements are **Ladruno-fork-only** — stock OpenSees
fails loud at the element line. Under partitioned/MPI emit, RBE2 routes to a
single canonical rank owning all slaves (raises if no rank owns the full set);
RBE3 can distribute across ranks.

## `g.loads` — load patterns & definitions

```python
with g.loads.case("dead"):            # context manager groups defs
    g.loads.gravity(...)
    g.loads.line(...)
```
```
# dimension-indexed (ADR 0050). point/surface are verb namespaces; line/volume are callables.
point.force(target, force, *, pg=None, label=None, tag=None, name=None)
point.moment(target, moment, *, ...)
point.force_closest(xyz, force, *, within=None, tol=None, ...)   point.moment_closest(xyz, moment, ...)
line(target, *, magnitude=None, direction=(0,0,-1), q_xyz=None, reduction=, target_form=, ...)
surface.pressure(target, magnitude, *, reduction=, target_form=, ...)   # scalar × face normal
surface.traction(target, vector, *, reduction=, target_form=, ...)      # free global vector / area
surface.shear(target, vector, *, reduction=, name=None)                 # strict in-plane (tangent); nodal-only
surface.force_resultant_center_mass(target, *, force=None, moment=None, magnitude=0.0, normal=False, direction=None, name=None)
volume(target, *, force_per_volume=(0,0,0), reduction=, target_form=, ...)
gravity(target, *, g=(0,0,-9.81), density=None, reduction=, target_form=, ...)
by_case(name) -> list[LoadDef]          cases() -> list[str]
```
Target resolution: `label → physical group → part label`.
`pg=`/`label=`/`tag=` short-circuit to a single source.

## `g.displacements` — prescribed motion (resolved into `fem.nodes.sp`)

Force-free sibling of `g.loads` (ADR 0050). Ownership: `g.constraints.bc`
= permanent homogeneous fixes; `g.displacements` = nonzero / pattern-bound
motion (a zero here is an allowed pattern-bound hold).

```
surface(target, *, dofs=None, disp_xyz=None, rot_xyz=None, magnitude=0.0, normal=False, direction=None, name=None)  # was g.loads.face_sp
point(target, *, dofs=None, values=None, name=None)   # prescribed value applied verbatim at each node
case(name)   by_case(name)   cases()
```

## `g.masses` — mass definitions (resolved into `fem.nodes.masses`)

```
point(target, *, mass, pg=None, label=None, tag=None, name=None)
line(target, *, mass_per_length, density, ...)
surface(target, *, density, thickness=None, ...)     volume(target, *, density, ...)
```

## `g.mesh_selection` — post-mesh selection sets

Named node/element subsets built **after** meshing (recorders,
post-processing, "these specific nodes"). The fluent `select(...).save_as(name)`
is the idiomatic path (same point-family chain as `g.model.select` /
`fem.nodes.select`); the tag-based verbs below are the lower-level surface.

```
select(*, level="node"|"element", dim=2, ids=None, name=None) -> MeshSelection   # chain → .save_as(name)
#   .in_box(lo,hi,*,inclusive=False) / .in_sphere(c,r) / .on_plane(point,normal,*,tol) / .nearest_to(p,*,count) / .where(fn) / | & - ^
add(dim, tags, *, name="", tag=-1) -> int                 # explicit ids: dim=0 nodes, 1/2/3 elements
from_physical(dim, name_or_tag, *, ms_name="", ms_tag=-1) -> int
union/intersection/difference(dim, tag_a, tag_b, *, name="", tag=-1) -> int
filter_set(dim, tag, *, name="", on_plane=None, in_box=None, in_sphere=None, closest_to=None, predicate=None, inclusive=False) -> int
sort_set(dim, tag, *, name="", key=...) -> int
node_ids(name) -> ndarray   element_ids(name) -> ndarray   names(dim=-1) -> list[str]
get_nodes(dim, tag) -> dict   get_elements(dim, tag) -> dict   summary() -> pd.DataFrame
```
`# src/apeGmsh/mesh/MeshSelectionSet.py:511 (select), :147 (add), :337 (from_physical)`

---

## `apeSees(fem)` — OpenSees bridge (see `opensees-bridge.md`)

Post-session. `from apeGmsh.opensees import apeSees`. Typed primitives
return **handles** passed by reference (no string types). **Not
fluent** — separate statements.

```python
ops = apeSees(fem, *, default_orientation=None, opensees=None)   # opensees: OpenSeesTarget | None
ops.model(*, ndm, ndf)

# typed namespaces — return handles:
m  = ops.nDMaterial.<Type>(**typed_kwargs)
m  = ops.uniaxialMaterial.<Type>(**typed_kwargs)    # e.g. Steel02: fy= not Fy=
s  = ops.section.<Type>(**typed_kwargs)
t  = ops.geomTransf.Linear|PDelta|Corotational(*, vecxz=None, orientation=None)
bi = ops.beamIntegration.Lobatto(*, section=s, n_ip=5)

# elements — pg= selection; body force / pressure are element params:
ops.element.<Type>(*, pg, material=m | section=s | transf=t, integration=bi, ...) -> ElementGroup

# supports / mass — RE-DECLARED on the bridge. MP constraints auto-emit;
# loads are OPT-IN via p.from_model(case) (ADR 0051 — NO g.loads auto-emit):
ops.fix(*, pg=None, nodes=None, dofs)               ops.mass(*, pg=None, nodes=None, values)
ts = ops.timeSeries.Linear|Constant|Path|Trig|Pulse|Ricker(...)
ts = ops.timeSeries.MomentStep|Yoffe(...)           # ADR 0062 normalized moment function S(t)
with ops.pattern.Plain(series=ts) as p:             # or UniformExcitation
    p.from_model("dead")                            # import a g.loads.case into the deck
    p.load(*, pg=None, node=None, forces)           # + ad-hoc bridge-authored load
    p.sp(*, pg=None, node=None, dof, value)
    p.moment_tensor(*, position, frame, M0=1.0,     # earthquake point source (ADR 0062 — RHS nodal forces)
        mech=dict(strike,dip,rake) | m_ij=<3×3>, t0=0.0, method="consistent"|"dipole", region=None)
ops.fault.from_shakermaker(fault, *, frame, M0, rise_time, peak_time, dt, t_total,
    length_scale, method="consistent", f_max=None, region=None) -> list[Plain]   # finite fault (MT-4)
ops.recorder.<Type>(...)                            ops.region(...)

# damping — domain-level (ADR 0053); resolve at emit, no held tag:
ops.damping.rayleigh(*, alpha_m=,beta_k=,... | ratio=,f_i=,f_j=,stiffness="initial"; on=None)
ops.damping.modal(ratios, *, modes)                 # bundles eigen; no modal_q
ops.damping.uniform|sec_stif|urd|urd_beta(*, ..., on=None, activate_time=, factor=)
#   on= attaches via region -damp; OR omit on= and pass the handle to a
#   -damp-capable element's damp= kwarg. ops.damping.* also on s.damping.* (staged).
# Loads reach the deck ONLY via p.from_model(case) or p.load — nothing
# auto-emits, so no 2x double-count trap. The deck is authoritative: the
# bridge applies exactly what you import and does NOT audit the geometry's
# case-list (no WarnUnconsumedModelLoads). A case you don't import is not
# applied; an import of a non-existent case is a no-op.
# NO mixing: a global ops.pattern.* + ops.stage(...) -> BridgeError.

# staged analysis (ADR 0034) — domainChange between stages:
with ops.stage("excavate") as s:                    # src/apeGmsh/opensees/apesees.py
    s.activate(...); s.fix(...); s.mass(...); s.region(...); s.recorder(...)
    s.damping.rayleigh(...); s.damping.uniform(..., on=)   # stage-bound (D5); no s.damping.modal
    with s.pattern(series=ts) as p: p.from_model("live")   # stage-scoped pattern (ADR 0051 BL-3)
    s.embedded(...); s.initial_stress(...); s.remove_sp(...); s.remove_bc(...); s.remove_element(...)
    s.set_time(...); s.set_creep(...); s.reset(...)
```

Flat emit / run verbs (each builds internally):

```
ops.build() -> BuiltModel
ops.tcl(path, *, run=False, bin=None, analyze_steps=None, analyze_dt=None, split=False, per_rank=False)
ops.py(path,  *, run=False, analyze_steps=None, analyze_dt=None, split=False)
ops.h5(path,  *, model_name=None, cuts=(), sweeps=())     # writes BOTH neutral + /opensees zones
ops.run(*, wipe=True)                                     # in-process LiveOpsEmitter; no analyze
ops.analyze(*, steps, dt=None) -> int
# per_rank=True (ADR 0061): partitioned model only → driver + ranks/rank<K>_<seq>.tcl
# fragments (each rank parses only its own); mutually exclusive with split=True.

# Remote SLURM (ADR 0060): emit → push (ssh) → sbatch → [wait → fetch], one call.
ops.run_remote(job_dir, *, cluster, np=None, name=None, deck="main.tcl", binary=None,
    walltime=None, analyze_steps=None, analyze_dt=None, wait=True, poll=15.0,
    timeout=None, overwrite=False) -> Job        # raises HPCError if status != COMPLETED
```

Which OpenSees runs — `OpenSeesTarget` (see `opensees-bridge.md` / `ladruno.md`):

```python
from apeGmsh.opensees import apeSees, OpenSeesTarget, OpenSeesCapabilities
ops = apeSees(fem, opensees=OpenSeesTarget(binary=None, python=None, require_fork=False))
ops.opensees                 # -> the bound OpenSeesTarget | None
ops.capabilities()           # -> OpenSeesCapabilities(source, has_fork, has_profiler, version) (LIVE probe)
ops.tcl(path, run=True, bin="...OpenSees.exe")     # bin= > target.binary > $OPENSEES_BIN > which
ops.py(path,  run=True, python="...python.exe")    # python= > target.python > $OPENSEES_VENV > which > sys.executable
# binary/python INERT for live run/analyze/eigen (can't swap import openseespy);
# require_fork=True asserts the LIVE build is the fork, failing loud at run().
# A target is NOT a fork switch — fork features stay gated at point of use; branch on ops.capabilities().
```
`# src/apeGmsh/opensees/apesees.py:4463 (tcl), :4524 (py), :4592 (h5), :4578 (run)`

**MP constraints now EMIT** (v2.0.0, ADR 0022 — *reversal of the old
"deferred" claim*). `equalDOF` / `rigidLink` / `rigidDiaphragm` /
`ASDEmbeddedNodeElement` auto-emit from `fem.nodes.constraints` /
`fem.elements.constraints` into the runnable Tcl/Py deck; the bridge
auto-adds `ops.constraints.Transformation()` when MP constraints are
present. **Do not hand-emit constraints the bridge now writes** (double
constraints / wrong stiffness).
`# verified: tests/test_mesh_editing_split_higher_order_lines.py::TestBridgeIntegration::test_bridge_accepts_split_frame_pg`

## Standalone modules — NOT session composites

Separate top-level imports, used after the session / bridge (not attributes
on `g`):

```python
# apeGmsh.hpc — remote SLURM submission (ADR 0060). Pairs with ops.run_remote.
from apeGmsh.hpc import Cluster, Job, JobStatus, ClusterConfig, HPCError
Cluster.load(name, path=None)                        # reads ~/.apegmsh/clusters.toml
cluster.submit(job_dir, *, np=None, name=None, deck="main.tcl", binary=None, walltime=None, overwrite=False) -> Job
cluster.ping() -> bool                               # cluster.render_batch_script(...) -> str
job.status() -> JobStatus  ; job.wait(*, poll=15.0, timeout=None) -> JobStatus
job.tail(n=50, *, stream="out"|"err") -> str  ; job.cancel()  ; job.fetch(dest=None) -> Path
Job.load(local_dir) -> Job                           # rehydrate from .apegmsh_job.json sidecar

# apeGmsh.sensitivity — finite-difference gradient / calibration driver.
from apeGmsh.sensitivity import Sensitivity, Param, Response
Sensitivity(forward, params, *, rel_step=1e-2, scheme="central"|"forward")   # engine-free scalar forward
Sensitivity.from_apesees(fem, *, build, params, response, steps, dt,         # live transient + capture
    runner=None, capture_path=None, rel_step=1e-2, scheme="central")
sens.gradient(at=None, *, rel_step=None, scheme=None) -> dict[str, float]
sens.step_study(param=None, *, at=None, rel_steps=None) -> list[(rel_step, grad)]   # plateau check
sens.solve(target, *, tol=1e-6, max_iter=50, damping=1.0) -> dict[str, float]      # 1-parameter only
Param(name=, value=, lower=None, upper=None)
Response(component=, pg=None, label=None, node=None, reduce="peak"|"rms"|"mean_abs"|"last"|"at_time", at_time=None, absolute=True)
```
`# src/apeGmsh/hpc/_cluster.py, _job.py ; src/apeGmsh/sensitivity/driver.py, spec.py`

## FEMData & persistence (see `fem-broker.md`, `results.md`)

```
fem = g.mesh.queries.get_fem_data(dim=3)
fem.info / fem.nodes / fem.elements / fem.inspect / fem.mesh_selection

FEMData.from_gmsh(dim=3, session=g, ndf=6, remove_orphans=False)
FEMData.from_msh("bridge.msh", dim=2)
FEMData.from_h5(path, *, root="/")                       # rebuilds + integrity-checks snapshot_id
fem.to_h5(path, *, model_name="", apegmsh_version="", ndf=0)   # neutral zone ONLY (no /opensees)
```
`# src/apeGmsh/mesh/FEMData.py:1548 (from_h5), :1603 (to_h5)`
`# verified: tests/test_femdata_from_h5.py::test_round_trip_nodes_and_elements`
`# verified: tests/test_femdata_to_h5.py::test_to_h5_writes_meta`

`fem.to_h5` / `g.save()` write **only** the neutral zone;
`apeSees(fem).h5(path)` writes neutral + opensees zones. `from_h5`
re-verifies `snapshot_id` and raises `MalformedH5Error` on mismatch.

## Quick reference: element type codes

For `fem.elements.select(element_type=...)`. Accepts alias (`"tet4"`),
Gmsh code (`4`), or Gmsh name (`"Tetrahedron 4"`).

| Code | Gmsh name      | Alias   | OpenSees typical mapping |
|------|----------------|---------|--------------------------|
| 1    | Line 2         | `line2` | `truss`, `elasticBeamColumn` |
| 2    | Triangle 3     | `tri3`  | `tri31` |
| 3    | Quad 4         | `quad4` | `quad`, `SSPquad`, `ShellMITC4`, `ShellDKGQ`, `ASDShellQ4` |
| 4    | Tetrahedron 4  | `tet4`  | `FourNodeTetrahedron` |
| 5    | Hexahedron 8   | `hex8`  | `stdBrick`, `SSPbrick`, `bbarBrick` |
| 6    | Prism 6        | —       | (not directly mapped) |
| 8    | Line 3 (quad)  | —       | demote via `split_higher_order_lines` |
| 9    | Triangle 6     | `tri6`  | `tri6n` (SixNodeTri), `BezierTri6` (Ladruno fork) |
| 10   | Quad 9         | `quad9` | — |
| 11   | Tetrahedron 10 | `tet10` | `TenNodeTetrahedron`, `BezierTet10` (Ladruno fork) |

Confirm availability with `fem.info.types` or
`fem.elements.type_table()` before filtering.
