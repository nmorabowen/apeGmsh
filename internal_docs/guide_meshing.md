# Meshing guide

Reference-style guide to the meshing layer of apeGmsh. This document is grounded
in the current source tree and mirrors the symbols that actually exist. It
assumes you are already familiar with the Part workflow described in
`guide_parts_assembly.md` and `guide_parts_vs_session.md`.

The meshing layer sits between geometry (`core/Model.py`, `core/Part.py`,
`core/_parts_registry.py`) and the FEM broker (`mesh/FEMData.py`,
`solvers/OpenSees.py`). Its job is to turn an OCC geometry — possibly assembled
from several parts — into a solver-ready mesh and expose enough handles
(physical groups, mesh selection sets, node/face maps) for `Constraints` and
the solver bridges to consume it.

The composites involved are attached to an `apeGmsh` session as:

- `g.model` — OCC geometry wrapper (`core/Model.py`)
- `g.parts` — instance registry (`core/_parts_registry.py`)
- `g.mesh` — thin composition container (`mesh/Mesh.py`) with seven focused
  sub-composites:
  - `g.mesh.generation` — generate, set_order, refine, optimize, set_algorithm
  - `g.mesh.sizing`     — global / per-entity size control
  - `g.mesh.field`      — `FieldHelper` (size fields)
  - `g.mesh.structured` — transfinite, recombine, smoothing, compound
  - `g.mesh.editing`    — clear, embed, periodic, STL, topology editing
  - `g.mesh.queries`    — get_nodes, get_elements, get_fem_data, quality_report
  - `g.mesh.partitioning` — partition / unpartition / renumber_*
- `g.physical` — physical group API (`mesh/PhysicalGroups.py`)
- `g.mesh_selection` — post-mesh selection sets (`mesh/MeshSelectionSet.py`)
- `g.constraints` — constraint definitions / resolver (`core/ConstraintsComposite.py`, `solvers/Constraints.py`)

Every action method on a mesh sub-composite returns `self` (the
sub-composite) so chaining works inside one namespace at a time. To chain
across composites, break the chain — this is a deliberate v1.0 design
choice in favour of a flat, discoverable API.


## 1. Parts and the session

apeGmsh separates two different uses of a Gmsh "session":

1. A **Part** (`core/Part.py`) owns its own isolated Gmsh session and is a pure
   geometry container. It only exists to build a shape and export it to STEP or
   IGES via `Part.save()`. Parts know nothing about meshing, physical groups,
   or constraints.
2. A **apeGmsh session** (the `apeGmsh` object users normally work with) owns the
   *assembly-level* Gmsh session — the one the mesher will actually run on. It
   wraps the same `gmsh.model.occ` API through `g.model`, and it keeps its own
   book-keeping in two places:
   - `g.model._metadata: dict[DimTag, dict]` — entity metadata (`kind`,
     cutting-plane normals, etc.). Labels live in `g.labels` (Gmsh PGs).
   - `g.parts.instances: dict[str, Instance]` — every imported or inlined
     *part instance* is registered here, with its entities grouped by dimension
     (`Instance.entities: dict[int, list[Tag]]`), translation/rotation, and
     bounding box.

Instances are created through five entry points on the `PartsRegistry`
composite:

| Entry point | File / line | Use case |
|---|---|---|
| `g.parts.add(part, *, label=None, translate=..., rotate=..., highest_dim_only=True)` | `_parts_registry.py` | Import a `Part` object (auto STEP round-trip) |
| `g.parts.import_step(file_path, *, label=None, translate=..., rotate=...)` | `_parts_registry.py` | Import a CAD file directly |
| `with g.parts.part(label): ...` | `_parts_registry.py` | Inline geometry block; entities created in the block are auto-tagged |
| `g.parts.register(label, dimtags)` | `_parts_registry.py` | Manually tag existing entities as an instance |
| `g.parts.from_model(label, *, dim=None, tags=None)` | `_parts_registry.py` | Adopt entities already sitting in `gmsh.model` |

All five converge on the same `Instance` dataclass. Once an instance exists,
the mesher and the constraint resolver can address its entities by *label*
instead of by raw Gmsh tag — this is what makes the downstream pipeline
robust to fragmentation, fusion, and re-tagging.

See `guide_parts_vs_session.md` for the rationale behind keeping the Part and
session sessions separate, and `guide_parts_assembly.md` for the full
five-phase workflow.


## 2. Conformal meshing: fuse, cut, intersect, fragment

Boolean operations live in `core/_model_boolean.py` and are exposed as methods
on `g.model`. All four share the same shape:

```python
g.model.boolean.fuse     (objects, tools, *, dim=3, remove_object=True, remove_tool=True, sync=True) -> list[Tag]
g.model.boolean.cut      (objects, tools, *, dim=3, remove_object=True, remove_tool=True, sync=True) -> list[Tag]
g.model.boolean.intersect(objects, tools, *, dim=3, remove_object=True, remove_tool=True, sync=True) -> list[Tag]
g.model.boolean.fragment (objects, tools, *, dim=3, remove_object=True, remove_tool=True, cleanup_free=True, sync=True) -> list[Tag]
```

`objects` and `tools` accept a flexible `TagsLike = Tag | list[Tag] | DimTag |
list[DimTag]`. The internal `_bool_op()` helper normalises them, calls the OCC
operation, then updates `Model._metadata` — consumed entities are unregistered
and surviving ones are re-registered with `kind` equal to the operation name.

Semantically:

- **`fuse`** returns the boolean union. Shared faces between the operands are
  removed and the result is a single body. Use this when two parts should
  become *one* part with a single label, not a pair of parts sharing an
  interface.
- **`cut`** returns `A \ B`. The default removes the tool; pass
  `remove_tool=False` to keep the cutter around.
- **`intersect`** returns `A ∩ B`. Useful for windowing a shape against a box
  or another solid.
- **`fragment`** is the conformal-meshing primitive. It splits every shape at
  every intersection and keeps *all* sub-volumes, including the pieces of the
  tools that are left over. The `cleanup_free=True` default also drops
  surfaces that do not bound a volume after the operation — those are the
  dangling remnants of cutting planes that sit outside the solid and would
  otherwise show up as stray 2D elements in the mesh.

**Conformal meshing in practice** almost always means "fragment, then mesh".
Without fragmentation, two touching parts will be meshed independently and
their interface nodes will not be shared, which breaks any constraint or
integration that crosses the interface. With fragmentation, the interface
becomes a shared surface in the OCC topology and the mesher produces coincident
nodes on both sides automatically.

`PartsRegistry` wraps this pattern in three higher-level operations so you do
not have to track tags by hand:

- **`g.parts.fragment_all(*, dim=None)`** (`_parts_registry.py`) —
  fragments every instance in the registry against every other instance at the
  requested dimension (auto-detected if `None`). After it runs, each
  `Instance.entities` is updated in place with the *new* surviving tags. Use
  this at the end of Phase 3 in the standard assembly workflow.
- **`g.parts.fragment_pair(label_a, label_b, *, dim=None)`** — same, but
  restricted to two instances. Useful when you only want one interface to be
  conformal and want to keep the rest of the assembly unfragmented for
  performance.
- **`g.parts.fuse_group(labels, *, label=None, dim=None, properties=None)`**
  (`_parts_registry.py`) — fuses a list of instances into a single new
  instance. The old instances are removed from the registry and a fresh one
  is created with the surviving tags. See `plan_fuse_group.md` for the design
  motivation.

A rule of thumb: **fragment** when the parts must share nodes but remain
distinct logical bodies (e.g. a steel gusset welded to a concrete column, two
materials meeting at an interface). **fuse_group** when the parts are really
one body and you want a single label downstream (e.g. a welded assembly of
many small STEP pieces that should be treated as one component).


## 3. Dimension and higher-dimensional meshing

Gmsh meshes are always indexed by a dimension between 0 and 3. apeGmsh exposes
the dimension explicitly rather than hiding it, because the same session
frequently carries elements of multiple dimensions at once (for example, a
3D solid with 2D shells glued to its surface, or 1D beam lines sharing nodes
with a 3D volume).

Generation is controlled by a single entry point:

```python
g.mesh.generation.generate(dim: int = 3) -> Mesh
```

- `dim=1` produces only 1D elements on curves.
- `dim=2` produces 1D + 2D elements up to surfaces. This is what you want for
  shell models and for meshing the surface of a solid without filling it.
- `dim=3` (the default) produces the full 1D + 2D + 3D cascade. 3D meshing
  internally runs the lower dimensions first, so the output always includes
  the surface and edge meshes the volume mesher used.

Once the mesh exists, most query methods accept `dim` as a filter with the
same convention: `dim=-1` means "all dimensions", and any positive value
restricts the query. The most useful ones are:

- `g.mesh.queries.get_nodes(*, dim=-1, tag=-1, ...) -> dict`
- `g.mesh.queries.get_elements(*, dim=-1, tag=-1) -> dict`
- `g.mesh.queries.get_fem_data(dim: int = 2) -> FEMData`

`get_fem_data` is the bridge to the FEM broker. Its `dim` argument selects the
*element* dimension you want the solver to see, not the dimension of the mesh
that was generated. A common pattern is to generate at `dim=3` and then extract
at `dim=2` for a shell model, or at `dim=3` for a solid model. The FEM broker
is source-agnostic on purpose: whether elements came from physical groups or
mesh selection sets, `FEMData` has the same shape and downstream code should
not special-case either source.

### Element order and uniform refinement

High-order elements are toggled globally through `set_order`:

```python
g.mesh.generation.set_order(order: int) -> Mesh
```

`order=1` is the default (linear). `order=2` promotes edges, faces and volumes
to their quadratic variants (9-node quads, 10-node tets, 20/27-node hexes
depending on recombination). Order changes must be applied *after*
`generate()`.

`g.mesh.generation.refine()` performs one round of uniform subdivision.
It is cheap for diagnostics but rarely what you want for production meshes —
prefer size fields (Section 5) for targeted refinement.

### What `generate()` inherits from Gmsh (the default contract)

`g.mesh.generation.generate(dim)` is intentionally a thin wrapper over
`gmsh.model.mesh.generate(dim)`. apeGmsh does **not** touch any `Mesh.*`
option behind your back on session startup — `g.begin()` only calls
`gmsh.initialize()` and `gmsh.model.add(name)`. That means: unless *you*
(or a previous call on this session) have set an option, `generate()` runs
with Gmsh's factory defaults.

This is a deliberate contract: every knob you care about is either left at
Gmsh's default *or* has a visible apeGmsh setter. Nothing in between.

The table below lists the options that actually shape the mesh, the Gmsh
default in effect when you call `generate()` on a fresh session, and the
apeGmsh setter (if any) that changes it. When there is no apeGmsh setter,
you can always fall through to `gmsh.option.setNumber(key, value)` yourself.

| Gmsh option | Default | Effect | apeGmsh setter |
|---|---|---|---|
| `Mesh.Algorithm`   | `6` (Frontal-Delaunay) | 2D algorithm, global | `g.mesh.generation.set_algorithm(tag, alg, dim=2)` (per-surface) |
| `Mesh.Algorithm3D` | `1` (Delaunay)         | 3D algorithm, global. Note: `"auto"` / `"default"` in the apeGmsh string table maps to `HXT` (10), not the raw Gmsh default | `g.mesh.generation.set_algorithm(0, alg, dim=3)` |
| `Mesh.ElementOrder`                | `1` (linear)     | Element order              | `g.mesh.generation.set_order(order)` (applied *after* `generate()`) |
| `Mesh.HighOrderOptimize`           | `0` (off)        | Curve high-order elements  | — (raw `gmsh.option`) |
| `Mesh.MeshSizeMin`                 | `0.0`            | Global floor on element size | `g.mesh.sizing.set_global_size(max, min)` / `set_size_global(min_size=...)` |
| `Mesh.MeshSizeMax`                 | `1e22`           | Global ceiling on element size | `g.mesh.sizing.set_global_size(max, min)` / `set_size_global(max_size=...)` |
| `Mesh.MeshSizeFactor`              | `1.0`            | Global multiplier applied to all characteristic lengths | — |
| `Mesh.MeshSizeFromPoints`          | `1` (on)         | Honour per-point `lc` values from STEP/IGES and `set_size` | `g.mesh.sizing.set_size_sources(from_points=...)` |
| `Mesh.MeshSizeFromCurvature`       | `0` (off)        | Refine with surface curvature | `g.mesh.sizing.set_size_sources(from_curvature=...)` |
| `Mesh.MeshSizeExtendFromBoundary`  | `1` (on)         | Propagate boundary sizes inward | `g.mesh.sizing.set_size_sources(extend_from_boundary=...)` |
| `Mesh.RecombineAll`                | `0` (off)        | Try to recombine triangles into quads globally | `g.mesh.structured.recombine()` |
| `Mesh.RecombinationAlgorithm`      | `1` (Blossom)    | Blossom-based full-quad merging | — |
| `Mesh.Recombine3DAll`              | `0` (off)        | Recombine tets into hexes globally | — |
| `Mesh.Smoothing`                   | `1`              | Number of Laplacian smoothing passes | `g.mesh.structured.set_smoothing(tag, n)` (per-surface) |
| `Mesh.Optimize`                    | `1` (on)         | Run the built-in tet optimiser after 3D meshing | `g.mesh.generation.optimize(...)` (explicit pass) |
| `Mesh.OptimizeNetgen`              | `0` (off)        | Netgen optimiser pass | `g.mesh.generation.optimize("Netgen", ...)` |
| `Mesh.OptimizeThreshold`           | `0.3`            | Quality threshold below which elements are optimised | — |
| `General.NumThreads`               | `1`              | Threads used by parallel mesh kernels (HXT uses this) | — |
| `Mesh.Binary`                      | `0` (ASCII)      | Output format when saving `.msh` | — |
| `Mesh.MshFileVersion`              | `4.1`            | `.msh` file format version | — |

Things worth remembering:

1. **2D vs 3D algorithm defaults differ.** Gmsh's raw default for
   `Mesh.Algorithm3D` is `1` (plain Delaunay), but apeGmsh's `"auto"` /
   `"default"` alias points at `HXT` (`10`) because HXT is the modern
   recommendation. If you never call `set_algorithm(..., dim=3)`, you get
   **Delaunay**, not HXT. Pass `set_algorithm(0, "hxt", dim=3)` explicitly
   if you want HXT.
2. **`Mesh.Optimize=1` is on by default.** A 3D `generate()` already runs
   one tet-optimiser pass without you asking. Calling
   `g.mesh.generation.optimize(...)` adds *another* pass on top of that.
3. **Size from points is on by default.** If you import STEP/IGES and your
   `set_global_size` appears to be ignored, the file is pushing per-point
   characteristic lengths through `Mesh.MeshSizeFromPoints`. Disable it
   with `g.mesh.sizing.set_size_sources(from_points=False)`.
4. **`MeshSizeMax = 1e22` is effectively "no ceiling".** If you forget to
   set a global size *and* there are no size fields *and* `from_points` is
   off, Gmsh will happily produce a one-element mesh of your whole domain.
5. **Order is set after generate().** `set_order(2)` elevates elements
   in-place; it is not an option that `generate()` reads. Calling it
   before `generate()` has no effect.
6. **apeGmsh never clears options between generations.** If you change
   `set_algorithm`, `set_global_size`, or call `set_size_sources`, those
   values stick on the Gmsh session for every subsequent `generate()`
   until you overwrite them or call `g.end()`.

If you need an option that is not in the table, reach straight through:

```python
import gmsh
gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5)
g.mesh.generation.generate(3)
```

This is always allowed — apeGmsh never hides Gmsh from you; it only
promotes the knobs that matter often enough to deserve a first-class name.


## 4. Mesh algorithms

apeGmsh exposes the Gmsh algorithm catalogues in three equivalent forms, and
`set_algorithm` accepts any of them. **Strings are the preferred form** — they
are easier to remember, they tolerate aliases and separator/case variation,
and typos raise a helpful `ValueError` listing every canonical name.

### The name-based API

The canonical lookup tables live in `mesh/Mesh.py` and are exported as
`apeGmsh.ALGORITHM_2D` and `apeGmsh.ALGORITHM_3D`:

```python
ALGORITHM_2D = {
    # canonical names
    "mesh_adapt": 1, "automatic": 2, "initial_mesh_only": 3,
    "delaunay":   5, "frontal_delaunay": 6, "bamg": 7,
    "frontal_delaunay_quads": 8, "packing_parallelograms": 9,
    "quasi_structured_quad":  11,
    # aliases
    "auto": 2, "default": 2, "frontal": 6,
    "quad": 8, "quads": 8, "quasi_structured": 11, "qsq": 11, ...
}

ALGORITHM_3D = {
    "delaunay": 1, "initial_mesh_only": 3, "frontal": 4,
    "mmg3d":    7, "r_tree": 9, "hxt": 10,
    # aliases
    "auto": 10, "default": 10, "automatic": 10, ...
}
```

Matching is case- and separator-insensitive, so `"Frontal-Delaunay"`,
`"frontal_delaunay"`, and `"FRONTAL DELAUNAY"` are equivalent.

For IDE autocomplete, the same names are exposed as string class constants
on `MeshAlgorithm2D` and `MeshAlgorithm3D`:

```python
from apeGmsh import MeshAlgorithm2D, MeshAlgorithm3D

MeshAlgorithm2D.QUADS       # -> "frontal_delaunay_quads"
MeshAlgorithm2D.QUAD        # -> "quasi_structured_quad"
MeshAlgorithm2D.AUTO        # -> "automatic"
MeshAlgorithm3D.HXT         # -> "hxt"
MeshAlgorithm3D.AUTO        # -> "hxt"
```

The legacy `Algorithm2D` / `Algorithm3D` `IntEnum`s are still exported and
still work — they are kept so that existing notebooks and scripts keep
running. New code should prefer the string form.

### set_algorithm

Selection happens through a single method whose behaviour depends on `dim`:

```python
g.mesh.generation.set_algorithm(tag: int, algorithm, *, dim: int = 2) -> Mesh
```

`algorithm` accepts:

- a **string** — looked up in `ALGORITHM_2D` / `ALGORITHM_3D`
- a `MeshAlgorithm2D` / `MeshAlgorithm3D` attribute (same strings)
- an `Algorithm2D` / `Algorithm3D` `IntEnum` member (legacy)
- a raw `int` — passed straight to Gmsh for forward compatibility

Examples (all four equivalent):

```python
g.mesh.generation.set_algorithm(surf_tag, "frontal_delaunay_quads")
g.mesh.generation.set_algorithm(surf_tag, "quads")                   # alias
g.mesh.generation.set_algorithm(surf_tag, MeshAlgorithm2D.QUADS)
g.mesh.generation.set_algorithm(surf_tag, Algorithm2D.FRONTAL_DELAUNAY_QUADS)

g.mesh.generation.set_algorithm(0, "hxt", dim=3)
g.mesh.generation.set_algorithm(0, "auto", dim=3)                    # also hxt
g.mesh.generation.set_algorithm(0, MeshAlgorithm3D.HXT, dim=3)
```

- With `dim=2`, the algorithm is set **per surface** via
  `gmsh.model.mesh.setAlgorithm(2, tag, algorithm)`. This lets you mix, for
  example, `FRONTAL_DELAUNAY_QUADS` on structured surfaces and plain
  `AUTOMATIC` everywhere else.
- With `dim=3`, the algorithm is set **globally** via
  `gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)`. Gmsh does not
  support per-volume 3D algorithm selection, so the `tag` argument is ignored.

Rules of thumb from the Gmsh docs as they apply here:

- `Algorithm2D.AUTOMATIC` / `FRONTAL_DELAUNAY` are the safe defaults for
  unstructured triangular meshes.
- `FRONTAL_DELAUNAY_QUADS` or `PACKING_PARALLELOGRAMS` are the preferred
  starting points for quad-dominant surface meshing; both still benefit from a
  subsequent `set_recombine`.
- `QUASI_STRUCTURED_QUAD` is the best-quality quad mesher but is the slowest
  and requires clean geometry.
- `Algorithm3D.HXT` is the modern default: fast, parallel, robust on large
  models. Fall back to `DELAUNAY` only for very small meshes or when `HXT`
  fails on pathological topology.
- `MMG3D` is reserved for remeshing / metric-based adaptation workflows.

Quad recombination and smoothing are controlled separately:

```python
g.mesh.structured.set_recombine(tag, *, dim=2, angle=45.0) -> Mesh
g.mesh.structured.recombine() -> Mesh
g.mesh.structured.set_smoothing(tag, val, *, dim=2) -> Mesh
```

`set_recombine` requests recombination on a single surface (or volume, when
`dim=3`) and the `angle` threshold controls how aggressively triangles are
merged into quads. `recombine()` is the global equivalent and takes no
arguments. `set_smoothing` applies `val` Laplacian passes to a surface.


## 5. Size control and mesh fields

apeGmsh gives you three layers of size control, from coarse to fine:

**Global bounds** — set a floor and ceiling on element size:

```python
g.mesh.sizing.set_global_size(max_size, min_size=0.0) -> Mesh
g.mesh.sizing.set_size_global(*, min_size=None, max_size=None)
```

The second form lets you change only one bound at a time. Both map to
`Mesh.MeshSizeMax` / `Mesh.MeshSizeMin` in Gmsh options.

**Size sources** — control which inputs Gmsh consults when deciding element
size at a point:

```python
g.mesh.sizing.set_size_sources(*, from_points=None, from_curvature=None,
                            extend_from_boundary=None) -> Mesh
```

A common gotcha after importing IGES or STEP is that the file carries
per-point characteristic lengths that override your `set_global_size` call.
Passing `from_points=False` (and usually `from_curvature=False`) makes the
global bound authoritative again.

**Per-point sizes:**

```python
g.mesh.sizing.set_size(tags, size, *, dim=0) -> Mesh
g.mesh.sizing.set_size_all_points(size) -> Mesh
g.mesh.sizing.set_size_callback(func) -> Mesh
```

The callback signature is `func(dim, tag, x, y, z, lc) -> float` and lets you
compute size in Python based on arbitrary criteria.

**Mesh fields** — `g.mesh.field` is a `FieldHelper` (`_mesh_field.py`) that
wraps the raw `gmsh.model.mesh.field` API with named convenience builders:

| Helper | Purpose |
|---|---|
| `field.distance(*, curves=None, surfaces=None, points=None, sampling=100)` | Shortest distance to a set of entities |
| `field.threshold(distance_field, *, size_min, size_max, dist_min, dist_max, sigmoid=False, ...)` | Ramp size between two distance thresholds |
| `field.math_eval(expression)` | Arbitrary MathEval field in `x, y, z` |
| `field.box(*, x_min, y_min, z_min, x_max, y_max, z_max, size_in, size_out, thickness=0)` | Different size inside and outside a box |
| `field.minimum(field_tags)` | Element-wise minimum of several fields |
| `field.boundary_layer(*, curves=None, points=None, size_near, ratio=1.2, n_layers=5, thickness=None, fan_points=None)` | BoundaryLayer field for wall-resolved meshes |

Once a field is built, register it as the active size field with
`field.set_background(tag)`, or as a boundary-layer field with
`field.set_boundary_layer_field(tag)`. The raw primitives
(`add`, `set_number`, `set_numbers`, `set_string`) are still available if you
need to use a field type that does not have a helper yet.

The canonical "refine near a feature" recipe is:

```python
d = g.mesh.field.distance(curves=crack_tip_curves)
t = g.mesh.field.threshold(d, size_min=0.1, size_max=5.0,
                               dist_min=0.5, dist_max=10.0)
g.mesh.field.set_background(t)
g.mesh.generation.generate(3)
```


## 6. Structured meshing: transfinite and recombination

Transfinite meshing produces structured (mapped) grids from topologically
quadrilateral or hexahedral regions. The three primitives mirror Gmsh:

```python
g.mesh.structured.set_transfinite_curve(tag, n_nodes, *, mesh_type="Progression", coef=1.0)
g.mesh.structured.set_transfinite_surface(tag, *, arrangement="Left", corners=None)
g.mesh.structured.set_transfinite_volume(tag, *, corners=None)
```

- On a curve, `n_nodes` counts endpoints. `mesh_type` is `"Progression"` for
  a geometric ratio (use `coef > 1` to cluster nodes at the start,
  `coef < 1` to cluster at the end) or `"Bump"` to cluster at both ends.
- On a surface, all four bounding curves must already carry compatible
  transfinite constraints. `arrangement` picks the triangle split pattern
  (`"Left"`, `"Right"`, `"AlternateLeft"`, `"AlternateRight"`). For a
  quad-only surface, also call `set_recombine(tag)` afterwards.
- On a volume, pass the eight corner point tags explicitly when the
  auto-detection fails.

For whole assemblies where you want Gmsh to *find* transfinite regions
itself, use:

```python
g.mesh.structured.set_transfinite_automatic(dim_tags=None, *, corner_angle=2.35,
                                 recombine=True) -> Mesh
```

`corner_angle` (in radians; default ≈ 135°) controls how sharp a vertex has
to be to count as a "corner". `recombine=True` automatically recombines the
detected surfaces so you get quads rather than triangles. Passing
`dim_tags=None` applies it to every entity in the model.

Structured meshing composes cleanly with the global algorithm: set
`Algorithm2D.AUTOMATIC` for the unstructured regions, then mark the
structured patches with transfinite constraints. Gmsh will use the
transfinite code path where applicable and the general algorithm everywhere
else.


## 6a. Driving mesh commands by physical group

Physical groups are named bags of OCC entity tags — they carry no meshing
semantics on their own, because every Gmsh mesh-control call
(`setAlgorithm`, `setRecombine`, `setTransfiniteCurve`, `setSize`, field
`CurvesList` / `SurfacesList`, …) takes raw entity tags. apeGmsh bridges the
gap with a single resolver on `g.physical` plus a small family of
`*_by_physical` wrappers on `g.mesh` that fan per-entity commands out over
every member of a named group. The goal is to let you write code in terms
of domain labels ("Concrete", "BeamColumnJoint", "StructuredFlanges")
without ever calling `gmsh.model.getEntitiesForPhysicalGroup` by hand.

### The resolver

```python
g.physical.entities(name_or_tag, *, dim=None) -> list[Tag]     # PhysicalGroups.py
```

- Pass a **name** (`"Concrete"`) and optionally a `dim`. With no `dim`,
  apeGmsh searches every dimension from 0 to 3 and returns the tags of the
  first match, so a name that is unambiguous across dimensions just works.
- Pass a raw PG tag and `dim` is required.
- Missing groups raise `KeyError`; passing a raw tag without `dim` raises
  `TypeError`. Both messages name the offending group so failures are
  immediately actionable.

The resolver composes cleanly with every mesh entry point that already
accepts a list of tags:

```python
# Drive a per-surface algorithm from a PG
for s in g.physical.entities("Concrete", dim=2):
    g.mesh.generation.set_algorithm(s, "frontal_delaunay_quads")
    g.mesh.structured.set_recombine(s)

# Feed a distance field directly from a PG
joint = g.physical.entities("BeamColumnJoint", dim=2)
d = g.mesh.field.distance(surfaces=joint)
t = g.mesh.field.threshold(d, size_min=0.02, dist_min=0.05,
                              size_max=0.30, dist_max=1.0)
g.mesh.field.set_background(t)

# Embed rebar curves pulled from a PG into a concrete volume
rebar_curves = g.physical.entities("Rebars", dim=1)
g.mesh.editing.embed(rebar_curves, in_tag=concrete_vol, dim=1, in_dim=3)
```

Because `g.parts.add_physical_groups()` (`_parts_registry.py`) already
creates one PG per registered instance, the resolver also doubles as a
"by part label" lookup for the common case — `g.physical.entities("col")`
and `g.parts.instances["col"].entities[dim]` give the same tags. Where the
PG route is strictly more expressive is when you want to group *across*
instances (e.g. collect every interface surface produced by
`fragment_all()` under one label) or *a subset of one* instance (e.g. only
the top faces of a slab).

### The `*_by_physical` convenience wrappers

For the very common "fan a per-entity mesh command out over a named PG"
pattern, `Mesh` exposes thin wrappers that do the resolution and the loop
internally. They all take the PG name as the first argument and the
dimension as a keyword, and they all return `self` so they chain:

| Wrapper | Fans out to | Notes |
|---|---|---|
| `g.mesh.generation.set_algorithm_by_physical(name, algorithm, *, dim=2)` | `set_algorithm` | `dim=2` is per-surface; `dim=3` still writes the global `Mesh.Algorithm3D` option, name is used only in the log |
| `g.mesh.structured.set_recombine_by_physical(name, *, dim=2, angle=45.0)` | `set_recombine` | Turn triangles into quads over every surface in the PG |
| `g.mesh.structured.set_smoothing_by_physical(name, val, *, dim=2)` | `set_smoothing` | Laplacian passes per surface |
| `g.mesh.sizing.set_size_by_physical(name, size, *, dim=0)` | `set_size` | Only effective on points; use a field for surface/volume sizing |
| `g.mesh.structured.set_transfinite_by_physical(name, *, dim, **kwargs)` | `set_transfinite_curve / _surface / _volume` | Picks the right variant from `dim`; forwards `**kwargs` untouched |

A compact recipe mixing several of them:

```python
# PG layout (pre-mesh, declared on geometry)
g.physical.add_surface(flange_surfs, name="Flanges")
g.physical.add_surface(web_surfs,    name="Web")
g.physical.add_curve  (edge_curves,  name="WebEdges")
g.physical.add_surface(joint_surfs,  name="BeamColumnJoint")

# Structured quad patches on the flanges
g.mesh.generation.set_algorithm_by_physical("Flanges", "frontal_delaunay_quads")
g.mesh.structured.set_recombine_by_physical("Flanges")

# A transfinite web, 40 nodes along every edge, clustered 1:1.1
g.mesh.structured.set_transfinite_by_physical(
    "WebEdges", dim=1, n_nodes=40, mesh_type="Progression", coef=1.1,
)
for s in g.physical.entities("Web", dim=2):
    g.mesh.structured.set_transfinite_surface(s)
    g.mesh.structured.set_recombine(s)

# Refinement zone around the joint
d = g.mesh.field.distance(
    surfaces=g.physical.entities("BeamColumnJoint", dim=2)
)
t = g.mesh.field.threshold(d, size_min=0.02, dist_min=0.05,
                              size_max=0.30, dist_max=1.0)
g.mesh.field.set_background(t)

g.mesh.generation.generate(3)
```

### Two caveats

The 3D-algorithm global limitation from Section 4 still holds:
`set_algorithm_by_physical("SoilBlock", "delaunay", dim=3)` is accepted,
but it ends up writing `Mesh.Algorithm3D` once (not once per volume), and
the PG name is only preserved in the log. It's still useful as a
self-documenting way to say "I picked Delaunay because of the soil
volume", but it does not give you per-volume 3D selection.

Second, everything above is strictly *pre-mesh*. `MeshSelectionSet` —
covered in the next section — is its post-mesh sibling and cannot drive
sizes, algorithms or transfinite constraints, because the mesh has to
exist before a selection set can be defined. If you want "these elements
one size, those elements another size", build the zones as physical
groups (or raw entity tags), feed them to `field.distance`/`field.box`/
`field.threshold`, and let the mesher produce the right element sizes
directly. Use `MeshSelectionSet` later, when you need to address specific
node or element subsets of the *finished* mesh for boundary conditions,
post-processing, or constraints.


## 7. Physical groups and mesh selection sets

Once the mesh exists you need named handles to address subsets of it —
boundary faces, node sets for constraints, element sets for material
assignment, and so on. apeGmsh has two composites for this, and they are
deliberately complementary.

**`g.physical` (PhysicalGroups, `mesh/PhysicalGroups.py`)** is the
geometry-driven route. It wraps `gmsh.model.addPhysicalGroup` and requires
`(dim, entity_tags)` — it tags *geometry entities*, and the mesher copies the
tag onto the elements the geometry owns. Physical groups survive writing and
re-reading a `.msh` file.

```python
g.physical.add(dim, tags, *, name="", tag=-1) -> Tag
g.physical.add_point(tags,   *, name="", tag=-1)
g.physical.add_curve(tags,   *, name="", tag=-1)
g.physical.add_surface(tags, *, name="", tag=-1)
g.physical.add_volume(tags,  *, name="", tag=-1)
g.physical.get_all(dim=-1) -> list[(dim, tag)]
g.physical.get_nodes(dim, tag) -> dict
g.physical.summary() -> DataFrame
```

`g.parts.add_physical_groups(dim=None)` (`_parts_registry.py`) is the
one-liner that creates one physical group per registered instance, so you get
named handles for every part without touching raw tags.

**`g.mesh_selection` (MeshSelectionSet, `mesh/MeshSelectionSet.py`)** is the
mesh-driven route. It identifies subsets directly on the mesh — by spatial
predicate, element type, or explicit IDs — and is the right tool for things
that do not correspond to a single OCC entity (a bounding box full of nodes, a
plane slicing through the middle of a volume, elements of a particular type).
It uses the same `(dim, tag) + name` identity contract as `g.physical`, so
downstream code can iterate over both in the same way.

```python
g.mesh_selection.add(dim, tags, *, name="", tag=-1) -> int
g.mesh_selection.add_nodes(*, name="", tag=-1, on_plane=..., in_box=...,
                                in_sphere=..., nearest_to=..., predicate=...) -> int
g.mesh_selection.add_elements(dim=2, *, name="", in_box=..., on_plane=...,
                                       by_type=..., predicate=...) -> int
g.mesh_selection.from_physical(dim, pg_tag) -> int
```

The dimension convention on a mesh selection set matches the *element*
dimension: `dim=0` is a node set, `dim=1/2/3` are line/surface/volume element
sets. `from_physical` is the interop bridge — it materialises a physical group
as a mesh selection set so you can then combine it with spatial queries.

Both composites are snapshotted into `FEMData` when you call
`get_fem_data()`. The broker treats `fem.physical` and `fem.mesh_selection`
identically; downstream solvers should not care which one produced a given
set.


## 8. The bridge between mesh and constraints

Constraints are declared **before** the mesh exists and resolved **after**.
This two-stage pipeline is what lets you describe intent ("tie the top of the
gusset to the face of the beam") in terms of part labels and geometry, while
the actual node tags, DOFs, weights and offsets are computed from the mesh
once it has been generated.

### Stage 1 — definition, on geometry

All definition methods live on `g.constraints` and return a lightweight
`ConstraintDef` dataclass that is appended to `constraint_defs`. They touch no
mesh data:

- Node-to-node: `equal_dof`, `rigid_link`, `penalty`
- Node-to-group: `rigid_diaphragm`, `rigid_body`, `kinematic_coupling`
- Node-to-surface: `tie`

All of them take *labels* (instance names) as arguments, never raw tags. This
is why fragmentation has to happen first: the labels must already point at the
conformal, fragmented entities.

### Stage 2 — resolution, on the mesh

After `g.mesh.generation.generate()` you extract an FEM view of the mesh and feed it to
the resolver:

```python
fem = g.mesh.queries.get_fem_data(dim=3)
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
face_map = g.parts.build_face_map(node_map)
records  = g.constraints.resolve(
    fem.nodes.ids, fem.nodes.coords,
    node_map=node_map,
    face_map=face_map,
)
```

- `build_node_map` returns a `dict[label, set[int]]` that assigns every mesh
  node to the instance(s) whose bounding box contains it. This is the
  geometry-to-mesh label lookup the resolver relies on.
- `build_face_map` returns a `dict[label, ndarray]` of surface-element
  connectivity per instance, for surface-based constraints (`tie`, future
  contact types).
- `resolve` walks the `constraint_defs` list and converts each one into one or
  more `ConstraintRecord`s with concrete node tags and DOFs. The records are
  solver-agnostic; the OpenSees bridge and any future solver bridge consume
  the same shape.

The resolver currently does **not** implement embedded-constraint resolution
— `ConstraintsComposite.py` raises `NotImplementedError` for that path.
Everything else in the definition catalogue is wired end to end.

This is also where the no-proxy rule matters: there is no "ConstraintEntity"
or "MeshedEntity" sitting between `Instance.entities` and the node/face maps.
The registry is the single source of truth; the resolver reads from it
directly.


## 9. A minimal end-to-end example

The assembly workflow collapses to the following sequence, with each step
mapping to one section of this guide:

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="demo") as g:
    # 1. Instances — either import parts or build inline
    col = g.parts.import_step("column.step", label="col")
    bm  = g.parts.import_step("beam.step",   label="bm",
                              translate=(0, 0, 3.0))

    # 2. Conformal topology — fragment so the interface becomes shared
    g.parts.fragment_all()

    # 3. Physical groups per part (explicit in v1.0)
    g.physical.add_volume(g.parts.instances["col"].entities[3], name="col")
    g.physical.add_volume(g.parts.instances["bm"].entities[3],  name="bm")

    # 4. Size + algorithm + field control
    (g.mesh.sizing
        .set_size_sources(from_points=False)
        .set_global_size(0.15))
    g.mesh.generation.set_algorithm(0, "hxt", dim=3)

    # 5. Generate the mesh at the target dimension
    (g.mesh.generation
        .generate(dim=3)
        .set_order(2))
    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)

    # 6. Declare constraints against part labels (pre-resolution)
    g.constraints.tie(master_label="col", slave_label="bm")

    # 7. Extract FEM data (resolves constraints automatically)
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info)
```

Every method shown above is a direct reference to the symbols listed earlier
in the guide. When in doubt, the definitions in
`src/apeGmsh/mesh/_mesh_*.py`, `src/apeGmsh/core/_parts_registry.py`,
`src/apeGmsh/core/_model_boolean.py` and
`src/apeGmsh/core/ConstraintsComposite.py` are the authoritative source.


## See also

- `guide_parts_vs_session.md` — when to use parts vs a direct session
- `guide_parts_assembly.md` — full five-phase assembly workflow
- `plan_fuse_group.md` — motivation and API for `fuse_group`
- `plan_mesh_selection_set.md` — architecture of `MeshSelectionSet`
- `plan_v2_unified_architecture.md` / `plan_v2_final.md` — the unified runtime
  that these composites live in
