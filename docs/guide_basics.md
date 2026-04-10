# pyGmsh basics

A first-contact guide to working inside a pyGmsh session. This is the
document to read before `guide_parts_assembly.md` or `guide_meshing.md`:
it covers only the fundamentals — how a session starts, how geometry is
built, and how the OCC boolean operations (`fragment`, `fuse`,
`cut`/`intersect`) are used to get to a conformal, mesh-ready model.

The guide is grounded in the current source:

- `src/pyGmsh/_session.py` — session lifecycle (`begin`/`end`)
- `src/pyGmsh/_core.py` — the `pyGmsh` composite container
- `src/pyGmsh/core/Model.py` + `_model_geometry.py` + `_model_boolean.py`
  — geometry creation and boolean operations

All code snippets assume `from apeGmsh import apeGmsh`.


## 1. Why initialize? The session lifecycle

Gmsh is a C library with a single, process-wide state. Before any API
call you must hand control to that state with `gmsh.initialize()`, create
a named model with `gmsh.model.add(...)`, and eventually release the
state with `gmsh.finalize()`. Forgetting any of the three is the single
most common source of Gmsh errors.

pyGmsh wraps this lifecycle behind two calls on a session object, and
supports two equivalent usage patterns. Both produce the exact same
result — pick whichever fits the situation better.

**Pattern A — explicit `begin()` / `end()`.** Useful in notebooks when
you want to keep the session alive across several cells, in interactive
debugging, or when the session lifetime has to be controlled by
something other than a lexical block (e.g. a class `__init__` /
`__del__`, a Flask request, a test fixture).

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="cantilever", verbose=True)
g.begin()   # gmsh.initialize() + gmsh.model.add("cantilever") + composite wiring

g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
g.mesh.generate(dim=3)
# ... you can keep using g across many cells / function calls ...

g.end()     # gmsh.finalize()
```

With this pattern **you own the cleanup**. If an exception is raised
between `begin()` and `end()`, `gmsh.finalize()` never runs, and the
next call to `gmsh.initialize()` anywhere in the process will inherit
a polluted state. In a notebook, the symptom is that a perfectly
correct cell suddenly errors because of a previous cell's crash. Wrap
sensitive sections in `try / finally`:

```python
g = apeGmsh(model_name="cantilever")
g.begin()
try:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
    g.mesh.generate(dim=3)
finally:
    g.end()
```

**Pattern B — context manager (`with`).** The preferred form in scripts
and one-shot notebook cells. Equivalent to Pattern A wrapped in the
`try / finally` above: `__enter__` calls `begin()`, `__exit__` calls
`end()` no matter how the block exits.

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="cantilever", verbose=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
    g.mesh.generate(dim=3)
# gmsh.finalize() runs here, even if the block raised
```

Both patterns go through the same `begin()` implementation, which does
three things in order:

1. Calls `gmsh.initialize()` and `gmsh.model.add(self.name)`.
2. Instantiates every composite listed in `pyGmsh._COMPOSITES` — `model`,
   `mesh`, `physical`, `mesh_selection`, `constraints`, `loads`,
   `opensees`, and so on — and attaches them as attributes on `g`.
   Those composites share a single `_registry` (the book-keeping dict
   that tracks every entity created through pyGmsh) so nothing in the
   pipeline needs to read raw Gmsh tags.
3. Marks the session as active. Any composite method that touches gmsh
   asserts `g._active` first; this is what prevents the "I forgot
   `begin()`" footgun.

`end()` finalises the Gmsh state and marks the session inactive. After
`end()` the composites still exist, but calling anything on them will
raise — the Gmsh state they referenced is gone.

Rule of thumb: reach for `with` unless you have a concrete reason not
to. The explicit form is there for interactive work and for cases
where the session has to outlive a single code block.

A session always uses the **OpenCASCADE kernel** (`gmsh.model.occ`). The
built-in kernel (`gmsh.model.geo`) is intentionally not wrapped: the
meshing, constraints, and FEM layers assume OCC semantics (persistent
entity identity across boolean operations, accurate bounding boxes,
STEP/IGES I/O). Mixing kernels would break the `_registry`.


## 2. Creating geometry

Geometry is created through `g.model`, which exposes the OCC kernel
behind a thin, self-documenting wrapper. Every `add_*` method:

- calls the underlying `gmsh.model.occ.add*` function,
- calls `gmsh.model.occ.synchronize()` unless `sync=False`,
- registers the new entity in `g.model._registry` with a `label`,
  `kind`, and its `DimTag`, and
- returns the **integer tag** of the new entity.

You therefore rarely touch `gmsh.model.occ` directly.

### Primitives

Solids, surfaces, curves, and points are available:

```python
with apeGmsh(model_name="demo") as g:
    # Points (dim = 0)
    p1 = g.model.geometry.add_point(0, 0, 0, mesh_size=0.1, label="origin")

    # Curves (dim = 1)
    p2 = g.model.geometry.add_point(1, 0, 0)
    line = g.model.geometry.add_line(p1, p2, label="bottom_edge")

    # Surfaces (dim = 2)
    plate = g.model.geometry.add_rectangle(0, 0, 0, 1.0, 0.5, label="plate")

    # Solids (dim = 3)
    box  = g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
    cyl  = g.model.geometry.add_cylinder(0.5, 0.5, 0, 0, 0, 10, radius=0.3, label="core")
    sph  = g.model.geometry.add_sphere(0.5, 0.5, 5, radius=0.2, label="notch")
```

Two things are worth noting:

1. **Labels are the primary addressing mechanism in pyGmsh.** The raw
   integer tags are unstable across boolean operations (fragment can
   split one box into several new tags), but labels can be re-resolved
   against the registry at any time. Always label anything you plan to
   reference later — a physical group, a boundary condition surface, a
   constraint partner.

2. **`sync=True` is the default.** Each call synchronises OCC before
   returning, which is slow for large models. When you are building a
   batch of entities in a tight loop, pass `sync=False` to every call
   except the last one — the final `sync=True` flushes the whole batch
   to the kernel at once.

### Sketched geometry

For non-primitive shapes (an I-beam cross-section, a gusset plate, a
fillet profile) the pattern is the standard Gmsh sketch workflow, just
with pyGmsh wrappers:

```python
# 1. Points
pts = [g.model.geometry.add_point(x, y, 0, sync=False) for x, y in coords]
# 2. Lines closing the contour
lns = [g.model.geometry.add_line(pts[i], pts[(i + 1) % len(pts)], sync=False)
       for i in range(len(pts))]
# 3. Curve loop -> surface
loop = g.model.geometry.add_curve_loop(lns, sync=False)
face = g.model.geometry.add_plane_surface([loop], label="I_web")
# 4. Extrude to a solid if needed (use gmsh.model.occ.extrude + re-register)
```

Extrusion, revolution, and transforms are exposed via
`g.model.transforms.extrude(...)`, `g.model.transforms.translate(...)`, `g.model.transforms.rotate(...)`,
and friends on `_model_transforms.py`. They follow the same
label + registry conventions as the primitives.


## 3. OCC operations: fragment, fuse, cut, intersect

Once you have more than one body in the model you are almost always
going to need a boolean operation. pyGmsh exposes four of them on
`g.model`, all implemented in `core/_model_boolean.py` through a single
internal helper `_bool_op` that:

- resolves integer tags to `DimTag`s (so you can pass labels or bare
  tags interchangeably),
- calls the matching `gmsh.model.occ.*` function,
- synchronises,
- cleans up the registry (consumed entities are removed; surviving
  ones are re-registered), and
- returns a `list[Tag]` of the surviving entities at the target
  dimension.

### fuse — union (A ∪ B)

`g.model.boolean.fuse(objects, tools)` merges bodies into one. Use this when
two parts are geometrically the *same material* and you don't care
about the interface between them — the classic example is gluing a
haunch onto a beam flange.

```python
beam    = g.model.geometry.add_box(0, 0, 0, 10, 0.3, 0.5)
haunch  = g.model.geometry.add_box(0, 0, 0.5, 2.0, 0.3, 0.3)
merged  = g.model.boolean.fuse(beam, haunch)        # -> [one volume tag]
```

After `fuse`, the interface surface between the two boxes is *gone*.
You cannot recover it, so you cannot put a physical group, a BC, or a
tie constraint on it. If you need the interface, use `fragment`
instead.

### cut — difference (A − B)

`g.model.boolean.cut(objects, tools)` removes the tool from the object. This
is how you drill bolt holes, carve notches, or subtract a void region
from a soil block.

```python
block  = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
hole   = g.model.geometry.add_cylinder(0.5, 0.5, 0, 0, 0, 1, radius=0.1)
block2 = g.model.boolean.cut(block, hole)            # block with a through-hole
```

### intersect — intersection (A ∩ B)

`g.model.boolean.intersect(objects, tools)` keeps only the overlap region.
This is useful for clipping one shape to the bounding volume of
another — for example, clipping a soil layer to a site footprint.

### fragment — split at intersections (keep everything)

`g.model.boolean.fragment(objects, tools)` is the most important one for FEM
work, and the one that trips people up first.

`fragment` computes all intersections between `objects` and `tools`
and then *splits every body at those intersections* without throwing
anything away. The result is a collection of sub-volumes (and
sub-surfaces, and sub-curves) that **share coincident topology**
wherever they touch — no gaps, no overlaps, no duplicated nodes.
This is the geometric precondition for a conformal mesh across a
multi-body model.

```python
soil     = g.model.geometry.add_box(-5, -5, -5, 10, 10, 5, label="soil")
footing  = g.model.geometry.add_box(-1, -1, -1, 2, 2, 1, label="footing")

# Fragment the soil against the footing. Both bodies survive, but
# the soil now has a matching face where the footing sits.
pieces = g.model.boolean.fragment(soil, footing)
```

Because fragmenting can multiply entities, the registry is updated
**in place**: consumed inputs are popped, new pieces are re-registered
with a derived label. After a `fragment` you should re-resolve labels
through `g.model._registry` (or through `g.parts` if you are working
with parts) rather than relying on the tags you had before the call.

`fragment` also does a small post-processing step. If the operation
leaves free-floating surface fragments — for example a cutting plane
that extended past the solid — it removes them when
`cleanup_free=True` (the default). Set it to `False` if you
deliberately want to keep those surfaces for a boundary condition or
a selection set.

### Choosing between fragment and fuse — the conformal question

This is the decision you need to make for every multi-body model:

| You want…                                                      | Use       |
|---------------------------------------------------------------|-----------|
| Two bodies merged into one material region                    | `fuse`    |
| Two bodies sharing a mesh-conformal interface, still separate | `fragment`|
| Two bodies remaining independent, tied with constraints       | neither   |

"Mesh-conformal" means the mesher produces a single set of nodes on
the shared face, so both bodies reference the same DOFs there. Without
`fragment`, each body would mesh its face independently and you would
end up with two unconnected node clouds that happen to be coincident
in space — a silent, catastrophic modelling error.

The rule of thumb in this project: **if two parts touch and should
transmit force through the contact, fragment them**. If they should
transmit force through an elastic or rigid link instead, leave them
alone and add a constraint in `g.constraints`.


## 4. Worked example: soil block + footing + column

End-to-end script showing initialization, geometry, and the three
OCC operations working together. The goal is a soil block with a
footing embedded in its top and a column rising out of the footing,
all prepared for a conformal mesh.

### 4a. Context-manager form (scripts, one-shot cells)

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="ssi_demo", verbose=True) as g:

    # --- Bodies ---------------------------------------------------------
    # Soil: 10 x 10 x 5 block with its top at z = 0
    soil    = g.model.geometry.add_box(-5, -5, -5, 10, 10, 5, label="soil")

    # Footing: 2 x 2 x 1 block, top flush with soil surface
    footing = g.model.geometry.add_box(-1, -1, -1, 2, 2, 1, label="footing")

    # Column stub above the footing (two halves that we will fuse)
    col_low  = g.model.geometry.add_box(-0.15, -0.15, 0,    0.3, 0.3, 1.0)
    col_high = g.model.geometry.add_box(-0.15, -0.15, 1.0,  0.3, 0.3, 2.0)

    # --- Fuse: two column segments become one body ---------------------
    column = g.model.boolean.fuse(col_low, col_high)
    # The interface at z = 1.0 is erased. Good: the column is one
    # homogeneous piece of concrete with no artificial seam.

    # --- Fragment: make soil + footing + column conformal --------------
    # Any order works; fragment is symmetric in objects/tools except
    # for the order of surviving tags.
    pieces = g.model.boolean.fragment(soil, [footing] + column)
    # After this call:
    #   - soil has a matching face with the footing bottom
    #   - footing has a matching face with the column base
    #   - no entity overlaps, no gaps, no free surfaces

    # --- Physical groups (for materials + BCs) -------------------------
    # Labels still resolve because fragment updated the registry.
    g.physical.add_from_label("soil",    name="Soil",    dim=3)
    g.physical.add_from_label("footing", name="Footing", dim=3)
    g.physical.add_from_label("column",  name="Column",  dim=3)

    # --- Mesh ----------------------------------------------------------
    g.mesh.set_size_global(0.5)
    g.mesh.generate(dim=3)

    # At this point the three volumes share nodes across their
    # fragment-generated interfaces; OpenSees will see a single
    # connected assembly.
```

### 4b. Explicit `begin()` / `end()` form (notebooks, multi-cell work)

The same model, written so the session stays open across what would
typically be several notebook cells. This is the pattern to use when
you want to inspect `g.model._registry` between steps, call
`g.plot.show()` mid-build, or re-run just the meshing cell without
rebuilding the geometry.

```python
from apeGmsh import apeGmsh

# --- Cell 1: open the session --------------------------------------------
g = apeGmsh(model_name="ssi_demo", verbose=True)
g.begin()

try:
    # --- Cell 2: build the bodies ----------------------------------------
    soil    = g.model.geometry.add_box(-5, -5, -5, 10, 10, 5, label="soil")
    footing = g.model.geometry.add_box(-1, -1, -1, 2, 2, 1,   label="footing")
    col_low  = g.model.geometry.add_box(-0.15, -0.15, 0,   0.3, 0.3, 1.0)
    col_high = g.model.geometry.add_box(-0.15, -0.15, 1.0, 0.3, 0.3, 2.0)

    # --- Cell 3: fuse the column halves ---------------------------------
    column = g.model.boolean.fuse(col_low, col_high)

    # --- Cell 4: fragment the whole assembly ----------------------------
    pieces = g.model.boolean.fragment(soil, [footing] + column)

    # --- Cell 5: physical groups + mesh ---------------------------------
    g.physical.add_from_label("soil",    name="Soil",    dim=3)
    g.physical.add_from_label("footing", name="Footing", dim=3)
    g.physical.add_from_label("column",  name="Column",  dim=3)

    g.mesh.set_size_global(0.5)
    g.mesh.generate(dim=3)

finally:
    # --- Final cell: release the Gmsh state ------------------------------
    g.end()
```

The `try / finally` around the body is what makes this pattern safe in
a notebook: if any cell raises, the `finally` block still runs and
cleans up the Gmsh state, so the next session you open starts from a
clean slate. If you skip the `try / finally`, remember to run `g.end()`
manually before re-running the "open the session" cell — otherwise the
second `gmsh.initialize()` call will fight with the first.

The three OCC operations each play a distinct role here:

- `fuse` on the two column segments is cosmetic — it removes an
  unnecessary internal face so the column is a single body.
- `fragment` on the full trio is structural — it enforces conformal
  topology so the mesher can produce a single, connected FEM model.
- `cut` and `intersect` did not appear, but they would come in if the
  soil block had a void (say, a buried tunnel: `cut(soil, tunnel)`)
  or if the footing had to be trimmed to a site polygon
  (`intersect(footing, site)`).

### When not to fragment

Fragmenting a 5000-entity assembly is expensive. If your bodies are
either (a) truly disjoint in space, or (b) intended to be tied with
`equalDOF` / rigid links rather than sharing nodes, skip `fragment`
and handle the coupling in `g.constraints` instead. The meshing guide
(`guide_meshing.md`) covers that path.


## 5. What to read next

- `docs/guide_parts_assembly.md` — how to build geometry in isolated
  `Part` sessions and assemble them in a master session via
  `g.parts.add(...)` and `g.parts.fragment_all()`.
- `docs/guide_meshing.md` — mesh sizing, physical groups,
  `MeshSelectionSet`, constraint resolution, and the FEM broker.
- `docs/plan_v2_unified_architecture.md` — the v2 architecture the
  composites are converging on.
