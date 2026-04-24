# First steps with apeGmsh

A conversational walkthrough for someone opening apeGmsh for the first
time. It intentionally moves slowly — the goal is to build a mental
model, not to dump the full API. For the complete reference see the
other guides (`guide_basics.md`, `guide_meshing.md`, …) and the
API pages.

This document is built up lesson-by-lesson from real teaching
sessions. Each lesson is short, answers one question, and leaves you
ready for the next.


## Lesson 1 — What apeGmsh is, and why it exists

The one-sentence version:

> apeGmsh lets you describe a structural model once — geometry,
> loads, constraints, masses — and hand it off to a solver (OpenSees
> today) without writing Gmsh API calls by hand.

Two ideas do all the work.

### 1.1  A session owns one model

You open a session with a `with` block. Inside it, the object `g`
exposes everything — geometry, meshing, loads, the solver bridge. No
global state, no stray `gmsh.initialize()` calls scattered around your
script.

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="hello", verbose=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
```

That's a full meshed cube in three lines.

### 1.2  Two ways to open a session

`apeGmsh` (and `Part`, which we'll meet later) supports two equivalent
lifecycle patterns. Pick whichever matches the situation.

**Style A — context manager.** Recommended for scripts. The session
closes cleanly on exit *and* on exceptions.

```python
with apeGmsh(model_name="hello") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
```

**Style B — explicit `begin()` / `end()`.** Recommended for Jupyter
notebooks, where you want the session to stay alive across cells so
you can inspect `g` between steps.

```python
g = apeGmsh(model_name="hello").begin()

g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
# ... more cells, more operations ...

g.end()
```

Both styles produce the same result. The `with` form is safer because
it closes the Gmsh state even if something raises in the middle; the
explicit form is more ergonomic when you are exploring interactively.

### 1.3  Composites group related actions

`g` is not a flat object with two hundred methods. It has ~16
attributes, each a focused slice of the API:

| Access | Purpose |
|---|---|
| `g.model` | Geometry — points, boxes, booleans, STEP I/O |
| `g.mesh` | Meshing — sizing, generation, structured, queries |
| `g.physical` | Named groups (`"Base"`, `"Body"`) that survive the pipeline |
| `g.constraints`, `g.loads`, `g.masses` | Pre-mesh declarations |
| `g.opensees` | The solver bridge |

And the big composites have **sub-composites**. You always write

```python
g.mesh.generation.generate(dim=3)     # correct
```

not

```python
g.mesh.generate(dim=3)                # will not work
```

This is deliberate: the prefix tells you *what kind* of operation it
is (generation vs. sizing vs. structured vs. queries) and keeps the
namespaces honest.

---

## Lesson 2 — The basic model workflow, and your four geometry avenues

### 2.1  The workflow, end to end

A full apeGmsh run has six stages:

1. **Create geometry** — primitives, CAD, sections, or Parts.
2. **Boolean operations** — fragment / fuse / cut to turn a pile of
   shapes into one conformal assembly.
3. **Define physical groups** — named handles (`"Base"`, `"Body"`,
   `"facade"`) you will refer to everywhere downstream.
4. **Declare pre-mesh attributes** — constraints, loads, masses.
   These are stored as definitions against physical groups / labels
   and resolved later.
5. **Mesh the model** — sizing, then `g.mesh.generation.generate(dim=N)`.
6. **Extract the FEM broker** — `fem = g.mesh.queries.get_fem_data(dim=N)`.
   The broker is an immutable snapshot of nodes, elements, and all
   resolved attributes. It is what you feed to OpenSees.

Stages 3 and 4 are slightly interchangeable — you can declare a load
on a physical group as soon as the group exists. The mesh does not
need to exist yet; apeGmsh only resolves the declaration to concrete
node IDs when you call `get_fem_data`.

### 2.2  Four geometry avenues

apeGmsh gives you four ways to put geometry into a session. They are
not alternatives — a real model usually mixes them. The useful
question is *which one should each part of my model come from?*

**Primitives — `g.model.geometry`.** Built-in shapes from the OCC
kernel: `add_box`, `add_cylinder`, `add_sphere`, `add_cone`,
`add_torus`, `add_wedge`, `add_rectangle`, plus lower-dim
(`add_point`, `add_line`, `add_arc`, `add_circle`, `add_spline`,
`add_bspline`, `add_plane_surface`, …). Good for regular geometry,
test models, footings, idealised columns.

```python
with apeGmsh(model_name="demo") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)        # x,y,z, dx,dy,dz
    g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 3, r=0.3)
```

**CAD import — `g.model.io`.** For real engineering geometry —
anything that already lives in SolidWorks, CATIA, Inventor, FreeCAD.

```python
g.model.io.load_step("foundation.step", label="foundation")
g.model.io.heal_shapes()     # fix tiny gaps / sliver faces
```

Catch: CAD files often come in with per-vertex `lc` hints that hijack
your mesh. See the meshing lesson (`set_size_sources(from_points=False)`).

**Parts — reusable sub-geometries.** A `Part` is a self-contained
geometry you build once and import many times. Parts own their own
Gmsh session, persist to STEP + a sidecar JSON, and carry their
labels with them across the STEP round-trip.

```python
from apeGmsh import apeGmsh, Part

col = Part("column")
with col:
    col.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3, label="shaft")

with apeGmsh(model_name="frame") as g:
    g.parts.add(col, label="col_A", translate=(0, 0, 0))
    g.parts.add(col, label="col_B", translate=(5, 0, 0))
    g.parts.fragment_all()    # make touching bodies conformal
```

Use Parts when the same sub-geometry appears more than once, or when
you want to build a library of pre-verified components.

**Parametric sections — `g.sections`.** Structural-steel shorthand.
Conceptually, **a section is a preset Part**: one call runs a
pre-written builder that deposits a correctly-proportioned member
with **pre-labelled sub-regions** (flanges, web, end faces) and
registers it through the same `Instance` mechanism Parts use. The
difference is that you do not write the geometry — the section knows
what a W-shape is.

```python
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col", lc=50,
)
# Labels created automatically:
#   col.top_flange, col.web, col.bottom_flange
#   col.start_face, col.end_face
```

Available today: `W_solid`, `rect_solid`, `W_shell`. Reach for a
section when the geometry is a standard structural shape; reach for a
Part when it is anything custom that you still want to reuse.

### 2.3  How the four avenues fit together

Primitives, CAD, sections, and Parts all deposit OCC solids (or
surfaces) into the same model. Once they are there, **boolean
operations** glue them into one assembly:

- `g.model.boolean.fragment(...)` — most common. Splits overlapping
  or touching bodies at their shared interfaces so they mesh
  conformally.
- `g.model.boolean.fuse`, `cut`, `intersect` — standard CSG when you
  actually want to merge or subtract material.

That is why step 2 of the workflow is "boolean operations" — it is
what turns *a pile of shapes* into *one meshable assembly*.

---

## Lesson 3 — Parts, Instances, and the session-as-assembly

apeGmsh borrows the **Abaqus mental model** and the source calls it
out explicitly:

> A **Part** is geometry only. An **assembly** imports Parts,
> positions them, fragments them, names them, meshes them, and feeds
> the solver.

In apeGmsh the session *is* the assembly. When you open
`with apeGmsh(...) as g:`, you are creating the assembly. Everything
you put into it — inline geometry, imported CAD, sections, external
Parts — lands inside that one assembly.

### 3.1  Part vs. Instance

These two words are often used interchangeably in CAD tools, but in
apeGmsh they have sharp meanings:

- **Part** — a template. Lives *outside* the session (its own Gmsh
  state, its own STEP file, its own sidecar JSON). It is re-usable
  across sessions and projects.
- **Instance** — a placement. Lives *inside* the session, in the
  parts registry (`g.parts._instances`). It is the assembly-level
  expression of some chunk of geometry.

Every placement, no matter where the geometry came from, produces
an **Instance**. Some Instances have a Part behind them; some do
not. They all follow the same rules inside the session.

The cleanest way to say it: **Part = template (outside). Instance =
placement (inside).**

### 3.2  Four entry points — all produce Instances

| Entry point | Where the geometry comes from |
|---|---|
| `g.model.geometry.add_*` + `g.parts.register(name, ...)` | Built inline in the session, then promoted |
| `g.model.io.load_step(...)` + `g.parts.from_model(name)` | Untracked entities already in the session (STEP/IGES import) |
| `g.parts.add(part_object)` | An external `Part` — its own Gmsh session, own STEP on disk |
| `g.sections.W_solid(...)`, `rect_solid`, `W_shell` | A pre-written builder — a "preset Part" |

`register` (since **v1.0.2**) accepts three mutually-exclusive ways
to point at the geometry, so you rarely need to hand-build dimtag
lists:

```python
# Raw dimtags (positional — backwards compatible)
g.parts.register("column", [(3, t_col)])

# Resolve from an apeGmsh label (Tier 1 naming)
g.parts.register("column", label="shaft")              # unambiguous dim
g.parts.register("column", label="shaft", dim=3)       # when a label spans dims

# Resolve from a physical group (Tier 2 naming)
g.parts.register("column", pg="Col_A")
```

Exactly one of `dimtags` / `label=` / `pg=` must be supplied; passing
zero or more than one raises `TypeError`.

All four deposit their result into the same `_instances` dictionary.
That is why the rest of the library only has to reason about
Instances — Parts are an (optional) way to *source* an Instance, not
a parallel concept.

### 3.3  The exclusive-ownership rule

An entity belongs to **at most one Instance**. The registry rejects
overlapping assignments at `register()` time. You can still have
entities that belong to no Instance — those are just raw geometry —
but you cannot have one entity in two Instances.

This is why the session is an assembly of *disjoint* Instances, not
overlapping ones. If you need overlapping named groupings — for
example, one name for "all steel" and another for "just the
columns" — use **labels** or **physical groups** instead. Those
allow many names on one entity.

### 3.4  Ground truth under the abstraction

All geometry flows through the **OpenCASCADE (OCC) kernel** no
matter which entry point created it. And everything bottoms out at
`(dim, tag)` pairs because that is the currency Gmsh itself speaks.

apeGmsh's naming system (Instances, labels, physical groups) is a
resolver layer *on top* of dimtags — never a replacement. At any
point you can fall through to raw `(dim, tag)` pairs and call Gmsh
directly. This is a deliberate design choice: the abstraction is
there to be ergonomic, not to lock you in.

---

## Lesson 4 — Tags, labels, and physical groups

apeGmsh gives you **three ways** to refer to an entity. They are not
alternatives — they are three layers stacked on top of each other.

### 4.1  The three layers at a glance

| | **Tag** | **Label** | **Physical group** |
|---|---|---|---|
| What it is | `int` assigned by the OCC kernel | `str` managed by apeGmsh | `str` registered with Gmsh |
| Example | `5` (volume #5) | `"shaft"`, `"col.top_flange"` | `"Body"`, `"Base"` |
| Paired with dim? | Yes — `(dim, tag)` | No — the label knows its own dim(s) | Yes, rigidly |
| Who sees it | Gmsh, apeGmsh internals | apeGmsh only | Gmsh **and the solver** |
| Survives booleans | No — OCC renumbers | Yes (centre-of-mass matching) | Yes (native Gmsh) |
| Can nest? | N/A | Yes — dotted names like `"col.web"` | No |
| Can overlap? | N/A | Yes — many labels on one entity | No (one entity = one PG per dim) |
| Created by | Every `g.model.geometry.add_*` call | `add_*(..., label="...")` or `g.labels.add(...)` | `g.physical.add_volume(...)` etc. |

### 4.2  The mental rule

> **Tag for math. Label for talking. Physical group for the solver.**

- **Tag** — the ground truth. Rarely held in a variable, because
  booleans invalidate it. Grab it, hand it to a label or PG, forget
  the integer.
- **Label** — your working vocabulary while building the model.
  Cheap, expressive, nestable, overlap-friendly. Make as many as
  you need.
- **Physical group** — your contract with the solver. When you write
  `g.opensees.elements.fix("Base", dofs=[1,1,1])`, only a physical
  group named `"Base"` makes that work. Create one deliberately,
  only when the solver needs to know.

### 4.3  The typical promotion pattern

```python
with apeGmsh(model_name="col") as g:
    # Label the entity while building — the tag is returned but we
    # don't hold onto it.
    g.model.geometry.add_box(0, 0, 0, 0.3, 0.3, 3, label="shaft")

    # Promote the label to a PG when the solver needs to see it.
    g.physical.from_label("shaft", name="Body")
```

`g.physical.from_label(...)` is the clean one-step shortcut. The
two-line equivalent — `entities = g.labels.entities("shaft", dim=3);
g.physical.add_volume(entities, name="Body")` — works too, but
`from_label` is idiomatic.

### 4.4  Why two tiers, not one?

Three reasons labels and physical groups are different concepts:

1. **Physical groups pollute the solver model.** Every PG becomes a
   named set that the solver sees. If every construction label
   became a PG, an OpenSees model might carry 80 meaningless
   names.
2. **Physical groups cannot nest.** Gmsh does not interpret the dot
   in `"col.top_flange"` — that's pure apeGmsh. Labels give you
   hierarchical naming for free, which matters once Parts get
   instantiated multiple times (`"col_A.top_flange"` vs
   `"col_B.top_flange"`).
3. **Physical groups cannot overlap.** One entity belongs to at most
   one PG *per dimension*. Labels let you simultaneously call a
   volume `"steel"`, `"column"`, and `"col_A"` — and query any of
   the three.

### 4.5  The dimension asymmetry (important)

Labels and physical groups handle dimensions very differently. This
trips people up, so it is worth calling out explicitly.

**A label is one string that can span multiple dimensions at
once.** You disambiguate at query time:

```python
# "col.end_face" may exist as both the surface (dim=2) and the
# curves on its perimeter (dim=1) — labels don't mind.
faces = g.labels.entities("col.end_face", dim=2)
edges = g.labels.entities("col.end_face", dim=1)
```

If you ask `g.labels.entities("col.end_face")` with no `dim=` and
the label lives at more than one dim, apeGmsh raises `ValueError`
asking you to be specific.

**A physical group is rigidly keyed by `(dim, pg_tag)`.** Gmsh's
own API forces the dim into the signature
(`gmsh.model.addPhysicalGroup(dim, tags)`). You *can* reuse the
same human-readable **name string** at two different dims:

```python
g.physical.add_volume(vol_tags,   name="col_A")  # (dim=3, name="col_A")
g.physical.add_surface(face_tags, name="col_A")  # (dim=2, name="col_A")
```

…but these are **two separate physical groups** that happen to
share a name. The solver sees two entries. `g.physical.add` upserts
on `(dim, name)`, not on `name` alone.

**Why Gmsh designed it this way.** The solver always needs to know
*what dim*: "fix DOFs on `Base`" only makes sense if `Base` is a
surface of nodes — not a volume and a curve simultaneously. Forcing
dim into PG identity matches how downstream solvers consume the
data. One group = one kind of thing to do.

### 4.6  Querying — labels vs physical groups

The query APIs are almost parallel, with one extra method on the PG
side to handle the multi-dim case:

| Operation | Labels | Physical groups |
|---|---|---|
| Entity tags at a specific dim | `g.labels.entities("shaft", dim=3)` | `g.physical.entities("Body", dim=3)` |
| Entity tags with inferred dim | `g.labels.entities("shaft")` *(raises if multi-dim)* | `g.physical.entities("Body")` *(raises if multi-dim)* |
| `(dim, tag)` pairs across all dims | *n/a — pick a dim* | `g.physical.dim_tags("Body")` |
| List all names | `g.labels.all_names()` | `g.physical.get_all(dim=-1)` |
| Reverse lookup: names for an entity | `g.labels.names_for(dim, tag)` | `g.physical.get_groups_for_entity(dim, tag)` |
| Summary table | — | `g.physical.summary()` *(returns a DataFrame)* |

Concrete multi-dim example:

```python
g.physical.add_volume(vol_tags,   name="col_A")
g.physical.add_surface(face_tags, name="col_A")

g.physical.entities("col_A", dim=3)  # [vol tags]
g.physical.entities("col_A", dim=2)  # [face tags]
g.physical.entities("col_A")         # raises ValueError — multi-dim
g.physical.dim_tags("col_A")         # [(3, v1), (2, f1), (2, f2), ...]
```

That last call is the honest cross-dim query — a flat list of
`(dim, tag)` pairs that can be handed straight to any Gmsh API or
to `g.parts.register(name, dimtags=...)`.

---

## Lesson 5 — Querying geometric entities

Lesson 4 covered the naming layer. But names are only one of
several ways to ask the model a question. On the **geometry side**
(pre-mesh), apeGmsh gives you three kinds of queries:

| Kind of question | Where to ask | Example |
|---|---|---|
| What is named *X*? | `g.labels`, `g.physical` | `g.labels.entities("shaft", dim=3)` |
| What is *near* or *touches* *X*? (topology) | `g.model.queries` | `g.model.queries.boundary([(3, t)])` |
| What is inside this region in space? (spatial) | `g.model.queries` | `g.model.queries.entities_in_bounding_box(...)` |

Post-mesh queries (nodes, elements, the FEM broker) are a separate
tier covered in a later lesson, once meshing is introduced.

### 5.1  Name queries

Fast, deliberate, human-readable — but only available for entities
you named up front. Covered in Lesson 4.

### 5.2  Topological queries — `g.model.queries`

When you don't have a name, but you know the shape of the
relationship. Pre-mesh, on OCC geometry:

```python
# Downward: the surfaces that bound a volume
g.model.queries.boundary([(3, vol_tag)])     # → [(2, s1), (2, s2), ...]

# Upward + downward adjacencies
g.model.queries.adjacencies(3, vol_tag)

# Physical measures
g.model.queries.bounding_box(3, vol_tag)
g.model.queries.center_of_mass(3, vol_tag)
g.model.queries.mass(3, vol_tag)             # volume / area / length
```

`boundary` is the one you reach for most. "Give me the faces of
this solid" is a constant need when wiring up BCs and loads.

### 5.3  Spatial queries — also `g.model.queries`

When you know *where* something is but not what it is. Classic use:
"grab every surface at z=0 so I can fix them."

```python
base_faces = g.model.queries.entities_in_bounding_box(
    xmin=-10, ymin=-10, zmin=-1e-3,
    xmax= 10, ymax= 10, zmax= 1e-3,
    dim=2,
)
# → list of (2, tag) pairs for surfaces inside that thin slab
```

A thin-slab bounding box ± a tolerance is the idiom for "everything
on this plane." Together with a physical group it becomes the
standard way to define boundary surfaces without knowing tags:

```python
base_faces = g.model.queries.entities_in_bounding_box(
    -10, -10, -1e-3, 10, 10, 1e-3, dim=2,
)
g.physical.add_surface([t for _, t in base_faces], name="Base")
```

`g.model.queries.registry` returns a DataFrame of every entity in
the model (dim, tag, bbox, label, PG) — the cleanest way to eyeball
what you actually have at any point.

`entities_in_bounding_box` is the low-level spatial primitive. For
composable filters (in a box *and* with a label *and* horizontal),
reach for the `g.model.selection` composite — see Lesson 6.

### 5.4  The unifying idea

Names (labels, physical groups) are really **cached queries**.
They are a way of saying *"remember this set for me now, so I do
not have to re-derive it by bounding box or by adjacency every
time."*

That is why the three styles chain together — you **derive** a set
with topology or spatial logic, then **cache** it behind a name for
the rest of the pipeline:

```python
# 1. Derive with a spatial query.
base_faces = g.model.queries.entities_in_bounding_box(
    -10, -10, -1e-3, 10, 10, 1e-3, dim=2,
)
# 2. Cache with a physical group — the name is now a stable
#    handle for everything downstream.
g.physical.add_surface(
    [t for _, t in base_faces], name="Base",
)
```

Later lessons will pick up the third step — *consume* — where the
cached name pulls mesh nodes and elements from the FEM broker once
the mesh exists.

---

## Lesson 6 — Selection: filters, set algebra, conversion

`g.model.queries.entities_in_bounding_box` answers one very
specific question. `g.model.selection` wraps that — and a dozen
other filters — behind a single fluent API. Whenever you find
yourself composing bbox / label / size checks by hand, reach for
the selection composite instead.

### 6.1  Entry points

```python
g.model.selection.select_points(*, <filters>)    # dim=0
g.model.selection.select_curves(*, <filters>)    # dim=1
g.model.selection.select_surfaces(*, <filters>)  # dim=2
g.model.selection.select_volumes(*, <filters>)   # dim=3
g.model.selection.select_all(dim=-1, *, <filters>)
```

Since **v1.0.3** every filter is an **explicit, keyword-only named
parameter** on each method — not an opaque `**kwargs`. IDE
autocomplete, `help()`, and static type checkers all see the
complete list. The same filter set is accepted by
`Selection.filter(...)` for refining an existing Selection.

All five methods return a **`Selection` object** — a rich wrapper
around a list of dimtags.

### 6.2  The filter families (all AND-combined)

Three families; mix as many as you want. Every filter defaults to
`None` (inactive). Combine them by passing any subset as keyword
arguments.

**Identity — "who are you?"**

```python
tags         = [5, 7, 9]          # keep only these tags
exclude_tags = [3]
labels       = "col_*"            # fnmatch pattern
labels       = ["steel", "col_*"]
physical     = "Body"             # members of this PG
kinds        = "box"              # registered OCC primitive kind
```

**Spatial — "where are you?"**

```python
in_box    = (x0, y0, z0, x1, y1, z1)
in_sphere = (cx, cy, cz, r)            # centroid within radius
on_plane  = ("z", 0.0, 1e-3)           # bbox intersects z=0.0 within tol
on_axis   = ("z", 1e-3)                # centroid lies on the z axis
at_point  = (x, y, z, tol)             # bbox contains (x, y, z) within tol
```

**Metric / orientation — "how big, which way?"** *(dim-specific;
silently skipped on other dims)*

```python
length_range = (0.1, 2.0)         # curves only (dim=1)
area_range   = (0.0, 10.0)        # surfaces only (dim=2)
volume_range = (1e-3, None)       # volumes only (dim=3)
aligned      = ("z", 5.0)         # curves within N degrees of the axis
horizontal   = True               # curves perpendicular to z
vertical     = True               # curves parallel to z
predicate    = lambda d, t: ...   # escape hatch for anything custom
```

### 6.3  Set algebra on Selections

Once you have Selections, combine them like sets:

```python
base     = g.model.selection.select_surfaces(on_plane=("z", 0.0, 1e-3))
top      = g.model.selection.select_surfaces(on_plane=("z", 3.0, 1e-3))
all_ends = base | top                     # union
sides    = g.model.selection.select_surfaces() - all_ends  # difference
shared   = base & top                     # intersection
only_one = base ^ top                     # symmetric difference
```

Plus `.filter(...)` to narrow further, `.limit(n)`,
`.sorted_by("area")`, etc.

### 6.4  Topology helpers on the composite

Selection-native versions of the walk-the-graph queries from
Lesson 5:

```python
# Faces of a Selection of volumes
faces = g.model.selection.boundary_of(my_volumes)

# Surfaces that touch these curves
surfaces = g.model.selection.adjacent_to(my_curves, dim_target=2)

# Nearest entities to a point
hit = g.model.selection.closest_to(0.0, 0.0, 3.0, dim=2, n=4)
```

### 6.5  Converting a Selection — the payoff

A Selection is where the **derive → cache** chain shines:

```python
base = g.model.selection.select_surfaces(on_plane=("z", 0.0, 1e-3))

base.dimtags               # [(2, t1), (2, t2), ...]
base.tags                  # (t1, t2, ...)
base.to_physical("Base")   # ← one-step promotion to a PG
base.to_dataframe()        # inspection table
base.bbox                  # aggregate bounding box
base.centers               # ndarray of centroids
```

`to_physical(name)` collapses the three-step chain from Lesson 5
into a single fluent chain:

```python
# Before — raw dimtag round-trip
base_faces = g.model.queries.entities_in_bounding_box(
    -10, -10, -1e-3, 10, 10, 1e-3, dim=2,
)
g.physical.add_surface([t for _, t in base_faces], name="Base")

# After — one declarative line
(g.model.selection
    .select_surfaces(on_plane=("z", 0.0, 1e-3))
    .to_physical("Base"))
```

### 6.6  `g.model.queries` vs `g.model.selection`

They overlap deliberately. Rough guide:

- **`g.model.queries`** — a single specific lookup; you want raw
  dimtags back. Good for one-off programmatic use.
- **`g.model.selection`** — composing filters, combining sets, or
  building something you will hand to a PG. Good for the
  declarative *"all horizontal surfaces at z=0 that are not on the
  top slab"* style of code.

---

## Lesson 7 — Boolean operations

Boolean operations combine or split solids. This is where raw
geometry becomes a meshable assembly. apeGmsh exposes them at
**two levels**, and knowing which to reach for is the whole lesson.

### 7.1  The four operations

Under the hood, every boolean is one of four OCC primitives:

| Operation | What it does | When to use |
|---|---|---|
| **`fuse(A, B)`** | A ∪ B — merge into one body; interface disappears | You want a monolithic part |
| **`cut(A, B)`** | A − B — subtract the tool from the object | Holes, chamfers, notches |
| **`intersect(A, B)`** | A ∩ B — keep only the overlap | Extracting the common volume |
| **`fragment(A, B)`** | Split everything at intersections; **keep all sub-pieces** | Conformal assemblies |

The distinction that matters most for FEM work is **fuse vs
fragment**:

```
  Before              fuse(A, B)            fragment(A, B)
┌─────┐┌─────┐       ┌───────────┐       ┌─────┬─────┐
│  A  ││  B  │  →    │   A ∪ B   │  or   │  A  │  B  │
└─────┘└─────┘       └───────────┘       └─────┴─────┘
 two bodies,         one body,            two bodies,
 touching            interface gone       sharing a face
```

`fuse` collapses the interface — there is no longer an A/B
distinction. `fragment` keeps the interface as a **shared face**:
two volumes, one boundary entity owned by both. The shared face is
what lets the mesh be continuous across the seam — a *conformal*
assembly.

### 7.2  Two levels — same OCC calls, different ergonomics

apeGmsh gives you both a low-level and a Parts-aware API. They
call the same `gmsh.model.occ.*` primitive underneath:

```python
# Low-level — takes dimtags, tags, labels, or PG names (since v1.0.4)
g.model.boolean.fragment([(3, t_slab)], [(3, t_col)])
g.model.boolean.fragment("slab", "col")       # resolved via g.labels / g.physical

# Parts level — Instance labels, plus extra bookkeeping
g.parts.fragment_pair("slab", "col")
g.parts.fragment_all()                # everything vs everything
g.parts.fuse_group(["fl_a", "fl_b", "web"], label="I_beam")
```

Since **v1.0.4**, both levels are equally safe: every boolean —
low-level *or* Parts-level — walks the OCC result map and updates
the Instance registry via the shared `_remap_from_result` helper.
The Instance cache cannot go stale under a boolean regardless of
which entry point you use.

The Parts-level methods still earn their keep by adding:

1. **Higher-level semantics** — `fragment_all` (everything vs
   everything), `fragment_pair` (just two instances),
   `fuse_group` (merge listed instances into one surviving
   Instance and drop the others from the registry).
2. **Orphan warnings** — `fragment_all` tells you when an
   untracked entity participates, so you know your Parts set is
   incomplete.
3. **Name discoverability** — autocompletion on `g.parts.*`
   shows the Instance-focused operations together.

So the choice between levels is ergonomic, not a safety question.

### 7.3  Which to reach for

| You have… | Use… |
|---|---|
| Raw inline geometry, no Parts | `g.model.boolean.fuse/cut/intersect/fragment` |
| Parts or sections and you want instance-level semantics (e.g. "merge these five into one survivor") | `g.parts.fragment_all` / `fragment_pair` / `fuse_group` |
| A mix | Either works — adopt with `g.parts.from_model(...)` first if you want the orphans tracked |

Both `g.model.boolean.*` and `g.parts.*` accept label names and
physical-group names since v1.0.4, so the naming layer you built
up in Lessons 3–4 works at either level.

### 7.4  What survives a boolean

| Thing | Survives? | Why |
|---|---|---|
| **Tags** | ❌ | OCC renumbers |
| **Labels** | ✅ | Stored as hidden PGs; `pg_preserved()` walks the result map |
| **Physical groups** | ✅ | Same mechanism — Gmsh-native |
| **Instance.entities cache** | ✅ (since v1.0.4) | `_remap_from_result` runs on both entry points |
| **Instance.bbox cache** | ✅ (since v1.0.4) | Same helper |

Labels survive because apeGmsh stores them as hidden physical
groups prefixed `_label:`. Every boolean is wrapped in
`pg_preserved()`, which asks Gmsh *"what was in this PG before?"*,
runs the boolean, and re-adds the equivalent post-boolean entities
to the same PG. So `g.labels.entities("shaft", dim=3)` re-resolves
correctly even after renumbering. The Instance cache used to be
the odd one out — not any more. See `plan_instance_computed_view.md`
under *Planning* for the deferred Plan B alternative that would
remove the cache altogether.

### 7.5  A worked example — slab + column, conformal

```python
with apeGmsh(model_name="frame") as g:
    # Build two separate solids.
    g.model.geometry.add_box(0, 0, 0, 5, 5, 0.3, label="slab")
    g.model.geometry.add_box(2, 2, 0.3, 1, 1, 3,   label="col")

    # Promote both to Instances (the labels become their identity).
    g.parts.register("slab", label="slab")
    g.parts.register("col",  label="col")

    # Make the interface shared.
    g.parts.fragment_all()

    # Labels still resolve — the column sits on a face shared with the slab.
    slab_tags = g.labels.entities("slab", dim=3)
    col_tags  = g.labels.entities("col",  dim=3)

    # Promote to PGs for the solver.
    g.physical.from_label("slab", name="Slab")
    g.physical.from_label("col",  name="Col")

    # The mesh will be continuous across the interface.
    g.mesh.sizing.set_global_size(0.2)
    g.mesh.generation.generate(dim=3)
```

If we used `fuse_group(["slab", "col"])` instead, slab and column
would merge into one volume — fine if they are the same material,
wrong if you wanted separate materials for each.

### 7.6  Pitfalls

- **Run `remove_duplicates` first on imported CAD.** STEP imports
  often have numerically-close but distinct vertices at touching
  faces. Call `g.model.queries.remove_duplicates(tolerance=...)`
  before fragmentation or you'll get microscopic sliver volumes.
  Tolerance is unit-dependent: mm → `1e-3`, metres → `1e-6`.
- **Don't hold raw tags past a boolean.** `t = add_box(...)`
  followed by a `fragment` leaves `t` pointing at nothing. Use
  labels, PGs, or re-query.
- **`fragment_all` warns about untracked entities.** If you see
  the warning, either adopt them with `g.parts.from_model()`
  first, or accept that they'll participate in fragmentation but
  not be tracked in the registry.
- **`fragment(cleanup_free=True)` is the default.** It drops
  dim-2 surfaces that don't bound a volume — useful for killing
  remnants of a cutting plane that extended past your solid. In
  pure 2D models it auto-skips (otherwise it would kill every
  surface in sight).

---

## Lesson 8 — CAD import: principles and pitfalls

CAD import gets its own lesson because the **syntax** is trivial
(`g.model.io.load_step(...)`) and the **discipline** is not.
Exporters introduce geometric imprecision, duplicated topology,
and hidden mesh-size hints that will bite you if you skip the
cleanup. Every post-import step is about cleaning up somebody
else's geometry before you can mesh it.

### 8.1  The mental model

Think of STEP / IGES as **faxed geometry**: the shape is right,
but the metadata is noisy. Shared vertices from adjacent surfaces
end up as two numerically-close-but-distinct points. Small
chamfer edges become degenerate. Closed shells ship open. Every
BRep point carries a tiny `lc` value the CAD tool used for its
own display sampling. You inherit all of it.

Three things you *almost always* do after import, plus one
sizing gotcha:

1. `remove_duplicates` — merge coincident entities
2. `heal_shapes` — fix degenerate edges / tiny faces / unsewn
   shells (optional but common)
3. `make_conformal` — fragment everything against itself so
   touching bodies share faces
4. `set_size_sources(from_points=False)` — ignore the exporter's
   per-point `lc` hints

### 8.2  Loading — `g.model.io.load_step` / `load_iges`

```python
imported = g.model.io.load_step(
    "bracket.step",
    highest_dim_only=True,   # default — return only top-dim entities
)
# imported → {3: [5, 6, 7]}   (one dict key per dimension present)

volumes = imported[3]
```

- **`highest_dim_only=True`** — the default. Returns volumes for
  solid models, surfaces for surface models. Use this unless you
  need sub-entities.
- **`highest_dim_only=False`** — returns every dim. Needed for
  wireframe frames (1D models) where the entities of interest
  are curves.

Sister methods: `save_step`, `save_iges`, `load_dxf`, `save_dxf`.
**Prefer STEP** — modern spec, preserves exact NURBS + tolerances.
IGES is legacy and routinely leaves small gaps at surface
junctions that you'll have to clean up.

### 8.3  `remove_duplicates` — merge coincident topology

CAD exporters write the shared vertex between two touching faces
**twice** — once per face. `removeAllDuplicates` walks every
dimension (points → curves → surfaces → volumes) and collapses
entities that are geometrically identical within a tolerance.

```python
g.model.queries.remove_duplicates(tolerance=1e-3)   # mm-scale model
```

**Tolerance is unit-dependent** — the single most common thing
to get wrong:

| Units | Typical tolerance |
|---|---|
| mm | `1e-3` (= 1 µm) |
| metres | `1e-6` |
| inches | `1e-5` |

Pick something at least two orders of magnitude finer than any
feature you care about, and at least an order of magnitude
coarser than the exporter's floating-point noise. If
fragmentation later produces microscopic sliver volumes, your
tolerance was too tight. If features disappear, too loose.

### 8.4  `heal_shapes` — fix degenerate and unsewn geometry

When `remove_duplicates` isn't enough — tiny faces, degenerate
edges, open shells that should be solids — reach for
`heal_shapes`:

```python
g.model.io.heal_shapes(
    tolerance=1e-3,
    fix_degenerated=True,
    fix_small_edges=True,
    fix_small_faces=True,
    sew_faces=True,       # reconnect open shells at shared edges
    make_solids=True,     # close healed shells into solids
)
```

You don't always need this. Reach for it when:

- You see unmeshed surfaces where you expected volumes (open
  shells).
- Gmsh warns about degenerate edges during meshing.
- The model is from an older IGES file.
- You're fragmenting and getting sliver volumes even with a
  generous `remove_duplicates` tolerance.

### 8.5  `make_conformal` — share faces between touching bodies

The crucial step. After `remove_duplicates`, adjacent bodies
still occupy overlapping space without any shared topology —
they're two bodies that happen to touch, not an assembly.
`make_conformal` fragments everything against everything so
every interface becomes a shared face.

```python
g.model.queries.make_conformal(
    dims=[1, 2, 3],      # default: all non-empty dims
    tolerance=1.0,       # ← deliberately LOOSE for CAD junctions
)
```

**The tolerance trap.** `make_conformal` and `remove_duplicates`
both take `tolerance`, but they're checking different things:

| Call | Checks | Tolerance scale |
|---|---|---|
| `remove_duplicates` | "Are these two points numerically the same?" | **Tight** — e.g. `1e-3` mm |
| `make_conformal` | "Do these two curves touch vs miss by a gap?" | **Loose** — e.g. `1.0` mm |

Exporters can leave visible-to-the-eye gaps at beam–column
joints (not noise, actual gaps from how the CAD tool modelled
the joint). The fragment needs tolerance wide enough to span
those gaps and treat them as intersections. If you pass the same
tight tolerance you used for `remove_duplicates`, the fragment
misses the joints and your assembly stays disconnected.

### 8.6  The sizing gotcha — `set_size_sources(from_points=False)`

CAD exporters bake per-vertex characteristic lengths into every
BRep point. Gmsh, helpfully, consults those during meshing. The
result:

```python
g.model.io.load_step("bracket.step")
g.mesh.sizing.set_global_size(5.0)       # sets Mesh.MeshSizeMax = 5.0
g.mesh.generation.generate(dim=3)        # produces tiny elements anyway  ❌
```

Because Gmsh takes the **minimum** of all size sources at each
node, and every imported point carries an `lc` from the exporter
that's smaller than your global. Your global is a ceiling that
was never hit.

Fix: tell Gmsh to ignore per-point sources before you set the
global size:

```python
g.mesh.sizing.set_size_sources(
    from_points=False,        # ← disables the exporter's per-point lc
    from_curvature=False,     # also turns off adaptive curvature refinement
    extend_from_boundary=False,
)
g.mesh.sizing.set_global_size(5.0)       # now actually governs  ✅
```

Universal when importing CAD. A good habit: call
`set_size_sources(from_points=False)` immediately after loading,
before any sizing.

### 8.7  Adopting the import as a Part

Once the geometry is clean, register it as an Instance so
downstream code can refer to it by name (see Lesson 3):

```python
g.model.io.load_step("bracket.step")
# ...cleanup...
g.parts.from_model("bracket")            # everything untracked → one Instance
```

Or be selective — pass `dim=` and/or `tags=` to pick specific
imported entities as the Instance.

### 8.8  The canonical CAD-import prelude

Six lines that you'll write the same way nearly every time:

```python
with apeGmsh(model_name="bracket") as g:
    # 1. Disable the exporter's per-point sizing before anything else.
    g.mesh.sizing.set_size_sources(from_points=False)

    # 2. Load.
    g.model.io.load_step("bracket.step")

    # 3. Clean topology (tight tolerance for coincidence).
    g.model.queries.remove_duplicates(tolerance=1e-3)

    # 4. Heal if needed (skip if the import was clean).
    # g.model.io.heal_shapes(tolerance=1e-3)

    # 5. Share interfaces (loose tolerance for CAD joints).
    g.model.queries.make_conformal(tolerance=1.0)

    # 6. Track as a Part so labels and constraints can reference it.
    g.parts.from_model("bracket")

    # Now mesh.
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
```

Skip any of steps 1, 3, 5 at your peril.

---

## Lesson 9 — Managing Parts, labels, and physical groups

You've seen how to *create* each of the three naming layers. Now
the management surface — listing, checking, renaming, deleting —
and the one rule that's consistent across all three.

### 9.1  The universal rule

> **Removing a name never removes geometry.**

Every "delete" method on Parts, labels, and PGs removes only the
*naming*. The underlying entities stay in the Gmsh session; they
just become orphans (with respect to that naming layer). Geometry
removal is a separate concern handled by
`g.model.queries.remove(...)`.

Cascade behaviour:

- `g.parts.delete("col")` → Instance gone, entities untouched,
  show up as "Untracked" in the viewer.
- `g.labels.remove("shaft")` → label PG gone, entities
  untouched.
- `g.physical.remove_name("Base")` → PG gone, entities untouched.

If you want the geometry gone too, you call the geometry-removal
method explicitly after.

### 9.2  Parts / Instances

```python
# Create        (Lesson 3)
g.parts.register("col", label="shaft")
g.parts.from_model("bracket")
g.parts.add(part_obj, label="col_A")
g.sections.W_solid(..., label="beam_1")

# List & inspect
g.parts.labels()              # list[str] of Instance names
g.parts.instances             # dict[str, Instance]
"col" in g.parts.instances    # existence check

# Get one
inst = g.parts.get("col")
inst.entities                 # {dim: [tag, ...]}

# Rename
g.parts.rename("col", "column_A")
# KeyError if "col" missing; ValueError if "column_A" already exists.

# Delete
g.parts.delete("column_A")
# Entities become orphan — next from_model() sweeps them up again.
```

The `labels()` method name is slightly confusing — these are
**Instance names**, not Tier-1 labels. Same word, different
concept. (The parts registry was written before the Tier-1
label system fully crystallised.)

### 9.3  Labels (Tier 1)

```python
# Create
g.model.geometry.add_box(..., label="shaft")         # at creation time
g.labels.add(3, [5, 7], name="steel_volumes")        # after the fact

# List & inspect
g.labels.get_all(dim=-1)      # list[str]
g.labels.summary()            # pandas DataFrame
g.labels.has("shaft", dim=3)  # bool — no exception

# Get entities
g.labels.entities("shaft", dim=3)          # list[tag]
g.labels.entities("shaft")                 # raises if multi-dim

# Reverse lookup
g.labels.labels_for_entity(3, 5)           # list[str] — labels covering this entity
g.labels.reverse_map(dim=3)                # dict[DimTag, str] — bulk

# Rename
g.labels.rename("shaft", "column_body")
g.labels.rename("shaft", "column_body", dim=3)   # scope to one dim

# Remove
g.labels.remove("shaft")                # all dims where it exists
g.labels.remove("shaft", dim=3)         # one dim only
```

Labels are multi-dim by construction (Lesson 4.5). Both `rename`
and `remove` default to "operate at every dim where this name
exists" — pass `dim=` to scope.

### 9.4  Physical groups (Tier 2)

```python
# Create
g.physical.add_volume(tags, name="Body")
g.physical.from_label("shaft", name="Body")

# List & inspect
g.physical.get_all(dim=-1)               # list[(dim, tag)]
g.physical.summary()                     # DataFrame

# Exists? (returns tag or None)
pg_tag = g.physical.get_tag(3, "Body")

# Get entities
g.physical.entities("Body", dim=3)       # list[tag]
g.physical.dim_tags("Body")              # list[(dim, tag)] — cross-dim

# Reverse lookup
g.physical.get_groups_for_entity(3, 5)   # list[pg_tag]

# Rename — BY (dim, tag), not by old name
g.physical.set_name(3, pg_tag, "Slab")

# Remove
g.physical.remove_name("Body")           # by name
g.physical.remove([(3, pg_tag)])         # by (dim, tag)
g.physical.remove_all()                  # nuclear
```

The rename API is the big departure: `set_name(dim, tag,
new_name)` takes the **physical-group tag**, not the old name.
That's a consequence of the Gmsh data model — a PG is identified
by `(dim, pg_tag)`, and the name is a property on it. It also
means if the same name exists at two dims (two different PGs),
you rename each one separately.

### 9.5  Side-by-side management table

| Operation | Parts | Labels | Physical groups |
|---|---|---|---|
| Create | `register`, `from_model`, `add`, `sections.*` | `label="..."` kw, `g.labels.add(dim, tags, name=...)` | `g.physical.add_*(tags, name=...)`, `from_label` |
| List names | `g.parts.labels()` | `g.labels.get_all()` | `g.physical.get_all()` *(returns (dim,tag))* |
| Summary | `g.parts.instances` *(dict)* | `g.labels.summary()` | `g.physical.summary()` |
| Exists? | `"X" in g.parts.instances` | `g.labels.has("X")` | `g.physical.get_tag(dim, "X") is not None` |
| Get entities | `g.parts.get("X").entities` | `g.labels.entities("X", dim=)` | `g.physical.entities("X", dim=)` |
| Rename | `g.parts.rename("old", "new")` | `g.labels.rename("old", "new", dim=)` | `g.physical.set_name(dim, tag, "new")` |
| Delete one | `g.parts.delete("X")` | `g.labels.remove("X", dim=)` | `g.physical.remove_name("X")` |
| Delete all | — | — | `g.physical.remove_all()` |
| Reverse lookup | iterate `instances` | `g.labels.labels_for_entity(dim, tag)` | `g.physical.get_groups_for_entity(dim, tag)` |

### 9.6  Gotchas

- **Rename is shallow.** `g.parts.rename("col_A", "col_B")`
  changes the Instance's registry key. It does **not** rename
  labels like `col_A.shaft` → `col_B.shaft` — those stay. If you
  want that cascade, do it yourself in a loop over
  `g.labels.get_all()`.
- **Parts rename cannot cross an existing name.** If `"col_B"`
  already exists, you get `ValueError`. No merging. Delete the
  destination first if that's what you want.
- **PG rename is per-(dim, tag), not per-name.** If `"col_A"`
  exists as both a volume-PG and a surface-PG (same name,
  different dims — Lesson 4.5), renaming means two `set_name`
  calls.
- **Label remove without `dim=` hits every dim.** If you
  declared a label at dims 2 and 3 and only wanted to drop the
  dim-3 instance, pass `dim=3`. Otherwise both go.
- **Delete-then-re-sweep is a valid pattern.** `g.parts.delete("X")`
  followed by `g.parts.from_model("Y")` adopts everything that
  was in X into Y. Useful when you want a rename *with*
  re-gathering of content.

---

## Lesson 10 — Meshing basics

With clean, named geometry in hand, meshing is the next step. The
good news: the common path is short. The subtle bits (fields,
transfinite, partitioning) are deferred to the meshing deep-dive
(`guide_meshing.md`) — this lesson covers what you actually need
on day one.

### 10.1  The mesh pipeline

```
geometry  →  sizing  →  generate  →  (order, optimize, refine)  →  renumber  →  get_fem_data
             g.mesh.sizing            g.mesh.generation            g.mesh.partitioning  g.mesh.queries
```

Every step is a sub-composite call. The whole flow fits in about
half a dozen lines once you've written it a few times — see §10.10
for the canonical form.

### 10.2  Sizing — the bedrock

Gmsh picks an element size at every node by taking the **minimum**
of every active size source. So "sizing" is really about
**constraining the upper bound** — your global is a ceiling, not a
target.

The three levers:

```python
# 1. Global band
g.mesh.sizing.set_global_size(max_size, min_size=0.0)

# 2. Per-entity size
g.mesh.sizing.set_size([(3, vol_tag)], 10.0)

# 3. Per-physical-group size
g.mesh.sizing.set_size_by_physical("WeldArea", 2.0)
```

Which size sources Gmsh consults (recall Lesson 8.6):

```python
g.mesh.sizing.set_size_sources(
    from_points=True,         # per-BRep-point lc — disable for CAD imports
    from_curvature=False,     # adaptive refinement near curves
    extend_from_boundary=True # propagate boundary sizes into the interior
)
```

For inline geometry, the defaults are usually fine. For CAD
imports, **disable `from_points`**.

**Underlying Gmsh options.** Every sizing call ends up as a
`gmsh.option.setNumber(...)`. Good to know in case you're reading
the Gmsh reference or want to set an option apeGmsh doesn't wrap:

| apeGmsh call | Gmsh option |
|---|---|
| `set_global_size(max=, min=)` | `Mesh.MeshSizeMax`, `Mesh.MeshSizeMin` |
| `set_size_sources(from_points=…)` | `Mesh.MeshSizeFromPoints` |
| `set_size_sources(from_curvature=…)` | `Mesh.MeshSizeFromCurvature` |
| `set_size_sources(extend_from_boundary=…)` | `Mesh.MeshSizeExtendFromBoundary` |
| `set_size([(dim, tag)], s)` | `gmsh.model.mesh.setSize(...)` (not an option — per-entity API call) |

### 10.3  Generation — one call, dim matters

```python
g.mesh.generation.generate(dim=3)
```

The `dim` parameter says **what dimension of elements to
produce**:

| `dim=` | Produces | Typical use |
|---|---|---|
| 1 | Edge mesh only | Wireframe / 1D frame |
| 2 | 2D elements (tris, quads) | Shell models; surface check of a solid |
| 3 | 3D elements (tets, hexes) | Solid / bulk models |

Lower dims are meshed automatically as prerequisites.
`generate(dim=3)` implicitly meshes curves and surfaces first,
then the volume.

You can call `generate(dim=2)` on a 3D solid to get just its
surface mesh — handy for sanity-checking the geometry before
paying the cost of a 3D mesh.

### 10.4  Element order

Default is **linear**. Elevate to quadratic after generation:

```python
g.mesh.generation.generate(dim=3)
g.mesh.generation.set_order(2)     # must call AFTER generate
```

`set_order(2)` adds mid-edge nodes to existing elements in place.
Common values: 1 (linear), 2 (quadratic), 3 (cubic).

!!! warning "Curved geometry + coarse mesh"
    When you elevate to order ≥ 2 on a curved surface, the
    newly-added mid-edge nodes are projected onto the underlying
    OCC geometry. On a coarse mesh over a sharply curved face,
    this projection can **flip the element's Jacobian** —
    producing invalid elements that OpenSees will reject at
    assembly. Two fixes: refine the mesh before `set_order`, or
    run `g.mesh.generation.optimize("HighOrder")` after elevation
    to untangle the bad elements. See §10.6.

### 10.5  Algorithm choice

Default works most of the time. When you need to override:

```python
# Per-surface for dim=2
g.mesh.generation.set_algorithm("WebSurface", "frontal_delaunay_quads")

# Global for dim=3 (pass tag=0)
g.mesh.generation.set_algorithm(0, "hxt", dim=3)
```

`set_algorithm` takes labels and PG names for the `tag` arg, so
you can target specific regions without handling tags.

**2D algorithms — `Mesh.Algorithm`:**

| Name | Gmsh code | Notes |
|---|---:|---|
| `automatic` *(alias: `auto`, `default`)* | 2 | Gmsh picks per-surface. Default. |
| `mesh_adapt` *(alias: `meshadapt`)* | 1 | Adaptive; good for curved surfaces with non-uniform sizing. |
| `initial_mesh_only` | 3 | Stops after the initial mesh — no optimisation. |
| `delaunay` | 5 | Classic Delaunay. Robust, fast. |
| `frontal_delaunay` *(alias: `frontal`, `front`, `tri`)* | 6 | Higher quality triangles; slower than Delaunay. |
| `bamg` | 7 | Anisotropic remesher — needs a background size field. |
| `frontal_delaunay_quads` *(alias: `quad`, `quads`)* | 8 | Full-quad output. |
| `packing_parallelograms` *(alias: `pack`, `packing`)* | 9 | Quad-dominant packing. |
| `quasi_structured_quad` *(alias: `qsq`)* | 11 | Experimental structured-quad generator. |

**3D algorithms — `Mesh.Algorithm3D`:**

| Name | Gmsh code | Notes |
|---|---:|---|
| `hxt` *(alias: `auto`, `default`, `automatic`)* | 10 | Default. Modern, fast tet mesher; recommended for large models. |
| `delaunay` | 1 | Classic Delaunay tets. Stable fallback. |
| `initial_mesh_only` | 3 | Stops after initial mesh. |
| `frontal` | 4 | Advancing-front tets; better quality, slower. |
| `mmg3d` *(alias: `mmg`)* | 7 | Remesher — needs a valid input mesh. |
| `r_tree` *(alias: `rtree`)* | 9 | Legacy R-tree based; rarely used. |

**Optimisation methods — `g.mesh.generation.optimize(method=...)`:**

```python
g.mesh.generation.optimize("Netgen")   # common
g.mesh.generation.optimize("HighOrder")  # for order >= 2 elements
```

Accepted strings: `""` (default), `"Netgen"`, `"HighOrder"`,
`"HighOrderElastic"`, `"HighOrderFastCurving"`, `"Laplace2D"`,
`"Relocate2D"`, `"Relocate3D"`, `"QuadQuasiStructured"`,
`"UntangleMeshGeometry"`. Available under the `OptimizeMethod`
constant class if you prefer typed names.

**Underlying Gmsh options:**

| apeGmsh call | Gmsh option / API |
|---|---|
| `set_algorithm(tag, alg, dim=2)` | `gmsh.model.mesh.setAlgorithm(2, tag, code)` (per-surface) |
| `set_algorithm(0, alg, dim=3)` | `Mesh.Algorithm3D` (global option) |
| `optimize(method)` | `gmsh.model.mesh.optimize(method, ...)` |

If you pass a name apeGmsh doesn't recognise it raises a
`ValueError` that lists every canonical name for the dimension —
no silent Gmsh errors downstream.

### 10.6  Optimisation — smoothing bad elements after the fact

Generation produces a mesh; optimisation tries to improve it.
Ships as a post-generation pass that nudges nodes around to
reduce sliver elements, fix inverted Jacobians, and smooth poor
shape functions.

```python
g.mesh.generation.optimize()                 # default pass
g.mesh.generation.optimize("Netgen", niter=3)
g.mesh.generation.optimize("HighOrder")      # for order ≥ 2 meshes
```

Signature:

```python
optimize(
    method: str = "",
    *,
    force: bool = False,
    niter: int = 1,
    dim_tags: list[(dim, tag)] | None = None,
)
```

- `method` — the algorithm. Empty string is Gmsh's default smoother.
- `force` — apply even to already-valid elements (otherwise Gmsh
  skips elements it considers OK).
- `niter` — number of passes. Diminishing returns past ~3.
- `dim_tags` — limit to specific entities. `None` = the whole mesh.

**When to reach for each method:**

| Method | Use when |
|---|---|
| `""` (default) | First pass, linear meshes, no special needs |
| `"Netgen"` | Tet-mesh quality improvement — most common for 3D solids |
| `"HighOrder"` | After `set_order(2)` fixes invalid mid-edge nodes |
| `"HighOrderElastic"` | `HighOrder` plus elastic relaxation — slower, often better |
| `"HighOrderFastCurving"` | Fast high-order curving near boundaries |
| `"Laplace2D"` | Classical 2D Laplace smoothing (tris/quads) |
| `"Relocate2D"` / `"Relocate3D"` | Minimise element deformation per-dim |
| `"QuadQuasiStructured"` | Clean up quad meshes from `quasi_structured_quad` |
| `"UntangleMeshGeometry"` | Nuclear option — fix tangled / inverted elements |

All accepted strings also live as typed constants under
`OptimizeMethod` (import from `apeGmsh` or `apeGmsh.mesh`) if you
prefer autocompletion.

**Where in the flow:** after `generate`, after `set_order` if used.
Before `renumber`.

```python
g.mesh.generation.generate(dim=3)
g.mesh.generation.set_order(2)
g.mesh.generation.optimize("HighOrder")   # fix any invalid mid-edge nodes
g.mesh.generation.optimize("Netgen")      # general quality cleanup
g.mesh.partitioning.renumber(dim=3, method="rcm")
```

If an optimisation pass can't improve the mesh it returns silently
— no error. Run `quality_report()` (§10.9) afterwards to see what
actually changed.

### 10.7  Renumbering — mandatory before `get_fem_data`

Gmsh's internal mesh numbering is non-contiguous after booleans
and meshing. OpenSees (and most other solvers) want **dense,
1-based IDs**. `renumber` provides that, and optionally reorders
for bandwidth or cache locality:

```python
g.mesh.partitioning.renumber(
    dim=3,              # element dimension to use for bandwidth + element renumbering
    method="rcm",       # "simple" | "rcm" | "hilbert" | "metis"
    base=1,             # OpenSees / Abaqus convention
)
```

| Method | What it does | When |
|---|---|---|
| `"simple"` | Just makes IDs contiguous | When you don't care about ordering — fastest |
| `"rcm"` | Reverse Cuthill-McKee — minimises matrix bandwidth | Default for direct solvers |
| `"hilbert"` | Hilbert space-filling curve — improves cache locality | Dense iterative solvers |
| `"metis"` | METIS graph-partitioner ordering | Preparation for parallel partitioning |

**Call it once, right before `get_fem_data`.** Renumbering after
the broker is built defeats the purpose.

#### What renumber actually renumbers

The name lives under `g.mesh.partitioning`, which makes it sound
like it touches geometry or partitioning. It does not. It calls
`gmsh.model.mesh.renumberNodes(...)` and
`gmsh.model.mesh.renumberElements(...)` — **mesh node tags and
mesh element tags only**. Everything else is untouched:

| Thing | Renumbered? |
|---|---|
| Mesh **node tags** | ✅ renumbered to dense `base, base+1, ...` |
| Mesh **element tags** | ✅ renumbered, same scheme |
| OCC **entity tags** (volume/surface/curve/point) | ❌ unchanged |
| **Labels** (Tier 1) | ❌ unchanged — they index OCC entities, not nodes |
| **Physical groups** (Tier 2) | ❌ unchanged — same reason |
| Instance registry (`g.parts.*`) | ❌ unchanged |

In other words, your naming layer is completely undisturbed. The
names you used to build the model (`"Base"`, `"Body"`, `"shaft"`)
keep pointing at the same OCC entities; only the *mesh nodes and
elements sitting on those entities* get new IDs. That is why
`fem.nodes.get(pg="Base")` still works after a `renumber` — the
PG→entity→nodes lookup chain is intact; only the last link
returns fresh integers.

### 10.8  The FEM broker — handing off to the solver

```python
fem = g.mesh.queries.get_fem_data(dim=3)
```

`fem` is the snapshot we've been mentioning since Lesson 3 — an
immutable `FEMData` object with `.nodes`, `.elements`, and the
resolved constraint / load / mass records organised underneath.
It's the single contract between the session and any downstream
solver.

Covered in detail in the next lesson. For meshing purposes, just
remember: `get_fem_data(dim)` gets you the handoff object.

### 10.9  Quality check

```python
g.mesh.queries.quality_report()       # returns a DataFrame
```

Reports element counts and quality metrics (Jacobian range,
skewness, etc.) grouped by physical group. Skim it before you
waste time running an analysis on slivers or inverted elements.

### 10.10  The canonical mesh flow

Six lines, every time (two of them optional):

```python
# Assume geometry + physical groups are already set up.

g.mesh.sizing.set_global_size(5.0)                   # 1. ceiling
g.mesh.generation.generate(dim=3)                    # 2. mesh
g.mesh.generation.set_order(2)                       # 3. (optional) quadratic
g.mesh.generation.optimize("Netgen")                 # 4. (optional) quality pass
g.mesh.partitioning.renumber(dim=3, method="rcm")    # 5. solver-ready IDs
fem = g.mesh.queries.get_fem_data(dim=3)             # 6. handoff
```

That is a complete, conformal, solver-ready mesh.

### 10.11  When defaults aren't enough

This lesson stays on the straight path. For more:

- **Adaptive sizing with fields** —
  `g.mesh.field.distance / threshold / box / boundary_layer /
  minimum`. Useful for refining near edges, around a weld, or in
  a boundary layer. See `guide_meshing.md`.
- **Structured / transfinite meshing** —
  `g.mesh.structured.set_transfinite_curve / surface / volume` +
  `recombine`. For hex-dominant meshes where element alignment
  matters. Same guide.
- **Parallel partitioning** —
  `g.mesh.partitioning.partition(n_parts=, method=)`. For
  OpenSeesMP / other MPI runs. See `guide_partitioning.md`.
- **Per-physical-group element types** —
  `g.opensees.elements.assign("Body",
  "FourNodeTetrahedron", ...)`. Covered when we get to the
  OpenSees bridge.

---

## Lesson 11 — The FEM broker

Every lesson since Lesson 3 has mentioned
`fem = g.mesh.queries.get_fem_data(...)`. Time to open it up. The
broker is the **single contract between apeGmsh and any
downstream solver**. Understand its shape and you understand how
to wire OpenSees (or anything else) to your model.

### 11.1  What the broker is

> An **immutable snapshot** of nodes, elements, and all
> pre-declared constraint / load / mass records, taken at a
> specific moment in the session's life.

Three words matter:

- **Immutable** — once built, it doesn't change. Modifying the
  session afterwards doesn't touch `fem`. Rebuild if you need to
  reflect changes.
- **Snapshot** — it's a copy of the data you need, not a live
  view into Gmsh. The Gmsh session can be closed and `fem` still
  works.
- **Contract** — every solver bridge consumes it through the same
  API. Code that reads `fem.nodes.get(pg=...)` works regardless
  of the solver on the other end.

```python
g.mesh.generation.generate(dim=3)
g.mesh.partitioning.renumber(dim=3, method="rcm")
fem = g.mesh.queries.get_fem_data(dim=3)     # ← the broker
```

### 11.2  The top-level shape

```
fem
├── .nodes           NodeComposite   — everything indexed by node ID
├── .elements        ElementComposite — everything indexed by element ID
├── .info            MeshInfo        — n_nodes, n_elems, bandwidth, elem_type_name
└── .inspect         InspectComposite — summary, tables, source tracing
```

Two big composites (**nodes** and **elements**) mirror each
other. Each carries its own IDs and coords/connectivity, plus
per-node and per-element records (constraints, loads, masses).

Why the split? Because different solver concepts live at
different levels:

| Level | What lives there |
|---|---|
| **Nodes** | Point forces, lumped masses, BCs, equalDOF constraints, rigid links |
| **Elements** | Pressure loads, body forces, surface ties, embedded constraints |

The broker mirrors this split. Pressure is `fem.elements.loads`;
a point force is `fem.nodes.loads`. No confusion about where a
record belongs.

### 11.3  Nodes — bulk access vs filtered access

**Bulk:** all nodes, in one shot.

```python
fem.nodes.ids              # ndarray(N,) — every node ID
fem.nodes.coords           # ndarray(N, 3) — every coordinate
fem.nodes.index(node_id)   # O(1) ID → row index lookup
```

**Filtered by name:** the common case. `get()` returns a
`NodeResult`:

```python
# By physical group
result = fem.nodes.get(pg="Base")

# By label
result = fem.nodes.get(label="shaft")

result.ids             # ndarray of node IDs in this subset
result.coords          # ndarray(k, 3) of their coords
result.to_dataframe()  # pandas
```

And crucially — **iterable**. The most common idiom in solver
code:

```python
for nid, xyz in fem.nodes.get(pg="Base"):
    ops.node(nid, *xyz)
    ops.fix(nid, 1, 1, 1)
```

One line per node, IDs and coordinates unpacked automatically.

#### Per-node record sets

The `.nodes` composite also carries resolved records — the output
of the two-stage pipeline (declare pre-mesh, resolve at broker
build time):

```python
fem.nodes.constraints     # NodeConstraintSet — equalDOF, rigidLink, node_to_surface, ...
fem.nodes.loads           # NodalLoadSet — point forces + moments
fem.nodes.masses          # MassSet — lumped nodal masses
fem.nodes.physical        # PhysicalGroupSet — PGs that survived into the broker
fem.nodes.labels          # LabelSet — labels that did the same
```

Each of these is iterable and/or filterable. We'll go deep on
them in the constraints / loads / masses lessons.

### 11.4  Elements — connectivity and type heterogeneity

Elements are trickier than nodes because a single model can
contain multiple element types (tets + hexes, tris + quads, etc.).
The broker handles this with two access patterns.

**Bulk (homogeneous mesh only):**

```python
fem.elements.ids             # ndarray(E,) — every element ID
fem.elements.connectivity    # ndarray(E, npe) — ONLY if mesh is homogeneous
```

If your mesh has mixed element types, `.connectivity` raises
`TypeError` — a single 2D array can't hold elements of different
node counts.

**Filtered by name:**

```python
result = fem.elements.get(pg="Body")    # GroupResult
for group in result:
    group.element_type       # e.g. "Tetrahedron4"
    group.ids                # ndarray for this element type
    group.connectivity       # ndarray(k, npe) for this element type
```

**The single-type fast path:** if you know a PG has exactly one
element type, `resolve()` flattens it:

```python
ids, conn = fem.elements.resolve(pg="Body", element_type="Tetrahedron4")
# Flat (ids, conn) tuple for the single type
```

This is the common OpenSees pattern — one element type per PG,
one call to get the connectivity.

#### Per-element record sets

Parallel to nodes:

```python
fem.elements.constraints     # SurfaceConstraintSet — tie, mortar, tied_contact
fem.elements.loads           # ElementLoadSet — pressure, body force
```

### 11.5  Introspection — "what's actually in here?"

Before you start writing a thousand lines of OpenSees code, check
what the broker actually captured:

```python
print(fem.inspect.summary())
# Compact text summary: node count, element count, PGs, labels,
# constraint kinds, load patterns, mass records.

fem.inspect.node_table()         # pandas DataFrame of all nodes
fem.inspect.physical_table()     # DataFrame of PGs + their member counts
fem.inspect.constraint_summary() # constraint kind counts + source names
```

Run `fem.inspect.summary()` the first time you build a broker on
a new model. It's the fastest way to catch *"oh I forgot to
promote that label to a PG"* kinds of mistakes.

### 11.6  `fem.info` — mesh statistics at a glance

```python
fem.info.n_nodes           # total node count
fem.info.n_elems           # total element count
fem.info.bandwidth         # matrix bandwidth (after renumber)
fem.info.elem_type_name    # e.g. "Tetrahedron4" for homogeneous meshes
```

Useful for logging, sanity checks, and deciding whether your
solver config is sensible for the problem size.

### 11.7  Why immutability matters

Because the broker is the solver contract, it has to be stable.
Consider:

```python
fem = g.mesh.queries.get_fem_data(dim=3)

# ... some time later, inside solver code ...
g.model.geometry.add_box(0, 0, 0, 1, 1, 1)   # session mutates
g.parts.fragment_all()                        # everything renumbers

for nid, xyz in fem.nodes.get(pg="Base"):    # still works, same IDs & coords
    ops.fix(nid, 1, 1, 1)
```

The solver loop runs against a *frozen* world. If `fem` were
live, that loop would see inconsistent state. By making it a
snapshot, apeGmsh gives the solver code a contract it can depend
on.

When you need the current state, build a new broker:

```python
fem = g.mesh.queries.get_fem_data(dim=3)    # fresh snapshot
```

### 11.8  The canonical handoff

Everything we've built so far meets the broker here:

```python
with apeGmsh(model_name="frame") as g:
    # ... geometry + PGs + constraints + loads + masses (pre-mesh declarations)
    # ... mesh generation + renumber

    fem = g.mesh.queries.get_fem_data(dim=3)

    # Solver side — consume via the broker
    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material("Concrete", "ElasticIsotropic",
                                         E=30e9, nu=0.2, rho=2400)
    g.opensees.elements.assign("Body", "FourNodeTetrahedron",
                               material="Concrete")

    # These read from fem internally
    g.opensees.ingest.loads(fem).masses(fem).constraints(fem)
    g.opensees.build()
```

The `ingest` methods are where the broker's constraint / load /
mass record sets get consumed — each iterates `fem.nodes.*` or
`fem.elements.*` and emits OpenSees commands. The next three
lessons open those record types up.

---

## Lesson 12 — Constraints

Constraints are where the solver world gets interesting. apeGmsh
has **fifteen** constraint kinds across five categories. Picking
the right one matters for both correctness and convergence. The
library uses a **two-stage pipeline** that decouples *what you
want* from *what the mesh makes concrete*.

### 12.1  The two-stage pipeline

**Stage 1 — declare, pre-mesh, against labels and PGs:**

```python
g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3], tolerance=1e-3)
```

At declaration time `"col"` and `"beam"` are just names. No mesh
exists yet. apeGmsh records the *intent*.

**Stage 2 — resolve, at broker build time, to concrete node pairs:**

```python
fem = g.mesh.queries.get_fem_data(dim=3)
# fem.nodes.constraints now carries NodePairRecord objects
# with resolved master_node, slave_node, dofs fields.
```

The resolver walks every declaration, matches entities to mesh
nodes, applies the tolerance test, and emits records into the
broker. The solver side (Lesson 15) reads those records and emits
`ops` commands.

Why the split? Because you can't talk about node IDs before
you've meshed, but you *can* talk about "col" and "beam." The
two-stage pipeline lets you write model-level intent early and
have it become solver-level commands automatically later.

### 12.2  The constraint taxonomy

Five categories, grouped by **what the constraint connects** and
**where the resolved record lives on the broker**:

| Category | Methods | Record type | Lives on |
|---|---|---|---|
| **Node-to-node** (1:1) | `equal_dof`, `rigid_link` *(beam/rod)*, `penalty` | `NodePairRecord` | `fem.nodes.constraints` |
| **Node-to-group** (1:N) | `rigid_diaphragm`, `rigid_body`, `kinematic_coupling` | `NodeGroupRecord` | `fem.nodes.constraints` |
| **Mixed-DOF** (node → surface, different meshes) | `node_to_surface`, `node_to_surface_spring` | `NodeToSurfaceRecord` *(with phantom nodes)* | `fem.nodes.constraints` |
| **Surface interpolation** | `tie`, `distributing_coupling`, `embedded` | `InterpolationRecord` | `fem.elements.constraints` |
| **Surface-to-surface** | `tied_contact`, `mortar` | `SurfaceCouplingRecord` | `fem.elements.constraints` |

Top three categories land on **nodes**; bottom two land on
**elements**. Pressure is an element load, a point force is a
node load — and the same split applies to constraints. A surface
interpolation *is* element-valued; a rigid link between two nodes
is node-valued.

### 12.3  The common declarations

Most models only use these four:

```python
# equal_dof — co-located nodes must have identical DOF values.
# Use when master and slave share nodes at the interface (e.g.
# after fragment_all creates a shared face).
g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3], tolerance=1e-3)

# rigid_link — master and slave move as a rigid body.
g.constraints.rigid_link("col_end", "beam_start", link_type="beam")
# link_type="beam" couples all 6 DOFs; "rod" couples translations only.

# tie — shape-function interpolation for NON-matching meshes.
g.constraints.tie(
    master_label="shell",
    slave_label="beam",
    master_entities=[(2, face_tag)],
    slave_entities=[(1, edge_tag)],
    dofs=[1, 2, 3, 4, 5, 6],
    tolerance=5.0,
)

# rigid_diaphragm — all member nodes move as a rigid diaphragm
# in-plane. Classic use: floor slab as a rigid horizontal
# diaphragm.
g.constraints.rigid_diaphragm(
    master_label="diaphragm_node",
    slave_label="column_tops",
    dofs=[1, 2, 6],   # in-plane translations + rotation about vertical
)
```

### 12.4  Choosing the right kind

| Situation | Reach for |
|---|---|
| Co-located nodes, same mesh (after `fragment_all`) | `equal_dof` — cheapest |
| Non-matching meshes at an interface | `tie` |
| Slave should follow master rigidly | `rigid_link` (1:1) or `rigid_body` (1:N) |
| 1D rebar embedded in a 3D solid | `embedded` |
| Beam-to-shell junction, spread load over multiple nodes | `distributing_coupling` |
| Floor slab acts rigid in-plane | `rigid_diaphragm` |
| Contact between two touching surfaces | `tied_contact` (small gap) or `mortar` (larger gap) |
| Node floats in master volume, not at a mesh node | `node_to_surface` — phantom nodes created automatically |

### 12.5  The node-to-surface special case

`node_to_surface` is the workhorse for mixed-DOF coupling where
the slave node floats in space, not on the master mesh. Example:
a beam end connecting to a solid column's face.

```python
g.constraints.node_to_surface(
    master="col_face",           # the 3D face
    slave="beam_end_node",        # the 1D beam's free end
    dofs=[1, 2, 3, 4, 5, 6],
    tolerance=1.0,
)
```

Behind the scenes apeGmsh creates **phantom nodes** on the master
surface at the barycentric projection of the slave, and ties them
to the slave via `equal_dof`. The phantom nodes must be emitted
first at the solver side (see §12.8).

Shared-edge mesh nodes are deduplicated, so each slave gets
exactly one phantom — no double constraints on shared boundaries.

### 12.6  How records land on the broker

**Node-level constraints** live on `fem.nodes.constraints`:

```python
# Flat iteration — every pair, with compound records auto-expanded
for pair in fem.nodes.constraints.pairs():
    pair.kind            # "equal_dof", "rigid_beam", ...
    pair.master_node     # int
    pair.slave_node      # int
    pair.dofs            # list[int]

# Grouped iteration — preferred for solvers with native multi-slave commands
for master, slaves in fem.nodes.constraints.rigid_link_groups():
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

for master, slaves in fem.nodes.constraints.rigid_diaphragms():
    ops.rigidDiaphragm(3, master, *slaves)

# Typed iteration — just equal_dof records
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)

# Raw compound records — when you need phantom_coords or extras
for nts in fem.nodes.constraints.node_to_surfaces():
    nts.master_node, nts.slave_node, nts.phantom_coords, ...
```

**Surface-level constraints** live on `fem.elements.constraints`:

```python
# Interpolation records — tie, distributing, embedded
for interp in fem.elements.constraints.interpolations():
    interp.slave_node
    interp.master_nodes     # list[int]
    interp.weights          # ndarray — shape-function weights
    interp.dofs

# Coupling records — tied_contact, mortar
for coup in fem.elements.constraints.couplings():
    ...
```

### 12.7  The `Kind` enum — no magic strings

```python
K = fem.nodes.constraints.Kind

for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
```

Bulk classification:

```python
K.NODE_PAIR_KINDS   # frozenset — all node-pair kind strings
K.SURFACE_KINDS     # frozenset — all surface kind strings
```

Thirteen named kinds total: `EQUAL_DOF`, `RIGID_BEAM`,
`RIGID_BEAM_STIFF`, `RIGID_ROD`, `RIGID_DIAPHRAGM`, `RIGID_BODY`,
`KINEMATIC_COUPLING`, `PENALTY`, `NODE_TO_SURFACE`,
`NODE_TO_SURFACE_SPRING`, `TIE`, `DISTRIBUTING`, `EMBEDDED`,
`TIED_CONTACT`, `MORTAR`.

### 12.8  Emission template for OpenSees

The order matters. Phantom nodes first, then node-level
constraints, then element-level:

```python
# 1. Phantom nodes from node_to_surface constraints
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# 2. Rigid links (grouped by master — covers every rigid kind)
for master, slaves in fem.nodes.constraints.rigid_link_groups():
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

# 3. Equal DOFs (includes expanded node_to_surface pairs)
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)

# 4. Rigid diaphragms
for master, slaves in fem.nodes.constraints.rigid_diaphragms():
    ops.rigidDiaphragm(3, master, *slaves)

# 5. Surface constraints
for interp in fem.elements.constraints.interpolations():
    ...
for coup in fem.elements.constraints.couplings():
    ...
```

**You almost never write this by hand.** `g.opensees.ingest
.constraints(fem)` does it for you. Lesson 15.

### 12.9  `tie` emission note — `ASDEmbeddedNodeElement`

apeGmsh emits `tie` as `ASDEmbeddedNodeElement` (a penalty
element), not as Lagrange multipliers. Default stiffness K=1e18.
If Newton fails to converge, drop it:

```python
g.opensees.ingest.constraints(fem, tie_penalty=1e12)
```

Quad-4 master faces are auto-split into triangles so the
shape-function interpolation is well-defined.

### 12.10  Pitfalls

- **`equal_dof` needs co-located nodes.** `tolerance=1e-3`
  covers floating-point noise; it does **not** cover real mesh
  mismatch. Use `tie` for non-matching meshes.
- **`tie` penalty defaults to 1e18.** Drop to 1e10–1e12 if
  Newton fails. The element only needs K >> parent element
  stiffness, not K → ∞.
- **`rigid_diaphragm` DOF choice is model-dependent.** For a
  horizontal floor slab in 3D: `dofs=[1, 2, 6]` (in-plane
  translations + rotation about vertical). For a 2D plane:
  `dofs=[1, 2, 3]`. Get this wrong and the diaphragm either
  under-constrains or over-constrains the slab.
- **`node_to_surface` phantom nodes must be emitted before
  equal_dof.** Order matters. `ingest.constraints` handles it
  automatically; if you emit by hand, walk `phantom_nodes()`
  first.
- **Multi-kind rigid links merge.**
  `fem.nodes.constraints.rigid_link_groups()` accumulates across
  `rigid_beam`, `rigid_rod`, `rigid_diaphragm`, `rigid_body`,
  `kinematic_coupling`, **and** expanded `node_to_surface`
  phantom links. One grouped iteration covers all rigid-style
  constraints.

---

## Lesson 13 — Loads

Loads share the same two-stage pipeline as constraints: declare
against labels and PGs pre-mesh, resolve to concrete
node/element records at broker build. Five factory methods cover
the common cases, and one context manager (`pattern`) groups them
into OpenSees-compatible load patterns.

### 13.1  The two-stage pipeline

```python
# Stage 1 — declare against named targets
g.loads.gravity("Concrete", g=(0, 0, -9.81), density=2400)
g.loads.surface("RoofSlab", magnitude=-3e3, normal=True)

# Stage 2 — at broker build, definitions become resolved records
fem = g.mesh.queries.get_fem_data(dim=3)
# fem.nodes.loads now has concrete NodalLoadRecord entries
# fem.elements.loads has ElementLoadRecord entries where applicable
```

Same idea as constraints: you don't know node IDs at declaration
time, but you do know the names — so the library records intent
and resolves later.

### 13.2  The `pattern` idiom — grouping loads

OpenSees organises loads into **load patterns** (dead, live,
wind, seismic, …). apeGmsh exposes this with a context manager:

```python
with g.loads.pattern("Dead"):
    g.loads.gravity("Concrete", density=2400)
    g.loads.line("Beams", magnitude=-2e3, direction=(0, 0, -1))

with g.loads.pattern("Live"):
    g.loads.surface("Slabs", magnitude=-3e3)
```

Every load declaration inside the `with` block is tagged with the
active pattern name. At the solver side,
`g.opensees.ingest.loads(fem)` emits one OpenSees `pattern` /
`load` block per name. Loads declared outside any `with` end up
in the default/untagged pattern.

```python
g.loads.patterns()   # list[str] of declared pattern names
```

### 13.3  The five factory methods

| Method | Applies to | Typical use |
|---|---|---|
| `point(target, force_xyz=, moment_xyz=)` | Nodes of `target` | Concentrated force or moment at specific points |
| `line(target, magnitude=, direction=)` | Curves | Distributed load along a beam |
| `surface(target, magnitude=, normal=True)` | Faces | Pressure (`normal=True`) or traction |
| `gravity(target, g=, density=)` | Volumes | Self-weight from density × gravity |
| `body(target, force_per_volume=)` | Volumes | Generic volumetric body force |
| `face_load(target, force_xyz=, moment_xyz=)` | Face centroids | Total load spread from centroid to face nodes |

All take the same target-resolution kwargs (§13.4) and all
respect the active `pattern`.

Concrete shapes:

```python
# Concentrated at the nodes of "TopAnchor"
g.loads.point("TopAnchor", force_xyz=(0, 0, -50e3))

# 2 kN/m downward on "Beams"
g.loads.line("Beams", magnitude=-2e3, direction=(0, 0, -1))

# 3 kN/m² normal-inward on "RoofSlab"
g.loads.surface("RoofSlab", magnitude=-3e3, normal=True)

# Self-weight of a concrete volume
g.loads.gravity("Concrete", g=(0, 0, -9.81), density=2400)

# Generic body force (e.g. seepage or centrifugal)
g.loads.body("Aquifer", force_per_volume=(0, 0, -9810))
```

### 13.4  Target resolution — four ways to name a thing

Every factory method accepts four targeting kwargs:

```python
g.loads.point("TopAnchor", force_xyz=(0, 0, -1))  # positional: label first, then PG
g.loads.point(label="TopAnchor", force_xyz=(0, 0, -1))   # explicit label
g.loads.point(pg="TopAnchor", force_xyz=(0, 0, -1))      # explicit PG
g.loads.point(tag=(0, 42), force_xyz=(0, 0, -1))         # raw (dim, tag)
```

Positional tries **label first, then PG** — same order as
`resolve_to_tags` uses elsewhere. Use explicit `label=` or `pg=`
when a name exists at both layers and you want to disambiguate.

### 13.5  Where loads land on the broker

The broker splits loads the same way it splits constraints — by
what they connect.

**`fem.nodes.loads` — `NodalLoadSet`:** concentrated forces and
moments resolved to specific nodes. Also how distributed loads
land when `target_form="nodal"` (the default).

```python
for load in fem.nodes.loads:
    load.node_id        # int
    load.force_xyz      # tuple[float, float, float] | None
    load.moment_xyz     # tuple[float, float, float] | None
    load.pattern        # pattern name
```

**`fem.elements.loads` — `ElementLoadSet`:** loads that stay
attached to elements (pressure on an element face, body force on
an element volume) when `target_form="element"`.

```python
for eload in fem.elements.loads:
    eload.element_id
    eload.load_type     # e.g. "Pressure", "BodyForce"
    eload.params        # dict — kind-specific parameters
    eload.pattern
```

#### `target_form` — which side of the broker

Every load factory accepts `target_form` with two valid values:

- `"nodal"` *(default)* — reduces to equivalent nodal forces via
  the selected `reduction` strategy. Lands on `fem.nodes.loads`.
  Works for every solver.
- `"element"` — stays attached to elements as a
  pressure / body-force record. Lands on `fem.elements.loads`.
  Only works with solvers that accept `ops.eleLoad` (OpenSees
  does).

#### `reduction` — how distributed loads become nodal values

For `target_form="nodal"`, pick how the continuous field
collapses onto mesh nodes:

- `"tributary"` *(default)* — each node gets the load integrated
  over its tributary area / volume. Fast, simple, always correct
  in the limit of fine meshes.
- `"consistent"` — the proper weak-form consistent load vector
  via shape functions. More accurate on coarse meshes,
  especially for moments.

You almost always want `"tributary"` unless you're comparing
against a benchmark that specifies consistent loading.

### 13.6  Emission template for OpenSees

```python
# Nodal loads — one per pattern
for pattern_name in g.loads.patterns():
    ops.pattern("Plain", pattern_tag, ts_tag)
    for load in fem.nodes.loads:
        if load.pattern != pattern_name:
            continue
        fx, fy, fz = load.force_xyz or (0, 0, 0)
        mx, my, mz = load.moment_xyz or (0, 0, 0)
        ops.load(load.node_id, fx, fy, fz, mx, my, mz)   # 3D, ndf=6

# Element loads (pressure, body force)
for eload in fem.elements.loads:
    ops.eleLoad(eload.element_id, eload.load_type, **eload.params)
```

Note: apeGmsh stores spatial vectors only; **it doesn't know
`ndf`**. You pick the slice that matches your model (3 components
for `ndf=3`, 6 for `ndf=6`). The `ingest.loads(fem)` helper does
this automatically using the `ndm`/`ndf` you set via
`set_model`.

### 13.7  The typical full declaration

```python
with apeGmsh(model_name="building") as g:
    # ... geometry + PGs ...

    with g.loads.pattern("Dead"):
        g.loads.gravity("Concrete", g=(0, 0, -9.81), density=2400)
        g.loads.gravity("Steel",    g=(0, 0, -9.81), density=7850)

    with g.loads.pattern("Live"):
        g.loads.surface("FloorSlabs", magnitude=-2.4e3, normal=True)

    with g.loads.pattern("Wind"):
        g.loads.surface("Facade", magnitude=1.5e3, normal=True)
        g.loads.point("WindApexPt", force_xyz=(5e4, 0, 0))

    # ... mesh + renumber + fem = get_fem_data ...

    g.opensees.ingest.loads(fem)    # emits all three patterns
```

Three patterns, each with any mix of `gravity` / `surface` /
`point` declarations. The solver side treats each pattern as an
independent load case — apply with a `timeSeries` + `pattern`
block in your analysis script.

### 13.8  Pitfalls

- **`surface` with `normal=True` vs `direction=`.** If
  `normal=True` (default), `magnitude` is a scalar pressure
  applied in the outward-normal direction. If `normal=False`,
  pass `direction=(dx, dy, dz)` for the traction axis. Don't mix
  — pick one.
- **Sign convention.** apeGmsh doesn't flip signs for you.
  Gravity with `g=(0, 0, -9.81)` is downward; a negative
  `magnitude` on `surface` is inward (opposite the outward
  normal). Write it as you'd expect to see it in equilibrium.
- **Loads declared outside a `pattern` go to the default
  pattern.** Fine for quick tests, surprising in a multi-case
  analysis. When in doubt, always open a
  `with g.loads.pattern(...)` block.
- **`target_form="element"` requires solver support.** Most
  generic solvers only accept nodal loads. Stick with the
  default `"nodal"` unless your solver specifically handles
  element load records.
- **Changing pre-mesh declarations doesn't invalidate an
  existing broker.** Rebuild
  (`fem = g.mesh.queries.get_fem_data(...)` again) after you
  edit load declarations, or your changes won't make it into the
  solver input.

---

## Lesson 14 — Masses

Masses are the third pre-mesh record type (after constraints and
loads). The API mirrors loads — four factories, same
target-resolution kwargs — but masses **only ever land on
nodes**, never elements. There is no element-level mass record;
everything reduces to nodal masses at resolution time.

### 14.1  The pipeline

Same two-stage contract:

```python
# Stage 1 — declare
g.masses.volume("Concrete", density=2400)
g.masses.point("TopAnchor", mass=1500)

# Stage 2 — resolve at broker build
fem = g.mesh.queries.get_fem_data(dim=3)
# fem.nodes.masses now carries per-node MassRecord entries
```

Mass declarations know nothing about nodes until the broker is
built. Declare against labels and PGs; the resolver lumps
everything onto the mesh nodes at extraction time.

### 14.2  The four factories

| Method | Applies to | Parameter |
|---|---|---|
| `point(target, mass=, rotational=)` | Nodes of `target` | `mass` (scalar) + optional `(Ixx, Iyy, Izz)` |
| `line(target, linear_density=)` | Curves | kg / m |
| `surface(target, areal_density=)` | Faces | kg / m² |
| `volume(target, density=)` | Volumes | kg / m³ |

All four take:

- The same four target-resolution forms as loads — positional,
  `label=`, `pg=`, `tag=`.
- `reduction="lumped"` *(default — currently the only supported
  mode).*
- `name=` — optional human-readable tag for the definition.

```python
# Point mass — 1500 kg plus rotational inertia at an anchor
g.masses.point(
    "TopAnchor",
    mass=1500,
    rotational=(50, 50, 100),   # (Ixx, Iyy, Izz)
)

# Line mass — 5 kg/m along every beam
g.masses.line("Beams", linear_density=5.0)

# Surface mass — 120 kg/m² on floor slabs (finishes + partitions)
g.masses.surface("FloorSlabs", areal_density=120)

# Volume mass — concrete at 2400 kg/m³
g.masses.volume("Concrete", density=2400)
```

### 14.3  The rotational-inertia argument

Only `point()` accepts rotational inertia. The other three derive
rotational moments from the translational integration and the
geometry's shape functions — you don't set them explicitly.

```python
# 6-DOF mass: mx=my=mz=1500, Ixx=50, Iyy=50, Izz=100
g.masses.point("Lumped", mass=1500, rotational=(50, 50, 100))

# 3-DOF — no rotation
g.masses.point("Lumped", mass=1500)
```

If `rotational=` is omitted, the broker's `MassRecord` has
`Ixx=Iyy=Izz=0`. OpenSees ignores the rotational components if
your model's `ndf` doesn't include rotations.

### 14.4  Where masses land on the broker

One destination: **`fem.nodes.masses` — `MassSet`**. Every
`MassRecord` carries a full 6-DOF mass vector:

```python
for m in fem.nodes.masses:
    m.node_id       # int
    m.mass          # (mx, my, mz, Ixx, Iyy, Izz) — always 6-tuple
```

The 6-tuple is always present even for models where some
components are zero (3D solids typically have `Ixx=Iyy=Izz=0`).
The solver side slices to match `ndf`.

There is no `fem.elements.masses`. Element-level mass isn't a
first-class record type because every solver ultimately needs
nodal mass to assemble the mass matrix; apeGmsh just does that
reduction eagerly, at resolution time.

### 14.5  Accumulation — declarations sum per node

If two declarations contribute to the same node, their masses
**add**. This is useful and also a trap.

```python
# Both contribute to the nodes on the slab
g.masses.volume("Slab", density=2400)          # structural mass
g.masses.surface("Slab", areal_density=120)    # finishes + partitions
```

A node on both the volume and its face picks up a share of both.
That's the intended behaviour — structural + non-structural mass,
integrated together, one `ops.mass(...)` call per node at
emission.

Declare the same mass twice by accident and you get double mass.
The broker does not de-duplicate by definition — every `_add_def`
call stores a distinct entry. `fem.nodes.masses.summary()` is
worth checking on complex models.

### 14.6  Introspection — did I get the mass I expected?

```python
# Total translational mass — sum of mx across all records
fem.nodes.masses.total_mass()

# Lookup a specific node
rec = fem.nodes.masses.by_node(42)
if rec is not None:
    print(rec.mass)       # (mx, my, mz, Ixx, Iyy, Izz)

# Full summary — one row per node
fem.nodes.masses.summary()   # DataFrame: node_id, mx, my, mz, Ixx, Iyy, Izz

# Indexed access
fem.nodes.masses[0]          # first MassRecord
```

`total_mass()` is the fastest sanity check — it should match
your hand-calc of total structural mass plus whatever
non-structural you added. If it's off by orders of magnitude, you
probably picked the wrong density units (kg/m³ vs t/m³).

### 14.7  Emission template for OpenSees

```python
for m in fem.nodes.masses:
    # 6-DOF — pass the full tuple, OpenSees slices as needed
    ops.mass(m.node_id, *m.mass)
```

That's it. No pattern grouping, no ordering concerns, no
reduction choice. The broker has done the accumulation; the
solver just consumes. `g.opensees.ingest.masses(fem)` does this
loop internally.

### 14.8  The typical full declaration

```python
with apeGmsh(model_name="tower") as g:
    # ... geometry + PGs ...

    # Structural mass from materials
    g.masses.volume("Concrete", density=2400)
    g.masses.volume("Steel",    density=7850)

    # Non-structural — finishes, partitions, mechanical loads
    g.masses.surface("FloorSlabs", areal_density=120)

    # Lumped equipment at specific anchors
    g.masses.point("MechanicalFloorMass",
                   mass=50_000,
                   rotational=(2e6, 2e6, 4e6))

    # ... mesh + renumber + fem = get_fem_data ...

    g.opensees.ingest.masses(fem)

    print(f"Total mass: {fem.nodes.masses.total_mass():,.0f} kg")
```

### 14.9  Pitfalls

- **Units must match across the session.** `density` in kg/m³,
  coordinates in metres, `ops.mass` eating kg — consistent SI.
  If you built geometry in millimetres, your density needs to be
  kg/mm³ (density × 1e-9) or your `total_mass()` comes out wrong
  by a factor of 10⁹. apeGmsh does not convert for you.
- **`rotational=` is only accepted by `point()`.** The other
  three factories compute rotational moments from the mass
  distribution implicitly. For a specific rotational inertia on
  a distributed mass, convert to an equivalent point mass at the
  centroid.
- **Accumulation is not idempotent.** Declaring
  `g.masses.volume("Concrete", density=2400)` twice gives you
  double mass. Review `summary()` or `total_mass()` to catch
  duplicates.
- **Rebuild the broker after editing mass declarations.** Same
  as with loads — the broker is a snapshot. Forgetting to
  rebuild `fem` after a density change means the solver sees
  stale data.
- **`ops.mass` with zero rotational components is fine** for
  solids (ndf=3 models don't use rotations), but beam/shell
  models care. Don't forget `rotational=` on lumped point masses
  when the host DOFs include rotations.

---

## Roadmap — what's still coming

One more lesson to complete the solver half:

| # | Lesson | Scope |
|---|---|---|
| 15 | **OpenSees bridge** | `set_model`, `materials.add_nd_material`, `elements.assign`, `elements.fix`, `ingest.loads(fem).masses(fem).constraints(fem)`, `build`, `export.tcl/py`. The full pipeline from broker to runnable solver. |

Two optional follow-ons:

- **Viewers** — `g.model.viewer()` for BRep inspection,
  `g.mesh.viewer(fem=fem)` for FEM overlays.
- **Worked end-to-end example** — one 40-line script exercising
  Lessons 1–15 on a real structural model.
