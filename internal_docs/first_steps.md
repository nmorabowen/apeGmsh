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
five lines once you've written it a few times.

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

### 10.5  Algorithm choice

Default works most of the time. When you need to override:

```python
# Per-surface for dim=2
g.mesh.generation.set_algorithm("WebSurface", "frontal_delaunay_quads")

# Global for dim=3 (pass tag=0)
g.mesh.generation.set_algorithm(0, "hxt", dim=3)
```

Main 2D algorithms: `delaunay` (default, robust),
`frontal_delaunay` (higher quality), `packing_of_parallelograms`
(quad-dominant), `frontal_delaunay_quads` (full-quad).

Main 3D algorithms: `delaunay` (default), `hxt` (much faster for
large tet meshes), `frontal`, `mmg3d`.

`set_algorithm` takes labels and PG names for the `tag` arg, so
you can target specific regions without handling tags.

### 10.6  Renumbering — mandatory before `get_fem_data`

Gmsh's internal tag numbering is non-contiguous after booleans,
labels, and PGs do their work. OpenSees (and most other solvers)
want **dense, 1-based IDs**. `renumber` provides that, and
optionally reorders for bandwidth or cache locality:

```python
g.mesh.partitioning.renumber(
    dim=3,              # the dim you'll later extract
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

### 10.7  The FEM broker — handing off to the solver

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

### 10.8  Quality check

```python
g.mesh.queries.quality_report()       # returns a DataFrame
```

Reports element counts and quality metrics (Jacobian range,
skewness, etc.) grouped by physical group. Skim it before you
waste time running an analysis on slivers or inverted elements.

### 10.9  The canonical mesh flow

Five lines, every time:

```python
# Assume geometry + physical groups are already set up.

g.mesh.sizing.set_global_size(5.0)                   # 1. ceiling
g.mesh.generation.generate(dim=3)                    # 2. mesh
g.mesh.generation.set_order(2)                       # 3. (optional) quadratic
g.mesh.partitioning.renumber(dim=3, method="rcm")    # 4. solver-ready IDs
fem = g.mesh.queries.get_fem_data(dim=3)             # 5. handoff
```

That is a complete, conformal, solver-ready mesh.

### 10.10  When defaults aren't enough

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

_More lessons will be appended here as the guide grows._
