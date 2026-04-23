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

_More lessons will be appended here as the guide grows._
