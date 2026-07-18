# The session

This page explains the object every apeGmsh script revolves around — the
session, `g`: what it owns, how its lifetime works, how a model flows through
it from first box to solver snapshot, and why a Part is a *template* while the
session is the one true model.

## One kernel, one model

Gmsh is a C library with a single, process-wide state. Working with it raw
means a ritual: initialize the runtime, add a named model, remember to finalize
— and forgetting any step is the classic source of mysterious Gmsh errors. The
session exists to own that ritual for you. When it opens, it brings up the Gmsh
runtime, creates a model under your `model_name`, and wires up every composite
you'll touch — `g.model`, `g.mesh`, `g.physical`, `g.loads`, `g.masses`,
`g.constraints`, `g.parts`, and friends. All of them are thin wrappers talking
to the same live kernel, and they share one metadata store and one label
registry, which is why a label you attach in `g.model` is visible to a load you
declare in `g.loads`. Every composite also checks that the session is actually
open before touching Gmsh, so the "I forgot to initialize" failure mode simply
can't reach you — you get a clear error instead of a crash deep inside the C
library.

One deliberate restriction is worth knowing about: a session always uses the
**OpenCASCADE kernel**. Gmsh's built-in kernel is intentionally not wrapped,
because everything downstream — label tracking across boolean operations,
accurate bounding-box queries, STEP import and export — assumes OCC semantics.
There is no kernel to choose; there is only the one that makes the rest of the
library work.

## Holding it open

There are two equivalent ways to run a session, and they differ only in who is
responsible for closing it. The form you'll use most is the context manager:

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="cantilever", verbose=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
    g.mesh.generation.generate(dim=3)
# gmsh.finalize() runs here, even if the block raised
```

The `with` block guarantees cleanup on every exit path — normal completion,
exception, keyboard interrupt. Reach for it in scripts and one-shot notebook
cells, which is to say: by default.

The second form is explicit `begin()` / `end()`. It exists because sometimes
the session's lifetime can't be a lexical block — a notebook where you build
geometry in one cell, inspect it in the next, and re-mesh in a third; a test
fixture; a class that owns a session across method calls. Then you open and
close it yourself, and the cleanup is your job:

```python
g = apeGmsh(model_name="cantilever")
g.begin()
try:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
    g.mesh.generation.generate(dim=3)
finally:
    g.end()
```

If an exception escapes between `begin()` and `end()` without that
`try / finally`, the kernel state stays alive and the session stays open — in a
notebook the symptom is a correct cell erroring because a previous cell
crashed. The rule of thumb: use `with` unless you have a concrete reason not
to, and when you do go explicit, wrap the body in `try / finally`.

Sessions also nest safely. The Gmsh runtime is reference-counted inside
apeGmsh, so opening a `Part` (which owns its own session — more below) while a
main session is running won't tear the outer session down when the inner one
closes. You don't have to think about this; it's the reason you don't have to
think about it.

After `end()` the composites still exist as Python objects, but the Gmsh state
they referenced is gone — calling anything on them raises. A closed session is
closed.

## Synchronization

Every geometry call — `add_box`, `add_point`, the transforms — flushes the OCC
kernel before returning, so the model is always in a consistent, queryable
state. That is the right default for interactive work and for any script where
you query geometry between calls. It is the wrong default for exactly one
situation: building hundreds of entities in a tight loop, where the per-call
flush dominates the runtime. For that, every geometry method accepts
`sync=False`; pass it to every call in the batch except the last, and the final
synchronizing call flushes the whole batch at once. This is a performance
lever, nothing more — the two forms produce identical models.

## From build to snapshot

A session has a direction. You build geometry and name it, you declare loads,
masses, and constraints against those names (the declare-then-resolve timing
from [the mental model](mental-model.md) — declarations are intent, recorded
before any node exists), you mesh, and then you ask for the snapshot:

```python
fem = g.mesh.queries.get_fem_data(dim=3)   # dim=2, or None for all dims
```

That call is more than a query — it is the hinge of the session's lifecycle.
Resolving your declarations against the real mesh produces the immutable
`FEMData` snapshot, and from that moment the snapshot, not the Gmsh kernel, is
the canonical model. So the session **freezes its build phase**: geometry,
meshing, and physical-group calls now raise a clear error instead of silently
desynchronizing the snapshot from the kernel it came from. What remains open is
exactly what operates *on* snapshots — composing modules with `g.compose`, and
the interface-bridging constraints (embedded ties, tied contact, `equalDOF`,
rigid links and diaphragms) that couple already-meshed regions. If you need to
change the geometry after the freeze, you rebuild or reload; you don't mutate.

The snapshot is also what persists. Pass `save_to="model.h5"` when you create
the session and `end()` writes the model automatically on the way out; call
`g.save()` to write it yourself at any point after the snapshot exists. A saved
model reopens with `apeGmsh.from_h5(...)` — a session born directly in the
frozen phase, with no Gmsh kernel at all, ready for composition and the solver
bridge. The mechanics are in
[Save & reload a model](../how-to/save-reload.md); the concept to keep is that
the session is *transient* and the snapshot is *durable*. The kernel is
scaffolding; `FEMData` is the building.

## A Part is a template; the session is the model

Everything above describes one session building one model directly. apeGmsh
also lets you author geometry in a **Part** — and understanding what a Part is
*not* is the fastest way to understand the design.

A Part is an isolated geometry unit with its own private Gmsh session. Inside
it you have the full geometry API — primitives, booleans, labels — and nothing
else: a Part cannot mesh, cannot declare loads or constraints, cannot talk to a
solver. That restriction is the point. A Part is a reusable *template* — a
standard column, a precast panel, a shape someone handed you as CAD — and
templates don't get to make solver-facing decisions. Those belong to the one
session that assembles the real model, which means you can re-mesh, re-load, or
re-constrain the assembly without ever touching the geometry definitions.

```python
from apeGmsh import Part, apeGmsh

# Build in isolation
col = Part("column")
col.begin()
col.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)
col.save()
col.end()

# Assemble: one template, two instances
g = apeGmsh(model_name="frame")
g.begin()
g.parts.add(col, label="col_1", translate=(0, 0, 0))
g.parts.add(col, label="col_2", translate=(6, 0, 0))
g.parts.fragment_all()
g.mesh.generation.generate(dim=3)
```

The Part travels as a STEP file — `save()` writes one you own and can
version-control; if you skip `save()`, the Part auto-persists itself to a
tempfile when its session closes, so it flows straight into `g.parts.add(...)`
either way. Each `add` places an *instance* of the template, and the instance's
labels are prefixed with its name so `col_1` and `col_2` stay individually
addressable. `fragment_all()` then splits the assembled bodies at every
intersection so touching instances share a conformal interface — one set of
nodes where they meet, which is what makes the assembly a single connected
model rather than bodies that merely coincide in space. (When you *want*
independent meshes coupled by constraints instead — beam-to-solid ties, contact
— you skip the fragment and declare the tie; see
[Tie non-matching meshes](../how-to/tie-meshes.md).)

You don't need a `Part` object to get this bookkeeping. Building inline, you
can wrap geometry in named blocks and get the same tracked instances:

```python
with g.parts.part("col_1"):
    g.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)

with g.parts.part("col_2"):
    g.model.geometry.add_box(6, 0, 0,  0.5, 0.5, 3.0)

g.parts.fragment_all()
```

So when do you reach for which? Parts earn their extra step when the geometry
is genuinely reusable (one template, many placements), when it comes from
external CAD, or when different people own different components and the STEP
file is the handoff. The direct session wins everywhere else: single-body
models, parametric sweeps that rebuild geometry every iteration, and meshing
workflows where you want to set structured-mesh constraints on entities the
moment you create them. Both paths converge on the same place — one session,
one mesh, one snapshot.

However you build, the hierarchy is fixed: Parts are templates, instances are
placements, and the session is the one true model. Even the `Assembly` builder
— a declarative front for composing already-saved `model.h5` modules, covered
in [Compose modules](../how-to/compose-modules.md) — resolves down to one
session in the end. Everything the solver will ever see passes through that
session exactly once, in the snapshot.

---

*Next: [Geometry & CAD](geometry-and-cad.md).*
