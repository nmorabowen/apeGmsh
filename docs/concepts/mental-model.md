# Core concepts (the mental model)

You already know FEM and OpenSees. apeGmsh isn't asking you to relearn either —
it's giving you one place to *describe* a model and a clean hand-off to the
solver. Six ideas carry the whole library. Read them once and the API stops
feeling like a pile of namespaces and starts feeling like a sentence you already
know how to say.

## 1. A session owns one Gmsh kernel

Everything starts with a session, `g`. It owns a single live Gmsh kernel for its
whole lifetime, and every namespace you touch — `g.model`, `g.mesh`, `g.loads`,
`g.masses`, `g.constraints` — is a thin wrapper talking to that one kernel. Open
it with a `with` block so it always closes cleanly:

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model") as g:
    ...  # the whole model is built in here
```

At the top level, the session *is* the assembly — there's no separate
`Assembly` object to juggle, and OpenSees is deliberately **not** a session
composite. The solver bridge comes later, after you have a snapshot.

## 2. Composites split by concern

The session doesn't put 200 methods on one object. It splits by *what you're
doing*, so the path you type reads like the intent behind it. Geometry lives
under `g.model` (which further splits into `geometry / boolean / transforms /
io / queries`), meshing under `g.mesh` (`generation / sizing / field /
structured / editing / queries / partitioning`), and the physics each get their
own home: `g.loads`, `g.masses`, `g.constraints`.

```python
g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
g.mesh.sizing.set_global_size(0.5)
g.mesh.generation.generate(dim=3)
```

Every method has a home. You almost never reach back into raw `gmsh.*` — and
when you wonder "where does X live?", the concern is usually the answer.

## 3. Labels, physical groups, and tags — the naming layers

Gmsh identifies entities by raw integer **tags**, and tags are brittle: a boolean
cut or a re-mesh can renumber them out from under you. apeGmsh adds two naming
layers on top so you almost never touch a tag again.

A **label** is the name you attach at geometry time — pass `label=` to any
`add_*` call. A **physical group** is the name that survives meshing and crosses
into the solver; you promote labels (or selections) to physical groups so loads,
supports, and elements can reference them.

```python
g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")  # label (geometry-time)
g.physical.add_volume("body", name="Body")                 # physical group (solver-facing)
```

The rule of thumb: **target everything by name, never by raw tag.** Idiomatic
apeGmsh is verbose-by-name on purpose — it's what makes a script survive a
geometry edit.

## 4. `.select()` turns coordinates into names

Some entities don't have a natural label — the *top face* of a box, every
surface *crossing a plane*, the node *nearest* a point. For those you select by
geometry and hand the result a name. `g.model.select(...)` starts a fluent chain:
pick a starting set, narrow it with spatial verbs, then a terminal bakes it into
a label or physical group.

```python
(g.model.select("body", dim=2)
   .in_box(0, 0, 2, 10, 5, 2)      # also: .on_plane / .crossing_plane / .nearest_to / .where
   .to_physical("TopFace"))         # or .to_label("top")
```

Now `"TopFace"` is just a name like any other — the rest of your script never
sees a tag. This is how spatial reasoning enters an otherwise name-driven model.

## 5. `FEMData` is the solver-agnostic snapshot

When the mesh is ready you ask for the snapshot:

```python
fem = g.mesh.queries.get_fem_data(dim=3)   # dim=2, or None for all dims
```

`FEMData` is an immutable, solver-agnostic record of the model: `fem.nodes`,
`fem.elements`, `fem.info`, plus the loads, masses, constraints, and per-node
DOF you declared. It needs no live Gmsh session — it round-trips through
`model.h5`, composes into larger assemblies, and is the *one thing* every solver
bridge consumes. OpenSees gets a first-class bridge; anyone else just reads the
snapshot. This is the seam that keeps modeling and solving independent.

## 6. Declare-then-resolve, and the typed bridge

Here's the timing that ties it together. You **declare** loads, masses, and
constraints *before* the mesh exists, against names — not nodes:

```python
with g.loads.case("dead"):
    g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
g.masses.volume("Body", density=2400)
```

There are no element tags or tributary loops here — just intent. The actual
nodes and elements don't exist yet. apeGmsh **resolves** those declarations
against the real mesh at `get_fem_data(...)`, folding them into the snapshot.

Then the typed `apeSees(fem)` bridge consumes that snapshot to write OpenSees.
You build the model with typed primitives — not raw `ops.*` strings — and
multi-point constraints (`equalDOF`, `rigidLink`, `rigidDiaphragm`,
`ASDEmbeddedNodeElement`) **emit automatically** from the snapshot. You declare
the tie; the bridge writes the deck:

```python
from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
# ... typed materials / sections / elements, then:
ops.run()
```

---

**Ready to build?** Walk the whole spine end to end — geometry to a verified
deflection in under 40 lines — in
[Your first model in 10 minutes](../tutorials/first-model.md).
