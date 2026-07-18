# The FEM broker (`FEMData`)

This page explains the immutable snapshot at the center of apeGmsh — what
`FEMData` contains, why it is frozen, and why it, and not the live Gmsh
session, is the contract every solver consumes.

## Why a broker at all

A live Gmsh session is a *conversation*: tags shift when you regenerate,
physical groups live behind API calls, and every downstream tool that wants
the mesh has to learn the same query dance — and keep the session alive while
it dances. That coupling is exactly what the broker breaks.
[The session page](session.md) showed you the hinge:

```python
fem = g.mesh.queries.get_fem_data(dim=3)   # dim=2, or None for all dims
```

What that call actually does is worth slowing down for. It walks the mesh
once and extracts nodes and elements into plain NumPy arrays. It snapshots
every physical group, label, and named mesh selection. And it *resolves* your
declarations — the loads, masses, supports, and constraints you declared
against names before any node existed — into concrete per-node and
per-element records. Declare-then-resolve, from
[the mental model](mental-model.md), lands here: this is the resolve step,
and the records are its output. The result is a `FEMData` object that needs
no live Gmsh session at all. You can close the session, ship the snapshot to
another process, or read it years later, and it answers the same questions
the same way.

A snapshot doesn't have to come from a session you built, either.
`FEMData.from_msh("bridge.msh", dim=2)` imports an external Gmsh mesh, and
partial snapshots can be synthesized from solver output files — useful for
post-processing, though those carry only what the source format knows
(nodes, elements, groups; no labels, no pre-mesh declarations).

## What the snapshot holds

The broker is organized by what a structural engineer reaches for — nodes
and elements — with everything else attached to whichever of the two it
belongs to:

```python
fem.nodes        # NodeComposite:    ids, coords, physical, labels,
                 #                   constraints, loads, sp, masses
fem.elements     # ElementComposite: per-type groups, physical, labels,
                 #                   constraints, loads
fem.info         # MeshInfo:         n_nodes, n_elems, bandwidth, types
fem.inspect      # summaries and DataFrames
```

The raw arrays are there when you want them — `fem.nodes.ids` is an
`ndarray(N,)` of node IDs, `fem.nodes.coords` an `ndarray(N, 3)` of
coordinates. The IDs deliberately use object dtype so iterating yields plain
Python `int`s, which C-extension solvers accept without complaint. Elements
are stored *per type*: a mesh that mixes tets and surface quads holds one
group per element type, each with its own `ids` and `connectivity` arrays.
Nothing is hard-coded to linear elements — a second-order mesh simply
carries wider connectivity rows, in Gmsh node ordering; any solver-specific
permutation belongs to the consuming adapter, not the broker.

Both composites share the two naming layers you built the model with:
`fem.nodes.physical` (the solver-facing physical groups) and
`fem.nodes.labels` (the geometry-time labels), plus `fem.mesh_selection`,
the snapshot of any named node or element sets you registered. A name that
meant something during the build means the same thing on the snapshot.

Then there are the resolved records — the part of the snapshot that has no
Gmsh counterpart at all, because it was born from your declarations:

- `fem.nodes.constraints` — node-pair constraints (`equal_dof`, rigid
  links, diaphragms), plus compound records that expand into pairs and any
  phantom nodes they generated;
- `fem.nodes.loads` and `fem.elements.loads` — nodal forces and element
  loads, grouped by the load case they were declared under;
- `fem.nodes.sp` — single-point records: homogeneous fixities and
  prescribed displacements;
- `fem.nodes.masses` — accumulated per-node mass vectors;
- `fem.elements.constraints` — surface-level ties and couplings, carried
  as interpolation weights ready to become MP constraints.

Interface ties declared through the higher-level surfaces — embedded
reinforcement, node embedment, contact — ride along as their own record
lists on `fem.elements`. You rarely touch them directly; the bridge consumes
them. The point is that they are *in the snapshot*: the model's physics
travels with its mesh, as data.

## Reading it

Selection on the broker is the same `.select()` chain you learned in
[Selection & queries](selection.md) — same seeds, same spatial verbs — only
the atoms are now node and element IDs frozen in the snapshot:

```python
for nid, xyz in fem.nodes.select(pg="Base").result():
    ops.node(nid, *xyz)

ids, conn = fem.elements.select(pg="Body").result().resolve()
ids, conn = fem.elements.select("Body").result().resolve(element_type="tet4")
```

A node selection materializes as `(id, xyz)` pairs with bulk `ids` /
`coords` arrays behind them; an element selection materializes per type, and
`.resolve()` flattens it to `(ids, connectivity)` when the selection is a
single type — or when you name the type you want out of a mixed one.

The record sets read just as directly. Each is iterable, and constraint
kinds are constants rather than magic strings, so a typo is a linter error
instead of a silently-skipped branch:

```python
K = fem.nodes.constraints.Kind
for c in fem.nodes.constraints.pairs():          # compounds auto-expand
    if c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)

for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass)
```

And when you want to *see* the model rather than emit it, `fem.inspect`
turns the same data into summaries — `fem.inspect.summary()` for the
one-liner, `constraint_summary()` / `load_summary()` / `mass_summary()` for
breakdowns that trace each record back to the name or definition that
produced it. The snapshot tells you not just what you have, but why you
have it.

## Why it's frozen

`FEMData` is immutable, and the immutability is load-bearing. Every
consumer — a solver bridge, a results file, a composed assembly — needs the
snapshot it was handed to still mean the same thing later, and the cheapest
way to guarantee that is to make "later" impossible to differ from "now".
That is also why the session freezes its build phase the moment the snapshot
exists (the lifecycle consequence covered in [The session](session.md)): a
kernel that kept mutating under a snapshot would make the snapshot a lie.

The guarantee is enforced, not just promised. Every snapshot carries a
deterministic content hash:

```python
fem.snapshot_id     # content-addressed identity, computed once and cached
```

Two snapshots with the same nodes, elements, groups, and records hash
identically; change anything and the hash changes. The hash is how a results
file is traced back to the exact mesh that produced it, and — as you'll see
below — how a reloaded file proves it wasn't corrupted or edited in transit.
An identity like that is only possible because the thing it identifies
cannot move.

## The solver contract

Because the snapshot is complete, self-contained, and dead, *any* solver can
consume it — that is the design's central claim. Nothing in `FEMData` knows
what OpenSees is. A hand-rolled consumer is just loops over the surfaces
above:

```python
for nid, xyz in fem.nodes.select().result():             # domain nodes
    ops.node(nid, *xyz)
for nid, xyz in fem.nodes.constraints.phantom_nodes():   # then phantoms
    ops.node(nid, *xyz)

for group in fem.elements.select().result():             # per-type groups
    for eid, conn in group:
        ops.element("FourNodeTetrahedron", eid, *conn, mat_tag)

for rec in fem.nodes.sp.homogeneous():                   # supports
    ...                                                  # rec.node_id, rec.dof

for load in fem.nodes.loads:                             # loads, per record
    fx, fy, fz = load.force_xyz or (0.0, 0.0, 0.0)
    ops.load(load.node_id, fx, fy, fz)
```

Swap the `ops.*` calls for Abaqus keywords or your own assembler and the
loops don't change — the broker's shapes are the contract, and they hold for
every consumer. In practice you will usually not write these loops at all:
the typed `apeSees(fem)` bridge consumes the same snapshot and adds the
judgment calls a raw loop leaves to you — which constraints auto-emit, how
load cases map to patterns, how per-node DOF counts are inferred. That
bridge is the next page's subject. What matters here is the direction of
the dependency: the bridge is *a* consumer of `FEMData`, not a privileged
partner. The seam between modeling and solving is the snapshot, and it cuts
all the way through.

## Persistence: the neutral zone

A snapshot that needs no live session is a snapshot that can live in a
file. The broker persists natively to HDF5:

```python
fem.to_h5("plate.h5", model_name="plate")   # write
fem = FEMData.from_h5("plate.h5")           # read back — no Gmsh anywhere
```

Everything described on this page rides along — nodes, per-type elements,
physical groups, labels, mesh selections, constraints, loads, masses, SP
records, and the interface-tie records. This is called the **neutral zone**:
the solver-agnostic model, and nothing else. No OpenSees content is written,
and its absence is deliberate — it is the "no solver loaded" signal. When
you want a fully enriched file, the bridge writes its own zone alongside the
neutral one (`apeSees(fem).h5(path)`); the two zones version independently,
and readers tolerate files written by the immediately prior schema version,
so a model saved by a slightly older apeGmsh still loads.

`from_h5` is fail-loud: it recomputes the rebuilt snapshot's `snapshot_id`
and verifies it against the hash stored in the file, raising rather than
handing back a silently-wrong model. The same layout also embeds inside
larger files — a composed `results.h5` carries the model under `/model/`,
and `FEMData.from_h5(path, root="/model")` rehydrates from there — which is
how a results file always knows exactly which mesh produced it.

Day to day you rarely call `to_h5` yourself: pass `save_to="model.h5"` to
the session and it autosaves the neutral zone on exit, or call `g.save()`
explicitly. The mechanics, options, and gotchas are in
[Save & reload a model](../how-to/save-reload.md); the full persistence API
is in the [FEMData reference](../api/fem.md).

## Composition over snapshots

Persistence makes one more thing possible: building *with* snapshots.
`apeGmsh.from_h5("host.h5")` reopens a saved model as a session with no
Gmsh kernel at all — geometry and meshing are gone, and what remains is
exactly what operates on frozen data: `g.compose(...)` grafts further saved
modules into the model, interface constraints couple the already-meshed
regions, and `g.save(...)` writes the assembly out again. Each composed
module leaves a provenance record on the result (`fem.composed_from`), and
every merged node and element remembers which module it came from — so even
a many-module assembly can trace each row to its source.

This is the immutability argument completing itself. Snapshots compose
*because* they are frozen: merging two live Gmsh sessions would mean
reconciling two mutable tag spaces, but merging two snapshots is arithmetic
on plain data. The recipe is in
[Compose modules](../how-to/compose-modules.md); the concept to keep is that
`FEMData` is not just the session's output — it is a first-class building
block in its own right.

---

*Next: [The OpenSees bridge](opensees-bridge.md).*
