# apeGmsh loads

A guide to defining loads in an apeGmsh session — concentrated forces,
distributed line and surface loads, gravity, and generic body forces.
This document is solver-agnostic: it describes the apeGmsh abstraction
and the data that ends up in the FEM broker, not any particular solver
adapter. The OpenSees bridge is mentioned only where the mapping is
illuminating; see `guide_fem_broker.md` for how a broker is handed to a
solver.

The guide is grounded in the current source:

- `src/apeGmsh/core/LoadsComposite.py` — the user-facing composite
- `src/apeGmsh/solvers/Loads.py` — `LoadDef`, `LoadRecord`, and
  `LoadResolver` (pure mesh math, no Gmsh)
- `src/apeGmsh/mesh/FEMData.py` — the `NodalLoadSet` / `SPSet` / `ElementLoadSet` that lands in the broker

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```


## 1. The two-stage pipeline: define, then resolve

Loads in apeGmsh follow the same two-stage pattern as constraints and
masses. You *define* loads before meshing against high-level targets
(part labels, physical groups, mesh selections), and the library
*resolves* them to node- and element-level records after the mesh
exists. You never write `nodeId → force` by hand.

The first stage is bookkeeping. When you call `g.loads.point(...)`,
`g.loads.line(...)`, or any other factory method, a `LoadDef` dataclass
is appended to `g.loads.load_defs`. It stores the intent — *"apply
−3 kN/m² to the slab named 'roof'"* — together with the active pattern
name, the reduction strategy (`"tributary"` vs `"consistent"`), and the
target form (`"nodal"` vs `"element"`). Nothing is computed yet, and
the load holds no mesh information because no mesh may exist.

The second stage runs during FEM extraction. When you call
`g.mesh.get_fem_data()`, the mesh composite walks `g.loads.load_defs`,
resolves every target to the relevant node / edge / face / volume
connectivity through Gmsh, and hands those raw arrays to a
`LoadResolver` which produces `NodalLoadRecord` or `ElementLoadRecord`
objects. The resolved records are stored in `g.loads.load_records` and
copied into `fem.nodes.loads` (nodal) and `fem.elements.loads` (element) as a `NodalLoadSet` / `ElementLoadSet`. These are the only objects the
solver adapter ever sees.

The practical consequence is that you can rebuild the mesh, refine it,
or swap out a part entirely, and your load script does not change — the
same definitions are re-resolved against the new mesh. It is also the
reason load definitions accept *names* rather than raw `(dim, tag)`
pairs: names survive remeshing; tags do not.


## 2. Targets: what a load can be attached to

Every load factory takes a `target` as its first argument. A target is
any of four things, resolved in this order by
`LoadsComposite._resolve_target`:

1. **A raw list of `(dim, tag)` tuples.** Escape hatch for low-level
   code. `g.loads.point([(0, 12)], force_xyz=(0, 0, -1000))`.
2. **A mesh selection name.** If `g.mesh_selection` has a stored set
   under that name, the load attaches to its nodes or elements directly.
   This is the only way to load a subset of an entity that was carved
   out by a post-mesh selection.
3. **A physical group name.** Looked up via
   `gmsh.model.getPhysicalGroups()` and `getPhysicalName`. Loads apply
   to every entity in the group.
4. **A part label.** Falls back to `g.parts.instances[label].entities`.

If a name matches nothing in any of the four layers, you get a
`KeyError` immediately at definition time, not during resolution. That
is deliberate: typos are cheap to fix before meshing and expensive to
chase after.

The four-layer lookup is the reason this guide recommends physical
groups or part labels for almost everything. Both are created from
geometry and survive any mesh refinement. Mesh selections are powerful
but coupled to a specific mesh state; use them when you genuinely need
to pick nodes by coordinate, proximity, or a post-mesh query, and
accept that the selection has to be rebuilt if the mesh changes.


## 3. Patterns

A load *pattern* is a string label that groups load definitions into a
logical set. Patterns map directly onto the concept of a load pattern
or load case in structural solvers: everything inside one pattern gets
applied together, scaled by the same factor, and a solver adapter is
free to emit one `timeSeries` / `pattern` pair per pattern name.

You group loads under a pattern with a context manager:

```python
with g.loads.pattern("dead"):
    g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
    g.loads.line("beams",      magnitude=-2.0e3, direction="z")

with g.loads.pattern("live"):
    g.loads.surface("slabs", magnitude=-3.0e3)

with g.loads.pattern("wind_X"):
    g.loads.surface("facade_west", magnitude=+1.2e3, normal=True)
```

Outside any `with` block, definitions fall into a pattern called
`"default"`. You can query what you have with `g.loads.patterns()` (a
list of pattern names in insertion order) and
`g.loads.by_pattern("dead")` (the definitions belonging to that
pattern). After resolution, the same grouping is preserved on the
resolved records, so the solver adapter can walk the broker one pattern
at a time:

```python
fem = g.mesh.get_fem_data()
for pat in fem.nodes.loads.patterns():
    # open a solver load pattern here
    for rec in fem.nodes.loads.by_pattern(pat):
        # emit one command per record (nodal loads)
        ...
    for rec in fem.elements.loads.by_pattern(pat):
        # emit one command per record (element loads)
        ...
```

Patterns are the only place in the load API that carries load-case
semantics. Everything else — reduction strategy, target form,
magnitude units — is orthogonal.


## 4. `reduction` and `target_form`: how a distributed load becomes numbers

Every distributed load (line, surface, gravity, body) takes two
keyword arguments that control how it is discretized onto the mesh:

- `reduction` — `"tributary"` (default) or `"consistent"`. Controls
  the *math*: how a distributed load becomes equivalent point loads.
- `target_form` — `"nodal"` (default) or `"element"`. Controls the
  *output shape*: a list of per-node force records, or a list of
  per-element load commands.

These are independent choices, and not every combination is supported
for every load type. The dispatch table lives at the top of
`LoadsComposite.py`; if you request an unsupported combination you get
a clear `ValueError` at definition time, not a silent miscalculation.

### Tributary vs consistent

*Tributary* reduction lumps the distributed load onto nodes by area
(or length) share. For a uniform line load `q` over an edge of length
`L`, each end node receives `q·L/2`. For a uniform surface pressure
over a triangular face of area `A`, each corner node receives `p·A/3`.
This is exact for uniform loads on straight / flat elements, trivial
to compute, and has the great virtue of being obviously correct when
you inspect the resulting nodal force vector by hand.

*Consistent* reduction integrates the load against the element shape
functions, so a distributed load on a higher-order element lands on
mid-side nodes with the correct weighting. For linear elements it
coincides with the tributary result (up to the sign convention on
nonuniform loads). The practical reason to pick `"consistent"` is
that you are using quadratic line / surface elements and care about
the moment produced by a linear pressure ramp, or you are matching a
textbook derivation.

As a rule of thumb: use `"tributary"` for every beam/truss model and
for linear solid / shell meshes. Reach for `"consistent"` when you
have quadratic elements *and* the load is nonuniform *and* you need to
match a closed-form solution. In all other cases `"tributary"` is
simpler and indistinguishable in the answer.

### Nodal vs element form

`target_form="nodal"` produces `NodalLoadRecord` objects — one force /
moment tuple per affected node. A nodal record is the most portable
output: every solver understands "apply this force to this node".
This is the default, and for point loads it is the only option.

`target_form="element"` produces `ElementLoadRecord` objects instead —
one record per element, carrying a `load_type` string
(`"beamUniform"`, `"surfacePressure"`, `"bodyForce"`, ...) and a
`params` dictionary. This maps onto native element-level load commands
(`eleLoad -beamUniform q_y q_x` in OpenSees, for example), which is
what you want when a beam element can carry the distributed load
natively: it preserves the moment distribution along the element
instead of collapsing it into two end forces.

Again, a rule of thumb: for beams loaded along their span, choose
`target_form="element"` so the solver sees a true beam load and
recovers the parabolic bending moment diagram. For surface tractions
on continuum elements, `target_form="nodal"` is almost always what you
want — element-level surface pressures are element-library-specific
and not universally supported.


## 5. Point loads

A point load applies the **same** force and/or moment to every node
that resolves from its target. The most common pattern is to target
a physical group that was defined on a single geometric point or
vertex:

```python
# A 10 kN downward tip load on a single-vertex physical group
g.physical.add_point("tip", [(0, vtx_tag)])

with g.loads.pattern("tip_load"):
    g.loads.point("tip", force_xyz=(0.0, 0.0, -10_000.0))
```

`force_xyz` is optional; so is `moment_xyz`. At least one must be
nonzero, and specifying a moment on a model whose DOF count is less
than 4 raises a `ValueError` during resolution (no rotational DOFs to
absorb it). A 2D model (`ndf=3`) accepts `force_xyz=(Fx, Fy)` and a
single scalar moment `moment_xyz=(Mz,)`.

Targeting anything with more than one node broadcasts the *same*
force to each of them, which is rarely what you want for a line or
area — that is what distributed loads are for. The only legitimate
use of a multi-node point-load target is applying an identical
reaction at a row of pinned nodes (e.g., a support jack).


## 6. Line loads

A line load is a distributed force per unit length along a 1-D
geometric entity (a curve) or the underlying beam / truss elements
that were meshed on it. You can specify it in two equivalent ways:

```python
# Scalar magnitude + direction
g.loads.line("beams", magnitude=-2.0e3, direction="z")

# Explicit per-component vector
g.loads.line("beams", q_xyz=(0.0, 0.0, -2.0e3))
```

`magnitude` is a scalar (force per unit length). `direction` is
either an axis name `"x" | "y" | "z"` or a 3-vector; it is not
normalized, so `direction=(0, 0, -1)` gives you −1 per unit length,
and `direction=(0, 0, -9.81)` gives you −9.81 per unit length.
`q_xyz` is the explicit vector form and takes precedence over
`magnitude`/`direction` when both are provided.

The `reduction`/`target_form` choices matter most here. A beam loaded
along its span is the classic case where element-form output is
worth it:

```python
with g.loads.pattern("live"):
    g.loads.line(
        "floor_beams",
        magnitude=-5.0e3, direction="z",
        reduction="consistent",
        target_form="element",
    )
```

With `target_form="element"` the broker contains one
`ElementLoadRecord` per beam element, carrying the original `q_xyz`
and an element id. An OpenSees adapter emits `ops.eleLoad("-ele",
eid, "-type", "-beamUniform", qy, qz, qx)` and the solver recovers a
proper parabolic moment diagram inside the element. With
`target_form="nodal"`, the same line load lumps into two end forces
per element and the internal moment diagram is piecewise linear —
still correct at the nodes, but wrong between them.


## 7. Surface loads

Surface loads apply a pressure or traction to a 2-D geometric entity
or the mesh faces sitting on it. The distinction between "pressure"
and "traction" is controlled by the `normal` flag:

```python
# Pressure (scalar, perpendicular to each face, positive into the face)
g.loads.surface("roof", magnitude=-3.0e3, normal=True)

# Traction (vector, same direction on every face regardless of normal)
g.loads.surface("facade_west", magnitude=+1.2e3, direction=(1, 0, 0),
                normal=False)
```

Pressure is the right model for wind, snow, water, or any load that
follows the surface orientation: if the roof is sloped, the load is
perpendicular to the slope without you having to resolve components.
Traction is the right model for a follower-force experiment or a
uniform "pull" on a face that does not care about curvature.

For continuum (solid or shell) meshes, stay with the default
`target_form="nodal"`. The tributary reduction gives each corner node
of a face its share of `p·A` and accumulates across faces that share
a node, and the result is exactly the equivalent nodal force vector
you would write by hand. Element-form surface loads exist (see the
dispatch table in `LoadsComposite.py`) but are solver-specific and
rarely worth the plumbing.


## 8. Gravity

Gravity is a body load with special affordances. Instead of writing
`g.loads.body(vol, force_per_volume=(0, 0, -ρ·g))` by hand, you write:

```python
g.loads.gravity("concrete_columns", g=(0, 0, -9.81), density=2400)
```

The `g` vector defaults to `(0, 0, -9.81)`, so the common case is a
one-liner:

```python
g.loads.gravity("concrete_columns", density=2400)
```

`density=None` is also valid and tells the solver bridge to read the
density from the assigned material or section — this is the
recommended path once you have materials defined, because it keeps
the load definition in sync with the material model and avoids the
classic bug of updating one and forgetting the other.

As with distributed loads, `reduction` and `target_form` control how
gravity becomes numbers. Tributary-nodal is the default and the
correct choice for almost every case: each corner of a volume element
gets `(ρ·g·V)/n_corners`. Consistent-nodal is the shape-function
integral, worth using only with quadratic volume elements. The
element-form output hands the solver a per-element body force it can
apply natively.

Gravity targets must resolve to *volumes* (3-D entities). Applying
`g.loads.gravity(...)` to a surface or curve target is a no-op: the
volume iterator in `_target_elements` silently skips non-3D dim-tags.
This is a deliberate choice — gravity on a shell is meaningful only
with a thickness and a material density, which belongs on the shell
section, not the load.


## 9. Generic body forces

Any body force that is *not* gravity — a magnetic body force, a
centrifugal acceleration, a thermal expansion driving force re-cast
as a body load — goes through `g.loads.body`:

```python
g.loads.body("rotor_volume", force_per_volume=(0, 0, -1.5e4))
```

The only difference from `gravity` is the absence of a density term:
`force_per_volume` is the full vector, already multiplied by whatever
density or scalar field produces it. If you find yourself writing
`force_per_volume=(0, 0, -density*9.81)` by hand, you want `gravity`
instead.


## 10. Face-concentrated loads

When you need to apply a force or moment to a solid face *without*
coupling it to a structural element, `face_load` distributes the
load directly to the face nodes — no reference node, no phantom, no
conditioning issues:

```python
with g.loads.pattern("tip"):
    g.loads.face_load(pg="tip_face", force_xyz=(0, 0, -10_000))
    g.loads.face_load(pg="tip_face", moment_xyz=(0, 100_000, 0))
```

**Force distribution:** the total force `F` is split equally among
all `N` nodes on the face: `f_i = F / N`.

**Moment distribution:** the moment `M` about the face centroid is
converted to statically equivalent nodal forces via a least-norm
solve. The solver builds a `6 × 3N` equilibrium matrix encoding
`Sum(f_i) = 0` and `Sum(r_i × f_i) = M`, then finds the minimum-norm
force vector `f = A^T (A A^T)^{-1} b`. The result is exact: the
nodal forces produce zero net force and the desired net moment.

Both contributions are combined and accumulated into `NodalLoadRecord`
entries — the same type produced by `point()` or `surface()`. The
downstream solver adapter sees no difference.

**When to use `face_load` vs `node_to_surface`:**

- Use `face_load` when you only need to apply a load or BC to a face
  and have no structural element at the reference point.
- Use `node_to_surface` when you need the reference node as a
  structural node (e.g., connecting a frame beam to a solid face).


## 11. Face-prescribed displacements

`face_sp` maps a rigid-body motion at the face centroid to per-node
prescribed displacements:

```python
# Homogeneous fix (all face nodes fixed in x, y, z)
g.loads.face_sp(pg="base_face", dofs=[1, 1, 1])

# Prescribed translation at centroid
g.loads.face_sp(pg="base_face", dofs=[1, 1, 1],
                disp_xyz=(0.01, 0, 0))

# Prescribed rotation about centroid
g.loads.face_sp(pg="base_face", dofs=[1, 1, 1],
                rot_xyz=(0, 0, 0.01))

# Combined
g.loads.face_sp(pg="base_face", dofs=[1, 1, 1],
                disp_xyz=(0.01, 0, 0), rot_xyz=(0, 0, 0.01))
```

For each face node, the displacement is computed as:

    u_i = disp_xyz + rot_xyz × r_i

where `r_i` is the arm vector from the face centroid to node `i`.

The result is a list of `SPRecord` entries stored in `fem.nodes.sp`.
A solver adapter emits them as `ops.fix()` (homogeneous) or
`ops.sp(node, dof, value)` (non-zero).

**Note:** `face_sp` lives on `LoadsComposite` (not on `ConstraintsComposite`)
because it produces boundary conditions, not multi-point constraints.
The `dofs` mask follows the same convention as `elements.fix()`.


## 12. Putting it all together

A realistic load script for a small building model reads like this,
and it is the shape of script you should aim for:

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="office_block")
g.begin()

# ... parts, physical groups, mesh ...

with g.loads.pattern("self_weight"):
    g.loads.gravity("rc_volumes", density=2400)
    g.loads.gravity("steel_volumes", density=7850)

with g.loads.pattern("superimposed_dead"):
    g.loads.surface("floors", magnitude=-1.5e3)   # finishes
    g.loads.line("facade_beams", magnitude=-3.0e3, direction="z",
                 target_form="element")

with g.loads.pattern("live"):
    g.loads.surface("floors",      magnitude=-3.0e3)
    g.loads.surface("roof",        magnitude=-1.0e3)

with g.loads.pattern("crane_point"):
    g.loads.point("crane_hook", force_xyz=(0, 0, -50_000))

# Resolve happens automatically inside get_fem_data()
fem = g.mesh.get_fem_data()

for pat in fem.nodes.loads.patterns():
    nodal   = fem.nodes.loads.by_pattern(pat)
    element = fem.elements.loads.by_pattern(pat)
    print(pat, len(nodal), "nodal records,", len(element), "element records")

g.end()
```

Three things about this script are worth noting. First, every load
targets a *name* — nothing references a Gmsh tag directly, so the
script survives any remesh. Second, patterns are used
aggressively: each one names a physically meaningful load case that
will eventually correspond to a `timeSeries` / `pattern` pair in the
solver. Third, the script is declarative: there is no loop over
nodes, no manual integration, no `if ndf == 3 else 6`. The
`LoadResolver` handles all of that when `get_fem_data()` runs.


## 13. Debugging loads

Loads are easy to get wrong and easy to inspect. Three cheap sanity
checks will catch almost everything:

```python
fem = g.mesh.get_fem_data()

# (a) how many records in each pattern?
for pat in fem.nodes.loads.patterns():
    nodal   = fem.nodes.loads.by_pattern(pat)
    element = fem.elements.loads.by_pattern(pat)
    print(pat, len(nodal), "nodal,", len(element), "element")

# (b) total applied force per pattern — should match hand calc
import numpy as np
for pat in fem.nodes.loads.patterns():
    total_f = np.zeros(3)
    total_m = np.zeros(3)
    for rec in fem.nodes.loads.by_pattern(pat):
        if rec.force_xyz  is not None: total_f += np.asarray(rec.force_xyz)
        if rec.moment_xyz is not None: total_m += np.asarray(rec.moment_xyz)
    print(pat, "ΣF =", total_f, "ΣM =", total_m)

# (c) visual check via the viewer
g.mesh.view.load_arrows(pattern="live")   # if implemented
```

Check (b) is the one that catches the most bugs: a gravity load whose
sign is inverted, a pressure applied to the wrong face orientation,
or a unit-mismatch in `density` (kg/m³ vs g/cm³) all show up as an
order-of-magnitude error in ΣF that you spot on the first read.

If a load definition succeeds but produces no records, the target
almost certainly resolved to the wrong entity dimension: a pressure
targeted at a volume label, a gravity targeted at a surface. The
`_target_*` methods in `LoadsComposite` filter by dimension and
silently drop mismatches, which is friendly for mixed-entity targets
but less friendly when you typed the wrong label.


## 14. What a solver adapter sees

A solver adapter never calls into `LoadsComposite`. It reads a
`NodalLoadSet` / `SPSet` / `ElementLoadSet` out of the broker and
walks them:

```python
fem = g.mesh.get_fem_data()

for pat in fem.nodes.loads.patterns():
    # open a solver load pattern / load case here
    for rec in fem.nodes.loads.by_pattern(pat):
        # rec.node_id, rec.force_xyz, rec.moment_xyz
        # Either may be None; each is a length-3 tuple when present.
        ...
    for rec in fem.elements.loads.by_pattern(pat):
        # rec.element_id, rec.load_type, rec.params
        ...
```

`NodalLoadRecord` stores spatial 3-vectors (`force_xyz`, `moment_xyz`)
and knows nothing about the solver's DOF count. The adapter maps
those vectors onto its DOF space at emit time — a 2D planar frame
slices `(Fx, Fy)` from `force_xyz` and `Mz` from `moment_xyz`; a 3D
frame uses all six; a 3D solid ignores moments.
`ElementLoadRecord.load_type` is a string — `"beamUniform"`,
`"surfacePressure"`, `"bodyForce"` — that a solver adapter switches on
to pick the right native command. Both records carry their `pattern`
and `name`, so the adapter can replay the grouping the user wrote.

**SP records** from `face_sp` live in a separate sub-composite:

```python
# Homogeneous fix records
for rec in fem.nodes.sp.homogeneous():
    ops.fix(rec.node_id, ...)

# Prescribed displacement records
for rec in fem.nodes.sp.prescribed():
    ops.sp(rec.node_id, rec.dof, rec.value)
```

The contract is deliberately dumb: a list of records, each fully
self-describing, no cross-references. That is what makes the broker
picklable, the solver adapter short, and unit tests possible without
Gmsh in the loop. Every bit of cleverness — tributary vs consistent,
target resolution, DOF padding — lives on the apeGmsh side of the
line, where it can be tested in isolation against a synthetic mesh.


## See also

- `guide_basics.md` — session lifecycle and geometry basics
- `guide_selection.md` — building physical groups and mesh selections
  that loads can target
- `guide_fem_broker.md` — how `NodalLoadSet` / `ElementLoadSet` end up
  in the broker and how a solver adapter walks them
