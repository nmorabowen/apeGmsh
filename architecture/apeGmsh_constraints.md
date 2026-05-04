---
title: apeGmsh Constraints
aliases: [constraints, apeGmsh-constraints, ConstraintsComposite, ConstraintDef, MPC, multi-point-constraint]
tags: [apeGmsh, architecture, constraints, MPC, composite, def, record, resolver, solver]
---

# apeGmsh Constraints

> [!note] Companion document
> This file documents the **constraints subsystem** — how
> multi-point kinematic relations between parts ("tie this column to
> that slab", "this node drives that surface") are captured pre-mesh
> and resolved into concrete node-pair / node-group / interpolation
> records on the frozen broker. It assumes you have read
> [[apeGmsh_principles]] (tenets **(ix)** "three class flavours",
> **(xi)** "pre-mesh mutable, broker frozen", **(xii)** "pure
> resolvers, impure composites"), [[apeGmsh_partInstanceAssemble]] for
> the part-label vocabulary (every constraint references two part
> labels), and [[apeGmsh_broker]] for where the records end up
> (`fem.nodes.constraints`). Related: [[apeGmsh_loads]] — the loads
> subsystem shares the two-stage layout but targets a single side;
> constraints always relate *two* sides.

Every constraint ultimately expresses a linear MPC equation

$$\mathbf{u}_{\text{slave}} = \mathbf{C}\,\mathbf{u}_{\text{master}}$$

for some transformation matrix $\mathbf{C}$. The subsystem's job is to
build $\mathbf{C}$ from geometric intent. Like the loads subsystem,
this is a strict define-then-resolve pipeline, but the constraint
model carries *more* invariants than loads: both sides must be named
part-label instances, surface-type constraints need a `face_map`
dependency injected at `resolve()` time, and the taxonomy partitions
defs into four levels (plus a mixed-DOF sidepath for solid-shell
coupling).

```
src/apeGmsh/core/ConstraintsComposite.py      ← factory + target resolution
src/apeGmsh/solvers/Constraints.py            ← public re-export shim
src/apeGmsh/solvers/_constraint_defs.py       ← Stage-1 Def dataclasses
src/apeGmsh/solvers/_constraint_records.py    ← Stage-2 Record dataclasses
src/apeGmsh/solvers/_constraint_geom.py       ← shape functions, spatial index,
                                                 Newton projection
src/apeGmsh/solvers/_constraint_resolver.py   ← ConstraintResolver (pure math)
src/apeGmsh/solvers/_kinds.py                 ← ConstraintKind enum
src/apeGmsh/solvers/_opensees_constraints.py  ← solver emission (not discussed here)
```

The session exposes the composite as `g.constraints`. As with loads,
that's the full public surface.

---

## 1. Why constraints are harder than loads

Loads map `(geometry) → (nodal / element records)` in a single pass.
Constraints map `(geometry_A, geometry_B) → (MPC equation between the
two meshes)`. Three things make this non-trivial:

| Concern                  | Loads                                | Constraints                                  |
| ------------------------ | ------------------------------------ | -------------------------------------------- |
| Target arity             | one side                             | two sides (master + slave)                   |
| Dependency on node_map   | optional fast path                   | **required** — identifies instance ownership |
| Dependency on face_map   | never                                | **required for surface-type constraints**    |
| Geometric operation      | quadrature                           | proximity search + shape-function projection |
| Pure-numpy resolver?     | yes                                  | yes (but spatial index is more complex)      |
| Produces                 | per-node forces                      | MPC matrix rows                              |

This is why constraints get their own doc: the story isn't just
"stored defs, later records", it's also "how the two sides find each
other, what the face map does, and why node_to_surface needs phantom
nodes".

---

## 2. Taxonomy — four levels + a sidepath

The doctrinal grouping comes from the header of
`solvers/Constraints.py` and is worth memorising. Each row maps
cleanly to a Def subclass, a composite factory, a resolver method,
and ultimately a record type.

### Level 1 — Node-to-Node (1 master, 1 slave)

| Factory           | Def                | Geometry                                       |
| ----------------- | ------------------ | ---------------------------------------------- |
| `equal_dof`       | `EqualDOFDef`      | co-located node pairs within `tolerance`       |
| `rigid_link`      | `RigidLinkDef`     | master point + slave entities; `link_type`     |
| `penalty`         | `PenaltyDef`       | co-located pairs + spring `stiffness`          |

### Level 2 — Node-to-Group (1 master, N slaves)

| Factory                | Def                    | Geometry                                |
| ---------------------- | ---------------------- | --------------------------------------- |
| `rigid_diaphragm`      | `RigidDiaphragmDef`    | master point + slaves on a plane        |
| `rigid_body`           | `RigidBodyDef`         | master point + all slaves               |
| `kinematic_coupling`   | `KinematicCouplingDef` | master point + slaves, explicit `dofs`  |

### Level 2b — Mixed-DOF (1 6-DOF master → N 3-DOF slaves)

| Factory                     | Def                         | Geometry                          |
| --------------------------- | --------------------------- | --------------------------------- |
| `node_to_surface`           | `NodeToSurfaceDef`          | phantom-node rigid-link path      |
| `node_to_surface_spring`    | `NodeToSurfaceSpringDef`    | phantom-node stiff-beam path      |

### Level 3 — Node-to-Surface (1 node ↔ 1 face)

| Factory                  | Def                         | Geometry                                |
| ------------------------ | --------------------------- | --------------------------------------- |
| `tie`                    | `TieDef`                    | slave node projected onto master face   |
| `distributing_coupling`  | `DistributingCouplingDef`   | master point distributes to slave surf. |
| `embedded`               | `EmbeddedDef`               | embedded nodes follow host field        |

### Level 4 — Surface-to-Surface

| Factory           | Def                | Geometry                                       |
| ----------------- | ------------------ | ---------------------------------------------- |
| `tied_contact`    | `TiedContactDef`   | per-slave-node interpolation (N tie records)   |
| `mortar`          | `MortarDef`        | Lagrange-multiplier operator **B** on Γ        |

---

## 3. Target model — part labels, not PGs

> [!important] Constraints use part labels, not the multi-tier chain that
> loads use.
> [[apeGmsh_loads]] resolves targets through five sources (raw DimTag,
> mesh selection, label PG, user PG, part label). **Constraints do
> not.** A constraint's master and slave are always **part-label
> instance keys** from `g.parts._instances`. This is enforced at
> factory time:
>
> ```python
> # ConstraintsComposite._add_def (core/ConstraintsComposite.py:129)
> for lbl in (defn.master_label, defn.slave_label):
>     if lbl not in parts._instances:
>         raise KeyError(...)
> ```

Why the asymmetry? A load applies to *one* region; a constraint ties
*two* things that each need a stable identity across mesh
refinements. Part labels are the coarsest identity apeGmsh offers —
one per placement. Loads can afford finer granularity (an arbitrary
named PG on some face); constraints cannot, because the resolver
needs to ask `g.parts.build_node_map(...)` which nodes belong to each
side.

Optional `master_entities` / `slave_entities` (list of `(dim, tag)`)
narrow the search to a subset of the part's entities — the typical
use is *"this part has many surfaces; only this one is the
interface"*:

```python
g.constraints.tie(
    master_label="column",
    slave_label="slab",
    master_entities=[(2, 13)],     # one interface face
    slave_entities=[(2, 17)],
)
```

### 3.1 The exceptions: `node_to_surface` and `embedded`

The `_add_def` guard skips label validation for both
`NodeToSurfaceDef` (and its spring variant) **and** `EmbeddedDef`
(`ConstraintsComposite.py:236`). Both reach into Gmsh by raw entity
tags rather than part-label instances:

* `NodeToSurfaceDef` — "master" is a bare geometry *point* entity tag
  and "slave" is a raw surface entity tag.
* `EmbeddedDef` — accepts explicit `host_entities` and
  `embedded_entities` keyword args (`_constraint_defs.py:271-274`),
  each a list of `(dim, tag)`. When omitted, the resolver falls back
  to `_entities_for_label(label)` which tries the `g.parts` instance
  registry first, then a physical-group name fallback
  (`ConstraintsComposite.py:1272-1299`).

For `node_to_surface` the composite's
`node_to_surface(master, slave, ...)` factory resolves these via
`_helpers.resolve_to_tags(...)` up-front and stores:

- `master_label`: stringified point entity tag (dim=0)
- `slave_label`: comma-separated list of surface entity tags (dim=2)

At resolve time, `_resolve_node_to_surface` queries Gmsh for the mesh
node on the point entity (there must be exactly one) and the union of
mesh nodes on all surface entities. Shared-edge nodes are naturally
deduplicated by the set union.

---

## 4. ConstraintsComposite — the factory layer

`ConstraintsComposite` holds three pieces of state:

```python
self._parent : _ApeGmshSession         # to reach g.parts at validate-time
self.constraint_defs   : list[ConstraintDef]     # append-only pre-mesh
self.constraint_records: list[ConstraintRecord]  # written by resolve()
```

Factory methods validate labels, build a Def, append. No Gmsh calls at
factory time — the composite queries Gmsh only at `resolve()`.

### 4.1 Two dispatch tables

The composite keeps *two* parallel tables keyed by Def type:

```python
_DISPATCH: dict[type, str]         # Def type  →  composite-level dispatch method
_RESOLVER_METHOD: dict[type, str]  # Def type  →  ConstraintResolver method name
```

The split exists because several Def subclasses share the same
composite-level lookup strategy but dispatch to different resolver
methods. For example, `EqualDOFDef`, `RigidLinkDef`, and `PenaltyDef`
all use `_resolve_node_pair` (same node-pair lookup logic) but end up
calling three different `ConstraintResolver.resolve_*` methods.

```
_DISPATCH groups by LOOKUP          _RESOLVER_METHOD groups by MATH
─────────────────────────────       ──────────────────────────────
EqualDOF    ─┐                     EqualDOF     → resolve_equal_dof
RigidLink   ─┼─→ _resolve_node_pair RigidLink   → resolve_rigid_link
Penalty     ─┘                     Penalty      → resolve_penalty

RigidDiaphragm  → _resolve_diaphragm  RigidDiaphragm → resolve_rigid_diaphragm
RigidBody       ─┐                    RigidBody      → resolve_kinematic_coupling
KinematicCoupl. ─┼─→ _resolve_kinematic  KinematicCoupling → resolve_kinematic_coupling
Distributing    ─┘                    Distributing  → resolve_distributing

Tie             → _resolve_face_slave   Tie         → resolve_tie
TiedContact    ─┐                     TiedContact  → resolve_tied_contact
Mortar         ─┼─→ _resolve_face_both Mortar       → resolve_mortar

NodeToSurface       ─┐                 NodeToSurface       → resolve_node_to_surface
NodeToSurfaceSpring ─┴─→ _resolve_node_to_surface  NodeToSurfaceSpring → resolve_node_to_surface_spring

Embedded → _resolve_embedded   Embedded → resolve_embedded
                                (implemented for tet4 / tri3 hosts via ASDEmbeddedNodeElement)
```

### 4.2 Five dispatch helpers

The private `_resolve_*` methods on the composite group by *what kind
of geometric primitive the resolver needs*:

| Helper                      | Needs          | Used by                             |
| --------------------------- | -------------- | ----------------------------------- |
| `_resolve_node_pair`        | two node sets  | EqualDOF, RigidLink, Penalty        |
| `_resolve_diaphragm`        | node union     | RigidDiaphragm                      |
| `_resolve_kinematic`        | two node sets  | RigidBody, KinematicCoupling, Distributing |
| `_resolve_face_slave`       | master faces + slave nodes | Tie                       |
| `_resolve_face_both`        | both faces + both node sets | TiedContact, Mortar      |
| `_resolve_node_to_surface`  | direct Gmsh queries | NodeToSurface / Spring         |
| `_resolve_embedded`         | host elems + embedded nodes (implemented) | Embedded         |

Each helper calls two low-level utilities:

- `_resolve_nodes(label, role, defn, node_map, all_nodes) -> set[int]`
  — prefers `master_entities`/`slave_entities` if provided (queries
  Gmsh directly), otherwise falls back to `node_map[label]`.
- `_resolve_faces(label, role, defn, face_map) -> ndarray`
  — prefers `*_entities` via `parts._collect_surface_faces(...)`,
  otherwise falls back to `face_map[label]`.

---

## 5. `resolve()` — the freeze

```python
recs = g.constraints.resolve(
    node_tags, node_coords,
    elem_tags=None, connectivity=None,
    *,
    node_map=None, face_map=None,
) -> NodeConstraintSet
```

Triggered by `Mesh.get_fem_data(...)` (same seam as loads). The
caller is expected to provide:

- `node_map: dict[part_label, set[int]]` — which mesh nodes belong to
  each part instance, built by `g.parts.build_node_map(...)`.
- `face_map: dict[part_label, ndarray(n_faces, n_nodes_per_face)]` —
  connectivity rows for each part's surfaces, built by
  `g.parts.build_face_map(node_map)`.

Both are **dependency-injected** — `ConstraintsComposite` never
imports `PartsRegistry`. The motivation is tenet **(xii)**: the
resolver stays pure. If you call `resolve()` without a `face_map` and
you have surface-type defs (`TieDef`, `TiedContactDef`, `MortarDef`),
the composite *warns* and those constraints silently resolve to
empty records:

```python
if has_face_constraints and face_map is None:
    raise TypeError(
        "Surface constraints are defined but face_map=None. "
        "Call resolve(..., face_map=parts.build_face_map(node_map)) "
        "or use Mesh.get_fem_data() which builds face_map automatically.")
```

The end-user rarely sees this because `Mesh.get_fem_data` builds both
maps before calling resolve.

---

## 6. ConstraintResolver — pure math

`ConstraintResolver` holds only raw arrays plus two precomputed
lookup dicts (`_node_to_idx`, `_elem_to_idx`). It never imports
gmsh, never touches the session. Two heavyweight helpers live next
to it in `_constraint_geom.py`:

- `_SpatialIndex` — a KD-tree-lite used to find closest slave candidates for
  a master point, and closest master-face candidates for a slave node.
- `_project_point_to_face` — Newton iteration that finds
  `(ξ, η)` on the master face whose physical coordinate is closest
  to the slave node, then evaluates `SHAPE_FUNCTIONS[face_type]` to
  get interpolation weights.

Resolver methods (one per row of the taxonomy):

| Method                            | Returns                      |
| --------------------------------- | ---------------------------- |
| `resolve_equal_dof`               | `list[NodePairRecord]`       |
| `resolve_rigid_link`              | `list[NodePairRecord]` (with `offset`)  |
| `resolve_penalty`                 | `list[NodePairRecord]` (with `penalty_stiffness`) |
| `resolve_rigid_diaphragm`         | `NodeGroupRecord` (with `plane_normal`)  |
| `resolve_kinematic_coupling`      | `NodeGroupRecord`            |
| `resolve_distributing`            | `NodeGroupRecord`            |
| `resolve_tie`                     | `list[InterpolationRecord]`  |
| `resolve_tied_contact`            | `SurfaceCouplingRecord`      |
| `resolve_mortar`                  | `SurfaceCouplingRecord` (dense operator) |
| `resolve_node_to_surface`         | `NodeToSurfaceRecord`        |
| `resolve_node_to_surface_spring`  | `NodeToSurfaceRecord` (variant flag) |

No resolver method ever opens Gmsh. They receive fully resolved node
sets / face matrices and produce records.

---

## 7. ConstraintRecord hierarchy (post-mesh, resolved)

Five record types, each with an `expand_to_pairs()` or
`constraint_matrix()` method that flattens the record down to the
atomic MPC equation the solver ultimately needs:

| Record type                | Master → slave shape              | Flattening method         |
| -------------------------- | --------------------------------- | ------------------------- |
| `NodePairRecord`           | 1 ↔ 1                             | `constraint_matrix(ndof)` |
| `NodeGroupRecord`          | 1 ↔ N                             | `expand_to_pairs()` → `list[NodePairRecord]` |
| `InterpolationRecord`      | N ↔ 1 (shape-fn weighted)         | `constraint_matrix(ndof)` |
| `SurfaceCouplingRecord`    | N ↔ M (composite of tie records or mortar op) | —             |
| `NodeToSurfaceRecord`      | 1 ↔ N via phantom nodes           | `expand()` → rigid_links + equalDOFs |

`NodePairRecord.constraint_matrix()` encodes the family of kinematics:

```
EQUAL_DOF / PENALTY:  C = selection matrix  (rows of identity)

RIGID_BEAM:                           RIGID_ROD:
  ┌ u_s ┐   ┌ I   -[r×] ┐ ┌ u_m ┐       u_s = u_m + θ_m × r
  │     │ = │            │ │     │       (θ_s free — only translations coupled)
  │ θ_s │   │ 0      I   │ │ θ_m │
  └     ┘   └            ┘ └     ┘
```

where `[r×]` is the skew-symmetric matrix of the rigid arm
`r = x_slave - x_master`. This is the single equation that powers
all of Level 1 and (via `NodeGroupRecord.expand_to_pairs()`) Level 2.

`InterpolationRecord` carries `master_nodes: list[int]`,
`weights: ndarray` (shape-function values on the projected parametric
coords), `projected_point`, and `parametric_coords`. The `weights`
sum to 1.0 (partition of unity) and the constraint equation is

$$u_{\text{slave}} = \sum_i w_i \, u_{\text{master}_i}.$$

---

## 8. `NodeToSurfaceDef` — why phantom nodes

The Level-2b sidepath deserves its own section because it's the
single constraint type that *creates mesh nodes at resolve time*.

Problem: a 6-DOF reference node (e.g., a fork support or moment
applicator) needs to couple to a solid surface whose nodes are
3-DOF. Direct `rigid_link` from 6-DOF master to 3-DOF slaves produces
an ill-conditioned reduced stiffness because the master rotation DOFs
have no on-diagonal stiffness contribution.

Solution: phantom nodes. The resolver:

1. For each real slave node, creates a **phantom** node at the same
   coordinates with 6 DOFs. Phantom tags start at
   `max_existing_tag + 1`.
2. Emits `rigid_beam` constraints from master → each phantom (this
   gives master rotations full 6×6 kinematic coupling).
3. Emits `equal_dof` on DOFs `[1,2,3]` from each phantom → original
   slave (translations coupled; rotations independent).

The `NodeToSurfaceRecord` bundles all three lists
(`phantom_nodes`, `rigid_link_records`, `equal_dof_records`) so the
solver can emit them in one block. `expand()` flattens to
`list[NodePairRecord]` in the correct order (rigid links first,
equalDOFs second).

### 8.1 The `_spring` variant

`node_to_surface_spring` swaps step 2 for a stiff `elasticBeamColumn`
element instead of a kinematic `rigidLink`. Use this when:

- The master carries **free rotational DOFs** that receive direct
  moment loading (e.g., a fork support on a solid end face), and
- The kinematic rotation coupling would leave the master rotation
  DOFs without direct stiffness, producing ill-conditioning.

Topology, factory signature, and slave deduplication are identical
to the constraint-based variant. The only difference is the solver
emission path — the bridge detects `NodeToSurfaceSpringDef` and
generates `ops.element('elasticBeamColumn', ...)` with a big `(A, E,
I, J)` for each master → phantom link, then `ops.equalDOF(master,
phantom, *dofs)` for the phantom → slave links.

---

## 9. End-to-end skeleton

```python
import apeGmsh as ape
import openseespy.opensees as ops

with ape.apeGmsh("connection.geo") as g:
    with g.parts.part("beam") as p:
        p.box(3, 0.2, 0.4)
    with g.parts.part("column") as p:
        p.box(0.4, 0.4, 3)

    g.parts.register_instances({"beam": (0, 0, 1.3),
                                "column": (0, 0, 0)})

    # ── Level 1 ───────────────────────────────────────────────
    g.constraints.equal_dof(
        master_label="beam", slave_label="column",
        tolerance=1e-3)

    # ── Level 2 ───────────────────────────────────────────────
    g.constraints.rigid_diaphragm(
        master_label="slab", slave_label="slab",
        master_point=(5, 5, 3),
        plane_normal=(0, 0, 1),
        constrained_dofs=[1, 2, 6])

    # ── Level 2b ──────────────────────────────────────────────
    g.constraints.node_to_surface(
        master="fork_support_point",    # str name of a geometry point
        slave="column_top_face",        # str name of a surface
        dofs=[1, 2, 3])

    # ── Level 3 ───────────────────────────────────────────────
    g.constraints.tie(
        master_label="beam", slave_label="column",
        master_entities=[(2, 5)],   # only this face is the interface
        slave_entities=[(2, 7)],
        tolerance=1.0)

    g.mesh.generate(size=0.1)
    fem = g.mesh.get_fem_data(dim=3)   # node_map + face_map built here

# Solver emission
for rec in fem.nodes.constraints:          # atomic iteration (default)
    if rec.kind == "equal_dof":
        ops.equalDOF(rec.master_node, rec.slave_node, *rec.dofs)
    elif rec.kind in ("rigid_beam", "rigid_rod"):
        ops.rigidLink(rec.kind.replace("rigid_", ""),
                      rec.master_node, rec.slave_node)

# Or compound iteration for NodeToSurface
for rec in fem.nodes.constraints.node_to_surfaces():
    for ph_tag, ph_coords in zip(rec.phantom_nodes, rec.phantom_coords):
        ops.node(ph_tag, *map(float, ph_coords))
    for p in rec.rigid_link_records:
        ops.rigidLink("beam", p.master_node, p.slave_node)
    for p in rec.equal_dof_records:
        ops.equalDOF(p.master_node, p.slave_node, *p.dofs)
```

See [[apeGmsh_broker#atomic-vs-compound]] for the iteration
convention — atomic vs compound iterators on a record set let the
solver pick between "flatten everything to node pairs" and
"preserve compound structure" (needed for `NodeToSurfaceRecord`
because you have to emit the phantom nodes before the rigid links).

---

## 10. Contributor rules

1. **Every new constraint is a Def + a `_DISPATCH` entry + a
   `_RESOLVER_METHOD` entry + a resolver method + (optionally) a
   record subclass.** The five tables are the source of truth;
   there is no "magic" that auto-infers any of them.
2. **Use part labels for master/slave.** The exceptions are
   `NodeToSurfaceDef`/`NodeToSurfaceSpringDef` (raw point/surface
   tags) and `EmbeddedDef` (entity-list keyword args). If your new
   constraint also bypasses label resolution, add it to the
   `isinstance(defn, (NodeToSurfaceDef, EmbeddedDef))` guard in
   `_add_def` (or refactor the guard into a method).
3. **The resolver stays pure.** No gmsh imports in
   `_constraint_resolver.py` or `_constraint_geom.py`. If you need a
   mesh query at resolve time, add it to the composite's
   `_resolve_*` helper and pass the resolved array down.
4. **Surface-type defs go into `_FACE_TYPES`.** The warning about
   `face_map=None` keys off this tuple; missing your def from it
   means silent failure when the caller forgets the face map.
5. **Records never carry `(dim, tag)`.** All resolved records
   reference mesh node tags and element tags, never geometry
   entities. Record flattening (`expand_to_pairs`,
   `constraint_matrix`) must also be OCC-free.
6. **Phantom tags start at `max_existing_tag + 1`.** If you add
   another def type that creates nodes at resolve time, use the
   same offset strategy — never hardcode a tag range. Collisions
   with solver-side elements are a footgun.
7. **`tolerance` is model-space, not parametric.** Every Def takes a
   `tolerance` in model units (metres, millimetres — whatever the
   user's Gmsh units are). Don't default to `1e-6` for surface
   searches: use the mesh characteristic length. The defaults in
   `_constraint_defs.py` reflect this (`tie` defaults to `1.0`,
   `equal_dof` defaults to `1e-6`).

---

## 11. Where it plugs in

- **Stored on the session** as `g.constraints` (a
  `ConstraintsComposite` instance, built in
  `apeGmsh._core.apeGmsh.__init__`; see [[apeGmsh_architecture]] §3
  on composites).
- **Called by** `Mesh.get_fem_data(...)` after meshing — the
  composite is passed the post-mesh `node_tags`, `node_coords`,
  `elem_tags`, `connectivity`, **plus** the `node_map` and
  `face_map` built from `g.parts`.
- **Stored as** `fem.nodes.constraints: NodeConstraintSet` on the
  broker — see [[apeGmsh_broker#fem.nodes|`fem.nodes`]] for the
  record-set conventions (atomic vs compound iterators).
- **Emitted by** `solvers/_opensees_constraints.py`, which
  dispatches on `record.kind` to produce `ops.equalDOF`,
  `ops.rigidLink`, `ops.rigidDiaphragm`, `ops.sp`, and (for
  `NodeToSurfaceRecord` / spring variant) the phantom-node +
  beam-element combo.

The constraint pipeline, summarised as an FSM:

```
factory (pre-mesh)
  ↓  label validation against g.parts._instances
store in constraint_defs
  ↓  wait for mesh
resolve (post-mesh, with node_map + face_map)
  ↓  spatial search + shape-function projection
NodeConstraintSet on fem.nodes.constraints
  ↓  atomic vs compound iteration
solver emission (OpenSees / Abaqus / …)
```

If you add a seventh arrow — say, post-resolve validation that
queries Gmsh — you've broken tenet **(v)** "broker is the boundary";
downstream of `from_gmsh(...)` no object references the OCC kernel.
