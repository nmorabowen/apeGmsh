# apeGmsh constraints

A guide to defining multi-point constraints in an apeGmsh session —
from simple DOF ties to surface coupling and mortar methods. This
document covers the apeGmsh abstraction; see `guide_fem_broker.md` for
how constraints land in the broker and get consumed by a solver.

Grounded in the current source:

- `src/apeGmsh/core/ConstraintsComposite.py` — the user-facing composite
- `src/apeGmsh/solvers/Constraints.py` — `ConstraintDef`, `ConstraintRecord`,
  and `ConstraintResolver`
- `src/apeGmsh/mesh/_record_set.py` — `NodeConstraintSet`, `SurfaceConstraintSet`

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```


## 1. The two-stage pipeline: define, then resolve

Constraints follow the same pattern as loads and masses. You *define*
constraints before meshing against high-level targets (part labels,
physical groups), and the library *resolves* them to node-level records
after the mesh exists.

```python
# Stage 1 — define (pre-mesh)
g.constraints.equal_dof("slab", "column_top", dofs=[1, 2, 3])

# Stage 2 — resolve (happens inside get_fem_data)
fem = g.mesh.queries.get_fem_data(dim=3)

# Resolved records are now on the broker
fem.nodes.constraints       # NodeConstraintSet  (node-pair types)
fem.elements.constraints    # SurfaceConstraintSet (surface types)
```

The definitions reference *names*, not tags — so they survive remeshing.


## 2. DOF numbering

All constraint definitions use **1-based DOF numbering**:

| DOF | Meaning |
|-----|---------|
| 1   | ux — translation X |
| 2   | uy — translation Y |
| 3   | uz — translation Z |
| 4   | rx — rotation X |
| 5   | ry — rotation Y |
| 6   | rz — rotation Z |

For a 3-DOF model (ndf=3, solids), only DOFs 1-3 exist.
For a 6-DOF model (ndf=6, frames/shells), all 6 are available.


## 3. Constraint types

### Level 1 — Node-to-node

**`equal_dof`** — tie co-located nodes so they share selected DOFs.
The most common constraint. Finds node pairs within a tolerance and
couples them.

```python
g.constraints.equal_dof(
    "slab", "column_top",
    dofs=[1, 2, 3],        # couple translations only
    tolerance=1e-6,
)
```

Use this when two parts share a boundary and you want continuity of
displacement across it. If you omit `dofs`, all DOFs are tied.

**`rigid_link`** — rigid bar coupling between a master node (or point)
and slave nodes. Two types:

```python
# "beam" — full 6-DOF rigid body motion (default)
g.constraints.rigid_link("center", "perimeter", link_type="beam")

# "rod" — translations only, rotations free
g.constraints.rigid_link("center", "perimeter", link_type="rod")
```

Use `rigid_link` for connecting a single master (e.g., column centroid)
to a ring of slave nodes. The difference from `equal_dof` is that
rigid links enforce rigid body kinematics — rotations at the master
produce translations at the slaves proportional to their offset.

**`penalty`** — soft spring approximation of `equal_dof`. Useful when
the solver has trouble with algebraic constraints (Lagrange multipliers)
and a spring coupling is more stable:

```python
g.constraints.penalty("part_A", "part_B", stiffness=1e10, dofs=[1, 2, 3])
```

The stiffness should be large enough to be effectively rigid but not so
large that it causes ill-conditioning. A rule of thumb: 10x to 1000x the
stiffest element in your model.


### Level 2 — Node-to-group

**`rigid_diaphragm`** — enforces in-plane rigidity at a floor level.
All slave nodes at the plane follow the master node's in-plane DOFs:

```python
g.constraints.rigid_diaphragm(
    "center_col",         # master label (the reference node)
    "floor_nodes",        # slave label (all floor nodes)
    plane_normal=(0, 0, 1),
    plane_tolerance=0.5,  # tolerance for "at the plane"
)
```

This is the classic floor-diaphragm constraint for building models.
The `plane_tolerance` filters slave nodes to only those within the
specified distance of the diaphragm plane.

**`rigid_body`** — full 6-DOF rigid body constraint. Every slave
follows the master as if welded:

```python
g.constraints.rigid_body("master_point", "slave_region")
```

Use this for modeling rigid blocks, pile caps, or foundation mats
that are much stiffer than the surrounding structure.

**`kinematic_coupling`** — generalized master-slave. You specify which
DOFs couple:

```python
g.constraints.kinematic_coupling(
    "center", "ring",
    dofs=[1, 2, 3],  # only translations
)
```


### Level 2b — Mixed-DOF coupling

**`node_to_surface`** — couples a 6-DOF node (e.g., a frame node) to a
3-DOF surface mesh (e.g., a solid). This is a compound constraint that
creates phantom nodes:

1. Duplicate slave positions as phantom nodes (6-DOF)
2. Rigid link from master to each phantom
3. EqualDOF from each phantom to the original slave (translations only)

```python
g.constraints.node_to_surface("frame_end", "solid_face")
```

For **pure BC application** (force/moment or prescribed displacement on a
face without a structural element at the reference), prefer
`g.loads.face_load()` / `g.loads.face_sp()` instead — they distribute
directly to face nodes and avoid the phantom node conditioning issue.
See `guide_loads.md` §10–11.

This is the constraint you need when connecting a beam/frame model
to a solid model. The phantom nodes bridge the DOF mismatch.


### Level 3 — Surface coupling

**`tie`** — surface tie via shape function interpolation. Slave nodes
project onto the closest master face and their DOFs are interpolated
from the master face nodes:

```python
g.constraints.tie("flange_surface", "web_surface", tolerance=1.0)
```

Use `tie` when two non-matching meshes share a boundary and you want
displacement continuity without requiring conformal meshing. The
tolerance controls how far a slave node can be from the master surface.

**`distributing_coupling`** — distributes a point load from a master
to a slave surface:

```python
g.constraints.distributing_coupling(
    "load_point", "bearing_surface",
    weighting="area",  # or "uniform"
)
```

**`embedded`** — embedded element constraint (reinforcement in concrete):

```python
g.constraints.embedded("concrete_volume", "rebar_lines")
```

Slave (embedded) nodes follow the displacement field of the host elements.


### Level 4 — Surface-to-surface

**`tied_contact`** — full surface-to-surface tie. Bidirectional check:

```python
g.constraints.tied_contact(
    "master_surface", "slave_surface", tolerance=1.0
)
```

More robust than `tie` for large non-matching meshes because it checks
projections in both directions.

**`mortar`** — mortar coupling via Lagrange multiplier space. The
mathematically rigorous option for non-matching meshes:

```python
g.constraints.mortar(
    "master_surface", "slave_surface",
    integration_order=2,
)
```

Use `mortar` when you need optimal convergence rates in the constraint
error (e.g., for academic benchmarks or problems where the interface
accuracy matters). For practical engineering, `tied_contact` or `tie`
are usually sufficient.


## 4. How constraints land in the broker

After `get_fem_data()`, constraints are split across two composites
based on what solver commands they produce:

**Node-level** (`fem.nodes.constraints`):
- `equal_dof` → `NodePairRecord`
- `rigid_link` → `NodePairRecord`
- `penalty` → `NodePairRecord`
- `rigid_diaphragm` → `NodeGroupRecord` (expands to pairs)
- `rigid_body` → `NodeGroupRecord`
- `kinematic_coupling` → `NodeGroupRecord`
- `node_to_surface` → `NodeToSurfaceRecord` (phantom nodes + pairs)

**Surface-level** (`fem.elements.constraints`):
- `tie` → `InterpolationRecord`
- `distributing_coupling` → `InterpolationRecord`
- `embedded` → `InterpolationRecord`
- `tied_contact` → `SurfaceCouplingRecord`
- `mortar` → `SurfaceCouplingRecord`


## 5. Consuming constraints in a solver

Use the `Kind` constants for linter-friendly comparisons — no magic
strings:

```python
K = fem.nodes.constraints.Kind

# 1. Create phantom nodes first
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# 2. Emit node-pair constraints (compound records expanded automatically)
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.RIGID_ROD:
        ops.rigidLink("rod", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
    elif c.kind == K.PENALTY:
        # custom penalty spring element
        ...

# 3. Surface constraints
for interp in fem.elements.constraints.interpolations():
    # build multi-point constraint from interpolation weights
    # interp.slave_node, interp.master_nodes, interp.weights
    ...
```


## 6. Introspection

```python
# What constraints do I have?
print(fem.inspect.constraint_summary())
# Node constraints (12 records):
#   equal_dof                  8  (source: 'slab_column_tie')
#   rigid_beam                 4  (source: 'node_to_surface coupling')
#   phantom nodes              4  (created by node_to_surface)
# Surface constraints (2 records):
#   tie                        2  (source: 'flange_web_tie')

# DataFrame summaries
fem.nodes.constraints.summary()       # kind, count, n_node_pairs
fem.elements.constraints.summary()    # kind, count, n_interpolations
```


## 7. Guidelines

- **Start with `equal_dof`** — it covers 80% of constraint needs.
- **Use `tie` for non-matching meshes** — it handles mesh incompatibility
  without conformal remeshing.
- **Use `node_to_surface` for beam-to-solid** — it bridges the DOF
  mismatch automatically.
- **Prefer `tied_contact` over `mortar`** for practical work — mortar is
  more accurate but harder to debug.
- **Check `phantom_nodes()`** — if you have `node_to_surface` constraints,
  phantom nodes must be created in the solver before emitting constraints.
- **Set `tolerance` carefully** — too tight and no pairs are found; too
  loose and you couple nodes that shouldn't be coupled.


## See also

- `guide_fem_broker.md` — how the broker organizes constraints
- `guide_loads.md` — the analogous two-stage pipeline for loads
- `guide_basics.md` — session lifecycle
