> **Absorbed** into `docs/concepts/constraints.md` (ADR 0079 P2-B). This guide is retained as working memory; the concepts page is canonical.

# apeGmsh constraints

A guide to defining multi-point constraints in an apeGmsh session —
from simple DOF ties to surface coupling and mortar methods. This
document covers the apeGmsh abstraction; see `guide_fem_broker.md` for
how constraints land in the broker and get consumed by a solver.

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```


## Tasks on this page

- [Tie co-located nodes (equal_dof)](#level-1-node-to-node) · [Couple a master to a node group](#level-2-node-to-group) · [Bridge beam-to-solid DOFs (node_to_surface)](#level-2b-mixed-dof-coupling) · [Tie non-matching surfaces](#level-3-surface-coupling) · [Surface-to-surface tie / mortar](#level-4-surface-to-surface) · [Consume constraints in a solver](#5-consuming-constraints-in-a-solver) · [Inspect what you declared](#6-introspection) · [Stage-bind a constraint (SSI)](#65-stage-binding-constraints-in-apesees-ssi-workflows)


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

**`kinematic_coupling`** — RBE2: a reference node rigidly drives a node
set, with per-slave DOF selectivity:

```python
g.constraints.kinematic_coupling(
    "center", "ring",
    dofs=[1, 2, 3],  # only translations; None ⇒ all the slave has
)
```

This emits the **Ladruno-fork `element LadrunoKinematicCoupling`**
(class tag 33012) — a penalty rigid-body driver carrying the correct
moment-arm transport `u_i = u_R + θ_R × d_i`, so an *offset* reference
node is coupled rigidly. It replaced the previous `equalDOF`-per-slave
expansion, which ignored the lever arm (correct only for coincident
nodes). **Fork-only:** the deck emits on any build, but running it needs
the Ladruno fork — stock OpenSees fails loud at the element line.

The penalty/enforcement knobs are exposed directly on the factory:
`k` (numeric or `"auto"` + `k_alpha`/`host`), `kr`, `enforce="penalty"|"al"`,
`bipenalty_dtcr` / `bipenalty_wcap` (explicit-dynamics penalty mass), and
`absolute` (skip the `g0` stress-free birth). `host` is given as a **FEM
element id**; the bridge translates it to the emitted OpenSees tag.

Under partitioned (OpenSeesMP) emit the element lands on a **single
canonical rank** (the rank where every slave is present; the reference
is ghost-declared there when foreign) — a slave set split across
partitions fails loud.

Reach for this when the region must move as a **rigid body** (loading
platen, rigid connection block). To *introduce a load at a point while
the region stays flexible*, use `distributing_coupling` (RBE3) instead.


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

When `master` is a string it is resolved via `resolve_to_tags(..., dim=0)` —
i.e. the name must refer to a geometric **point** entity (a vertex), not a
curve/surface/volume label. `slave`, in contrast, is resolved at `dim=2`
(surface entities). See `ConstraintsComposite.py:837`.

Note: `node_to_surface` (and `embedded`, below) bypass the strict
``part-label`` validation that other constraint factories run in
`_add_def` — both definition types are allowed to carry bare entity
tags / mixed labels because they are looked up later through their
own resolvers (`ConstraintsComposite.py:236`).

For **pure BC application** (force/moment or prescribed displacement on a
face without a structural element at the reference), prefer
`g.loads.surface.force_resultant_center_mass()` / `g.displacements.surface()` instead — they distribute
directly to face nodes and avoid the phantom node conditioning issue.
See `guide_loads.md` §10–11.

This is the constraint you need when connecting a beam/frame model
to a solid model. The phantom nodes bridge the DOF mismatch.

**`node_to_surface_spring`** — same topology as `node_to_surface`, same
call signature, but the master → phantom links are emitted as stiff
`elasticBeamColumn` elements instead of kinematic `rigidLink('beam', ...)`
constraints:

```python
g.constraints.node_to_surface_spring("frame_end", "solid_face")
```

Use this variant when the master carries **free rotational DOFs** (e.g. a
fork support on a solid end face) that receive direct moment loading. The
constraint-based `node_to_surface` can produce an ill-conditioned reduced
stiffness matrix in that case because the master rotation DOFs only get
stiffness through kinematic constraint back-propagation, with nothing
attaching directly to them. Stiff beams give those rotations a real
elastic stiffness path and the matrix conditioning recovers.

Pick `node_to_surface` when the master rotations are themselves
constrained or carry no moment; pick `node_to_surface_spring` when the
master is a free rotation node receiving moments.

**Emission contrast.** The plain `node_to_surface` variant emits its
master→phantom links through `fem.nodes.constraints.phantom_nodes()`
plus `pairs()` (kinematic `rigidLink('beam', ...)`). The spring
variant routes the same topology through
`fem.nodes.constraints.stiff_beam_groups()` instead — each group
becomes one `elasticBeamColumn` element per master/phantom pair. The
factory entry point lives at `ConstraintsComposite.py:906`.


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

> For a step-by-step recipe, see [How-to: Tie non-matching meshes](../how-to/tie-meshes.md).

**Tie-force recovery (`enforce="equation"`).** A tie emitted on the exact
kinematic route (`g.constraints.tie(..., enforce="equation")`, the
`LadrunoProjection`-handled constraint variant — see
`internal_docs/handoff_equation_tie_adr0068.md`) can report the interface
force it carries, the apeGmsh analogue of LS-DYNA `*DATABASE_NCFORC`:

```python
# live query — single value from the last projection step of a live analyze
ops = apeSees(fem)
...                                          # build + LadrunoProjection handler
ops.analyze(steps=1, dt=1e-3)
f = ops.ladruno_projection_tie_force(slave_node, dof=1)   # f = M(a_raw - a_proj)

# recorded time history — vec3/node, explicit analyses only
ops.recorder.Ladruno(file="ties.ladruno",
                     nodal_responses=("constraintTieForce",))
# ... after the run:
results.nodes.get(component="constraint_tie_force_x", ids=[slave_node])
```

Both routes are **fork-only** (they need the `LadrunoProjection` handler /
the `ladruno` recorder; a stock build fails loud). The live query works under
implicit and explicit analyses; the recorder channel is explicit-only (it is
scattered each commit by `CentralDifferenceLadruno`).

**`distributing_coupling`** — RBE3: distribute a load at a reference
point over a node set while the set stays **flexible**:

```python
g.constraints.distributing_coupling(
    "load_point", "bearing_surface",
    weighting="area",  # or "uniform"
)
```

This emits the **Ladruno-fork `element LadrunoDistributingCoupling`**
(class tag 33011), replacing the prior `NotImplementedError` stub: the
reference (dependent) node R is the weighted-average rigid-body fit of
the independent set, and a force/moment at R distributes as a
statically-equivalent pattern (`Σ Fᵢ = F`, `Σ rᵢ × Fᵢ = M`) **adding no
stiffness** to the independents. It is the flexible counterpart of
`kinematic_coupling` (RBE2, which holds the set rigid). **Fork-only:**
stock OpenSees fails loud at the element line.

`weighting="uniform"` ⇒ equal weights (the element default).
`weighting="area"` ⇒ apeGmsh computes each independent node's
**tributary area** over the slave surface (each face's area split
equally among its nodes — the same lumping model as `g.loads` surface
tributary resolution) and emits `-w w1..wN`, so a force at R distributes
like a uniform traction. An independent node on no slave face fails loud.

The same penalty/enforcement knobs as `kinematic_coupling` are exposed
(`k`/`k_alpha`/`host`, `kr`, `enforce`, `bipenalty_dtcr`/`bipenalty_wcap`,
`absolute`). Note R is **massless by construction**, so an explicit run
needs `bipenalty_dtcr` (or `bipenalty_wcap` with a `host`) or the stable
step collapses to zero.

**`embedded`** — embedded element constraint (reinforcement in concrete):

```python
g.constraints.embedded("concrete_volume", "rebar_lines")
```

Slave (embedded) nodes follow the displacement field of the host elements.

Implemented end-to-end: at resolution time `_resolve_embedded`
(`ConstraintsComposite.py:1218`) collects the host tet4/tri3 elements
and the embedded-curve nodes, drops embedded nodes that coincide with
a host corner (already rigidly attached via shared connectivity), and
hands the remainder to `resolver.resolve_embedded` which lands real
`InterpolationRecord`s on `fem.elements.constraints`.


### Level 4 — Surface-to-surface

**`tied_contact`** — full surface-to-surface tie. Bidirectional check:

```python
g.constraints.tied_contact(
    "master_surface", "slave_surface", tolerance=1.0
)
```

More robust than `tie` for large non-matching meshes because it checks
projections in both directions.

**`mortar`** — **not implemented; raises `NotImplementedError`.** A true
mortar method needs a Lagrange-multiplier space and surface integration
of the coupling operator over intersected segments; the previous
implementation did none of that (it was a collocation tie mislabelled
MORTAR, with a unit-dependent hardcoded tolerance) and was removed
rather than shipped plausible-but-wrong. Use `tied_contact` for a
non-matching collocation tie.


## 4. How constraints land in the broker

After `get_fem_data()`, constraints are split across two composites
based on what solver commands they produce:

**Node-level** (`fem.nodes.constraints`):
- `equal_dof` → `NodePairRecord`
- `rigid_link` → `NodePairRecord`
- `penalty` → `NodePairRecord`
- `rigid_diaphragm` → `NodeGroupRecord` (expands to pairs)
- `rigid_body` → `NodeGroupRecord`
- `kinematic_coupling` → `NodeGroupRecord` (emits the fork
  `element LadrunoKinematicCoupling`, not `equalDOF` pairs; carries an
  optional `CouplingControl` with the penalty knobs)
- `node_to_surface` → `NodeToSurfaceRecord` (phantom nodes + pairs)
- `node_to_surface_spring` → `NodeToSurfaceRecord` (phantom nodes + stiff-beam pairs)

**Surface-level** (`fem.elements.constraints`):
- `tie` → `InterpolationRecord` (emits `ASDEmbeddedNodeElement`)
- `distributing_coupling` → `InterpolationRecord` (emits the fork
  `element LadrunoDistributingCoupling`; `weights` carries the
  tributary areas under `weighting="area"`, `None` = uniform; optional
  `CouplingControl`)
- `embedded` → `InterpolationRecord` (emits `ASDEmbeddedNodeElement`)
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


## 6.5 Stage-binding constraints in apeSees (SSI workflows)

When using `apeSees` to emit a multi-stage analysis deck, any
constraint can be **claimed by name** for a specific stage so it
emits inside that stage's block instead of the global pre-stage
MP-constraint pass. This is what you want when the constraint
references nodes / elements that only come online via
`s.activate(pgs=[...])` in a later stage — emitting globally would
reference nodes that don't exist at parse time and crash OpenSees.

Name the constraint when you declare it:

```python
# At apeGmsh time — name= is the contract for stage routing later:
g.constraints.embedded(
    host_label="Rock", embedded_label="Lining",
    name="lining_embed", stiffness=1e8,
)
```

Then inside the stage block, claim by that name:

```python
# At apeSees bridge time:
with ops.stage(name="install_lining") as s:
    s.activate(pgs=["Lining"])
    s.embedded(name="lining_embed")    # claim — emits here, not globally
    s.analysis(...)
    s.run(n_increments=10, dt=0.1)
```

Available CLAIM verbs on `_StageBuilder`: `s.embedded`, `s.tie`,
`s.distributing`, `s.equal_dof`, `s.rigid_link`,
`s.rigid_diaphragm`, `s.kinematic_coupling`, `s.node_to_surface`,
`s.node_to_surface_spring`. `s.tied_contact` and `s.mortar` are
deferred — see [_DEFERRED.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/_DEFERRED.md).
Forgetting to claim is caught at build time by the V1 ownership-
tier validator with an actionable offender list.

Full guide: `guide_opensees.md` §4.4 "Multi-point constraints" and
ADR 0034 §5a "Stage-bound constraints via CLAIM-by-name".


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
- `guide_opensees.md` §4.4 — MP constraint emit + stage-binding in
  `apeSees` (SSI-2.D extension)
- [How-to: Tie non-matching meshes](../how-to/tie-meshes.md) — recipe for
  `tie` / `tied_contact` between non-conformal parts


??? note "For maintainers — source map"

    - `src/apeGmsh/core/ConstraintsComposite.py` — the user-facing composite
    - `src/apeGmsh/solvers/_constraint_defs.py` — `ConstraintDef` subclasses
    - `src/apeGmsh/solvers/_constraint_records.py` — `ConstraintRecord` subclasses
    - `src/apeGmsh/solvers/_constraint_resolver.py` — `ConstraintResolver`
    - `src/apeGmsh/solvers/_constraint_geom.py` — geometry helpers shared by the resolver
    - `src/apeGmsh/solvers/_kinds.py` — `ConstraintKind` (and `LoadKind`) constants
    - `src/apeGmsh/solvers/Constraints.py` — re-export shim that surfaces the names
      above under the historical import path
    - `src/apeGmsh/mesh/_record_set.py` — `NodeConstraintSet`, `SurfaceConstraintSet`
