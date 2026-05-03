# Constraints — `g.constraints`

Solver-agnostic kinematic-constraint engine. Constraints are
**declared on geometry** (part labels, optional entity scopes) and
**resolved on the mesh** by [`g.mesh.queries.get_fem_data`][apeGmsh.mesh._mesh_queries._Queries.get_fem_data].

## Two-stage pipeline

Stage 1 — **declare** before meshing. The factory methods on
`g.constraints` (`equal_dof`, `rigid_link`, `tie`, …) store
[`ConstraintDef`][apeGmsh.solvers._constraint_defs.ConstraintDef]
dataclasses describing intent at the geometry level. These
definitions carry no node tags and survive remeshing.

Stage 2 — **resolve** after meshing.
[`ConstraintResolver`][apeGmsh.solvers._constraint_resolver.ConstraintResolver]
walks the def list and produces concrete
[`ConstraintRecord`][apeGmsh.solvers._constraint_records.ConstraintRecord]
objects (actual node tags, weights, offset vectors). Records land on
the FEM broker:

| Record family             | Lives on                       |
| ------------------------- | ------------------------------ |
| `NodePairRecord`          | `fem.nodes.constraints`        |
| `NodeGroupRecord`         | `fem.nodes.constraints`        |
| `NodeToSurfaceRecord`     | `fem.nodes.constraints`        |
| `InterpolationRecord`     | `fem.elements.constraints`     |
| `SurfaceCouplingRecord`   | `fem.elements.constraints`     |

## Constraint taxonomy

Five tiers, ordered by topology:

| Tier         | Methods                                                                                                   | Record family               |
| ------------ | --------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1 — Pair     | [`equal_dof`](#tier-1-node-to-node), [`rigid_link`](#tier-1-node-to-node), [`penalty`](#tier-1-node-to-node) | `NodePairRecord`            |
| 2 — Group    | [`rigid_diaphragm`](#tier-2-node-to-group), [`rigid_body`](#tier-2-node-to-group), [`kinematic_coupling`](#tier-2-node-to-group) | `NodeGroupRecord`           |
| 2b — Mixed   | [`node_to_surface`](#tier-2b-mixed-dof), [`node_to_surface_spring`](#tier-2b-mixed-dof)                   | `NodeToSurfaceRecord`       |
| 3 — Surface  | [`tie`](#tier-3-node-to-surface), [`distributing_coupling`](#tier-3-node-to-surface), [`embedded`](#tier-3-node-to-surface) | `InterpolationRecord`       |
| 4 — Contact  | [`tied_contact`](#tier-4-surface-to-surface), [`mortar`](#tier-4-surface-to-surface)                      | `SurfaceCouplingRecord`     |

All constraints ultimately express the linear MPC equation
`u_slave = C · u_master`. Tiers differ in **how** `C` is built:
node co-location (Tier 1), kinematic transformation around a master
point (Tier 2), shape-function interpolation (Tier 3), or numerical
integration on the interface (Tier 4).

## Target identification

Most methods identify their master and slave sides by **part label**
(a key of `g.parts._instances`). `_add_def` validates both labels
against the registry and raises `KeyError` on a typo.

Optional `master_entities` / `slave_entities` arguments (lists of
`(dim, tag)`) narrow the search to a subset of the part's entities —
useful when a part has many surfaces and only one is the interface.

**Exceptions** to the part-label scheme:

* [`node_to_surface`](#tier-2b-mixed-dof) and
  [`node_to_surface_spring`](#tier-2b-mixed-dof) take **bare tags**
  instead — the master is a Gmsh point entity (`dim=0`) and the slave
  is one or more surface entities (`dim=2`).
* [`embedded`](#tier-3-node-to-surface) uses `host_label` /
  `embedded_label` to mirror Abaqus's vocabulary; the lookup logic
  otherwise matches the part-label scheme.

## Worked example

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="frame") as g:
    # ... geometry + Parts already imported ...

    # Tier 1 — co-located nodes share x/y/z
    g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3])

    # Tier 2 — slab nodes follow a centre-of-mass node
    g.constraints.rigid_diaphragm(
        "slab", "slab_master",
        master_point=(2.5, 2.5, 3.0),
        plane_normal=(0, 0, 1),
    )

    # Tier 3 — non-matching shell-to-solid interface
    g.constraints.tie(
        "shell_floor", "solid_column",
        master_entities=[(2, 17)],
        slave_entities=[(2, 41)],
        tolerance=5.0,
    )

    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Grouped emission — accumulates rigid_beam / rigid_diaphragm /
    # node_to_surface phantom links by master node.
    for master, slaves in fem.nodes.constraints.rigid_link_groups():
        for slave in slaves:
            ops.rigidLink("beam", master, slave)
```

## Composite

::: apeGmsh.core.ConstraintsComposite.ConstraintsComposite
    options:
      members_order: source
      show_bases: false
      heading_level: 3

## Base class

All Stage-1 definitions inherit from
[`ConstraintDef`][apeGmsh.solvers._constraint_defs.ConstraintDef] —
a thin dataclass carrying `kind`, `master_label`, `slave_label`, and
an optional friendly `name`. Subclasses add their kind-specific
parameters.

::: apeGmsh.solvers._constraint_defs.ConstraintDef
    options:
      heading_level: 3

## Tier 1 — Node-to-Node

Pairwise constraints between **co-located** nodes. The resolver
matches master-side nodes against slave-side nodes within
`tolerance` and emits one `NodePairRecord` per match.

::: apeGmsh.solvers._constraint_defs.EqualDOFDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.RigidLinkDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.PenaltyDef
    options:
      heading_level: 3

## Tier 2 — Node-to-Group

One master node drives many slave nodes through a kinematic
transformation about a master point. Use these for floor diaphragms,
lumped rigid bodies, or any cluster sharing a chosen DOF subset.

::: apeGmsh.solvers._constraint_defs.RigidDiaphragmDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.RigidBodyDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.KinematicCouplingDef
    options:
      heading_level: 3

## Tier 2b — Mixed-DOF

A 6-DOF master node coupled to 3-DOF slave nodes (typically a beam
end framing into a solid face). The resolver duplicates each slave
to a 6-DOF phantom node so that rotational kinematics can propagate
through a rigid arm before being equal-DOF-coupled to the original
3-DOF slave.

Two variants:

* [`NodeToSurfaceDef`][apeGmsh.solvers._constraint_defs.NodeToSurfaceDef]
  emits the master → phantom link as a kinematic
  `rigidLink('beam', …)` constraint. Cheap and exact.
* [`NodeToSurfaceSpringDef`][apeGmsh.solvers._constraint_defs.NodeToSurfaceSpringDef]
  emits it as a stiff `elasticBeamColumn` element. Use this when the
  master has free rotational DOFs that receive direct moment loading
  — the constraint variant can produce an ill-conditioned reduced
  stiffness matrix in that case.

::: apeGmsh.solvers._constraint_defs.NodeToSurfaceDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.NodeToSurfaceSpringDef
    options:
      heading_level: 3

## Tier 3 — Node-to-Surface

A slave node is constrained to the displacement field of a master
surface or volume through shape-function interpolation. Handles
non-matching meshes, distributed loads, and embedded reinforcement.

::: apeGmsh.solvers._constraint_defs.TieDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.DistributingCouplingDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.EmbeddedDef
    options:
      heading_level: 3

## Tier 4 — Surface-to-Surface

Bidirectional surface couplings. Use these when neither side can be
clearly picked as finer than the other and you want a symmetric
treatment.

::: apeGmsh.solvers._constraint_defs.TiedContactDef
    options:
      heading_level: 3

::: apeGmsh.solvers._constraint_defs.MortarDef
    options:
      heading_level: 3

## Records

Resolved records — what the FEM broker exposes after meshing.

::: apeGmsh.solvers._constraint_records
    options:
      heading_level: 3

## Resolver

::: apeGmsh.solvers._constraint_resolver.ConstraintResolver
    options:
      heading_level: 3

## Module shim

The top-level [`apeGmsh.solvers.Constraints`][] module re-exports all
public names from the `_constraint_*` modules for backwards
compatibility. Module-level docstring contains the canonical
taxonomy.

::: apeGmsh.solvers.Constraints
    options:
      members: false
      heading_level: 3
