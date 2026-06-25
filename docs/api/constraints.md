# Constraints — `g.constraints`

Solver-agnostic kinematic-constraint engine. Constraints are
**declared on geometry** (part labels, optional entity scopes) and
**resolved on the mesh** by [`g.mesh.queries.get_fem_data`][apeGmsh.mesh._mesh_queries._Queries.get_fem_data].

## Two-stage pipeline

Stage 1 — **declare** before meshing. The factory methods on
`g.constraints` (`equal_dof`, `rigid_link`, `tie`, …) store
[`ConstraintDef`][apeGmsh._kernel.defs.constraints.ConstraintDef]
dataclasses describing intent at the geometry level. These
definitions carry no node tags and survive remeshing.

Stage 2 — **resolve** after meshing.
[`ConstraintResolver`][apeGmsh._kernel.resolvers._constraint_resolver._resolver.ConstraintResolver]
walks the def list and produces concrete
[`ConstraintRecord`][apeGmsh._kernel.records._constraints.ConstraintRecord]
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

Six tiers, ordered by topology:

| Tier         | Methods                                                                                                   | Record family               |
| ------------ | --------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1 — Pair     | [`equal_dof`](#tier-1-node-to-node), [`rigid_link`](#tier-1-node-to-node), [`penalty`](#tier-1-node-to-node) | `NodePairRecord`            |
| 2 — Group    | [`rigid_diaphragm`](#tier-2-node-to-group), [`rigid_body`](#tier-2-node-to-group), [`kinematic_coupling`](#tier-2-node-to-group) | `NodeGroupRecord`           |
| 2b — Mixed   | [`node_to_surface`](#tier-2b-mixed-dof), [`node_to_surface_spring`](#tier-2b-mixed-dof)                   | `NodeToSurfaceRecord`       |
| 3 — Surface  | [`tie`](#tier-3-node-to-surface), [`distributing_coupling`](#tier-3-node-to-surface), [`embedded`](#tier-3-node-to-surface) | `InterpolationRecord`       |
| 4 — Contact  | [`tied_contact`](#tier-4-surface-to-surface)                                                              | `SurfaceCouplingRecord`     |
| 5 — Fork     | `contact`, `mortar` (deprecated alias for `contact(formulation="mortar", tie=True)`)                      | `ContactRecord`             |

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
[`ConstraintDef`][apeGmsh._kernel.defs.constraints.ConstraintDef] —
a thin dataclass carrying `kind`, `master_label`, `slave_label`, and
an optional friendly `name`. Subclasses add their kind-specific
parameters.

::: apeGmsh._kernel.defs.constraints.ConstraintDef
    options:
      heading_level: 3

## Tier 1 — Node-to-Node

Pairwise constraints between **co-located** nodes. The resolver
matches master-side nodes against slave-side nodes within
`tolerance` and emits one `NodePairRecord` per match.

::: apeGmsh._kernel.defs.constraints.EqualDOFDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.RigidLinkDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.PenaltyDef
    options:
      heading_level: 3

## Tier 2 — Node-to-Group

One master node drives many slave nodes through a kinematic
transformation about a master point. Use these for floor diaphragms,
lumped rigid bodies, or any cluster sharing a chosen DOF subset.

::: apeGmsh._kernel.defs.constraints.RigidDiaphragmDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.RigidBodyDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.KinematicCouplingDef
    options:
      heading_level: 3

## Tier 2b — Mixed-DOF

A 6-DOF master node coupled to 3-DOF slave nodes (typically a beam
end framing into a solid face). The resolver duplicates each slave
to a 6-DOF phantom node so that rotational kinematics can propagate
through a rigid arm before being equal-DOF-coupled to the original
3-DOF slave.

Two variants:

* [`NodeToSurfaceDef`][apeGmsh._kernel.defs.constraints.NodeToSurfaceDef]
  emits the master → phantom link as a kinematic
  `rigidLink('beam', …)` constraint. Cheap and exact.
* [`NodeToSurfaceSpringDef`][apeGmsh._kernel.defs.constraints.NodeToSurfaceSpringDef]
  emits it as a stiff `elasticBeamColumn` element. Use this when the
  master has free rotational DOFs that receive direct moment loading
  — the constraint variant can produce an ill-conditioned reduced
  stiffness matrix in that case.

::: apeGmsh._kernel.defs.constraints.NodeToSurfaceDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.NodeToSurfaceSpringDef
    options:
      heading_level: 3

## Tier 3 — Node-to-Surface

A slave node is constrained to the displacement field of a master
surface or volume through shape-function interpolation. Handles
non-matching meshes, distributed loads, and embedded reinforcement.

### Embedded — host mesh compatibility

`g.constraints.embedded(host_label, embedded_label, ...)` accepts any
standard structural mesh on the host side. The collector decomposes
non-simplex and higher-order hosts into linear sub-tris / sub-tets
using corner nodes only, then dispatches to the existing C++
`ASDEmbeddedNodeElement` (which accepts 3- or 4-node retained sets).

| Gmsh etype       | Code  | Host-side decomposition                                  |
| ---------------- | ----- | -------------------------------------------------------- |
| tri3 (CST)       | 2     | identity (1 tri per host)                                |
| tet4             | 4     | identity (1 tet per host)                                |
| quad4            | 3     | 2 tris via (0,2) diagonal split                          |
| hex8             | 5     | 6 right-handed Kuhn tets (shared main diagonal)          |
| prism6           | 6     | 3 tets                                                   |
| pyramid5         | 7     | 2 tets                                                   |
| tri6 (LST)       | 9     | corners only → 1 tri (midsides discarded)                |
| tet10            | 11    | corners only → 1 tet                                     |
| pyramid13        | 14    | corners only → 2 tets                                    |
| quad8 / quad9    | 16/10 | corners only → 2 tris                                    |
| hex20            | 17    | corners only → 6 Kuhn tets                               |
| prism15          | 18    | corners only → 3 tets                                    |

Sub-element rows are **virtual** — they do not correspond to elements
in the gmsh mesh. They exist purely as a coupling-layer fabrication
so the linear-shape-function coupling of `ASDEmbeddedNodeElement`
works against any supported host topology.

#### The linear-coupling contract (`host_coupling="linear"`)

The embedded coupling is always **linear** over 3 or 4 corner nodes,
regardless of the host's native interpolation order. An LST plate's
quadratic curvature, a hex8's bilinear twist mode, a quad9's
biquadratic field — none are seen by the embedded node. The embed
sees only the linear corner-to-corner stretch of whichever sub-tri /
sub-tet contains it.

`EmbeddedDef.host_coupling` is a reserved keyword that pins this
behaviour. Only `"linear"` is currently accepted. The keyword is
reserved (not just documented) so a future `"trilinear"` /
`"biquadratic"` option — which would require a new OpenSees element
class supporting N-node retained sets — can land without breaking
existing models.

#### Warning on midside-bearing hosts

The first time the collector decomposes a host that carries midside
nodes (tri6, tet10, quad8, quad9, hex20, prism15, pyramid13), one
`UserWarning` fires per `(etype, entity)` pointing at the
linear-coupling consequence. Acknowledge by setting
`host_coupling="linear"` explicitly on the `embedded(...)` call.

If you chose LST / quad8 / hex20 specifically for curvature fidelity,
the embed will not give it to you — either accept the linear coupling
or wait for the `HostProjector` work (deferred; see ADR 0036).

#### Per-hex coupling asymmetry

Two embedded nodes inside the same hex8 may couple to **different**
4-corner subsets depending on which of the 6 Kuhn sub-tets contains
each one. This is geometrically correct under linear coupling but
can surprise readers of the resolved records. The Kuhn decomposition
is symmetric (orientation-independent across adjacent hexes), so
there is no neighbour-hex-dependence in the choice.

#### Mixed-dim host fail-loud

A host part / physical group that combines 2D entities (shell, quad
plate) and 3D entities (brick, tet volume) raises at collection
time. The linear coupling cannot pick between sub-tris and sub-tets
deterministically (kNN centroid search would dispatch based on
opaque proximity, which is opaque physics). Split the host into two
separate `g.constraints.embedded(...)` calls — one for the 2D part,
one for the 3D part.

#### Off-host fail-loud

An embedded node that falls outside every host sub-element by more
than `EmbeddedDef.tolerance` (default `1.0` from the factory; the
class default is `0.0` for strictly-inside) raises naming the
offending slave node and its barycentric excess. Either fix the
geometry / mesh so the embed lies inside the host, or widen
`tolerance=` explicitly if extrapolation is intentional.

See [ADR 0036](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0036-embedded-host-decomposition.md)
for the full decision record (Kuhn-table orientation invariants,
alternatives rejected, `HostProjector` RFC deferral).

#### Example — rebar in hex-meshed concrete

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="rc_block") as g:
    # ... CAD import, parts, etc. ...

    # Hex-meshed concrete host, line-meshed rebar curve
    g.constraints.embedded(
        host_label="concrete_block_hex",
        embedded_label="rebar_curve",
        stiffness=1.0e8,        # STKO-parity penalty (ADR 0035)
        # host_coupling="linear" is the default; setting it
        # explicitly acknowledges the linear-coupling contract
        # if your host carries midside nodes.
    )

    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    # Embedded records land on fem.elements.constraints as
    # InterpolationRecord; each rebar node couples to 4 of the
    # 8 corners of the hex that contains it (one of 6 Kuhn
    # sub-tets — see the per-hex asymmetry note above).
```

::: apeGmsh._kernel.defs.constraints.TieDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.DistributingCouplingDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.EmbeddedDef
    options:
      heading_level: 3

## Tier 4 — Surface-to-Surface

Bidirectional surface couplings. Use these when neither side can be
clearly picked as finer than the other and you want a symmetric
treatment.

::: apeGmsh._kernel.defs.constraints.TiedContactDef
    options:
      heading_level: 3

::: apeGmsh._kernel.defs.constraints.MortarDef
    options:
      heading_level: 3

## Records

Resolved records — what the FEM broker exposes after meshing.

::: apeGmsh._kernel.records._constraints
    options:
      heading_level: 3

## Resolver

::: apeGmsh._kernel.resolvers._constraint_resolver._resolver.ConstraintResolver
    options:
      heading_level: 3

## Module shim

The top-level [`apeGmsh.core.ConstraintsComposite`][] module re-exports all
public names from the `_constraint_*` modules for backwards
compatibility. Module-level docstring contains the canonical
taxonomy.

::: apeGmsh.core.ConstraintsComposite
    options:
      members: false
      heading_level: 3
