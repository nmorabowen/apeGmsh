# Masses — `g.masses`

Solver-agnostic nodal-mass definitions, records, and resolver.
Masses are **declared on geometry** as concentrated values or
densities (linear, areal, volumetric) and **accumulated to nodes**
after meshing by [`g.mesh.queries.get_fem_data`][apeGmsh.mesh._mesh_queries._Queries.get_fem_data].

## Two-stage pipeline

Stage 1 — **declare** before meshing. The four factories on
`g.masses` ([`point`][apeGmsh.core.MassesComposite.MassesComposite.point],
[`line`][apeGmsh.core.MassesComposite.MassesComposite.line],
[`surface`][apeGmsh.core.MassesComposite.MassesComposite.surface],
[`volume`][apeGmsh.core.MassesComposite.MassesComposite.volume])
store [`MassDef`][apeGmsh.solvers.Masses.MassDef] dataclasses on
geometric targets.

Stage 2 — **resolve** after meshing.
[`MassResolver`][apeGmsh.solvers.Masses.MassResolver] converts each
def to per-node contributions, then the composite **accumulates**
contributions across overlapping defs so each node ends up with at
most one [`MassRecord`][apeGmsh.solvers.Masses.MassRecord]. Records
land on `fem.nodes.masses` as a `MassSet` (an iterable of
`MassRecord` with `total_mass()` / `by_node()` helpers — see
[FEM Broker](fem.md)).

Each record carries a length-6 vector
`(mx, my, mz, Ixx, Iyy, Izz)`; the OpenSees bridge slices it to the
model's `ndf` when emitting `ops.mass(...)` commands (rotational
components are dropped for `ndf < 4`).

## No patterns

Unlike [loads](loads.md), masses are **not** grouped under named
patterns. Mass is intrinsic to the model — there is one nodal mass
per node regardless of which load pattern is active. Multiple mass
definitions targeting overlapping nodes simply accumulate.

## Lumped vs. consistent reduction

Each factory accepts `reduction="lumped"` (default) or
`reduction="consistent"`:

* **lumped** — element total mass split equally among the corner
  nodes. Diagonal mass matrix; cheap and stable for explicit
  dynamics.
* **consistent** — proper consistent mass matrix integration.
  Today this is only meaningful for line elements (returns the
  `ρ_l L / 6 · [[2, 1], [1, 2]]` consistent matrix). Surface and
  volume paths fall through to lumped because tri3 / quad4 / tet4
  / hex8 with constant density have the same diagonal-summed
  per-node share. The separate paths exist so higher-order
  elements (tri6, quad8, tet10, hex20) can be wired in without
  changing the public API.

## Avoiding double-counting

apeGmsh emits explicit `ops.mass(...)` commands. If your OpenSees
material or section also carries a non-zero `rho`, those
contributions add to whatever this composite emits. Either:

* keep `rho=0` on the material and let `g.masses` carry all
  inertia, **or**
* skip the matching [`volume`][apeGmsh.core.MassesComposite.MassesComposite.volume]
  / [`surface`][apeGmsh.core.MassesComposite.MassesComposite.surface]
  call and let the material handle it.

Pair `g.masses.volume(..., density=ρ)` with
`g.loads.gravity(..., density=ρ)` for matching gravity body weight.

## Target identification

Targets follow the same flexible scheme as
[`LoadsComposite`](loads.md). Pass `pg=` / `label=` / `tag=` to
bypass auto-resolution.

## Worked example

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="frame") as g:
    # ... geometry + Parts ...

    # Lumped mass at the top of a tower
    g.masses.point("Antenna", mass=350.0)

    # Cladding mass spread along an exterior edge (kg/m)
    g.masses.line("PerimeterEdge", linear_density=85.0)

    # Slab self-mass via shell areal density (ρ·t = kg/m²)
    g.masses.surface("Slab", areal_density=2400.0 * 0.20)

    # Steel column self-mass via material density (kg/m³)
    g.masses.volume("Columns", density=7850.0)

    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    for m in fem.nodes.masses:
        ops.mass(m.node_id, *m.mass[:3])    # ndm=3 only

    print("Total mass:", fem.nodes.masses.total_mass())
```

## Composite

::: apeGmsh.core.MassesComposite.MassesComposite
    options:
      members_order: source
      show_bases: false
      heading_level: 3

## Definitions

::: apeGmsh.solvers.Masses.MassDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Masses.PointMassDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Masses.LineMassDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Masses.SurfaceMassDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Masses.VolumeMassDef
    options:
      heading_level: 3

## Resolved record

::: apeGmsh.solvers.Masses.MassRecord
    options:
      heading_level: 3

## Resolver

::: apeGmsh.solvers.Masses.MassResolver
    options:
      heading_level: 3
