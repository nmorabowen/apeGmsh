# Loads — `g.loads`

Solver-agnostic load definitions, records, and resolver. Loads are
**declared on geometry** (with optional pattern grouping) and
**resolved on the mesh** by [`g.mesh.queries.get_fem_data`][apeGmsh.mesh._mesh_queries._Queries.get_fem_data].

## Two-stage pipeline

Stage 1 — **declare** before meshing. The factory methods on
`g.loads` (`point`, `point_closest`, `line`, `surface`, `gravity`,
`body`, `face_load`, `face_sp`) store
[`LoadDef`][apeGmsh.solvers.Loads.LoadDef] dataclasses describing
intent at the geometry level. The active
[`pattern`][apeGmsh.core.LoadsComposite.LoadsComposite.pattern]
context tags every def created inside it.

Stage 2 — **resolve** after meshing.
[`LoadResolver`][apeGmsh.solvers.Loads.LoadResolver] converts each
def to a list of resolved records. Records land on the FEM broker
according to type:

| Record family             | Lives on                       | Emitted by                              |
| ------------------------- | ------------------------------ | --------------------------------------- |
| `NodalLoadRecord`         | `fem.nodes.loads`              | tributary / consistent reductions       |
| `ElementLoadRecord`       | `fem.elements.loads`           | `target_form="element"` (eleLoad style) |
| `SPRecord`                | `fem.nodes.sp`                 | `face_sp` only                          |

## Patterns

Loads (and only loads — not constraints, not masses) are grouped
under named patterns via the
[`pattern`][apeGmsh.core.LoadsComposite.LoadsComposite.pattern]
context manager:

```python
with g.loads.pattern("Dead"):
    g.loads.gravity("Slab", density=2400)
    g.loads.line("BeamEdge", magnitude=-15e3)

with g.loads.pattern("Live"):
    g.loads.surface("Slab", magnitude=-2.5e3, normal=False)
```

Defs declared outside any `pattern` block belong to the implicit
`"default"` pattern. Downstream solvers emit one
`timeSeries`/`pattern` block per group.

## Reduction & emission form

Three of the distributed-load factories (`line`, `surface`,
`gravity`, `body`) accept two orthogonal flags that change how the
load is converted to records:

| `reduction`     | `target_form` | Effect                                                                              |
| --------------- | ------------- | ----------------------------------------------------------------------------------- |
| `"tributary"`   | `"nodal"`     | Default. Length/area/volume-weighted nodal lumping — one `NodalLoadRecord` per node |
| `"consistent"`  | `"nodal"`     | Shape-function (Gauss-quadrature) integration — required for higher-order elements  |
| `"tributary"`   | `"element"`   | Skip nodal lumping entirely; emit one `ElementLoadRecord` per element               |
| `"consistent"`  | `"element"`   | Same as above; the solver's element handles the integration                         |

Use **element form** for beam-element line loads
(`eleLoad -beamUniform`), shell pressures handled inside the
element, or any solver-side load that you don't want decomposed at
the apeGmsh layer.

## Target identification

All factory methods accept a flexible positional `target` argument
plus three explicit keyword overrides (`pg=`, `label=`, `tag=`) that
pin the lookup source. The auto path tries, in order: raw
`(dim, tag)` list → mesh selection → label → physical group → part
label. The first match wins. See the
[`LoadsComposite`][apeGmsh.core.LoadsComposite.LoadsComposite]
class docstring for the full disambiguation rules.

## Worked example

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="frame") as g:
    # ... geometry + Parts already imported ...

    with g.loads.pattern("Dead"):
        g.loads.gravity("Slab", density=2400)            # body load
        g.loads.line("BeamEdge", magnitude=-15e3,        # distributed
                     direction=(0, 0, -1),               # line load
                     reduction="tributary")

    with g.loads.pattern("Push"):
        g.loads.point_closest(                           # snaps to
            xyz=(5.0, 2.5, 3.0), within="Slab",          # nearest
            force_xyz=(120e3, 0.0, 0.0),                 # mesh node
        )

    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Pattern-by-pattern emission into OpenSees
    for pat in g.loads.patterns():
        ops.timeSeries("Linear", pat_tag(pat))
        ops.pattern("Plain", pat_tag(pat), pat_tag(pat))
        for r in fem.nodes.loads.by_pattern(pat):
            ops.load(r.node_id, *(r.force_xyz or (0,)*3))
```

## Composite

::: apeGmsh.core.LoadsComposite.LoadsComposite
    options:
      members_order: source
      show_bases: false
      heading_level: 3

## Base classes

::: apeGmsh.solvers.Loads.LoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.LoadRecord
    options:
      heading_level: 3

## Concentrated loads

Concentrated forces and moments — applied either to nodes that
already exist on a named target, or to the mesh node nearest a
world coordinate.

::: apeGmsh.solvers.Loads.PointLoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.PointClosestLoadDef
    options:
      heading_level: 3

## Distributed loads

Length-, area-, or volume-distributed loads. All four accept the
`reduction` × `target_form` flags described above.

::: apeGmsh.solvers.Loads.LineLoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.SurfaceLoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.GravityLoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.BodyLoadDef
    options:
      heading_level: 3

## Face load and face SP

Face-centroid versions used when you want to apply a centroidal
force/moment or prescribed motion to a whole face without
introducing a reference node and a coupling constraint.

::: apeGmsh.solvers.Loads.FaceLoadDef
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.FaceSPDef
    options:
      heading_level: 3

## Resolved records

What ends up on the FEM broker after meshing.

::: apeGmsh.solvers.Loads.NodalLoadRecord
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.ElementLoadRecord
    options:
      heading_level: 3

::: apeGmsh.solvers.Loads.SPRecord
    options:
      heading_level: 3

## Resolver

::: apeGmsh.solvers.Loads.LoadResolver
    options:
      heading_level: 3
