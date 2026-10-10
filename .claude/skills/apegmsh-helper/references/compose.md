# Model composition: Assembly (ADR 0117)
<!-- skill-freshness: verified against apeGmsh AS5-c (2026-10-10) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

`Assembly` stitches independently-built, *saved* `model.h5` files into one
larger model by tag-offsetting + namespacing each instance's entities — no
re-meshing, no re-running geometry. ADR 0117 is the contract; ADR 0038 still
describes the merge engine underneath (tag windows, namespacing, the verifier).

Mental model:
- Build a part once; write it with `ops = apeSees(fem); ops.model(...); ops.h5("part.h5")`
  (a plain `g.save()` file has no `/opensees` model and `bridge()` refuses it).
- `Assembly(...).instance(label, "part.h5", translate=, rotate=)` places it, namespaced.
- Tie instances with the assembly verbs on dotted ports `"{instance}.{pg}"`.
- `bridge(ndm=, ndf=)` returns one `apeSees`; declare fixes/loads/analysis on it.
- `asm.h5("assembly.h5")` writes the archive; every reader opens it as a `model.h5`.

---

## Assembly v2 — instances of saved models (ADR 0117)

**The one path for multi-file models.** The v1 `g.compose`,
`FEMData.compose`, `apeGmsh.compose` and `Assembly.add / couple(part_a,
part_b, ports=) / materialize` were removed without deprecation (ADR 0117
D7, AS5-c). Published how-to: `docs/how-to/assemble-saved-models.md`.

```python
# verified: docs/how-to/assemble-saved-models.md (run as a script on the Ladruno fork; the equation tie needs openseespy >= 3.8.0 or the fork)
from math import pi
from apeGmsh.assembly import Assembly, AssemblyError

asm = (
    Assembly("stack")
    .instance("lower", "block.h5")                       # a saved model.h5 WITH /opensees
    .instance("upper", "block.h5",                       # same file again: read once
              rotate=((0.0, 0.0, 1.0), pi), translate=(200.0, 200.0, 400.0))
    .tie("lower.top", "upper.bot", enforce="equation", dofs=[1, 2, 3])
    .node("cap", (100.0, 100.0, 800.0))                  # assembly-owned reference node
    .rigid_link("cap", "upper.top", link_type="rod")
)
ops = asm.bridge(ndm=3, ndf=3)                           # one ordinary apeSees
(cap,) = ops.fem.nodes.select(label="cap").ids
ops.fix(pg="lower.bot", dofs=(1, 1, 1))                  # analysis content: on the bridge
...
ops.tcl("stack.tcl")             # serial deck when no instance has a rank (== flat=True)
asm.h5("stack.h5")               # = ops.h5 + /assembly zone
again = Assembly.from_h5("stack.h5")   # re-lists instances/nodes/ties only
```

- **Source** = a `model.h5` written by `ops.h5(...)` (neutral zone + `/opensees`):
  build the part in its own session, declare `model`, materials and element
  specs on `apeSees(fem)`, `ops.h5("part.h5")`. Extract with
  `get_fem_data(dim=None)` when a tie will use its faces.
- **Names.** No host: every name a file owns becomes `{label}.{name}` (PG
  `top` → `lower.top`, material `steel` → `lower.steel`). Labels: non-empty,
  no `.` `/` whitespace, no leading/trailing `_`. Instance labels, reference
  nodes and tie names share ONE namespace. `rotate=((ax, ay, az), theta_rad)`
  is about the origin and applied before `translate`.
- **Verbs** (ports `"{instance}.{pg|label}"`; a reference node is a bare name):
  `tie` (as `g.constraints.tie`; `enforce="equation"` exact, needs Lagrange),
  `equal_dof` (co-located nodes), `rigid_link(master, slave, link_type="beam"|"rod")`,
  `rigid_diaphragm` (6 DOF nodes in 3-D, so not on ndf-3 bricks),
  `embedded(host, embedded)`, `couple(target, kind="kinematic"|"distributing",
  reference=)` (RBE2/RBE3: Ladruno-fork elements, stock refuses). `node(name,
  coords)` declares a reference node: FEM id `k` for the `k`-th, read back by
  `label=`; its ndf is the bridge's unless `ops.ndf(tag, ndf=6)`.
- **Carry rule (D4): model content travels, analysis content is the
  assembly's.** Travels: mesh, groups, labels, intra-instance constraints,
  `/rebar_elements`, and from `/opensees` the materials, sections,
  transforms, beam integrations, element-attached dampings and element
  specs. Stays behind: fixes, masses, patterns, time series, recorders,
  stages, analysis. Declare those on the bridge.
- **Ranks (D6).** No `partition_rank` anywhere → serial, `tcl()` writes no
  `getPID`. `instance(..., partition_rank=k)` on EVERY instance, one per
  rank, ranks `0 .. n-1` → one `getPID` block per rank; reference nodes on
  rank 0; cross-instance MP lines on every owning rank (INV-9). The
  partitioned deck auto-emits `ParallelPlain` / `Mumps` with serial fallbacks
  unless you declare numberer/system.
- **Archive (D5).** `asm.h5(path)` refuses before `bridge()`, after a new
  declaration, or on a `from_h5` result. `Assembly.from_h5` never opens the
  instance files and cannot `bridge()` / `h5()`; rebuild the model with
  `OpenSeesModel.from_h5(path)`.
- **Refused (`AssemblyError`)**: `contact` / `interface` across instances
  (declare them inside the source); element rows whose args vary inside a
  PG (per-row selector #1542); a damping attached by region, global or in
  a stage; a material/section/transform/integration/element type not yet
  rehydrated (the error lists the supported ones: today uniaxial
  `Elastic`, nD `ElasticIsotropic`, `Elastic` / `ElasticMembranePlateSection` sections,
  `stdBrick`, `FourNodeTetrahedron`, `ShellMITC4`, `elasticBeamColumn`,
  `forceBeamColumn`, `dispBeamColumn`); a source whose `/composed_from`
  modules carry a rank (checked at `instance()` and again at `bridge()`); a
  ranked instance of an assembly archive; a tie/coupling that couples
  nothing (INV-7).
- **An element type outside the roster** (e.g. hex20 for the mixed-order
  route below, `FourNodeQuad`): write the source without that spec and
  declare it on the bridge by dotted PG (`ops.element.FourNodeQuad(pg="a.Rock", ...)`).


---

## Reading composed files — inspect / list / tree

The compose **readers** stay on the session and open any composed file,
an assembly archive included:

```python
# verified: tests/test_compose_facade.py
g = apeGmsh.from_h5("stack.h5")       # an archive written by asm.h5(...)
g.compose_list()    # tuple[ComposedModule, ...] — one per instance
g.compose_tree()    # tuple[ComposeTreeNode, ...] — nested-compose hierarchy
info = g.compose_inspect("part.h5")   # metadata-only read of any file
# keys: fem_hash, neutral_schema_version, tag_span_max, pg_inventory,
#       label_inventory, record_counts, composed_from, compose_tree, properties
```

`FEMData.from_h5("stack.h5")` gives the merged broker, with
`fem.composed_from[label]` per instance and a `module_label` on every merged
node and element.

## Namespacing — `'{instance}.{pg}'`

There is no host: every instance's physical groups, labels and named
materials/sections become `{instance}.{name}`. The `pattern` field of a load
record is NOT namespaced. Instance `k` (1-based) of a source whose ids are
below 10^6 starts at `k * 1_000_000`.

## Nested assemblies + depth cap

Instancing an assembly archive nests it. The depth cap is fixed at 3
(`DEFAULT_MAX_COMPOSE_DEPTH`): `ComposeDepthExceededError` ("would exceed the
maximum compose depth (3)"), raised at `bridge()`. The separator alternates
by depth (depth 1 = `.`, depth 2 = `/`, ...); storage is a flat graft and
`compose_tree()` re-derives the hierarchy. A ranked archive cannot be
instanced.

## apeGmsh.from_h5 — chain-phase sessions (no gmsh)

`apeGmsh.from_h5(path)` rebuilds a session from the neutral zone with **no
gmsh state**: `g.mesh.queries.get_fem_data()`, the compose readers and
`g.save(...)` work; `g.model.*` / `g.mesh.generation.*` raise
`ChainPhaseError`. Interface-bridging defs still route onto the FEM there
(ADR 0041: `tie`, `tied_contact`, `embedded`, `equal_dof`, `rigid_link`,
`rigid_diaphragm`, point loads/masses) and fail loud (`KeyError` for a
misspelled label, `ValueError` for a tie that resolves nothing). For
multi-file models use the `Assembly` verbs instead.

## Errors and warnings

`AssemblyError` (`from apeGmsh.assembly import AssemblyError`) for every
declaration, archive or `bridge()` refusal. The merge engine's typed errors
(`apeGmsh.mesh._compose`: `ComposeError` base, `ComposeDepthExceededError`,
`ComposeNamespaceCollisionError`, `ComposeTagCollisionError`, ...) surface
from `bridge()`. Warnings, emitted by `bridge()`:
- `ComposeFilterWarning` — an instance's stages / time series / patterns are
  dropped (analysis content is the bridge's).
- `ComposeDroppedStreamWarning` — a non-empty neutral stream that does not travel.
- `ComposeInterfaceSizeWarning` — interface-class constraint count > 50 000.

## Viewing composed models — Module color mode

The viewer colors by source module via a **string-keyed** mode, NOT a `ColorMode`
enum member. `set_mode` accepts `'Module'`, `'Module: Root'`, `'Module: Leaf'`
(`src/apeGmsh/viewers/ui/mesh_tabs.py:122`). Any reference to `ColorMode.MODULE` as an
enum is wrong.

---

## Independent meshes per part — mixed element types AND orders (ADR 0085/0086)

**Separate files are the ONLY route to mixed element order.** `set_order` is
global within a gmsh session, so hex20 in one region + hex8 in another is
impossible in one mesh pass — and a `Part` cannot help (it is a geometry
template with no mesh composite, by deliberate contract; ADR 0085). The unit
of authorship is a **full session per part**, assembled with `Assembly`:

```python
# verified: tests/test_meshable_part_route.py, tests/test_mortar_tie_compose.py
with apeGmsh(model_name="ribs", verbose=False) as g:
    ...                                          # geometry + volume/surface PGs
    g.mesh.recipe.structured(size=4.0, fallback="strict")
    g.mesh.generation.set_order(2, bubble=False) # hex20 — THIS part only
    fem = g.mesh.queries.get_fem_data(dim=None)  # dim=None: ties need dim-2 groups
ops = apeSees(fem); ops.model(ndm=3, ndf=3); ops.h5("part_ribs.h5")

asm = (Assembly("fuse")
       .instance("cover", "part_cover.h5")       # hex8
       .instance("rb", "part_ribs.h5")           # hex20
       .tie("cover.WeldFace", "rb.WeldRoot", dofs=[1, 2, 3],
            method="mortar", enforce="equation"))
ops = asm.bridge(ndm=3, ndf=3)
# hex20 is outside the rehydrate roster: declare it on the bridge by dotted PG.
```

Rules (measured on the Cerro Lindo rung-4 fuse):
1. **Extract every part with `get_fem_data(dim=None)`** — the tie resolvers
   need the dim-2 element groups.
2. **`enforce="equation"` for weld ties** — exact (−0.01 %); needs the Lagrange
   handler + an unsymmetric system.
3. **`method="mortar"` across an ORDER-mismatched interface** (ADR 0086) —
   collocation pins quad8 slave nodes to a bilinear master field and
   over-constrains (fuse: rib-root fixity a = 0.195 vs 0.230, K +7.7 %); the
   dual mortar restores it and is master/slave symmetric (closed form passes
   in both orderings). Requirements: flat coincident interface, convex facets,
   `enforce="equation"`; tri6 SLAVE facets refused (swap sides); every
   degenerate case is a hard `MortarTieError`. Subtlety: on grids that NEST
   (2.5 on 5.0, aligned) dual mortar provably coincides with collocation —
   the difference appears when slave facets straddle master face boundaries.
   The mortar math is apeGmsh-side numpy; the fork's `LadrunoTie -mortar` is
   tri3/quad4-only and is NOT the emit target.
4. Loads/BCs belong on the **bridge** (an instance's load patterns are
   dropped with a `ComposeFilterWarning`; neutral fixes/masses/SP cases travel
   only through the opt-in `ops.fix_from_model()`, `ops.mass_from_model()`,
   `p.from_model(case)`).
