# ADR 0126 — Structured panel mesher: rectangular wall panels and line members meshed as shared quad grids, without fragment

**Status:** Proposed (2026-10-26). No implementation yet. It adds a pure
planner (`apeGmsh.mesh.panels.plan_panels`) and one session verb
(`g.mesh.structured.panels`). It changes no Emitter Protocol, no H5 schema and
no `FEMData` constructor. Slices are in "Slices"; the ETABS parity mode (D8)
and the ETABS-import hook (D9) are later slices with their decisions recorded
here.

**Sources read for this ADR:** `mesh/_mesh_structured.py`
(`set_transfinite_curve`, `set_transfinite_surface` with `corners=`,
`set_recombine`); `core/_model_geometry.py` (`add_point`, `add_line`,
`add_curve_loop`, `add_plane_surface`, all OCC); `interop/etabs_import.py`
(`import_structural_model`, its shared-edge map); the YOO tower builder
(`YOO Guayaquil/yoo/yoo/sle/gmsh_mesh.py` at `f314caf`, and its
`sle/validation/ETABS_MESH.md`); PR #2573 (the physical-group name index)
and its "Not changed" list.

## Context

A wall-dominated building model from ETABS is a set of planar rectangular
panels (wall areas, one per storey and wall segment) and line members
(beams, columns, braces) that end on panel edges. ETABS meshes each wall area
on its own grid: `ceil(L / h)` equal divisions along the length and
`ceil(H / h)` up the height, with `h` the maximum element size (ETABS's
`WALLMESHMAXSIZE`). Where two areas' grids do not meet node to node, ETABS
ties them with edge constraints instead of re-meshing.

apeGmsh has two ways to mesh such a model, and neither fits:

- **Build OCC geometry and fragment it.** The YOO tower (ETABS RevK, 1,288
  panels, 8,604 frames, 46 storeys) means a fragment over about 13k entities,
  which grows close to N² and was never shown to finish. Gmsh's own mesh of
  the fragmented surfaces does not reproduce ETABS's grid either.
- **`interop/etabs_import.py`.** It already builds conforming geometry
  without fragment: one point per ETABS joint and one line per joint pair,
  reused by frames and area boundaries. It then leaves meshing to Gmsh's
  default algorithm, so its walls get an unstructured mesh, not ETABS's grid.

The YOO project solved this outside apeGmsh, with public apeGmsh calls only
(`gmsh_mesh.py`). It keeps one registry point per location (snap 1 cm), one
curve per point pair, and splits every frame at each registered point on it.
It then makes each panel a transfinite surface with explicit corners, every
curve transfinite with 2 nodes, and recombines to quads. Its station rule
gives each panel its own `ceil` grid, then adds a station wherever a seed
point (support, frame end, floor or ramp vertex, wall corner) lies on a panel
edge. That station is shared by every panel in the same wall plane (along)
or the same storey (up). The rule always conforms, but it spreads stations
further than conformity needs. Measured on RevK by YOO:

| Station rule | Panels with extra stations (along / up) | Extra stations (along / up) | Quads |
|---|---|---|---|
| each area's own `ceil(L/0.5)` grid (ETABS) | – | 0 | 43,012 |
| own-edge forced points only (does **not** conform) | 40 / 33 | 40 / 44 | 43,647 |
| minimal spreading (D3) | 104 / 33 | 104 / 66 | 44,053 |
| YOO today (plane-wide / storey-wide) | 138 / 132 | 138 / 264 | 46,189 |

About two thirds of YOO's extra quads come from over-spreading, not from real
forced points. Its smallest segment is a 0.09 m sliver, because a forced
point lands close to a `ceil` station and both are kept. Its `_panel` helper
takes a panel's bounding box and never checks that the panel is a rectangle.

Facts from the sources that shaped the decisions:

- A transfinite surface with `corners=` needs every bounding curve
  transfinite and the same node count on its two opposite logical sides.
  Built from 2-node curves, one per station segment, a side's node count is
  its station count. So **a station on one edge of a panel must also be on
  the opposite edge**, and **a station on a shared edge is on both panels'
  edges**, since the edge is one chain of curves. These two links are all
  conformity needs.
- A Gmsh line element follows its curve's direction, and a frame's local
  axes, gravity components and cardinal-point offsets assume `i → j`. YOO
  once lost 17 % of the dead load to one reversed curve.
- `g.physical.add_*` costs O(G) per name lookup today, so the tower's ~10k
  groups cost O(G²). YOO uses raw `gmsh.model.addPhysicalGroup` to avoid
  it. PR #2573 indexes the names. Its "Not changed" list records the next
  wall: `_fem_extract.extract_physical_groups` and the `model.h5` capture at
  `end()` call `getEntitiesForPhysicalGroup` once per group, O(E) each, so
  `end()` took 14 s with 6,000 groups.

## Decision

### D1 — Two pieces: a pure planner and a session verb

```python
from apeGmsh.mesh.panels import plan_panels

plan = plan_panels(
    panels={"PN_W1@S12": corners_w1, ...},   # name -> (4, 3) corners, in loop order
    members={"FR_B114": (xyz_i, xyz_j), ...}, # name -> line member, i -> j
    points={"SUP_7": xyz, ...},               # hard points (supports, floor vertices)
    size=0.5,
)
print(plan.report())                          # D6
with apeGmsh(model_name="tower") as g:
    built = g.mesh.structured.panels(plan)    # D7
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=None)
```

`plan_panels` runs no Gmsh. It takes coordinates and names, and returns a
frozen `PanelPlan`: the registry points, each panel's corner points and its
stations along and up, each member's chain of points, and the report. A plan
can be tested, compared and printed without a session, and the station rule
(D2–D5) lives in one place.

`g.mesh.structured.panels(plan)` turns a plan into session entities. It is
the only piece that touches Gmsh, and it holds no meshing logic of its own.

`plan_panels` lives in a new public module, `apeGmsh.mesh.panels`, rather
than on the session, because it needs none. The verb goes in
`g.mesh.structured`, next to the transfinite and recombine calls it is made
of.

### D2 — Topology without fragment: a point registry and one curve per point pair

Every coordinate the plan uses (panel corners, member ends, hard points, and
the stations of D3–D4) is registered through a spatial hash with
`snap=0.01` (model length units). Two coordinates within `snap` are one
point. The plan refuses an input where `snap` would merge two corners of one
panel.

A point is **on** a panel edge or a member when it lies within
`on_tol=0.015` of the segment and strictly between its ends. Every member is
split at every registered point on it, so a member's chain of points is
sorted from `i` to `j`. A member whose `i` and `j` snap to one point is
refused.

The verb creates one OCC point per registry point, and one line per
unordered point pair. Member curves are made first, each oriented along its
member (`i → j`). Panel loops then reuse those curves in whatever direction
they were made, signed as `etabs_import.curve_loop` already does. Two panels
that share an edge, and a member lying on a panel edge, therefore share
curves and nodes by construction, and no boolean operation runs. Every curve
is transfinite with 2 nodes, so a member is one `line2` element per segment
between registered points.

The planner refuses, before any session work:

- a member that crosses a panel's interior or lies inside its face;
- a hard point inside a panel's face, off its edges;
- two coplanar panels that overlap in area.

v1 has no interior meshing lines.

### D3 — Stations spread only as far as conformity requires (default)

Each panel has two **edge pairs**: its two length edges, and its two height
edges. An **edge class** is a union-find class over panel edges, built from
two links:

1. the two edges of one edge pair (the transfinite requirement);
2. two panel edges that are collinear (within `on_tol`) and overlap by more
   than `snap` (the shared-edge requirement).

All the edges of a class are parallel, so a station is a scalar
`s = p · d` along the class direction `d`. Each edge covers an interval
`[a, b]` of `s`. The class's station set is the union of three things:

- every member's own `ceil` grid: `ceil((b - a) / size)` equal divisions of
  its interval;
- every member's end points, which places a T-junction's corners on the
  longer edge;
- every forced point on any member: a registered point that lies on the
  edge, such as a member end, a hard point or another panel's corner.

The set is sorted and merged with `min_gap=0.02` (D4). Each edge then
carries exactly the stations of its class inside its interval.

A station therefore reaches every edge that must carry it, and no other.
This replaces YOO's plane-wide and storey-wide sharing (the table above:
44,053 quads against 46,189). With no forced points and no partial overlaps,
every class holds identical intervals, and the result is each area's own
`ceil` grid, ETABS's grid exactly.

The class build is near-linear. Edges are bucketed by direction and line, and
then by interval overlap, so there is no all-pairs scan. RevK's 1,092
horizontal and 1,104 vertical shared edges are the expected count. They are
YOO's `check_shared_wall_edges` total, and they are a test of the class
build.

### D4 — Forced points snap onto a nearby grid station before inserting

In each class, a forced point within `snap_fraction × size` of a `ceil`
station of that class moves that station onto the point
(`snap_fraction=0.3` by default). Otherwise, the point is inserted, and the
two segments beside it are re-divided by `ceil(seg / size)`, so no segment
grows longer than `size`. End points and other forced points never move. A
move or insert is done once per class, so every edge of the class sees the
same result.

`snap_fraction=0` turns snapping off. The plan then keeps both stations,
merges only those within `min_gap`, and allows slivers (YOO's current
behaviour).

Snapping is the default because this mode puts element quality ahead of
ETABS's equal divisions. D8 is the mode for ETABS's grid. A snap is recorded
in the report (D6), with the distance it moved.

### D5 — v1 meshes planar rectangles and refuses everything else

A panel is four corners in loop order. The plan checks that they are
coplanar within `on_tol`, that the four angles are right angles within 1e-3
rad, and that no side is shorter than `2 × snap`. It refuses a panel that
fails, by name and with the failed check. It never takes a bounding box.

Any orientation is accepted: a sloped or skewed-in-plan rectangle stays a
structured grid. Openings, non-rectangular panels and panels with more than
four corners are refused in v1. RevK has none: all 1,288 are vertical,
axis-aligned rectangles. A fallback (recombined unstructured quads with fixed
boundary stations, flagged in the report) is a separate decision for when a
model needs it.

### D6 — The plan reports what it did

`PanelPlan.report()` is a frozen record, and `str()` gives a readable
summary. It holds:

- per panel, the stations along and up, each tagged `grid`,
  `forced:<point or member name>`, `spread:<class id>` or
  `snapped:<point name> (<distance>)`;
- totals: the panel, quad, member-segment and registry-point counts, and the
  extra stations against each area's own `ceil` grid;
- the smallest segment (with panel and side), and the worst aspect ratio
  (with panel);
- every refusal of D2 and D5, when `plan_panels(..., strict=False)` collects
  them instead of raising on the first. The default is `strict=True`.

The quad count is known before meshing, because each panel is
`(n_along) × (n_up)` cells.

### D7 — The session verb: entities, transfinite, recombine, groups and checks

`g.mesh.structured.panels(plan)` requires a session with no model entities.
Adding panels to an existing model without fragment would leave the two
non-conforming, silently. It then:

1. creates the points, the member curves, the remaining panel curves, the
   curve loops and the plane surfaces (D2), with one `sync` at the end;
2. sets every curve transfinite with 2 nodes, each panel transfinite with
   its four corner points as `corners=`, and each panel recombined;
3. adds one physical group per panel name (dim 2) and per member name
   (dim 1), through `g.physical.add_*`, which needs PR #2573;
4. returns `PanelsBuilt`: the plan and name → entity tag for panels and
   members.

After `generate`, `PanelsBuilt.check(fem)` asserts three things:

- each panel group holds only `quad4`, with the planned count;
- each member group holds `line2` in `i → j` order;
- no element of a member is reversed against its member.

It raises with the names that fail. It is a separate call because meshing is
the caller's. The verb's docstring and the guide put it right after
`generate`.

The verb sets no element order and no renumbering. `generate`, `renumber`
and `get_fem_data` stay the caller's, as on the normal route.

### D8 — ETABS parity mode (a later slice)

`plan_panels(..., mode="etabs")` gives each area its own `ceil` grid and
never spreads a station. A forced point or a neighbour's station that misses
a panel's grid is not inserted. The planner records it as a **tie**: the
point, and the panel edge it must be tied to. The verb then makes separate
curves wherever two panels' grids differ along a shared edge (no shared
nodes there). It declares the ties through the existing constraint
declarations (`g.constraints.tie` or the mortar `interface`; which one is
that slice's decision). This is what ETABS does: its probe P1 shows
mismatched edges tied, not re-meshed. On RevK the mode must reproduce 43,012
quads exactly, so a model can be compared with ETABS mesh for mesh.

Conforming (D3) stays the default. The maintainer's rule is that physical
behaviour comes before ETABS parity, and parity mode exists for comparison
runs. D4's snapping does not apply in this mode.

### D9 — The ETABS-import hook (a later slice)

`interop.etabs_import.import_structural_model(..., wall_mesh="panels",
size=0.5)` builds a `PanelPlan` from the `StructuralModel`'s rectangular
areas, its frames and its joints, and calls D7 in place of its own geometry
build. It keeps its section-and-modifier physical groups by adding them as
unions of the per-member groups. Its existing build stays the default until
the panel route has run on a production model. A non-rectangular area under
`wall_mesh="panels"` is refused by name (D5).

## Slices

0. **Prerequisites, outside this ADR.** PR #2573 (the name index), and the
   follow-up its "Not changed" list names: `extract_physical_groups` and the
   `end()` capture must fetch entities for all groups in one pass instead of
   one `getEntitiesForPhysicalGroup` call per group. Without the second fix,
   a ~10k-group tower stalls in extraction whatever the mesher does.
1. **Planner (D1–D6).** `apeGmsh.mesh.panels`, pure, with unit tests on
   synthetic layouts:
   - a single panel's `ceil` grid;
   - a T-junction;
   - a beam end on a shared edge, spread to both panels and to their
     opposite edges, and to no other panel;
   - snap and no-snap on a point 0.1 h from a station;
   - every refusal of D2 and D5;
   - determinism under input reordering.
2. **Session verb (D7)** and `PanelsBuilt.check`. A live test meshes a
   two-storey, three-wall frame with beams, and asserts:
   - quads only, at the planned counts;
   - shared nodes along each shared edge;
   - member elements running `i → j`.

   A scale test meshes a synthetic tower of 1,288 panels and must run within
   a small factor of raw Gmsh on the same entities.
3. **RevK acceptance (YOO).** YOO replaces `gmsh_mesh.py` with D1/D7, and
   reproduces the D3 row of the table (44,053 quads, 104 / 66 extra
   stations).
4. **Parity mode (D8)**, with 43,012 quads on RevK as its acceptance, and T1
   run in that mode against ETABS.
5. **ETABS-import hook (D9).**

## Rejected alternatives

- **Fragment the OCC panels and members, then set transfinite.** Fragment
  grows close to N² at ~13k entities, and its output still needs station
  logic to make opposite sides match. The shared-curve construction gets
  conformity for free.
- **YOO's plane-wide and storey-wide stations.** They always conform, but
  they add 2,136 quads on RevK that no conformity requirement asks for (the
  table). D3 gives conformity with the minimum.
- **A ready-made mesh through ADR 0125's discrete import.** It would put the
  mesh in the session, but every project would rewrite the station rule and
  the curve bookkeeping, as YOO did. Here apeGmsh owns the rule, and Gmsh
  still meshes.
- **A fallback mesher for non-rectangles in v1.** No current model needs it.
  It is a second mesher, so it waits for its own decision (D5).
- **One verb that does both planning and building.** The station rule would
  then need a Gmsh session to test, and no report could exist before
  building. D1 splits them.

## Consequences

- A wall-dominated ETABS model gets ETABS's grid wherever the grids already
  conform, and the minimum extra stations elsewhere, through the normal
  `get_fem_data` route. Loads, constraints, masses, H5 and the viewers work
  as on any session.
- YOO drops its own mesher (`gmsh_mesh.py`) and its raw
  `gmsh.addPhysicalGroup` calls.
- The planner's tolerances (`snap`, `on_tol`, `min_gap`, `snap_fraction`)
  are part of the public signature, in model length units. A model in
  millimetres must scale them. The docstring says so, and slice 1 decides
  whether they are derived from `size` instead.
- A member that ends inside a panel face, an opening or a non-rectangular
  panel is refused, not meshed. Such models wait for interior meshing lines
  or the D5 fallback.

## Cross-references

- ADR 0125 — discrete mesh import (Proposed, PR #2570, not yet on `main`),
  the route for meshes made entirely outside apeGmsh.
- [ADR 0111](0111-stko-translator.md) — STKO translator, the other mesh-faithful
  importer.
- PR #2573 — physical-group name index (prerequisite, slice 0).
- `interop/etabs_import.py` — the shared-edge map D2 generalises; D9 hooks
  into it.
