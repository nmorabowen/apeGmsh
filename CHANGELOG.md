# Changelog

## v1.2.0 — Results viewer: Gauss contour, scrubber animation, shape-function catalog

Six PRs landing on top of v1.1.0's Results rebuild. `ContourDiagram`
gains two new rendering paths so element-level Gauss data flows
straight through the dialog → diagram → substrate pipeline; the time
scrubber gets real Play / FPS / loop controls; the shape-function
catalog grows from 5 to 12 element types so quadratic and prism
meshes are first-class. **All 6 PRs merged green** — 377 viewer
tests + 1876 non-viewer tests pass on main.

PRs in this release: [#35], [#36], [#37], [#38], [#39], [#42].

### ADDED — `ContourDiagram` Gauss paths

- **Element-constant Gauss contour** ([#37]). New `gauss_cell` rendering
  path activated when `n_gp == 1` per element (CST / tri31, hex8 with
  one-point integration, etc.). Reads via
  `results.elements.gauss.get(component=...)`, paints `cell_data` on
  a substrate submesh extracted by element IDs. Removes the manual
  nodal-averaging step the plate notebook was using.
- **GP→nodal extrapolation for higher-order integration** ([#39]).
  New `gauss_node` path activated when `n_gp > 1`. The slab is
  projected onto the linear-corner shape functions via the
  Moore–Penrose pseudo-inverse (`pinv(N)`), then accumulated into a
  per-node sum + count for cross-element averaging. New module
  [`apeGmsh.results._gauss_extrapolation`] with two public entry
  points: `extrapolate_gauss_slab_to_nodes(slab, fem)` and
  `per_element_max_gp_count(slab)`.
- **`ContourStyle.topology` field** ([#37]). User-facing knob with
  three values:
    * `"auto"` (default) — prefer nodal data when both composites have
      the requested component; fall through to Gauss otherwise.
    * `"nodes"` — force the nodal-scalar path (point data).
    * `"gauss"` — force the Gauss path; cell-vs-node sub-decision is
      made internally based on `n_gp`.
- **Topology dropdown in the Add Diagram dialog** ([#38]). Visible
  only when the selected kind is Contour. The Component combo
  populates from the union of nodes + gauss components under
  `"auto"`, and from the picked composite under `"nodes"` / `"gauss"`.

### ADDED — Time scrubber animation ([#36])

- **Play button** drives a `QTimer` at `1000 / fps` ms; each tick
  advances one step via `director.set_step(...)`. The scrubber stays
  slider-passive — it only updates the slider via the Director's
  `on_step_changed` callback, never directly.
- **FPS spinner** (1–60, default 30). Live — changing it while
  playing updates the timer interval without disturbing the run.
- **Loop modes** combo: `"once"` (stop at last step), `"loop"`
  (wrap to step 0), `"bounce"` (reverse direction at boundaries,
  never wraps).
- **Stops on stage change** automatically; a fresh stage may have
  a different step count, so the scrubber refreshes and waits for
  the user to press Play again.

### ADDED — Shape-function catalog expansion ([#42])

`SHAPE_FUNCTIONS_BY_GMSH_CODE` grows from 5 entries to 12. New types
covering everything you'd hit by setting
`gmsh.model.mesh.setOrder(2)` plus `wedge6` for extruded / layered
meshes:

| Code | Type   | Notes                                          |
|------|--------|------------------------------------------------|
| 6    | wedge6 | Linear prism (tri × line tensor product)       |
| 9    | tri6   | Quadratic triangle                             |
| 10   | quad9  | Lagrangian biquadratic quad                    |
| 11   | tet10  | Quadratic tet                                  |
| 12   | hex27  | Lagrangian triquadratic hex                    |
| 16   | quad8  | Serendipity quadratic quad                     |
| 17   | hex20  | Serendipity quadratic hex                      |

Node orderings match Gmsh's published convention (cross-checked
against the ASCII diagrams in `gmsh-4.15.1/src/geo/M{Triangle,
Quadrangle,Tetrahedron,Hexahedron,Prism}.h`), so a connectivity row
read straight from a Gmsh-generated mesh works without any
reordering. Pyramids (`pyr5` / `pyr14`) and `line3` are deliberately
out of scope — pyramids have a known apex singularity worth avoiding
for a first pass, and `line3` is rare in OpenSees output.

For GP→nodal extrapolation the higher-order types fall back to
their **linear counterpart** (tri6 → tri3, quad8/9 → quad4,
tet10 → tet4, hex20/27 → hex8). Reasons: the substrate is built
from linear cells (mid-side / face / center nodes are dropped in
`build_fem_scene`), so non-corner extrapolations are never painted;
and `pinv` on the full higher-order N matrix produces a
non-constant nodal field for a constant GP input
(minimum-norm regularization of the under-determined system),
which is wrong for visualization.

### REFACTORED — single-source diagram topology routing ([#35])

- New `Diagram.topology: str` class attribute; each subclass declares
  it next to `kind`. The Add Diagram dialog's `_KIND_TO_TOPOLOGY`
  table is now derived from those attributes:
  ```python
  _KIND_TO_TOPOLOGY = {
      entry.kind_id: entry.diagram_class.topology for entry in _KINDS
  }
  ```
  Previously the dict was hand-maintained alongside the per-class
  composite-reader calls and could drift.

### CHANGED — `ContourDiagram` internals

- Three internal effective-topology values replace the previous two:
  `"nodes"` / `"gauss_cell"` / `"gauss_node"`. Dispatch is decided
  at attach time after a single step-0 read used both for the n_gp
  probe and the initial scatter.
- Cross-element discontinuities are smoothed by the nodal averaging
  in the `gauss_node` path. Standard post-processor behaviour
  (STKO, ParaView). A future per-element subdivision path can
  preserve discontinuities — out of scope here, the `gauss_cell`
  scaffolding stays in place to keep that door open.

### Test coverage

| PR | New tests | Notes |
|----|----------:|-------|
| [#35] | 2  | Pinning test for the eight kind→topology mappings |
| [#36] | 10 | State-machine + timer + stage-change-stop coverage |
| [#37] | 10 | Auto resolution, attach, in-place mutation, multi-GP rejection (later refactored to extrapolation in [#39]) |
| [#38] | 11 | Visibility per kind, component listing per topology, end-to-end run() spec construction |
| [#39] | 11 | Linear-field round-trip on hex8 + 2×2×2 GPs (`atol=1e-12`), shared-face averaging, time-axis preservation, in-place mutation on the new path |
| [#42] | 37 | 5 invariants × 7 types: Kronecker delta, partition of unity, linear precision, dN-sum, FD cross-check |
| **Total** | **81** |  |

### Known follow-ups (not scheduled)

- **Discontinuity-preserving Gauss contour** — subdivides each
  multi-GP element into linear sub-cells, samples at sub-vertices,
  renders cell-data per sub-cell. Preserves jumps at material
  interfaces. ~500–800 LOC, design-discussion-first decision.
- **Hex27 face/center node ordering** — verified self-consistent in
  the shape-function math, but the assumed Gmsh ordering for nodes
  20–26 isn't independently validated against a real Gmsh hex27
  mesh. One-row fix in `_HEX27_LAGRANGE_INDEX` if a real mesh
  surfaces a mismatch.
- **Pyramid shape functions** — `pyr5` and `pyr14`. Add when needed.

[#35]: https://github.com/nmorabowen/apeGmsh/pull/35
[#36]: https://github.com/nmorabowen/apeGmsh/pull/36
[#37]: https://github.com/nmorabowen/apeGmsh/pull/37
[#38]: https://github.com/nmorabowen/apeGmsh/pull/38
[#39]: https://github.com/nmorabowen/apeGmsh/pull/39
[#42]: https://github.com/nmorabowen/apeGmsh/pull/42

## v1.1.0 — Results: backend-agnostic FEM post-processing system rebuild

Wholesale rebuild of the `apeGmsh.results` module. The legacy
in-memory `Results` carrier (a thin VTK-feeder bound to live numpy
arrays) is replaced by a lazy disk-backed reader plus a unified
composite API that mirrors `FEMData`. Recording flows through three
execution strategies — Tcl/Py recorders, in-process domain capture,
and an MPCO bridge — all driven by one declarative
`g.opensees.recorders` spec. **987 tests pass** end-to-end including
the El Ladruno OpenSees Tcl subprocess integration. See PR #12 and
`internal_docs/Results_architecture.md` for full design.

### ADDED — `apeGmsh.results`

- **Native HDF5 schema + I/O.** `NativeWriter` / `NativeReader`
  round-trip nodes, Gauss points, fibers, layers, line stations, and
  per-element forces. Stages are first-class (`kind="transient"` /
  `"static"` / `"mode"`). Multi-partition stitching transparent to
  the reader. Embedded FEMData snapshot in `/model/` — including
  `MeshSelectionStore` — so result files are self-contained.
- **`Results` composite class** mirroring `FEMData`. Selection
  vocabulary `pg=` / `label=` / `selection=` / `ids=`. Stage scoping
  via `results.stage(name)`; mode access via `results.modes`.
  Soft FEM coupling with hash-validated `bind()`.
- **`compute_snapshot_id(fem)`** deterministic content hash —
  ties recorder specs ↔ result files ↔ FEMData snapshots.
- **MPCO reader.** `Results.from_mpco(path)` reads existing STKO
  `.mpco` files through the same composite API. Partial FEMData
  synthesis from MPCO `MODEL/` group (nodes + elements +
  region-derived PGs).
- **`g.opensees.recorders`** declarative spec composite.
  Standalone class, no parent ref, no gmsh dependency. `.nodes`,
  `.elements`, `.line_stations`, `.gauss`, `.fibers`, `.layers`,
  `.modal` declaration methods. `spec.resolve(fem, ndm, ndf)`
  expands shorthand components, validates per category, locks
  `fem_snapshot_id`.
- **Three execution strategies driven by the spec:**
  - **Strategy A** — `g.opensees.export.tcl/py(..., recorders=spec)`
    emits `recorder Node/Element ...` commands + HDF5 manifest
    sidecar. `Results.from_recorders(spec, output_dir, fem=fem)`
    parses output `.out` files into native HDF5 with a cache layer
    at `<project_root>/results/`.
  - **Strategy B** — `with spec.capture(path, fem) as cap:` wraps
    an openseespy analyze loop, querying `ops.nodeDisp` etc. per
    record. Multi-record merge with NaN-fill when records target
    disjoint node sets. `cap.capture_modes()` writes one mode-kind
    stage per `ops.eigen` mode.
  - **Strategy C** — `recorders_file_format="mpco"` dispatches to
    a single `recorder mpco` line aggregating all records.
    Validated via subprocess against the El Ladruno OpenSees Tcl
    build.
- **Element capability flags** on `_ElemSpec`: `has_gauss`,
  `has_fibers`, `has_layers`, `has_line_stations` plus a
  `supports(category)` helper. All 16 entries in `_ELEM_REGISTRY`
  annotated.
- **`MeshSelectionStore`** name-based lookups: `names()`,
  `node_ids(name)`, `element_ids(name)` — mirrors
  `PhysicalGroupSet`'s API.
- **Architecture doc** `internal_docs/Results_architecture.md`
  (single canonical reference).

### CHANGED — Results module API (BREAKING)

- **`Results.from_fem(fem, point_data=..., cell_data=...)` removed.**
  Use `Results.from_native(...)`, `from_mpco(...)`,
  `from_recorders(...)`, or hand-construct via `NativeWriter` for
  the in-memory case.
- **`fem.viewer()`** raises `NotImplementedError` until the viewer
  rebuild project. The new flow will go through the rebuilt
  composite API.
- **`g.mesh.viewer(point_data=..., cell_data=...)`** raises
  `NotImplementedError`. The mesh-only paths (`g.mesh.viewer()` and
  `g.mesh.viewer(results=path)` for a `.vtu`/`.pvd` file) still
  work unchanged.
- **Public exports** under `apeGmsh.results`: `Results`,
  `ResultsReader`, `NativeReader`, `MPCOReader`, `ResultLevel`,
  `StageInfo`, `BindError`, and the slab dataclasses
  (`NodeSlab`, `ElementSlab`, `LineStationSlab`, `GaussSlab`,
  `FiberSlab`, `LayerSlab`).

### DEFERRED — element-level transcoding

Element-level records (`gauss` / `fibers` / `layers` /
`line_stations` / `elements`) work end-to-end on the **declaration
and emission** side. The **read/decode** side is stubbed:

- `MPCOReader.read_gauss/fibers/layers/...` returns empty slabs.
- `RecorderTranscoder` skips element records silently.
- `DomainCapture.step()` raises `NotImplementedError` for element
  categories.

All three need the same missing piece: a per-element-class
response-metadata catalog. Plan in
`internal_docs/plan_element_transcoding.md` (Phase 11a).
**Nodal results work everywhere today.**

### NEW FIXTURE

- `tests/fixtures/results/elasticFrame.mpco` — 400 KB binary,
  12 nodes / 11 elastic frame elements / 10 transient steps /
  2 model stages. Used by the MPCO reader + integration tests.

---

## v1.0.9 — Viewer: higher-order rendering + filter overhaul (WIP)

Lands the viewer fixes and refactor scaffolding from PR #11.  Higher-
order elements (Q8/Q9, Tri6, Tet10, etc.) no longer render as VTK's
sub-triangle tessellation fans.  The dim filter actually hides actors
now, and node display scopes to the dim filter.  Step 5 (corner /
midside / bubble node differentiation) is deferred to the next release.

### FIXED — viewer

- **Q9 / higher-order surface fill** — the fill layer is now built from
  linearized corner-only cells (`mesh_scene.GMSH_LINEAR`), so a Q9 quad
  renders as a single quad and a Tri6 as a single triangle.  31 element
  types covered including P3 / P4 and bubble variants; unknown types
  warn instead of being silently dropped.
- **Dim filter (1D/2D/3D checkboxes)** — was overridden in
  `mesh_viewer._on_mesh_filter` setup so it only set the pick mask;
  now also flips fill / wire / node-cloud actor visibility per dim.
- **Phantom wireframe on Reveal** — `VisibilityManager._rebuild_actors`
  now rebuilds the wire actor alongside the fill on hide / reveal, so
  hidden entities lose their edges and revealed entities regain them.
- **BRep surface fill for higher-order meshes** — `brep_scene` got the
  same linearization treatment for Tri6 / Quad8 / Quad9.

### CHANGED — viewer

- New **wireframe layer** built via `extract_all_edges()` per dim>=2,
  registered as `EntityRegistry.dim_wire_actors`.  Replaces VTK's
  built-in `show_edges` (which rendered the higher-order cell
  tessellation, not the FE element boundary).  Clipping plane,
  visibility manager, and dim filter all participate.
- **Per-dim node clouds** — single global `node_actor` replaced by
  `EntityRegistry.dim_node_actors` keyed by dim, each containing the
  nodes used by entities of that dim (with `includeBoundary=True`).
  The dim filter now scopes node display: unchecking 1D drops 1D-only
  nodes, but boundary nodes shared with a visible 2D dim stay.
- **Tree right-click Hide / Isolate / Reveal-all** — added to BRep
  `SelectionTreePanel` and `BrowserTab` (group + entity rows).  Backed
  by new `VisibilityManager.hide_dts(dts)` / `isolate_dts(dts)`
  programmatic counterparts of the pick-driven `hide()` / `isolate()`.

### ADDED — `viewers.core.visibility` doc

- Spelled out the **filter state model** in the module docstring:
  cosmetic dim toggle (`SetVisibility`), entity hide
  (`VisibilityManager._hidden`), and clipping (render-time) are three
  independent layers, intentionally not unified.

## v1.0.8 — Embedded-node constraint resolver (`ASDEmbeddedNodeElement`)

Closes Phase 11b.  Replaces the `NotImplementedError` on
`ConstraintsComposite._resolve_embedded` with a working resolver, so
embedded-rebar and similar non-conforming inclusions can be expressed
without fragmenting the host mesh.

### ADDED — `solvers/_constraint_resolver.py`

- `_barycentric_tri3(p, p0, p1, p2)` — barycentric coordinates of `p`
  inside a 2D triangle, with projection onto the triangle's plane for
  off-plane points.
- `_barycentric_tet4(p, p0, p1, p2, p3)` — same for a 3D tetrahedron.
- `ConstraintResolver.resolve_embedded(...)` — given embedded nodes and
  host element connectivity, locates each embedded node in its host
  via KD-tree spatial indexing + barycentric coordinates, then emits
  `InterpolationRecord` shape-function couplings that match
  `ASDEmbeddedNodeElement` kinematics.

### ADDED — integration

- `ConstraintsComposite._resolve_embedded` now dispatches to the new
  resolver, collects host element connectivity from a labeled master
  region (tri3 / tet4), filters out embedded nodes that coincide with
  host corners, and returns the coupling records.
- `examples/EOS Examples/15_embedded_rebar.py` rewritten to use the
  embedded path instead of the old fragment-based conformal rebar.

### ADDED — tests

- `tests/test_constraint_resolver.py` — 4 cases (tri3 interior + corner,
  tet4 centroid, multi-element search).

### ADDED — regression coverage

- `tests/test_target_resolution.py` — locks in `FEMData.nodes.get()` /
  `.elements.get()` `target=` precedence (`label > PG`) and raw
  `[(dim, tag)]` passthrough.
- `tests/test_boolean_ops.py` — guards the 2D `fragment(cleanup_free=True)`
  bug so it can't regress.
- `tests/test_parts_advanced.py` — covers `g.parts.add(part, label=...,
  translate=...)` on an unlabeled Part (no sidecar).

### ADDED — infrastructure

- `pyproject.toml` `[tool.pytest.ini_options]` with
  `pythonpath = ["src"]`, so pytest run from a worktree picks up the
  worktree's source instead of the editable install pointing at the
  main checkout.

---

## v1.0.7 — Selection upgrades + `set_transfinite_box`

Polish pass on the v1.0.6 selection API.  Eliminates the hand-rolled
patterns that kept showing up in scripts (two-step boundary queries,
`_apply_hex` helpers, manual node-count-by-axis loops) and adds the
predicates and combinators users were reaching for.

### ADDED — boundary helpers

- `g.model.queries.boundary_curves(tag)` — returns every unique
  curve on the boundary of an entity.  Encapsulates the two-step
  query (faces → individual face boundaries with `combined=False` →
  deduplicate) that's needed because Gmsh's `getBoundary(vol,
  recursive=True)` skips dim=1 and goes straight to vertices.
  Accepts a label, PG name, int tag, dimtag, or list of any.
- `g.model.queries.boundary_points(tag)` — symmetric helper for
  the eight corner points of a volume.

### ADDED — `select()` upgrades

- `select()` now accepts a label string with a `dim=` keyword:
  `select('box', dim=2, on={'z': 0})` resolves the label, walks
  to dim 2, and applies the predicate — no manual `boundary()`
  call beforehand.
- `not_on=` and `not_crossing=` negation predicates.  Same
  signed-distance computation as the positive forms; useful for
  *all faces except the bottom* style queries.
- The `Selection.to_label()` call on a mixed-dim selection no
  longer triggers the labels-composite collision warning — using
  the same name across multiple dims is the documented intent
  here.

### ADDED — `Selection` ergonomics

- Set operations: `selection | other` (union with deduplication),
  `selection & other` (intersection), `selection - other`
  (difference).  All three preserve the back-reference to
  `_Queries` so chaining keeps working.
- `selection.partition_by(axis=None)` groups entities by their
  dominant bounding-box axis.  Returns a `dict[str, Selection]`
  keyed by `'x'`, `'y'`, `'z'`, or — if `axis=` is given — a
  single `Selection`.  Semantics are dim-aware:
  - **curves** group by the *largest* extent (curve direction);
  - **surfaces** group by the *smallest* extent (perpendicular /
    normal direction).

### ADDED — primitive factories

- `g.model.queries.plane(z=0)`, `plane(p1, p2, p3)`, and
  `plane(normal=..., through=...)` build a `Plane` you can pass to
  any `select(on=..., crossing=...)` call (positive or negated).
- `g.model.queries.line(p1, p2)` builds a `Line`.
- Define a primitive once, reuse across many selections — useful
  when the same cutting plane appears in several queries.

### ADDED — `set_transfinite_box`

- `g.mesh.structured.set_transfinite_box(vol, *, size=None, n=None,
  recombine=True)` — collapses the full transfinite-hex setup
  (curve node counts, face transfinite + recombine, volume
  transfinite) into a single call.  Accepts either `size=` (target
  element size; node counts derived per edge from
  `round(length / size) + 1`) or `n=` (uniform node count per
  edge).  Pass `recombine=False` for a transfinite tet mesh
  instead of hex.

### CHANGED

- `examples/EOS Examples/22_geometric_selection.ipynb` rewrites
  the 3-D section to use `set_transfinite_box`,
  `select('box', dim=2, ...)`, the `plane()` factory,
  `not_on=`, set operations, `partition_by`, and chained
  `to_label / to_physical` — every v1.0.7 feature is exercised.

### FIXED

- `g.model.queries.bounding_box(tag, dim=N)`,
  `center_of_mass(tag, dim=N)`, and `mass(tag, dim=N)` now honour
  `dim` as an explicit hint when ``tag`` is a bare integer.
  Previously these went through `resolve_to_single_dimtag` →
  `resolve_dim`, which always searches dimensions 3 → 0 and
  returns the first match — so on a model containing both volume
  1 and curve 1, `bounding_box(1, dim=1)` silently returned the
  volume's bounding box.  Bare ints are now passed straight to
  the corresponding Gmsh OCC call at the requested dim.  String
  labels and `(dim, tag)` tuples still go through resolution.
- `g.model.geometry.slice` now passes its plane reference as an
  explicit `(2, plane_tag)` dimtag to the downstream
  `cut_by_surface` / `cut_by_plane` calls.  Previously it passed
  a bare int, which triggered `resolve_dim` to scan the live
  Gmsh model — and because `add_axis_cutting_plane` is called
  with `sync=False`, the new plane wasn't yet visible to
  `getEntities(2)`, causing the resolver to fall through to the
  curves and fail with `"surface ref N resolved to dim=1"`.
  Together these two fixes recover ≈14 previously-failing tests
  in `test_geometry_cutting`, `test_sections`, and
  `test_part_anchors`.

### INTERNAL

- New `_Queries._resolve_to_dimtags(tag)` helper consolidates the
  string / int / dimtag resolution path used by `boundary_curves`,
  `boundary_points`, and the new `select()` label-string branch.
- `_select_impl` now takes `not_on` / `not_crossing` and inverts
  the predicate result (`hit ^ invert`).  The four kwargs are
  mutually exclusive — exactly one must be passed.
- `Selection` is parameterised on `DimTag` (a `tuple[int, int]`
  alias).  Method signatures use proper type hints throughout.
- 29 new test cases in `tests/test_selection.py` covering the new
  helpers, the negation predicates, set operations, `partition_by`
  for both curves and surfaces, the primitive factories, and
  `set_transfinite_box`.  Total: 55 cases passing.

---

## v1.0.6 — Geometric selection API (`g.model.queries.select`)

### ADDED

- `g.model.queries.select(entities, on=..., crossing=...)` filters
  curves, surfaces, or volumes by a geometric predicate. Replaces
  the noisy `entities_in_bounding_box(xmin,ymin,zmin,xmax,ymax,zmax)`
  pattern with a readable description of *what* you want.
- Predicates work on the bounding-box corners of each candidate:
  - `on=` — every corner within `tol` of the primitive (entity lies
    on it).
  - `crossing=` — corners exist on both sides of the primitive
    (entity straddles it). Same signed-distance computation
    underlies both.
- Primitive formats — no imports needed:
  - `{'z': 0}` → axis-aligned plane z = 0.
  - `[(p1), (p2)]` (2 points) → infinite line, for cutting 2-D
    geometry.
  - `[(p1), (p2), (p3)]` (3 points) → infinite plane through 3
    non-collinear points. Use for surfaces and volumes.
- `select()` returns a `Selection` — a `list` subclass with three
  chainable methods:
  - `.select(...)` — filter further (AND logic when stacked).
  - `.to_label(name)` — register every entity as a label, grouped
    by dimension.
  - `.to_physical(name)` — register every entity as a physical
    group, grouped by dimension.
  Each returns `self` so you can keep chaining: `select → label →
  select again → physical`.
- `Selection.__repr__` describes the count by dimension and reminds
  the user how to chain — IDE autocomplete + the repr are the only
  discovery surface needed.
- New curriculum notebook
  `examples/EOS Examples/22_geometric_selection.ipynb` walks the
  full workflow: predicate intro → stacking → unstructured baseline
  → transfinite quad mesh of a plate → 3-D hex of a box.
- Companion script
  `examples/example_unstructured_and_transfinite.py` shows the
  unstructured-vs-transfinite contrast for two adjacent boxes.
- API docs page extended at `docs/api/model.md` with `Selection`,
  `Plane`, and `Line` (the latter two documented as internal but
  exposed so the format reference is auto-generated from
  docstrings).

### INTERNAL

- New module `src/apeGmsh/core/_selection.py` holds `Plane`, `Line`,
  the `_parse_primitive` dispatcher, the `_select_impl` core, and
  the `Selection` class. `Plane` and `Line` are not part of the
  public API — they are only constructed by `_parse_primitive` from
  raw user input passed to `select()`.
- `Selection` carries a back-reference to the originating `_Queries`
  so `.select()`, `.to_label()`, and `.to_physical()` can route to
  the session's `labels` / `physical` composites without the user
  having to thread context.
- Tests in `tests/test_selection.py` (26 cases) cover predicates in
  2-D and 3-D, primitive parsing (including degenerate / collinear /
  coincident input), the Selection chain, label / PG registration
  for both single-dim and mixed-dim selections, and error paths.

---

## v1.0.5 — Line loads with `normal=True` (radial / curve-perpendicular pressure)

### ADDED

- `g.loads.line(..., normal=True)` applies a pressure perpendicular
  to each edge instead of along a fixed direction.  Useful for
  internal/external pressure on curved 2-D boundaries (Lamé-style
  problems, fluid-loaded arcs, etc.) where the cartesian direction
  varies along the curve.
- Default sign comes from Gmsh's surface boundary orientation —
  `gmsh.model.getBoundary([(2, surface)], oriented=True)` tells the
  composite which side of the curve the structure sits on, so
  `magnitude > 0` always pushes *into* the structure (matching
  `g.loads.surface(..., normal=True)`).
- Optional `away_from=(x0, y0, z0)` reference point overrides the
  Gmsh path — flips the in-plane normal so it points away from that
  point.  Use for ambiguous cases (a curve bounding two surfaces, or
  a free curve not bounding any surface) or when you want to be
  explicit.
- Both `reduction="tributary"` (default) and `reduction="consistent"`
  honour `normal=True`.
- New worked example
  `examples/EOS Examples/example_plate_pyGmsh_v2.ipynb` rewrites the
  thick-walled-cylinder Lamé problem on top of the new API —
  replaces ~30 lines of manual consistent-force integration on the
  inner arc with a single `g.loads.line(pg='Pressure', magnitude=p,
  normal=True)` declaration.

### INTERNAL

- New resolver methods `LoadResolver.resolve_line_per_edge_tributary`
  and `resolve_line_per_edge_consistent` accept a list of
  `(edge, q_xyz)` items so the composite can pre-compute per-edge
  force-per-length vectors (which vary along curved boundaries).
  Constant-direction line loads still use the original
  `resolve_line_*` methods unchanged.
- Per-edge normal computation lives in `LoadsComposite` (it needs
  Gmsh queries for the boundary-orientation path); the resolver
  stays pure-math.

---

## v1.0.4 — Low-level booleans preserve Instances + accept label/PG refs

### FIXED

- `g.model.boolean.{fuse,cut,intersect,fragment}` now keep
  `Instance.entities` consistent when called directly on tags that
  happen to belong to a tracked Instance.  Previously the remap only
  ran inside `g.parts.fragment_all` / `fragment_pair` / `fuse_group`,
  so a low-level boolean left Instance entries pointing at consumed
  tags.  The remap-from-result walk has been extracted into
  `PartsRegistry._remap_from_result` and every OCC boolean call site
  (both `_bool_op` and the Parts-level methods) now routes through
  that single implementation.

### ADDED

- `g.model.boolean.*` accepts label names and user physical-group
  names in `objects=` / `tools=`, matching the input shape of
  `g.physical.add`.  Strings resolve via the shared resolver: label
  (Tier 1) first, then user PG (Tier 2).  Raw tags, dimtags, and
  mixed lists still work.

### INTERNAL

- New `resolve_to_dimtags` helper in `apeGmsh.core._helpers` —
  companion to `resolve_to_tags` that emits `(dim, tag)` pairs.
  Handles labels / PGs that span multiple dimensions without the
  caller having to coerce a single dim.
- Plan B (`Instance.entities` as a computed label-backed property)
  was weighed against this conservative fix and deferred; see
  `internal_docs/plan_instance_computed_view.md` for the signals that
  would trigger revisiting it.

---

## v1.0.0 — Clean Architecture (breaking)

v1.0 bundles two breaking changes: the package rename and the Model
composition refactor. A full find-replace migration guide is at
[`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md).

### BREAKING

- **Package renamed**: `pyGmsh` → `apeGmsh`
  - `from pyGmsh import pyGmsh` → `from apeGmsh import apeGmsh`
  - `class pyGmsh(_SessionBase)` → `class apeGmsh(_SessionBase)`
  - Companion app `apeGmshViewer` keeps its name (only its internal
    imports of our theme module were updated)

- **Model methods split into five sub-composites** (composition replaces
  mixin inheritance):
  - `g.model.geometry.*` — 19 primitive builders (add_point, add_line,
    add_box, add_cylinder, etc.)
  - `g.model.boolean.*` — fuse, cut, intersect, fragment
  - `g.model.transforms.*` — translate, rotate, scale, mirror, copy,
    extrude, revolve, sweep, thru_sections
  - `g.model.io.*` — load/save STEP, IGES, DXF, MSH, heal_shapes
  - `g.model.queries.*` — bounding_box, center_of_mass, mass, boundary,
    adjacencies, entities_in_bounding_box, remove, remove_duplicates,
    make_conformal, registry
  - `g.model.sync()`, `g.model.viewer()`, `g.model.gui()`,
    `g.model.launch_picker()`, `g.model.selection` stay flat on Model

- **Rename `g.mass` → `g.masses`** for consistency with the other
  plural composites (`g.loads`, `g.parts`, `g.physical`, `g.constraints`,
  `g.mesh_selection`)
  - `fem.mass` → `fem.masses`
  - Class names (`MassesComposite`, `MassSet`, `MassDef`, `MassRecord`)
    unchanged

- **Removed legacy aliases**:
  - `g.initialize()` / `g.finalize()` → use `g.begin()` / `g.end()`
  - `g._initialized` → use `g.is_active`
  - `g.model_name` → use `g.name`

- **Removed deprecated methods**:
  - `g.model.viewer_fast()` / `g.mesh.viewer_fast()` → use `viewer()`
    (always fast now)
  - `g.parts.add_physical_groups()` → explicit
    `g.physical.add_volume(inst.entities[3], name=...)`
  - `g.opensees.add_nodal_load()` → use `g.loads.point()` in a
    `g.loads.pattern()` block
  - `g.mesh_selection.add_nodes(nearest_to=...)` → `closest_to=`

- **Removed convenience delegates on the session**:
  - `g.remove_duplicates()` → `g.model.queries.remove_duplicates()`
  - `g.make_conformal()` → `g.model.queries.make_conformal()`

- **Removed property on `_SessionBase`**:
  - `_parent.model_name` → `_parent.name`

### FIXED

- Pylance / static analyzers no longer lose track of Model methods.
  Composition makes every method statically discoverable through
  the sub-composite classes (`_Geometry`, `_Boolean`, `_Transforms`,
  `_IO`, `_Queries`), each of which is a concrete class with explicit
  methods. No MRO walking across 5 mixin files.

### INTERNAL

- `_GeometryMixin` → `_Geometry`
- `_BooleanMixin` → `_Boolean`
- `_TransformsMixin` → `_Transforms`
- `_IOMixin` → `_IO`
- `_QueriesMixin` → `_Queries`

Each sub-composite now takes a `model` reference in `__init__` and
accesses Model state via `self._model._log(...)`,
`self._model._register(...)`, `self._model._as_dimtags(...)`,
`self._model._registry`, instead of inheriting state.

### MIGRATION

See [`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md) for the complete
find-replace table and an automated migration script.

**v0.3.0** is the last `pyGmsh` release (pre-rename safety tag).
**v0.3.1** is the transitional `apeGmsh` release (rename only).
**v1.0.0** is the new architecture (composition + cleanups).

---

## v0.3.1 — Package rename (transitional)

- Renamed `pyGmsh` package directory to `apeGmsh/` on disk
- All internal imports, class name, config entries updated
- Examples and docs still reference old API (deferred to v1.0)
- Safety release — rename is isolated from the architectural
  refactor that follows in v1.0

## v0.3.0 — Last pyGmsh release (safety tag)

Safety checkpoint before the package rename and v1.0 refactor. This
is the final tag under the `pyGmsh` name. If you need the old API,
pin to this version.

---

## v0.2.x — Loads, Masses, Viewer overlays

- New `g.loads` composite with pattern context managers
- New `g.mass` composite (renamed to `g.masses` in v1.0)
- `fem.loads` / `fem.mass` auto-resolved by `get_fem_data()`
- Loads/masses/constraints overlays live on `g.mesh.viewer(fem=...)` —
  they are mesh-resolved concepts and never landed on `g.model.viewer()`

## v0.2.0 — Composites architecture

- Assembly absorbed into `apeGmsh` as composites:
  - `g.parts` (PartsRegistry)
  - `g.constraints` (ConstraintsComposite)
- MeshSelectionSet + \_mesh\_filters spatial query engine
- Viewer rebuild: BRep / mesh viewers unified around EntityRegistry,
  PickEngine, ColorManager, VisibilityManager
- Catppuccin Mocha theme across all viewers
