# ADR 0041 — Chain-phase routing for geometry-intensive constraints (`embedded`, `tied_contact`)

**Status:** ACCEPTED 2026-05-27. Authored as DRAFT during a multi-agent
design review session; the 8 open questions were resolved with the
worker's recommendations adopted verbatim. See **Decisions (resolved
2026-05-27)** at the end of this file for the locked answers.
Renumbered 0039 → 0041 at acceptance to honor reservations for
ADR 0039 (`embedded host decomposition primitive`) and ADR 0040
(`g.loads.imposed_strain` eigenstrain primitive) — both deferred per
ADR 0038 but anticipated future ADRs that warrant the lower numbers.

This ADR proposes the design for Compose v1.1-A.2; the
[v1.1-A worker report (PR #380)](https://github.com/nmorabowen/apeGmsh/pull/380)
explicitly deferred `EmbeddedDef` and `TiedContactDef` to a separate
ADR because the architectural choices for the lifted geometric
primitives are load-bearing. No code has shipped under this ADR yet —
Phase 1 (Kuhn decomposition lift) follows acceptance.
Builds on [ADR 0036](0036-embedded-host-decomposition.md) (the build-
phase Kuhn decomposition this ADR proposes to lift) and
[ADR 0038](0038-compose-model-composition.md) (the chain-phase router
and `FEMDataSource` introduced in compose v1).

## Context

Compose v1 ([ADR 0038](0038-compose-model-composition.md)) ships
`g.compose(source_h5, label=...)` — load a previously-saved apeGmsh
model into the current session and continue working on the composed
broker. Once the session enters chain phase (`g._fem is not None`),
geometry mutation is frozen by `ChainPhaseError` but the eight
`g.constraints.*` callables remain callable per the contract in ADR
0038 Context paragraph:

> Interfaces between host and composed module are necessarily non-
> conformal (the module was meshed independently of the host) and are
> bridged by the existing constraint primitives —
> `g.constraints.embedded(...)` per ADR 0036,
> `g.constraints.tied_contact(...)`, `g.constraints.equalDOF(...)`,
> `g.constraints.rigid_link(...)`.

Compose v1.1-A
([PR #380](https://github.com/nmorabowen/apeGmsh/pull/380)) wired
chain-phase routing for the three node-only-resolution defs that
needed only the existing `FEMDataSource` surface (`nodes_for(target)`,
`has_target(target)`):

* `EqualDOFDef` → `NodePairRecord` via `ConstraintResolver.resolve_equal_dof`
* `RigidLinkDef` → `NodePairRecord` via `ConstraintResolver.resolve_rigid_link`
* `RigidDiaphragmDef` → `NodeGroupRecord` via `ConstraintResolver.resolve_rigid_diaphragm`

The router branch in
[`_chain_phase_router.py:190-203`](../../../_kernel/resolvers/_chain_phase_router.py)
dispatches per def type; each branch builds the resolver with
`(node_tags, node_coords)` arrays from `fem.nodes` and runs the same
pure-numpy resolver methods the build path uses. No gmsh state
required.

### The deferral

`EmbeddedDef` and `TiedContactDef` were explicitly deferred. PR #380's
worker report:

> These need geometry-surface queries `FEMDataSource` doesn't expose
> today:
>
> - **EmbeddedDef** needs Kuhn decomposition for hex/prism/pyramid
>   host elements (currently in `_collect_host_subelements` reading
>   from `gmsh.model.mesh.getElementsByType`) — would need to lift
>   off gmsh state
> - **TiedContactDef** needs face-connectivity synthesis from volume
>   elements (currently from `gmsh.model.mesh.getElementFaceNodes`) +
>   boundary face mapping per surface PG
>
> Both are substantive refactors (~1500+ lines if folded in). Defer
> to v1.1-A.2 — possibly its own ADR for the face-connectivity
> layer.

The callable contract is preserved: both defs are appended to
`ConstraintsComposite.constraint_defs`, the bump-counter advances,
and the next build-phase `get_fem_data()` re-extraction would resolve
them. In chain phase (e.g. post-`apeGmsh.from_h5(...)`) the next
extraction takes the chain-phase short-circuit and `_fem` is handed
back as-is — the def is stored but never applied.
Tests in PR #380's `TestDeferredDefsFallBackCleanly` lock the
callable-contract-preserved-but-not-applied behaviour.

### What the build-phase resolution path looks like today

The two defs flow through `ConstraintsComposite._resolve_embedded`
([ConstraintsComposite.py:1527-1584](../../../core/ConstraintsComposite.py))
and `_resolve_face_both`
([ConstraintsComposite.py:1467-1473](../../../core/ConstraintsComposite.py))
respectively:

**`_resolve_embedded`:**

1. `_collect_host_subelements(host_entities)` — calls
   `gmsh.model.mesh.getElements(dim, tag)` for each host entity,
   dispatches per Gmsh etype, decomposes hex8 / hex20 / prism6 /
   prism15 / pyramid5 / pyramid13 into corner-only sub-tets via the
   `HEX8_TO_6_TETS` / `PRISM6_TO_3_TETS` / `PYRAMID5_TO_2_TETS`
   module-level tables, decomposes quad4 / quad8 / quad9 into corner-
   only sub-tris, accepts tri3 / tri6 / tet4 / tet10 directly, and
   emits one `UserWarning` per (etype, entity) when a higher-order
   host hits the decomposition path. Returns an `ndarray(F, 3 | 4)`
   of virtual sub-element connectivity (corner-only).
2. `gmsh.model.mesh.getNodes(dim, tag)` for each embedded entity,
   union into `embedded_nodes: set[int]`.
3. `resolve_embedded(defn, host_elems, embedded_nodes)` on the
   resolver
   ([_resolver.py:726-858](../../../_kernel/resolvers/_constraint_resolver/_resolver.py))
   — builds a centroid KD-tree over host elements, for each embedded
   node finds the K=16 nearest centroids, evaluates `_barycentric_tri3`
   or `_barycentric_tet4` against each candidate, picks the lowest-
   excess hit, fails loud if `best_excess > defn.tolerance`, emits
   one `InterpolationRecord` per resolved embed.

**`_resolve_face_both` → `resolve_tied_contact`:**

1. `_resolve_faces(label, "master", defn, face_map)` — `face_map` is
   precomputed at build time by
   `PartsRegistry.build_face_map(node_map)`
   ([_parts_registry.py:883-903](../../../core/_parts_registry.py))
   which calls `_collect_surface_faces()` to pull every dim=2 surface
   element's connectivity via `gmsh.model.mesh.getElements(dim=2, tag)`
   and then filters per-part by node-ownership. The result is a
   `dict[label → ndarray(F, n_per_face)]`.
2. Same for the slave side.
3. `resolve_tied_contact(defn, m_faces, s_faces, m_nodes, s_nodes)` on
   the resolver
   ([_resolver.py:572-617](../../../_kernel/resolvers/_constraint_resolver/_resolver.py))
   — builds a `TieDef` from the `TiedContactDef`, calls
   `resolve_tie(tie_fwd, master_face_conn, slave_nodes)` to project
   each slave node onto the master faces (KD-tree centroid candidates
   + Newton iteration via `_project_point_to_face`), emits
   `InterpolationRecord` instances, wraps them in a
   `SurfaceCouplingRecord`.

### What FEMData carries today vs. what the lift needs

FEMData's `ElementComposite`
([FEMData.py:601-720](../../../mesh/FEMData.py)) carries:

* `_groups: dict[int, ElementGroup]` keyed by Gmsh etype code. Each
  `ElementGroup` carries `element_type` (ElementTypeInfo with
  `name`/`code`/`dim`/`order`/`npe`), `ids: ndarray(N,)`,
  `connectivity: ndarray(N, npe)`.
  ([payloads.py:117-186](../../../_kernel/payloads.py))
* `physical: PhysicalGroupSet` — Tier-2 PGs keyed by name.
* `labels: LabelSet` — Tier-1 labels keyed by name.
* `constraints: SurfaceConstraintSet` — already-resolved surface
  records (round-tripped through H5).
* `loads`, `partitions`, `_part_elem_map`, `_module_label` — not
  load-bearing for this ADR.

Both 2D and 3D ElementGroups are present in a typical SSI extract
(`get_fem_data(dim=None)` is the default;
[_fem_extract.py:52-82](../../../mesh/_fem_extract.py) extracts
elements for every dim Gmsh has, then unions referenced nodes). So
the face-connectivity for `tied_contact` is **already in FEMData** —
it's the dim=2 ElementGroups, no synthesis required. The deferral
narrative's "face-connectivity synthesis from volume elements" turns
out to be unnecessary for the typical case; what's needed is a
node-ownership filter over the existing dim=2 groups, mirroring
`PartsRegistry.build_face_map`.

What is NOT in FEMData:

* The Kuhn decomposition tables and the per-etype dispatch logic —
  these live on `ConstraintsComposite` as module-level constants
  ([ConstraintsComposite.py:101-130](../../../core/ConstraintsComposite.py))
  and as the `_collect_host_subelements` static method
  ([ConstraintsComposite.py:1633-1810](../../../core/ConstraintsComposite.py)).
  Both call `gmsh.model.mesh.*` and cannot run in chain phase
  unchanged.
* A `surface_pg → boundary_face_rows` map. Not strictly needed if we
  filter the existing dim=2 ElementGroups by node ownership; needed
  only if we want pre-computed acceleration.

### The user-value framing — SSI cimbra workflows

The canonical SSI workflow in
[ADR 0034 §5c](0034-stage-bound-bcs-and-recorders.md) installs
a cimbra (lining) inside Stage 2 via `domainChange` onto an
already-equilibrated rock state. The compose path is:

```python
g = apeGmsh.from_h5("soil_composed.h5")
g.compose("cimbra.h5", label="cimbra")
g.constraints.embedded(host="soil", embed="cimbra")  # ← silently no-ops today
g.compose("rebar.h5", label="rebar")
g.constraints.tied_contact(master="soil_interface",
                            slave="cimbra_interface")  # ← silently no-ops today
g.save("combined.h5")
```

Steps 3 and 5 today succeed at the API level (the def is appended)
but the constraint is never resolved into `_fem`. When the bridge
later walks `fem.elements.constraints` it sees no embedded /
tied_contact records and emits nothing — the SSI deck silently lacks
the interface coupling, the user gets divergence on the second
analyze step, and the failure mode is opaque (no error, no warning,
just wrong physics). Closing this gap is the explicit goal of
v1.1-A.2.

## Decision

The proposal below is structured around the four design questions
raised in the v1.1-A.2 kickoff. **Each subsection presents ONE
recommendation with reasoning.** The "Open questions for user
review" section at the end calls out the points that need explicit
user sign-off before Phase 1 begins.

### 1. Lift Kuhn decomposition into a geometric utility module

**Recommendation:** new file
`src/apeGmsh/_kernel/geometry/_host_decomposition.py`. Move the three
Kuhn-tet tables (`HEX8_TO_6_TETS`, `PRISM6_TO_3_TETS`,
`PYRAMID5_TO_2_TETS`) and the per-etype dispatch function out of
`ConstraintsComposite` into this module as a pure-numpy function:

```python
# src/apeGmsh/_kernel/geometry/_host_decomposition.py

HEX8_TO_6_TETS: ndarray         # exact same table as today
PRISM6_TO_3_TETS: ndarray       # exact same table as today
PYRAMID5_TO_2_TETS: ndarray     # exact same table as today

def decompose_hosts_to_subelements(
    groups: Iterable[tuple[int, ndarray]],   # (gmsh_etype, conn_array)
    *,
    warn_higher_order: Callable[[int, str], None] | None = None,
) -> ndarray:
    """Return ndarray(F, 3 | 4) of virtual tri / tet sub-element rows.

    Per-etype dispatch identical to ADR 0036's decomposition table:
    tri3 / tet4 identity; quad4 / quad8 / quad9 → corner tris;
    hex8 / hex20 → 6 Kuhn tets; prism6 / prism15 → 3 tets;
    pyramid5 / pyramid13 → 2 tets; tri6 / tet10 → corner-only.

    Caller passes (etype, conn) pairs already extracted from any
    source (gmsh.model.mesh.getElements OR FEMData ElementGroup);
    function is source-agnostic and gmsh-free.

    Mixed-dim host fail-loud + higher-order warning identical to
    ADR 0036's `_collect_host_subelements` invariants — the
    ``warn_higher_order`` callback receives (etype_code, descriptor)
    so the caller controls the warning's stacklevel / message.
    """
    ...
```

`ConstraintsComposite._collect_host_subelements` becomes a thin
adapter: pull `(etype, conn)` pairs via `gmsh.model.mesh.getElements`
for each host entity, then delegate to
`decompose_hosts_to_subelements(...)`. Identical numerical output;
build-phase byte-for-byte deck parity locked by an existing test in
`tests/test_embedded_decomposition.py`.

The chain-phase variant in `FEMDataSource` (§3 below) pulls
`(code, conn)` pairs from the broker's
`ElementComposite._groups[code].connectivity` instead, then calls the
same `decompose_hosts_to_subelements(...)` function.

**Why a geometric utility, not a sibling adapter in
`_constraint_resolver/`:**

Two reasons. First, the Kuhn tables and the per-etype dispatch are
pure geometry — they have no notion of "constraint resolver" or
"FEMData" or "gmsh session". Putting them in `_constraint_resolver/`
makes them harder to reuse from other resolvers that may need host
decomposition later (e.g. a future `g.loads.body_force_on_host(...)`
that picks up integration points on the same Kuhn sub-tets, or a
distributed-coupling primitive that needs the same corner-node
projection). `_kernel/geometry/` is the natural home — same layering
as `_kernel/_label_prefix`, `_kernel/payloads`, `_kernel/defs`.

Second, the lift is a **rename + module move**, not a redesign. The
function body is the existing per-etype switch from
`ConstraintsComposite._collect_host_subelements`, parameterised to
accept `(etype, conn)` pairs from any source rather than calling
`gmsh.model.mesh.getElements` directly. A sibling adapter would
duplicate the switch; a geometric utility lets both adapters
(build-phase gmsh-backed, chain-phase FEMData-backed) call one
function.

**Alternative considered and rejected:**

* `_constraint_resolver/_geom.py` already exists
  ([_geom.py:1-50](../../../_kernel/resolvers/_constraint_resolver/_geom.py))
  and carries `SHAPE_FUNCTIONS`, `_SpatialIndex`,
  `_project_point_to_face`, `_is_inside_parametric`. Adding the Kuhn
  tables here is geographically tempting (one module for all
  constraint geometry) but locks the decomposition into the resolver
  layer. Future non-constraint consumers (loads, masses, viewer slice
  visualisations) would have to import across the resolver boundary,
  which violates the `_kernel/` leaf-only convention from ADR 0015.

### 2. `FEMDataSource` API extension — minimal element-side surface

**Recommendation:** add two methods to `FEMDataSource`. Keep the
Protocol on the existing four-method shape — the Protocol stays
narrow per ADR 0038's "FEMDataSource is the narrow adapter"
commitment; the new methods land on the `FEMDataSource` concrete
class only.

```python
class FEMDataSource:
    # ... existing nodes_for / node_ids / node_coords / has_target ...

    def host_subelements_for(
        self, target: str,
    ) -> ndarray:
        """Return ndarray(F, 3 | 4) of virtual sub-element rows
        suitable as ``host_elems`` for ConstraintResolver.resolve_embedded.

        Resolves ``target`` to a set of element ids via the same
        Tier 1 → Tier 2 walk as :meth:`nodes_for`, but on the
        element side (skipping the node-side tiers).  Pulls the
        (etype, conn) pairs for the resolved elements and delegates
        to :func:`decompose_hosts_to_subelements`.

        Raises ``KeyError`` when ``target`` resolves to no elements
        and ``ValueError`` (matching the build-phase message) when
        the elements carry no embeddable host types.
        """
        ...

    def boundary_faces_for(
        self, target: str,
    ) -> ndarray:
        """Return ndarray(F, n_per_face) of surface element connectivity
        for the given target.

        Resolves ``target`` element-side (Tier 1 → Tier 2), then
        filters the dim=2 :class:`ElementGroup` rows by node-ownership
        (every row whose nodes all belong to the target's node set is
        kept).  Mirrors :meth:`PartsRegistry.build_face_map` semantics
        but works off the broker's existing dim=2 ElementGroups —
        no face synthesis from volume elements.

        Raises ``KeyError`` when ``target`` resolves to no entities,
        and ``ValueError`` mirroring the build-phase
        ``_resolve_faces`` message when no surface rows survive the
        ownership filter.
        """
        ...
```

The build-phase `GmshSource` peer
([_source.py:207-266](../../../_kernel/resolvers/_source.py)) does
NOT need to grow these methods — the build path keeps using
`_collect_host_subelements` / `_collect_surface_faces` directly via
`ConstraintsComposite`. The chain-phase router is the only consumer
that needs the new surface, and only on the FEMData adapter.

**Why exactly two methods (not more granular, not bulk):**

* `host_subelements_for(target)` matches the resolver's input shape
  (`ndarray(F, 3 | 4)`) and replaces one explicit call site
  (`_collect_host_subelements`). Splitting it into
  `elements_for(target) -> {npe: conn}` + `decompose(...)` exposes
  internal intermediate state for no caller benefit — the chain-
  phase router doesn't have a use case for the raw per-type element
  dict that the decomposition consumes.
* `boundary_faces_for(target)` mirrors the resolver's input shape
  (`ndarray(F, n_per_face)`) and replaces the second explicit call
  site (`face_map[label]` after `parts.build_face_map`). Same
  reasoning — splitting into `surface_elements_for(target)` +
  `filter_by_nodes(...)` would expose intermediate state with no
  reuse beneficiary.

**Why not extend the `ResolverSource` Protocol:**

The Protocol's purpose per its docstring is "the small surface that
session composites use to look up node tags, coordinates, and PG /
label resolutions". Adding element-side queries broadens it past
that scope and forces every future foreign adapter (LS-DYNA, xDMF,
Exodus per ADR 0026 §"Future adapters") to either implement element
queries or stub them. The Protocol stays at four methods; the two
new methods are concrete-class only.

**Alternative considered and rejected:**

* Lazy materialisation. The methods could cache results across calls
  (one `boundary_faces_for("soil")` may be followed by another call
  with the same target). Defer caching until a benchmark shows it
  matters — chain-phase resolution is bounded by the def count
  (typically <50 per session) and the cost is dominated by the
  resolver's KD-tree builds, not the source queries.

### 3. Chain-phase router branches — narrow dispatch over the new surface

**Recommendation:** add two branches to `route_def_to_fem` in
`_chain_phase_router.py`, parallel to the existing
`_route_equal_dof` / `_route_rigid_link` / `_route_rigid_diaphragm`
shape:

```python
# In _chain_phase_router.py

def _route_embedded(fem, source, defn) -> FEMData:
    """Resolve ``EmbeddedDef`` against ``fem`` and append interpolation records."""
    from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver

    host_elems = source.host_subelements_for(defn.master_label)
    embedded_nodes = set(int(n) for n in source.nodes_for(defn.slave_label))
    # Mirror build-phase: drop nodes that coincide with host corners.
    host_corner_nodes = set(int(t) for t in np.unique(host_elems))
    embedded_nodes -= host_corner_nodes
    if not embedded_nodes:
        return fem

    resolver = _build_resolver(fem, ConstraintResolver)
    records = resolver.resolve_embedded(defn, host_elems, embedded_nodes)
    new_fem = fem
    for rec in records:
        new_fem = new_fem.with_constraint(rec)
    return new_fem


def _route_tied_contact(fem, source, defn) -> FEMData:
    """Resolve ``TiedContactDef`` against ``fem`` and append surface-coupling record."""
    from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver

    m_faces = source.boundary_faces_for(defn.master_label)
    s_faces = source.boundary_faces_for(defn.slave_label)
    m_nodes = set(int(n) for n in source.nodes_for(defn.master_label))
    s_nodes = set(int(n) for n in source.nodes_for(defn.slave_label))

    resolver = _build_resolver(fem, ConstraintResolver)
    record = resolver.resolve_tied_contact(
        defn, m_faces, s_faces, m_nodes, s_nodes,
    )
    return fem.with_constraint(record)
```

Plus the two new dispatch lines in `route_def_to_fem` at
[`_chain_phase_router.py:200`](../../../_kernel/resolvers/_chain_phase_router.py):

```python
if isinstance(defn, EmbeddedDef):
    return _route_embedded(fem, source, defn)

if isinstance(defn, TiedContactDef):
    return _route_tied_contact(fem, source, defn)
```

The router's existing build-phase shape stays — `try_chain_phase_route`
catches `KeyError`/`TypeError` so a missing target still falls back
to the bump-counter pattern with the existing documented gap.
Resolver-side fail-louds (e.g.
`resolve_embedded`'s "slave node lies outside every host element")
propagate as `ValueError`, NOT caught — these are real errors the
user must address, just like the build-phase path raises them today.

**Worth noting:** `_build_resolver` currently passes only
`(node_tags, node_coords)`. `resolve_embedded` doesn't need
`(elem_tags, connectivity)` — the host elements are passed
explicitly as a function argument (`host_elems`). Same for
`resolve_tied_contact` (`m_faces` / `s_faces`). So `_build_resolver`
does NOT need to grow; the two new router branches use the existing
two-argument resolver constructor.

### 4. Face-connectivity synthesis — NOT needed for the typical case

**Recommendation:** do not add a `BoundaryFaceSet` composite or any
volume-face synthesis pass. The dim=2 surface ElementGroups are
already in FEMData via the default `get_fem_data(dim=None)`
extraction. `FEMDataSource.boundary_faces_for(target)` filters them
by node-ownership — exactly the same pattern as
`PartsRegistry.build_face_map`.

This was the architectural fork the kickoff flagged as unresolved.
The survey closed it:

* `_fem_extract.extract_groups` at
  [_fem_extract.py:52-82](../../../mesh/_fem_extract.py) extracts
  every dim Gmsh has when `dim is None`. The default
  `g.mesh.queries.get_fem_data()` uses `dim=None`
  ([_mesh_queries.py:122-138](../../../mesh/_mesh_queries.py)).
* H5 round-trip preserves all ElementGroups (per ADR 0023's per-zone
  schema).
* `apeGmsh.from_h5(...)` rebuilds FEMData with every ElementGroup the
  saved session extracted.

So as long as the saving session used the default extraction (every
SSI-style workflow does), the dim=2 surface elements are present in
chain phase. No synthesis pass needed.

**The narrow gap that remains:** if a user explicitly calls
`g.mesh.queries.get_fem_data(dim=3)` before `g.save(...)`, the saved
broker carries only dim=3 elements. A chain-phase
`tied_contact(master="...", slave="...")` then has no dim=2 groups
to filter and `boundary_faces_for` raises a clear error. The
user-visible remedy is to re-extract with `dim=None` before saving,
or to use `tied_contact` only in build phase. Documented in
`boundary_faces_for`'s docstring as a known gap; not worth a synthesis
pass that would have to invert volume connectivity into face rows for
this rare misconfiguration.

A future PR can add `synthesize_boundary_faces_from_volume(fem)` if
a real user hits this — purely additive to the design here, gated
on demand. Tracked in `_DEFERRED.md`.

### Two concrete examples

**Example 1 — embedded inside compose:**

```python
g = apeGmsh.from_h5("soil_composed.h5")
g.compose("cimbra.h5", label="cimbra")
g.constraints.embedded(
    host_label="soil",
    embedded_label="cimbra.interface_nodes",
    tolerance=1e-4,
)
g.save("combined.h5")
```

Flow:

1. `ConstraintsComposite.embedded(...)` builds `EmbeddedDef`,
   appends to `constraint_defs`.
2. `try_chain_phase_route(session, defn)` calls
   `route_def_to_fem(fem, defn)`.
3. New `EmbeddedDef` branch fires:
   `source.host_subelements_for("soil")` → resolves "soil" to
   element-side label, pulls (etype, conn) pairs from
   `fem.elements._groups`, calls
   `decompose_hosts_to_subelements(...)` → `ndarray(F, 4)` Kuhn
   sub-tets.
4. `source.nodes_for("cimbra.interface_nodes")` → `ndarray(N,)` of
   embedded node ids.
5. `resolver.resolve_embedded(defn, host_elems, embedded_nodes)` →
   list of `InterpolationRecord`s.
6. `fem.with_constraint(rec)` per record → new FEMData snapshot.
7. `session._fem = new_fem`, cache marked fresh.
8. `g.save(...)` persists the new constraint records via the
   existing H5 writer.

**Example 2 — tied_contact inside compose:**

```python
g = apeGmsh.from_h5("soil_composed.h5")
g.compose("cimbra.h5", label="cimbra")
g.constraints.tied_contact(
    master_label="soil.interface",
    slave_label="cimbra.interface",
    tolerance=1e-3,
)
```

Flow:

1. Def appended, router fires.
2. New `TiedContactDef` branch: `source.boundary_faces_for(...)`
   pulls dim=2 ElementGroups from `fem.elements._groups`, filters
   rows whose nodes all belong to "soil.interface" → master face
   array. Same for slave.
3. `resolver.resolve_tied_contact(defn, m_faces, s_faces, m_nodes,
   s_nodes)` → one `SurfaceCouplingRecord` wrapping per-slave
   `InterpolationRecord`s.
4. `fem.with_constraint(rec)` → new FEMData snapshot.

## Invariants

**INV-1 — Callable contract from ADR 0038 unchanged.**
`g.constraints.embedded(...)` and `g.constraints.tied_contact(...)`
remain callable in both build phase and chain phase. The bump-
counter path stays as the safety net for any def shape this router
does not cover. The two new router branches replace the deferred
no-op with proper transform routing, but they do NOT change the
public API and do NOT change build-phase behaviour.

**INV-2 — Build phase byte-identical.**
`ConstraintsComposite._collect_host_subelements` becomes a thin
adapter that delegates to
`decompose_hosts_to_subelements(...)`. The function body is the
existing per-etype switch, parameterised to accept `(etype, conn)`
pairs from any source. Build-phase callers see identical resolved
records, identical warnings, identical fail-louds. Locked by the
existing test suite in `tests/test_embedded_decomposition.py` and
the build-phase deck-parity tests.

**INV-3 — `FEMDataSource` stays narrow.**
The `ResolverSource` Protocol stays at four methods (`node_ids`,
`node_coords`, `nodes_for`, `has_target`). Element-side queries land
on the `FEMDataSource` concrete class only — the build-phase
`GmshSource` peer does not need to implement them. Foreign-format
adapters (LS-DYNA, xDMF, Exodus per ADR 0026) inherit the four-
method Protocol unchanged.

**INV-4 — Geometry mutation still frozen.**
`split_higher_order_lines` (ADR 0037) and any future broker mesh
mutation remain blocked by `ChainPhaseError` in chain phase. The new
router branches are READ-only against FEMData; they call
`fem.with_constraint(record)` which produces a new immutable snapshot
but never mutates the underlying gmsh state (because there is no
gmsh state in chain phase).

**INV-5 — No H5 schema bump.**
The two new router branches produce existing
`InterpolationRecord` / `SurfaceCouplingRecord` instances that already
have H5 serialisation per the broker's existing per-zone schema
(ADR 0023). No new H5 fields, no schema version bump.

**INV-6 — Higher-order warning + mixed-dim fail-loud preserved.**
ADR 0036's four mitigations (host_coupling kwarg, Kuhn orientation
property test, mixed-dim host fail-loud, higher-order warning) flow
through the new `decompose_hosts_to_subelements` function. The
warning fires exactly once per (etype, target) — the build-phase
warning per (etype, entity) maps to the chain-phase warning per
(etype, target-label) since the target is the natural granularity in
chain phase (no entity tags exist).

## Implementation phases

The implementation breaks into five sub-phases. Each is independently
shippable; the user can merge after any phase and the model continues
working (the deferred phases stay on bump-counter fallback).

### Phase 1 — Lift Kuhn decomposition into geometric utility (~250 LOC)

**Scope:** new file
`src/apeGmsh/_kernel/geometry/_host_decomposition.py` carrying the
three Kuhn tables and the `decompose_hosts_to_subelements(...)`
function. Rewire
`ConstraintsComposite._collect_host_subelements` as a thin adapter
that pulls (etype, conn) pairs from gmsh and delegates. Test surface:
all existing `tests/test_embedded_decomposition.py` cases pass
unchanged.

**Dependency:** none.

**"Useful alone?":** Yes — the lift is a refactor that makes the
decomposition reusable. No new user-visible behaviour, but the new
module is in place for Phase 3.

### Phase 2 — `FEMDataSource.host_subelements_for(target)` (~150 LOC + tests)

**Scope:** add the new method on `FEMDataSource` (concrete class
only; Protocol unchanged). Element-side label/PG resolution mirroring
`nodes_for`'s Tier 1 → Tier 2 walk. Delegates to Phase 1's
`decompose_hosts_to_subelements(...)`. Test surface:
`tests/test_phase_v1_1_a2_femdatasource.py` — round-trip parity
(build a model, save, load, assert `host_subelements_for("...")`
returns ndarray rows identical to a build-phase
`_collect_host_subelements` over the same label).

**Dependency:** Phase 1.

**"Useful alone?":** Marginally — exposes the surface but nothing
calls it yet.

### Phase 3 — `EmbeddedDef` chain-phase routing (~200 LOC + tests)

**Scope:** new `_route_embedded(fem, source, defn)` helper in
`_chain_phase_router.py`. Branch in `route_def_to_fem`. Test
surface: `TestEmbeddedChainPhase` in
`tests/test_phase_v1_1_a2_chain_phase_router.py` —
`EmbeddedDef` resolved in chain phase produces the same
`InterpolationRecord`s as the build-phase path (golden parity);
empty embedded node set → no-op; tolerance miss → `ValueError`
(matching build-phase message).

**Dependency:** Phases 1 + 2.

**"Useful alone?":** Yes — closes the embedded gap for SSI cimbra
workflows. tied_contact still deferred but the most common SSI use
case (embedded rebar / cimbra into compositionally-loaded soil)
works end-to-end.

### Phase 4 — `FEMDataSource.boundary_faces_for(target)` (~120 LOC + tests)

**Scope:** add the second concrete-class method. Pulls dim=2
ElementGroups, filters by node-ownership, returns `ndarray(F,
n_per_face)`. Test surface: golden parity against build-phase
`PartsRegistry.build_face_map(node_map)[label]` output.

**Dependency:** none (independent of Phase 1+2+3).

**"Useful alone?":** Marginally — exposes the surface but nothing
calls it yet. Could ship before Phase 3 if scheduling demanded.

### Phase 5 — `TiedContactDef` chain-phase routing (~150 LOC + tests)

**Scope:** new `_route_tied_contact(fem, source, defn)` helper.
Branch in `route_def_to_fem`. Test surface:
`TestTiedContactChainPhase` — golden parity with build-phase
`resolve_tied_contact` output; empty face set → fail-loud (matching
`_resolve_faces` message); H5 round-trip locks the constraint.

**Dependency:** Phase 4.

**"Useful alone?":** Yes — closes the tied_contact gap. Combined
with Phase 3, the full v1.1-A.2 scope ships.

### Sequencing recommendation

Ship Phase 1 + 2 + 3 as one PR (the embedded slice) and Phase 4 + 5
as a second PR (the tied_contact slice). Two PRs let the user
review the embedded path in isolation before the slightly-more-
involved tied_contact path lands. Total implementation footprint
~870 LOC + tests; tractable in two sessions.

## Rejected alternatives

**Defer indefinitely.** Same as v1.1-A.2's pre-shipped state today.
The callable contract per ADR 0038 is preserved; only chain-phase
routing is missing. Acceptance criterion: when a user actually hits
this in an SSI workflow. The kickoff doc explicitly names cimbra
SSI as that user; deferring further delays the workflow that
motivated compose v1 in the first place. **Rejected.**

**Composed-results writer pre-resolution.** Resolve embedded /
tied_contact at H5 write time so the saved broker carries pre-
resolved records, then chain-phase loads just see records (no
resolution needed). Pros: chain-phase router stays narrow. Cons: the
write-time resolution would still need the lifted geometric
utilities (Phase 1), and it shifts the user's mental model — `embedded`
calls between `save()` and the next user-action would still no-op.
The chain-phase router has to handle the "user composed a new module
then added a constraint" case regardless. **Rejected** because the
work is the same and the write-time variant is strictly less
expressive.

**Stage-bound constraints route.** Per
[ADR 0034 §5a](0034-stage-bound-bcs-and-recorders.md), `s.embedded
(name=...)` already exists as a CLAIM-by-name path that defers an
already-resolved constraint to a stage block. This is orthogonal to
chain-phase routing — the constraint must FIRST be resolved into
`fem.elements.constraints` (the apeGmsh build-time or compose-time
step), THEN claimed into the stage at apeSees-emit time. ADR 0034
explicitly notes "by apeSees bridge time the resolver dependencies
are typically gone and the resolved records already live on the
FEMData broker". The CLAIM path does NOT close the gap that v1.1-A.2
addresses; it consumes its output. **Not an alternative — they
compose.**

**Per-record-set widening (skip `FEMDataSource`).** Move the new
queries from `FEMDataSource` into `ConstraintsComposite` itself — a
phase-aware `_collect_host_subelements_chain` method that walks
`fem.elements._groups` when `self._parent._fem is not None`. Pros:
no `FEMDataSource` extension; the resolver doesn't learn the new
source surface. Cons: violates ADR 0038's "FEMDataSource is the
narrow adapter" commitment by leaking the chain-phase logic back
into the composite, and breaks the foreign-adapter future (an
LS-DYNA-backed FEMData via `from_h5("model.h5_lsdyna")` should be
able to drive chain-phase routing through the source adapter
interface without `ConstraintsComposite` knowing the source format).
**Rejected.**

**Add the new methods to the `ResolverSource` Protocol.** Force
every source — including the future LS-DYNA / Exodus / xDMF adapters
hinted at in ADR 0026 — to implement element-side queries. Pros:
uniform consumer code (the router doesn't switch on adapter type).
Cons: foreign adapters that lack a host-decomposition concept (a
beam-only LS-DYNA model has no Kuhn tets to decompose) would either
stub the method (raise NotImplementedError, which the router would
have to catch — same as today's bump-counter fallback) or
re-implement the decomposition (a lift that doesn't belong on a
foreign adapter). **Rejected** in favour of concrete-class methods;
the Protocol stays at four methods.

**Add a `BoundaryFaceSet` composite to FEMData.** Make face
connectivity an explicit FEMData attribute with a `surface_pg →
face_rows` mapping, persisted to H5, so chain-phase queries are
O(1). Pros: faster lookups; no per-call filter pass. Cons: it's a
new H5 schema field (bump from per-zone schema X.Y to X.(Y+1) per
ADR 0023), and existing models that didn't carry it would need a
backward-compat read path. The filter pass on the existing dim=2
ElementGroups is O(F × log N) and unmeasured-but-fast for typical
SSI face counts (<5000). **Rejected** as a premature optimisation;
gated on a real benchmark showing it matters.

**Synthesize boundary faces from volume connectivity.** Implement
`_synthesize_faces_from_volume(fem)` that inverts each volume
ElementGroup's connectivity into the canonical face rows per Gmsh
element type, then filters by surface PG. Pros: handles the
"saved with `dim=3` only" edge case. Cons: substantial code surface
(per-etype face-extraction tables, orientation handling), real
debugging risk, and the use case (user explicitly extracted dim=3
only, then later composed and tied_contact'd) is rare and has a
clear user-visible remedy (re-extract with `dim=None`). **Rejected**
for the v1.1-A.2 scope; deferred per
[_DEFERRED.md](../_DEFERRED.md).

## Consequences

### Positive

* **SSI cimbra workflows ship.** The canonical compose-then-tie
  pattern in ADR 0034 §5c works end-to-end on the chain. The user's
  current workaround (build all constraints in build phase before
  the first `g.save()`, then never add constraints in chain phase)
  goes away.
* **Build phase byte-identical.** The lift in Phase 1 is a refactor
  with golden parity; build-phase decks match pre-lift output. No
  user-visible change in any non-chain-phase workflow.
* **The `_collect_host_subelements` rename to a `_kernel/geometry/`
  utility opens follow-on uses** — future constraint primitives
  (distributed coupling on hosts, body force on Kuhn sub-tets,
  visualisation slices that walk sub-tets for solid-element fringes)
  can consume the same function without going through
  `ConstraintsComposite`.
* **`FEMDataSource` grows narrowly.** Two concrete-class methods,
  no Protocol bump. Foreign adapters per ADR 0026 stay unaffected.
* **No H5 schema bump, no new neutral-zone fields, no bridge
  changes.** The output records are existing
  `InterpolationRecord` / `SurfaceCouplingRecord` shapes.

### Negative (acknowledged)

* **Dim=3-only-saved broker has no chain-phase tied_contact path.**
  Documented in `boundary_faces_for`'s docstring; the user-facing
  remedy is to re-extract with `dim=None` before saving. Volume-
  to-face synthesis deferred per `_DEFERRED.md`.
* **The lift adds a new `_kernel/geometry/` module.** One more
  module in the kernel surface; one more place future contributors
  have to look when chasing constraint geometry. Mitigated by the
  module being a leaf (zero internal-apeGmsh imports, just numpy).
* **`FEMDataSource` is now phase-aware in a load-bearing way.** It
  was always the chain-phase adapter, but adding two query methods
  makes the chain-phase router depend on FEMData carrying a complete
  element table (across all dims). Build-phase callers continue to
  use `_collect_host_subelements` / `_collect_surface_faces`
  directly; the asymmetry is documented.

### Neutral

* **Implementation footprint ~870 LOC + tests across two PRs.**
  Comparable to v1.1-A (~500 LOC); not a multi-month project.
* **Test surface grows by two test files** — one for `FEMDataSource`
  extensions (Phases 2 + 4), one for the router branches (Phases 3
  + 5). Both small and focused.

## Decisions (resolved 2026-05-27)

The 8 questions raised during DRAFT review were resolved with all
worker recommendations adopted verbatim. Captured here for future
session continuity — these decisions are locked unless an explicit
ADR amendment supersedes them.

1. **Module location for Kuhn decomposition.** **Accepted: new
   subdir `src/apeGmsh/_kernel/geometry/_host_decomposition.py`.**
   Rationale: the decomposition isn't constraint-specific — future
   loads, masses, viewer slices can reuse. Putting it under
   `_constraint_resolver/` would mis-scope it.

2. **ADR number.** **Accepted: renumber to 0041** to honor the
   Phase 3 closeout's reservations of 0039 (`embedded host
   decomposition primitive`) + 0040 (`g.loads.imposed_strain`
   eigenstrain primitive). Both are deferred per ADR 0038 but
   anticipated future ADRs that warrant the lower numbers.

3. **`FEMDataSource` API split.** **Accepted: two concrete-class
   methods** — `host_subelements_for(target)` +
   `boundary_faces_for(target)`. Rationale: single-responsibility —
   embedded only needs host_subelements; tied_contact only needs
   boundary_faces. Combining them forces every caller to know
   post-filter logic. Two methods also document intent at the API.

4. **Protocol vs concrete-class only.** **Accepted: concrete-class
   only.** Methods land on `FEMDataSource` (concrete); leave
   `ResolverSource` Protocol unchanged at its four narrow methods.
   Rationale: the Protocol's job is the narrow node-resolution
   contract every reader can fulfil; element-side decomposition is
   a FEMData-specific capability. Widening the Protocol would force
   foreign adapters (LS-DYNA d3plot, xDMF, Exodus per ADR 0026)
   into either stubs or refusal-to-conform — same anti-pattern as
   the H5ModelReader "Protocol doesn't widen for results data"
   decision.

5. **`boundary_faces_for` and the dim=3-only-saved broker.**
   **Accepted: raise clean `ValueError` with documented remedy**
   ("no dim=2 ElementGroups in FEMData; re-extract with `dim=None`
   and re-save"). The volume-to-face synthesis pass is deferred —
   substantial code (~300–500 lines) for a rare misconfiguration.
   Users hitting it have a clear recovery path. If/when synthesis
   turns out to be needed (a real user with a real `dim=3`-only
   model), Phase 6 schedules then. YAGNI.

6. **PR sequencing.** **Accepted: two PRs** — Phases 1+2+3
   (embedded slice) and Phases 4+5 (tied_contact slice). Rationale:
   embedded is the more common SSI use case; shipping it standalone
   unblocks users earlier. Tied_contact is more specialized.
   Matches the prior compose work's slice pattern.

7. **Build-phase rename ripple.** **Accepted: audit + move
   cleanly if no consumers; otherwise re-export from the new
   module.** Phase 1's first task is the audit — survey for
   `from apeGmsh.core.ConstraintsComposite import HEX8_TO_6_TETS`
   (or similar) across the codebase + tests. If clean, move the
   constants outright; if consumers exist, add re-export aliases
   in `ConstraintsComposite` for backward compat.

8. **`host_subelements_for` warning channel.** **Accepted: per
   `(etype, target)`.** Threading gmsh entity context into FEMData
   purely for warning messages is gold-plating; the (etype, target)
   pair gives the user enough information to find the offending
   elements. Mirrors how other chain-phase errors phrase themselves
   (e.g., the v1.1-A "Constraint label '...' resolves to neither a
   label nor a physical group" message).

### Phase 1 unblocking criteria

With all 8 decisions resolved, Phase 1 (Kuhn decomposition lift)
spawns as the next worker task. Sequencing:

- **Phase 1**: lift Kuhn decomposition off gmsh → new module at
  `_kernel/geometry/_host_decomposition.py` + tests + import-shim
  audit per decision 7
- **Phase 2**: `FEMDataSource.host_subelements_for` + tests
- **Phase 3**: `EmbeddedDef` chain-phase routing in `_chain_phase_router.py` + integration tests
- **PR boundary** (per decision 6)
- **Phase 4**: `FEMDataSource.boundary_faces_for` (filter dim=2 ElementGroups by node-ownership) + tests
- **Phase 5**: `TiedContactDef` chain-phase routing + integration tests
- **PR boundary**

## Followups (not blocking)

* **Volume-to-face synthesis** (Phase 6) — gated on a real user
  hitting the dim=3-only-saved gap. Tracked in `_DEFERRED.md`.
* **`host_subelements_for` result caching** — gated on a benchmark
  showing it matters for SSI workflows with many compose-then-tie
  cycles.
* **PR #380 cleanup follow-on** — `_compose_bundles` cleanup
  flagged in #366's worker report (still allocated but unused);
  unchanged here, can ride in Phase 1 if convenient.

## References

* PR #380 (v1.1-A — first compose chain-phase router shipping) —
  [https://github.com/nmorabowen/apeGmsh/pull/380](https://github.com/nmorabowen/apeGmsh/pull/380),
  body excerpted in Context above for the deferral rationale.
* [`_chain_phase_router.py`](../../../_kernel/resolvers/_chain_phase_router.py)
  — existing router; new branches land here.
  * `try_chain_phase_route` at lines 55–87.
  * `route_def_to_fem` dispatch at lines 90–203 (deferral comment
    at lines 35–43).
  * `_route_equal_dof` / `_route_rigid_link` / `_route_rigid_diaphragm`
    at lines 206–275 — the shape v1.1-A.2's new branches will mirror.
  * `_build_resolver` at lines 278–289 — used unchanged.
* [`_source.py`](../../../_kernel/resolvers/_source.py) —
  `FEMDataSource` concrete class at lines 91–204. `nodes_for` Tier 1
  → Tier 2 walk at lines 116–171; `_nodes_from_element_ids` at lines
  183–204.
* [`_resolver.py`](../../../_kernel/resolvers/_constraint_resolver/_resolver.py)
  — build-phase resolvers; pure-numpy, no gmsh imports.
  * `resolve_embedded` at lines 726–858.
  * `resolve_tied_contact` at lines 572–617.
  * `_barycentric_tri3` / `_barycentric_tet4` at lines 52–111.
* [`_geom.py`](../../../_kernel/resolvers/_constraint_resolver/_geom.py)
  — existing geometry primitives (shape functions, spatial index);
  proposed Kuhn-decomposition module sits next to this in
  `_kernel/geometry/`.
* [`ConstraintsComposite.py`](../../../core/ConstraintsComposite.py)
  — build-phase composite.
  * `HEX8_TO_6_TETS` / `PRISM6_TO_3_TETS` / `PYRAMID5_TO_2_TETS`
    tables at lines 101–130 (the lift targets).
  * `_collect_host_subelements` at lines 1633–1810 (the function to
    refactor as an adapter).
  * `_resolve_embedded` at lines 1527–1584 (the build-phase call
    site).
  * `_resolve_face_both` at lines 1467–1473 (the tied_contact build
    call site).
  * `_resolve_faces` at lines 1403–1443 (uses `face_map[label]`).
  * `_add_def` chain-phase routing call at lines 332–347.
* [`_parts_registry.py`](../../../core/_parts_registry.py) —
  `PartsRegistry.build_face_map` at lines 883–903;
  `_collect_surface_faces` at lines 1211–1249 (the mirror that
  `boundary_faces_for` replaces in chain phase).
* [`FEMData.py`](../../../mesh/FEMData.py) —
  `ElementComposite` at lines 601–720, `with_constraint` at line
  1697.
* [`payloads.py`](../../../_kernel/payloads.py) — `ElementGroup` at
  lines 117–186 (the dim=2 surface groups
  `boundary_faces_for` filters).
* [`_fem_extract.py`](../../../mesh/_fem_extract.py) —
  `extract_groups` at lines 40–106 (the default `dim=None`
  extraction that ensures dim=2 ElementGroups are in chain-phase
  FEMData).
* [ADR 0034](0034-stage-bound-bcs-and-recorders.md) §5a — stage-
  bound CLAIM-by-name constraints; composes with this ADR but does
  not replace it.
* [ADR 0036](0036-embedded-host-decomposition.md) — embedded-host
  decomposition (the build-phase Kuhn machinery this ADR lifts).
* [ADR 0038](0038-compose-model-composition.md) — compose v1; the
  callable contract preserved per the Context paragraph and the
  chain-phase router introduced here.
* Project memory:
  `project_compose_v1_1_a_2_adr_kickoff.md`,
  `project_compose_v1_1_chain_phase_router_kickoff.md`.
