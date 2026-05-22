# ADR 0024 — `Emitter.region` Protocol widening for MPCO `pg=` filtering

**Status:** Accepted (Cerro Lindo Tier-1 feature work, late-May 2026).
Closes the LHS-scaling pain point for MPCO output and widens the
`Emitter` Protocol (the explicit "architecture event" the Protocol's
header documents).

## Context

The MPCO recorder writes a self-describing HDF5 `.mpco` file that is
the highest-information output format OpenSees produces — STKO /
apeGmsh consumers read it natively for fiber-level stress / strain
histories and section-force diagrams.

The MPCO recorder **records the whole model by default** (per the
authoritative mpco-recorder skill). The only filter the upstream
`OPS_MPCORecorder` parser accepts is `-R $regTag` — a reference to a
pre-declared OpenSees `region`. There is no direct `-node ... -ele
...` filtering on the recorder line itself.

The Cerro Lindo cimbra calibration runs an LHS envelope of ~1800
non-linear pushovers with `section.fiber.stress` recorded per element.
At whole-model scope this is **218 MB per run → ~400 GB total**. With
a region filter targeting only hombros / crown the same study fits in
~10 GB.

Before this ADR, the apeGmsh bridge had **no way** to declare a
`region` from `apeSees`. The `Emitter` Protocol
(`opensees/emitter/base.py`) carried no `region()` method, the
`MPCO` recorder primitive carried no `nodes_pg=` / `elements_pg=`
selectors, and the build pipeline had no fan-out to materialize PG
selectors into a region declaration. Users who needed MPCO output
filtered by an apeGmsh physical group either wrote raw OpenSees Tcl
by hand alongside their bridge-driven deck (breaking the typed-
primitive contract) or recorded the whole model and discarded the
unwanted output downstream (the 218 MB → 5 MB lever they couldn't
pull).

The header of `opensees/emitter/base.py` documents that widening the
Protocol is *"an architecture event"*: every concrete emitter
(`TclEmitter`, `PyEmitter`, `LiveOpsEmitter`, `H5Emitter`,
`RecordingEmitter`) must grow the new method, and the H5 schema must
gain a new group. This ADR is that event. ADR 0022 (MP constraint
emission, Phase 7b) is the immediate precedent for adding to the
Protocol with the same shape of ceremony.

## Decision

### Widen the `Emitter` Protocol — one new method

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def region(self, tag: int, *args: int | float | str) -> None: ...
```

`args` carries the raw OpenSees flag tail (`-node n1 n2 ...`,
`-ele e1 e2 ...`, `-eleOnly`, `-nodeOnly`, `-eleRange first last`,
etc.). One `region` Tcl command can mix `-node` and `-ele` in the
same call; MPCO's `-R $tag` then filters both nodal and element
output through the region's combined membership.

Implementation on all five concrete emitters:

| Emitter | `region(tag, *args)` |
|---|---|
| `TclEmitter` | Append `region $tag $args...` (single line) |
| `PyEmitter` | Append `ops.region(tag, *args)` line |
| `LiveOpsEmitter` | Call `ops.region(tag, *args)` directly on `openseespy.opensees` |
| `H5Emitter` | Append a `RegionRecord(tag, args)` to the internal buffer; persisted under `/opensees/regions/region_NNN` at write time |
| `RecordingEmitter` | Append `("region", (tag, *args), {})` to `self.calls` |

### Extend the `MPCO` recorder primitive with four selector kwargs

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class MPCO(Recorder):
    # ... existing fields ...
    nodes:       tuple[int, ...] | None = None
    nodes_pg:    str            | None = None
    elements:    tuple[int, ...] | None = None
    elements_pg: str            | None = None
    _region_tag: int            | None = None   # build-pipeline-set
```

Pairwise mutex: `nodes=` xor `nodes_pg=`; `elements=` xor `elements_pg=`.

Asymmetric-filter guard: a node-only filter (`nodes` or `nodes_pg`)
combined with `elem_responses` is **refused at construction time** —
OpenSees `MeshRegion::setNodes()` does *not* auto-derive elements, so
MPCO filtered through a node-only region would silently produce an
empty element stream. The symmetric direction is refused for API
uniformity (auto-derived nodes from elements vary by element type;
forcing the user to be explicit is safer).

Empty-PG guard: the build pipeline raises `BridgeError` early when
any of the selectors resolves to zero ids against the FEM snapshot —
OpenSees rejects an empty region at runtime, so emitting a bare
`region $tag` is a guaranteed downstream failure.

### Build-pipeline fan-out

`emit_recorder_spec` in `opensees/_internal/build.py` is a 3-line
dispatcher: a `RecorderDeclaration` branch and a single call to
`spec.materialize(emitter, fem, tags)._emit(emitter, tag)`.  Every
recorder primitive carries its own `materialize(emitter, fem, tags) ->
Recorder` hook (the default on `Recorder` returns `self` unchanged;
`Node` / `Element` resolve `pg=`; `MPCO` resolves the four filter
selectors and emits the region).

`MPCO.materialize` does the following when any filter selector is set:

1. Resolves `nodes_pg` / `elements_pg` via the existing
   `expand_pg_to_nodes` / `expand_pg_to_elements` helpers, copies
   `nodes` / `elements` if explicit.
2. Allocates one fresh region tag from the bridge's `TagAllocator`
   (`tags.allocate("region")` — new namespace key; regions have
   their own `TaggedObjectStorage` in OpenSees, so no collision
   with element / material / section tags).
3. Emits `emitter.region(tag, *args)` once. The args list carries
   `-node n1 n2 ...` when nodes are present, `-ele e1 e2 ...` when
   elements are present, and both when both sides are populated.
4. Returns the spec via `dataclasses.replace` with the selectors
   cleared and `_region_tag=$tag` populated; the dispatcher then
   calls `_emit` on the returned spec which appends `-R $tag` to
   the MPCO command.

The bridge's `BuiltModel.emit` passes its `TagAllocator` through to
`emit_recorder_spec(..., tags=tags)` — this is the only difference
in the build orchestration. Other recorder kinds (Node, Element,
MPCO without filters, RecorderDeclaration) ignore the kwarg.

### H5 zone — write-side only

The `H5Emitter.region()` method appends a `RegionRecord` to an
internal buffer; `_write_regions(f)` persists each as a
`/opensees/regions/region_NNN` sub-group with:

- `tag` attribute (the OpenSees region tag)
- `params` dataset (the variable-length tail recorded by `_write_param_array`)

The schema bump from 2.8.0 → **2.9.0** is additive (ADR 0023 minor).
The intervening 2.8.0 bump (embeddedNode `embedding_ele`→`cnode`
field rename, see `emitter/h5.py`'s version-history block) landed
separately on `main` while this work was in flight; this ADR's
zone-add bump rides on top.  Per the two-version reader window,
both 2.8.x and 2.9.x files are accepted at the OPENSEES zone.  Old
2.8.x readers ignore the new group and lose only the filter
round-trip.

**The read side is deliberately deferred.**  Regions have a single
consumer today — MPCO `-R $tag` filtering — and that consumer
regenerates them from `nodes_pg` / `elements_pg` on every emit. A
round-trip accessor (`om.regions()`) plus a replay step in
`_replay_into` would round-trip data we always regenerate; the
initial implementation shipped both but the follow-up audit rolled
the read-side surface back.  The H5 group remains as a write-side
debugging artifact; if a second consumer emerges (Rayleigh damping
by region, element load patterns scoped to a region, ...) the
accessor + replay can be re-introduced at that point.

## Invariants

- **INV-1** — `Emitter.region` is on the Protocol; every existing
  and future emitter implements it. AST tests pin this.
- **INV-2** — The build pipeline allocates region tags in the
  `"region"` namespace of `TagAllocator`. Regions have their own
  storage in OpenSees, so this namespace is independent of
  elements / sections / materials.
- **INV-3** — MPCO with **any** filter selector goes through
  `MPCO.materialize`; the recorder's own `_emit` refuses with
  `NotImplementedError` if reached directly with `pg=` still set
  (defense-in-depth against bypass of the build pipeline).
- **INV-4** — A region's contents are *not* inferred at re-emit
  time; the `params` payload persisted under `/opensees/regions/`
  is replayed verbatim. The replay is byte-stable for any pair of
  `to_h5` / `from_h5` calls.
- **INV-5** — Asymmetric-filter combinations
  (`nodes_pg=` + `elem_responses` with no element filter, or the
  symmetric form) are refused at construction time. Silent-empty
  output is impossible.
- **INV-6** — Empty-resolution PGs are refused at build time
  with a clear `BridgeError`. No bare `region $tag` line ever
  reaches an emitter.

## Consequences

- LHS-scaling for MPCO output: filtered output drops 218 MB → ~5 MB
  per pushover on the Cerro Lindo workload, ~400 GB → ~10 GB at
  N=1800.
- The H5 zone (`/opensees/regions/`) is write-side only; the read
  surface (`om.regions()` accessor and `_replay_into` replay step)
  is deferred until a second consumer materializes. MPCO+pg models
  driven through `apeSees(fem)` get their region declared and the
  MPCO line correctly carries `-R $tag` on the first build; the H5
  round-trip case (`OpenSeesModel.from_h5(p).build("tcl", q)`) is
  the deferred branch.
- `regions` is in `MODEL_HASH_EXCLUDED_CHILDREN` alongside `cuts`
  and `sweeps`: the broker does not load the zone, so a
  `from_h5 → to_h5` cycle produces a file without regions and
  `model_hash` would otherwise drift. Excluding the group keeps
  lineage clean across the cycle.
- ADR 0023's per-zone schema versioning policy is followed (additive
  new zone → minor bump 2.8.0 → 2.9.0).
- The `region()` Protocol method is general-purpose; future use
  cases beyond MPCO filtering (Rayleigh damping by region, element
  load patterns scoped to a region, recorder Node/Element via
  region instead of explicit ids) can reuse the same Protocol entry
  point. This ADR closes the Protocol widening, not the set of
  consumers.

## Cross-references

- ADR 0022 — MP constraint emission fan-out (precedent for Protocol
  widening with the same ceremony).
- ADR 0023 — Per-zone schema versioning (the minor-bump rule
  applied here).
- ADR 0021 — Lineage chain (the model_hash determinism this ADR
  preserves through the replay step).
- ADR 0019 — `OpenSeesModel` read-side broker (the regions
  read-surface would follow the same per-record-kind pattern as
  `om.materials()`, `om.sections()`, etc. when the trigger arrives).
