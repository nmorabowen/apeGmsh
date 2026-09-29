# ADR 0019 — `OpenSeesModel` — read-side broker for `model.h5`, distinct from `apeSees`

**Status:** Accepted (Phase 3 of the major architectural refactor, May 2026).
Complements [ADR 0011](0011-h5-as-fourth-emit-target.md); complements
[ADR 0018](0018-modeldata-vanilla-opensees-enrichment.md); preserves
both their invariants.

## Context

[ADR 0011](0011-h5-as-fourth-emit-target.md) made `model.h5` the
model-definition archive and mitigated *"H5 becomes the primary
contract instead of the in-memory bridge"* by *"treating the H5
strictly as an emit output, never an input — the bridge does not read
its own H5 back."* That mitigation locks `apeSees` as a one-way
producer: there is no `apeSees.from_h5`, and there will not be one.

[ADR 0018](0018-modeldata-vanilla-opensees-enrichment.md) added
`ModelData` as a *second writer front-door* (orientation-only, vanilla
side-feeder), explicitly preserving 0011's invariant: *"`ModelData` is
not the bridge … the bridge still never reads its own output."*
`ModelData` is *structurally separate* from `apeSees`; that separation
is what lets it round-trip without violating 0011.

Today's read surface for the `/opensees/` zone is
`opensees/emitter/h5_reader.py`. It is intentionally dict-style: every
accessor returns attribute dicts and `numpy` arrays, never typed Python
objects. It serves the viewer (per [ADR 0014](0014-viewer-is-pure-h5-consumer.md))
and ad-hoc inspection, and that is all it is asked to do.

The major architectural refactor's chain-forward goal —
*Results carries Model carries FEM* (see [ADR 0020](0020-results-carries-opensees-model.md))
— needs more than dict bags. It needs a typed Python broker that
mirrors `/opensees/` contents, is queryable, is diffable, and survives
across a subprocess hop without re-walking HDF5 from raw bytes. Doing
this on `apeSees` would repeal 0011. Doing it on `ModelData` would
break 0018's INV-5 (orientation-only scope; "no materials / sections /
patterns / recorders / analysis / constraints / loads / masses").

The resolution is a third class, on the read side only.

## Decision

### The class — `OpenSeesModel`, read-only broker

Add `apeGmsh.opensees.OpenSeesModel` in
`apeGmsh/opensees/opensees_model.py`. A frozen Python object that
mirrors the `/opensees/` zone of one `model.h5` and carries an embedded
`FEMData` for the neutral zone.

```python
om = OpenSeesModel.from_h5(path)              # rehydrate
om.to_h5(path)                                 # round-trip
om.build(target)                               # re-emit: 'tcl' | 'py' | 'live' | 'h5'

om.fem                                         # the embedded FEMData (lazy-bound)
om.materials() / om.sections() / om.transforms() / om.beam_integration()
om.patterns() / om.recorders() / om.cuts() / om.sweeps()

om.lineage                                     # (fem_hash, model_hash) per ADR 0021
om.snapshot_id                                 # composite hash
```

`OpenSeesModel` is `@dataclass(frozen=True)` on the outer wrapper; its
typed-record collections are exposed through read-only views (no
`.append`, no in-place mutation). The wrapper carries opaque hashes —
it does **not** recompute lineage from the in-memory record graph on
every access. Recompute-on-write is governed by [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md).

### Distinct from `apeSees` and `ModelData` — three roles, three classes

The write side stays asymmetric per ADR 0018 INV-5:

- **`apeSees(fem)`** — full bridge, declarative authoring of every
  OpenSees concept (materials, sections, patterns, recorders, analysis,
  constraints, loads, masses). Write surface; never reads H5.
- **`ModelData(fem)`** — vanilla side-feeder; orientation-only.
  Write surface; reads H5 narrowly through its own `from_h5` enrich
  round-trip (per 0018), never to drive emission.
- **`OpenSeesModel`** — read-side broker for an archived `model.h5`.
  **Read-mostly**; `build(target)` re-emits, but the input is the
  rehydrated record graph, not the bridge's typed-primitive API.

This ADR introduces no new mutable surface. There is no
`OpenSeesModel.add_material(...)`, no `.add_pattern(...)`, no
`.add_recorder(...)`. To author or modify, return to `apeSees(fem)`
(full bridge) or `ModelData(fem)` (orientation-only). To read or
re-emit an archived model, use `OpenSeesModel`.

### Invariants

**INV-1.** `apeSees` does not gain `from_h5`. ADR 0011's
*"the bridge does not read its own H5 back"* is unchanged. Any future
PR that adds `apeSees.from_h5` must repeal 0011, and `OpenSeesModel`
exists precisely to remove the temptation.

**INV-2.** `OpenSeesModel` is **not** a unification of `apeSees` and
`ModelData`. It is a third role on the read side. The write-side
asymmetry mandated by ADR 0018 INV-5 stays. Repealing INV-2 means
repealing 0018.

**INV-3.** `OpenSeesModel` holds no `h5py` write surface. Schema
authority remains with `H5Emitter` (per ADR 0018's "schema authority
stays single"). `to_h5` delegates to the shared composer
(`_compose_model_h5`, introduced in 0018 §Decision) — `OpenSeesModel`
constructs an `H5Emitter`, populates it from its record graph, and
calls the composer. Future schema bumps flow to `OpenSeesModel` for
free.

**INV-4.** `OpenSeesModel.fem` is lazy-imported. The module-level
import graph for `apeGmsh.opensees` does not gain a new eager edge to
`apeGmsh.mesh`. `FEMData` is bound inside `from_h5` and inside the
record-collection methods that need it; the
`tests/test_import_dag_polarity.py` tripwire (per ADR 0015 §2) must
stay green when `OpenSeesModel` lands.

**INV-5.** `build(target)` is for *re-running an exact archived deck*.
Tag identity (the integer tags emitted by `ops.element`,
`ops.uniaxialMaterial`, …) may **diverge** from a fresh
`apeSees(fem).run()` because the bridge's `TagAllocator` allocations
are lost across the H5 round-trip. The rehydrated `OpenSeesModel`
re-allocates tags deterministically from its record graph, but those
tags are not promised to be byte-identical to the bridge's first run.
Documented loudly in the class docstring; users who need tag stability
must capture the bridge's `BuiltModel` from `apeSees.build()` directly
and not round-trip through H5.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Repeal ADR 0011 and add `apeSees.from_h5`** | Violates the bridge-stays-write-only invariant the codebase has held since Phase 4. Round-tripping the bridge means reverse-compiling an internal IR (typed primitives + `TagAllocator` + emit-time fan-out) that was never designed as a serialization format. The H5 schema's `__deviation__` attrs (one `geomTransf` group per `geomTransf(...)` call, not per element) tell you the bridge↔H5 mapping is one-way by construction. |
| **Unify `apeSees` and `ModelData` behind a single `Model` class with a `.mode` flag** | Already rejected in [ADR 0018 §Alternatives §3](0018-modeldata-vanilla-opensees-enrichment.md). A wide writer surface forced symmetry between two asymmetric mandates (full bridge vs orientation-only enrichment); inevitable feature-parity drift. Re-proposing it for the read side reintroduces the same problem on the other axis. |
| **Keep `h5_reader.py` as the only read surface** | Makes ADR 0020's *"Results carries Model"* impossible without `Results` carrying dict bags. Loses queryability (no `om.materials().filter(kind=...)`), loses diffability (no `diff(om_a, om_b)`), loses ergonomic API. The viewer already pays a per-call walk cost through `h5_reader`; the chain-forward path needs amortization. |
| **Mutable `OpenSeesModel` with `.add_material(...)` etc.** | Duplicates `apeSees`'s authoring surface; invites two ways to construct the same model; creates the exact parity drift 0018 §Alt 3 already rejected. The frozen read-side broker is the value — mutation lives on the write-side classes (`apeSees` / `ModelData`) and nowhere else. |
| **Embed `OpenSeesModel`'s record types inside `apeGmsh.opensees.emitter._records`** instead of a new module | The emitter's record types are bridge-internal (they exist to bridge typed primitives into HDF5 bytes). Exposing them publicly under `apeGmsh.opensees` couples the public read-side API to the bridge's internal record library — exactly the coupling 0018 §Alt 2 already rejected for `ModelData`. `OpenSeesModel` carries its own read-side record types (typed views over the rehydrated record graph), symmetric with `ViewerData`'s `_records.py` (per ADR 0014). |

## Consequences

**Positive:**

- ADR 0011 preserved verbatim: `apeSees.from_h5` will not exist, and
  the temptation to add it is structurally removed (the read-side need
  is now answered by `OpenSeesModel`).
- ADR 0018 INV-5 preserved: `ModelData` stays scoped to
  orientation-only; the broader read need is answered elsewhere.
- The chain-forward goal (ADR 0020: `Results → OpenSeesModel → FEM`)
  becomes possible without violating any prior ADR.
- Read API is Pythonic: typed records, queryable record collections,
  diffability. Future tooling — `diff_models(om_a, om_b) → list[str]`,
  *"what materials does this model use?"*, *"what recorder topology
  changed between runs?"* — becomes a one-liner instead of a manual
  `h5_reader` walk.
- Three roles, three classes, one mandate each. `apeSees` writes;
  `ModelData` side-feeds orientation; `OpenSeesModel` reads. The
  asymmetry is intentional and named.

**Negative:**

- Third "Model"-named concept in the codebase joining the existing
  four:
  - `g.model` — geometry composite (`apeGmsh.core.Model`)
  - `ops.model` — OpenSees domain command (vanilla openseespy)
  - `BuiltModel` — frozen artifact produced by `apeSees.build()`
  - `ModelData` — vanilla side-feeder (ADR 0018)
  - **`OpenSeesModel`** — this ADR
  Accepted: the name `OpenSeesModel` is searchable and unambiguous.
  Docstrings, error messages, and ADR prose must always use the full
  `OpenSeesModel` (not bare `Model`) to disambiguate. A new
  `tests/test_naming_disambiguation.py` style grep would help; not in
  scope here.
- `build('live')` tag-identity divergence from `apeSees.run()`
  (INV-5). Documented; downstream tooling that depends on bridge-time
  tag stability must not go through H5 round-trip.
- One new public class to maintain. Mitigated by INV-3 (schema
  authority elsewhere) and INV-4 (no new eager import edge).
- The frozen-wrapper + read-only-views idiom is one more pattern to
  learn (it joins `FEMData`, `BuiltModel`, `ViewerData`). Accepted:
  every "read-side broker" in the codebase will look the same after
  this.

## References

- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — H5 strictly as emit output; this ADR's INV-1 preserves it
  unchanged.
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the consumer-side precedent for a read-only adapter
  (`ViewerData`); `OpenSeesModel` is the broader-scope analog.
- [decisions/0018-modeldata-vanilla-opensees-enrichment.md](0018-modeldata-vanilla-opensees-enrichment.md)
  — three writer roles, schema authority on `H5Emitter`; this ADR's
  INV-2 / INV-3 preserve it.
- [decisions/0020-results-carries-opensees-model.md](0020-results-carries-opensees-model.md)
  — the immediate consumer of `OpenSeesModel`; the chain
  `Results → OpenSeesModel → FEMData`.
- [decisions/0021-lineage-chain-replaces-snapshot-id.md](0021-lineage-chain-replaces-snapshot-id.md)
  — the `lineage` / `snapshot_id` surface this class exposes.
- [decisions/0015-label-pg-separate-registries-kernel-leaf.md](0015-label-pg-separate-registries-kernel-leaf.md)
  — the import-DAG polarity invariant INV-4 protects.
- [phase-8-untangle.md](../phase-8-untangle.md) §7 closure — the
  open question this ADR resolves.
