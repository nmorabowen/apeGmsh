# ADR 0043 — Connectivity graph + flexible (split) emit for model chaining

**Status:** PROPOSED 2026-05-28 (partially resolved 2026-05-28).
Authored as a **discussion draft** to open the design space — no code
has shipped under it. The four architecture-binding open questions
(Q1 / Q2 / Q8 / Q9) are now **locked** — see **Decisions (resolved
2026-05-28)** below; the remaining questions (Q3 / Q4 / Q5 / Q6 / Q7 /
Q10, all mode-B or fragment-detail) stay open and will be settled as
phase-1 (mode A) lands. Promotion to ACCEPTED follows the
[ADR 0041](0041-chain-phase-geometry-constraints.md) draft→accept flow.

Builds on / consumes:
[ADR 0038](0038-compose-model-composition.md) (`g.compose`, the
`FEMDataSource` chain-phase surface),
[ADR 0041](0041-chain-phase-geometry-constraints.md) (chain-phase
routing for `embedded` / `tied_contact`),
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) (the lineage
DAG this ADR reuses as the graph backbone),
[ADR 0032](0032-explicit-only-per-node-ndf.md) /
[ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) (`g.node_ndf`, the
interface ndf-reconciliation primitive),
[ADR 0008](0008-three-emit-targets.md) /
[ADR 0011](0011-h5-as-fourth-emit-target.md) (the emit targets this
ADR restructures), and the SSI staging line
([ADR 0028](0028-initial-stress-via-parameter-ramping.md)–
[ADR 0034](0034-stage-bound-bcs-and-recorders.md)) (the sequential
data-flow mechanisms mode B reuses).

## Context

Today the emit pipeline writes **one monolithic deck**:
`apeSees(fem).tcl(path)` / `.py(path)` accumulate every line into a
single buffer and write it to a single file
(`apesees.py:4076`). The deck carries, in one topological stream:
`model` → materials/sections/transforms/time-series → nodes →
elements → constraints → patterns/loads → recorders → analysis chain
→ `analyze`.

That is the correct *default* — self-contained, reproducible, runs
with one `OpenSees deck.tcl`. It hurts in two ways:

1. **Scale.** A real mesh produces a file dominated by `node` /
   `element` bulk — megabytes where the logic worth reading or editing
   is a few dozen lines lost at the bottom. You cannot regenerate just
   the analysis or recorders against frozen geometry, and diffing two
   model variants is all geometry noise.
2. **Composition.** The longer-term goal is an *ANSYS-Workbench-style*
   feature: **node-connect parts of the model / mesh / bridge to chain
   models**. A monolithic single-file emit has no seam at which a
   "part" begins or ends, and no place for a cross-part interface to
   live.

### The one OpenSees fact that dictates the design

OpenSees has **one global `Domain`** and a **flat integer tag
namespace**. Every `node` / `element` / `uniaxialMaterial` / `section`
registers into that single mutable object keyed by an int tag — there
are no namespaces. And the lifecycle verbs are global and
destructive: `wipe` (nukes everything), `wipeAnalysis` (clears only
analysis objects), `loadConst -time 0.0` (freezes loads),
`remove` (deletes by tag), `domainChange` (rebuilds the DOF map after
mid-analysis topology edits).

Everything below follows from this: chaining parts into one solve is
**tag-namespacing + interface constraints**; the lifecycle verbs must
be **owned by a single driver**, never by a composable fragment.

### This is an evolution, not greenfield

The pieces already exist and compose:

* `g.compose()` ([ADR 0038](0038-compose-model-composition.md)) already
  merges modules into one broker — the model-definition half of
  monolithic coupling.
* The chain-phase router ([ADR 0041](0041-chain-phase-geometry-constraints.md))
  already defers `embedded` / `tied_contact` / `equalDOF` /
  `rigid_link` to a second phase, after both sides exist — that *is*
  "the interface edge emits after both nodes."
* The lineage chain `fem_hash → model_hash → results_hash`
  ([ADR 0021](0021-lineage-chain-replaces-snapshot-id.md)) is already a
  **DAG** — the dependency backbone a Workbench graph needs.
* `g.node_ndf` ([ADR 0032](0032-explicit-only-per-node-ndf.md) /
  [0033](0033-s2-emit-wiring-per-node-ndf.md)) already reconciles
  mixed ndf at a shell↔solid interface.
* The SSI staging line ([ADR 0028](0028-initial-stress-via-parameter-ramping.md)–
  [0034](0034-stage-bound-bcs-and-recorders.md)) already does
  upstream-state → downstream-input (initial stress, staged
  activation) — the mechanisms sequential coupling needs.

## Decision (proposed)

Three layered pieces. (4) is the concrete phase-1 deliverable; (1)–(3)
frame what it must not paint into a corner.

### 1. The phase-bucket model — deck order *is* the merge contract

Define the canonical emit phases as **named buckets**. Each bucket is
one dependency tier; the deck is any topological linearization of the
tiers (OpenSees accepts many — there is no single magic total order,
only "definition before reference"):

```
model -ndm -ndf
─ definitions   materials · sections · geomTransf · beamIntegration · timeSeries
─ nodes         (+ mass — a node property, NOT a load; lives here)
─ elements
─ boundary      fix  (homogeneous SP constraints)
─ interface     equalDOF · rigidLink · rigidDiaphragm · ASDEmbeddedNode · tied
─ patterns      pattern { load · sp · eleLoad }
─ recorders     ← MUST precede analyze, or they capture nothing
─ analysis      constraints/numberer/system/test/algorithm/integrator/analysis
─ analyze       ← the only line whose position is rigid; dead last
```

The payoff: **the same bucket list is both the deck structure and the
`g.compose` merge order.** Composing parts = bucket-wise concatenation
of `definitions … mass + intra-part fix` across parts, then the
cross-part `interface` block, then the global `patterns / recorders /
analysis / analyze`. The "global model" is then defined exactly as:

> `global model = ⋃(per-part phase blocks) + interface edges + global analysis`

### 2. The connectivity graph — parts = nodes, interfaces = edges

Introduce a first-class **connectivity graph**: nodes are Parts /
composed modules; an edge carries `{part_a, part_b, port_a node-set,
port_b node-set, coupling}` where `coupling ∈ {shared, equalDOF,
rigidLink, rigidDiaphragm, embedded, tied_contact, data-handoff}`. The
**`interface` phase is the serialization of the graph's edges**.

This graph is the object a future Workbench UI edits. Critically, the
emitter is only *one* consumer of it — `g.compose` and the viewer
(`ColorMode.MODULE` already tracks module provenance) are the others.
The lineage DAG already gives us the node-identity + dependency
backbone for free.

### 3. Two coupling modes (they want different file shapes)

**A — Monolithic coupling.** All parts in ONE Domain, joined by
interface constraints, solved together. File shape = driver + per-part
sourced fragments + interface block (below). Node-connection = shared
nodes or `equalDOF` / `rigidLink` / `ASDEmbeddedNode` / `tied_contact`
(all already routed by [ADR 0041](0041-chain-phase-geometry-constraints.md)).
ndf mismatch resolved by `g.node_ndf`. **One analysis chain marches
the whole Domain.** This is the near-term extension of `g.compose` +
chain-phase routing.

**B — Sequential / Workbench data-flow.** Upstream is solved; its
*results* become downstream inputs. Separate decks / separate Domains
with a **data artifact on the edge**. OpenSees has first-class
mechanisms per flavor: submodeling (record interface-DOF history →
replay as `sp` + `timeSeries Path` boundary motion), DRM
(`H5DRMLoadPattern` — free-field → effective boundary forces; the HDF5
hand-off is the `model.h5` lineage pattern), initial-stress / staging
([ADR 0028](0028-initial-stress-via-parameter-ramping.md) /
[0030](0030-stage-bound-topology-activation.md)), and `database` /
`restore` checkpointing. This is the larger Workbench leap (scheduler
+ typed result ports + hand-off format), and is **explicitly out of
phase 1**.

### 4. Split emit — the concrete phase-1 deliverable

Add a `split=` / `parts=` emit mode to `apeSees.tcl` / `.py`. **The
single-file deck stays the default** (don't break the reproducible
artifact).

* **Per-part fragment** emits only buckets *definitions → mass + its
  own intra-part `fix`*. **No `model`, no `wipe`, no interface, no
  `analyze`.** (The same discipline that made the `_LiveRecorderSink`
  safe — a fragment that calls `wipe` can't be composed.)
* **Driver** (`deck.tcl`) emits `model`, `source`s each part fragment
  (Tcl `source` shares global scope — zero runtime cost), then the
  `interface` block (from the graph edges), then loads, recorders,
  analysis chain, `analyze`.
* **Tag namespacing per part** via reserved base offsets is the
  load-bearing contract — without it `source partB.tcl` silently
  clobbers `partA`'s tags. (See Open Question 1.)

## Proposed API surface

Everything anchors on **one object — the assembly graph**; both
coupling modes are operations on it. Ports are **label / PG names**,
never tags (the surface `oriented_elements` / `recorders` already
take). This is a sketch to argue against, not a locked signature.

### The graph + two edge kinds

```python
from apeGmsh import Assembly          # naming open: Assembly | Workbench | Project

asm = Assembly("cerro_lindo")

# nodes — each is a saved model (model.h5) or a live session
pier = asm.add("pier", "pier.h5")
soil = asm.add("soil", "soil.h5")
```

**A-edge — `couple`** (spatial: parts → one Domain). Resolves to
`g.compose(...)` + the chain-phase-routed constraint primitives
([ADR 0041](0041-chain-phase-geometry-constraints.md)); materializing
the `couple` edges yields one composed `model.h5`:

```python
asm.couple(soil, pier, kind="tied_contact", ports=("soil.top", "pier.base"))
asm.couple(deck, pier, kind="equalDOF",     ports=("deck.support", "pier.top"), dofs=(1,2,3))
```

**B-edge — `chain`** (temporal: solved state → next setup). The
restart edge:

```python
gravity = asm.add("gravity", "mine.h5", analysis=build_gravity)
seismic = asm.add("seismic", "mine.h5", analysis=build_seismic)

asm.chain(gravity, seismic, transfer="committed_state")        # flavor 3: same model, exact resume
asm.chain(global_, submodel, transfer="displacement",          # flavor 2: submodeling
          ports=("global.cut", "submodel.boundary"))
```

Key reuse: **`transfer=` draws on the recorder canonical vocabulary**
(`"displacement"`, `"stress"`, …) — because a B-edge *is* an upstream
recorder output re-applied downstream as an initial condition. Upstream
`recorders` write the field → `results.h5` → downstream consumes via
`initial_stress` / prescribed `sp`+`Path`. `transfer="committed_state"`
is the total-state special case (flavor 3, FE_Datastore).

### Materializing

```python
asm.emit("deck.tcl", split="parts")   # driver + per-part fragments (mode A)
asm.run()                             # scheduler: topo-run nodes, move artifacts along B-edges
asm.graph()                           # inspect — reuses compose_tree / the lineage DAG
```

### Restart surfaces

Flavor 3 (exact resume), paired with `model.h5` so rebuild is
automatic (`OpenSeesModel.build` replays stored tags deterministically,
[ADR 0019](0019-opensees-model-read-side-broker.md) INV-5 — the
rebuild-topology half OpenSees `restore` requires but does not
provide):

```python
res  = sees.run(...)                       # long run
res.checkpoint("step_5000")                # FE_Datastore save + model.h5 pair
sees = apeSees.from_checkpoint("step_5000")  # rebuild topology + ops.restore
sees.run(...)                              # resume exactly
```

Flavor 2 (branch from state) is the `chain` B-edge above — lossy by
design (you pick the transfer field), version-robust, inspectable.

### What exists vs. what's new

| Surface | Status |
|---|---|
| `couple` edge → composed model | ◑ `g.compose` + chain-phase constraints exist; needs the edge object + split emit |
| `asm.graph()` inspect | ◑ `compose_tree` + lineage DAG exist |
| split emit (`split="parts"`) | ✗ new — phase-1 deliverable |
| `chain` edge + scheduler | ✗ new — the Workbench leap |
| `transfer=` via recorder vocab → `initial_stress`/`sp` | ◑ both ends exist; needs typed transfer-field + wiring |
| `checkpoint` / `from_checkpoint` (FE_Datastore) | ✗ new — thin `ops.database/save/restore` wrap, fenced best-effort |

Mode A is mostly **assembling primitives that already exist** behind
one graph object; mode B (especially flavor-3 restart) is the
genuinely **new subsystem** — a scheduler + a typed transfer-field.
This reinforces Open Question 2's recommendation: A first.

## Alternatives considered

1. **Keep monolithic-only.** Rejected as the *ceiling* (blocks
   chaining, hurts at scale) but **retained as the default** — the
   self-contained single deck is the right artifact for the common
   case.
2. **Node/element bulk in a data file** (script reads CSV / numpy).
   The bigger lever for pure file size, but Tcl has no clean tabular
   reader and it adds real machinery. Deferred — orthogonal to
   chaining.
3. **Per-rank partition files.** Already handled inside the single
   deck via `partition_open` `if {[getPID]==K}` guards
   ([ADR 0027](0027-cross-partition-mp-constraints.md)); orthogonal to
   the part-split seam.
4. **Hand-stacked tags with required non-overlap** (no offset
   allocator). Cheaper but pushes a global-uniqueness obligation onto
   the user / every part author. Captured as Open Question 1, not
   pre-decided.

## Consequences

**Positive.**

- The phase-bucket list unifies three things that are today implicit:
  deck order, the `g.compose` merge order, and the part-fragment
  boundary. One concept, three consumers.
- The connectivity graph reuses the lineage DAG and the chain-phase
  router — the Workbench feature lands as a graph/scheduler layer over
  existing primitives, not a rewrite of emit.
- Split emit is a pure restructuring behind a flag; the default
  single-file path (and its byte-output) is untouched.

**Negative / risks.**

- **Tag namespacing becomes a hard contract** (Open Question 1) — the
  single highest-risk decision; a wrong call silently clobbers tags
  across parts.
- Two emit shapes (single vs split) to test and keep consistent.
- Mode B (sequential) introduces a scheduler + a hand-off artifact
  format — a genuinely new subsystem, deliberately fenced out of phase
  1 to keep the blast radius bounded.

## Open questions (the discussion agenda)

1. **Tag allocation across parts** — global base offsets assigned at
   compose time, or keep apeGmsh's `fem_eid`-stable tags and *require*
   non-overlap by construction (fail loud on collision)? Highest-risk
   contract; everything else depends on it.
2. **A or B first?** Monolithic (A) is a near-term extension of
   `g.compose` + chain-phase routing. Sequential (B) is the bigger
   Workbench leap. Recommendation: A first (it forces tag-namespacing,
   which B also needs), B as a separate track once the graph/port
   model exists.
3. **Is the connectivity graph persisted** (a zone in `model.h5`? a
   sidecar?) or runtime-only, reconstructed from the lineage DAG +
   declared interfaces?
4. **Fragment file layout** — `parts/<name>.tcl` + a `deck.tcl` driver;
   one directory per model? Mirror exactly for the `.py` target?
5. **Edge ownership** — does an edge carry the coupling *type +
   params*, or just the two port node-sets, with the coupling resolved
   at emit by the existing constraint resolvers ([ADR 0041](0041-chain-phase-geometry-constraints.md))?
6. **Interface ndf** — auto-reconcile via `g.node_ndf`, or require the
   user to declare it on the edge?
7. **Mode B hand-off format** — does the edge artifact reuse
   `model.h5` / `results.h5` + lineage, or need a dedicated
   transfer-field schema (interface DOF history, effective forces)?
8. **Round-trip** — must a split deck read back (rehydrate from
   fragments), or is split a write-only export (like `.tcl` today)?
   Recommendation: write-only; `model.h5` remains the canonical
   round-trip surface.
9. **One graph object or two?** A single `Assembly` carrying both
   `couple` (A) and `chain` (B) edges (recommended — it *is* the
   Workbench mental model), or a spatial `Assembly` (A) separate from a
   temporal `Pipeline` (B)?
10. **What is a B-edge node** — a `model.h5` file (scheduler runs
    subprocesses, `transfer` is a file artifact) or a live
    `apeSees` / `Results` (scheduler chains in-process, `transfer` is
    an in-memory hand-off)? Decides the scheduler's execution model and
    the transfer-field's serialization.

## Decisions (resolved 2026-05-28)

The four architecture-binding questions are locked; the rest stay open.

- **Q2 — sequencing → Mode A first.** Build the monolithic path (split
  emit + `couple`) before the sequential/restart subsystem. A is
  mostly assembling primitives that already exist (`g.compose` +
  chain-phase routing) and it forces the tag-namespacing contract that
  B also needs. B (scheduler + typed transfer-field) is a separate
  later track.
- **Q1 — tag allocation → base offsets at compose.** An allocator
  assigns each part a reserved tag range at compose time, consistent
  with [ADR 0038](0038-compose-model-composition.md)'s existing
  "namespaced tag-offset import". Robust by construction; the user
  never reasons about cross-part collisions. (Note the interaction
  with the `tag == fem_eid` convention recorders / `ModelData` rely on:
  offsets shift tags away from raw `fem_eid`, so any recorder /
  orientation join over a *composed* model must resolve through the
  per-part offset map, not assume identity. Tracked for phase-1.)
- **Q9 — graph object → a single `Assembly`.** One object carries both
  `couple` (A) and `chain` (B) edges — the literal Workbench mental
  model. No separate `Pipeline` type.
- **Q8 — round-trip → write-only.** Split decks are emit-only, like
  `.tcl` / `.py` today; `model.h5` remains the canonical round-trip
  surface. No fragment parser.

**Still open:** Q3 (graph persistence), Q4 (fragment file layout),
Q5 (edge ownership / coupling params), Q6 (interface ndf),
Q7 (mode-B transfer-field format), Q10 (B-edge node = file vs live
session) — all mode-B or fragment-detail, to be settled as phase-1
lands.

### Phase 1 (mode A) scope

Per Q1/Q2/Q9, the first slice keys the split off the **existing
`g.compose` module labels** — fragments are one file per composed
module plus a driver, with base-offset tags already produced by the
ADR 0038 import. No new `Assembly` object is required for the initial
split-emit slice; the explicit `Assembly` graph + `couple`-edge
declaration layers on top once split emit is proven.

## References

- [decisions/0038-compose-model-composition.md](0038-compose-model-composition.md)
  — `g.compose`; the merge engine this ADR's phase-buckets formalize.
- [decisions/0041-chain-phase-geometry-constraints.md](0041-chain-phase-geometry-constraints.md)
  — chain-phase routing; the interface edges serialize through it.
- [decisions/0021-lineage-chain-replaces-snapshot-id.md](0021-lineage-chain-replaces-snapshot-id.md)
  — the lineage DAG reused as the connectivity-graph backbone.
- [decisions/0027-cross-partition-mp-constraints.md](0027-cross-partition-mp-constraints.md)
  — partition emit (orthogonal split axis).
- [decisions/0032-explicit-only-per-node-ndf.md](0032-explicit-only-per-node-ndf.md)
  — `g.node_ndf`, interface ndf reconciliation.
- [decisions/0028-initial-stress-via-parameter-ramping.md](0028-initial-stress-via-parameter-ramping.md)
  / [0030](0030-stage-bound-topology-activation.md) — sequential
  data-flow (mode B) mechanisms.
- [../emitter.md](../emitter.md) — the Emitter Protocol the split mode
  drives.
