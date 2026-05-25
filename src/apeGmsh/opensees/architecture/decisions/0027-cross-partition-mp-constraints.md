# ADR 0027 — Cross-partition MP-constraint emission policy

**Status:** Accepted (2026-05-22, P3 of the partition-emission work
stream). Extends [ADR 0022](0022-mp-constraint-emission-fanout.md) to
the partitioned-emit case and locks the policy P4 will implement.

## Context

ADR 0022 closed the §3.3 deferral by widening the `Emitter` Protocol
with five MP-constraint methods (`equalDOF`, `rigidLink`,
`rigidDiaphragm`, `embeddedNode`, `mp_constraint_comment`) and
introducing the build-time fan-out
`emit_mp_constraints(emitter, fem)` at
`opensees/_internal/build.py:1284`. That helper walks the broker's
constraint collections **unconditionally** — every MP constraint is
emitted exactly once into a single, undivided deck.

The current work stream introduces partitioned models. P1 (weighted
partitioning on `g.mesh.partitioning`) and P2 (`PartitionRecord` +
`PartitionSet` composite on `FEMData`) together produce
`fem.partitions: tuple[PartitionRecord, ...]` describing the rank
each node and element belongs to. When `len(fem.partitions) > 1` the
emitter must emit Tcl/Py blocks scoped per rank (the
`if {[getPID] == K}` blocks consumed by OpenSeesMP), and the MP-fan-out
pass from ADR 0022 needs a policy for the case where a constraint
straddles ranks — its master and slave (or master and any one of its
slaves) live on different partitions.

Three known straddle scenarios drive the policy:

1. **rigidLink across columns on different ranks.** A column-stripe
   partition assigns adjacent column lines to different ranks; a
   rigid link between two column tops crosses the boundary.
2. **rigidDiaphragm whose floor master is on rank K but the slaved
   column tops are scattered.** The most common case under any
   column-distributed partition: one master node, slaves spread across
   every rank that owns at least one column in that floor.
3. **Embedded / tied surface coupling whose host element and embedded
   node are on different ranks.** ASDEmbeddedNodeElement carries an
   element tag (the host) and a cnode (the embedded node); the
   partitioner can place these on different ranks even for the same
   physical contact pair.

The MP-fan-out from ADR 0022, the `region()` widening from [ADR
0024](0024-emitter-protocol-widen-region.md), the per-zone schema
versioning from [ADR 0023](0023-per-zone-schema-versioning.md), and
the `eigen()` widening from
[ADR 0025](0025-emitter-protocol-widen-eigen.md) all assume an
unpartitioned emit pass today. None of them carry a rank-aware
emission contract. This ADR is the architectural decision that
defines that contract; P4 is the implementation.

The Protocol header in `opensees/emitter/base.py` continues to call
widenings *"an architecture event"*; this ADR records the event for
the two new emission scoping methods P4 will add (`partition_open`,
`partition_close`) and pins the per-emitter MP-constraint replication
rule.

## Decision

**Replicate cross-partition MP constraints on every owning rank.**
Foreign node tags referenced by a replicated constraint are declared
on each rank's emit block (`node(tag, *xyz, ndf=...)`) before the
constraint line. Tag and coordinate values are broker-derived and
identical across ranks. The rule applies per ADR 0022's five
MP-constraint Protocol methods as follows:

- **`equalDOF(master, slave, *dofs)`** — emit on **both** owning
  ranks. Both ranks declare the foreign node via
  `node(tag, *xyz, ndf=...)` before the `equalDOF` line. The rank
  that natively owns the node skips the redeclaration; the foreign-
  side rank emits it.

- **`rigidLink(kind, master, slave)`** — same rule as `equalDOF`.
  Emit on both owning ranks; foreign node declared on the
  non-owning side.

- **`rigidDiaphragm(perp_dir, master, *slaves)`** — emit on **every**
  rank that owns **any** slave node. The master node is foreign on
  all-but-one rank and is declared via `node(master, *xyz, ndf=...)`
  on each foreign-side rank. Slaves are partitioned by ownership: each
  rank declares any foreign slaves it does not natively own before the
  `rigidDiaphragm` line. The full `rigidDiaphragm` command line is
  emitted verbatim and identically on every owning rank — slaves
  are not sharded.

- **`embeddedNode(ele_tag, cnode, *rnodes)`** — bound to the rank
  that owns `ele_tag` (the host element). The constraint is emitted
  **only** on that rank. Foreign `rnodes` and a foreign `cnode` are
  declared on the host-element rank via `node(tag, *xyz, ndf=...)`
  before the `element ASDEmbeddedNodeElement ...` line. Slave-side
  ranks **do not** emit this constraint — element ownership is the
  single source of truth (see §"ASDEmbeddedNodeElement ownership"
  below).

- **`mp_constraint_comment(name)`** — emit on every rank where the
  associated constraint is emitted. The comment precedes its
  constraint line on each rank, so cross-rank log diffs remain
  symmetric and grep-friendly. Comment text is identical across
  ranks.

## Phantom-node policy

Per ADR 0022 INV-3, `NodeToSurfaceRecord.phantom_nodes` are emitted
via `emitter.node(phantom_id, *xyz)` **before** any constraint
references them. Phantom nodes carry `ndf=6` and broker-deterministic
coordinates.

Under partitioning the rule extends: a phantom node is emitted on
**every** rank that hosts any constraint referencing it. The phantom
tag and its coordinates are produced **once** at build time by the
broker (the `NodeToSurfaceRecord` snapshot already carries
`phantom_nodes` and `phantom_coords`) — they are **not** allocated
per-rank. This means the same phantom tag and the same `*xyz` tuple
appear in the `node(...)` line on every owning rank's block. Phantom
emission still precedes the constraint emission within each rank's
block (the within-rank ordering invariant from ADR 0022 INV-3 is
preserved one rank at a time).

## ASDEmbeddedNodeElement ownership

ASDEmbeddedNodeElement is semantically a constraint (per ADR 0022's
fifth Protocol method, `embeddedNode`) but technically an element —
it carries an element tag and lives in OpenSees's element domain. For
partitioning purposes **element ownership wins**: the rank that owns
`ele_tag` is the unique emitter for this constraint. This matches
the general element-fan-out partition guard P4 will apply to the
existing `emitter.element(...)` path; embeddedNode is the only
MP-constraint method that participates in element-side ownership
rather than in the replicate-on-both rule.

Practical consequence: a tied-contact pair whose host element and
embedded node sit on different ranks emits **once**, on the
host-element rank, with the embedded node declared as foreign there.
The other rank's block is silent on this constraint — there is no
mirror emission. OpenSeesMP's element-side handler resolves the
embedded node tag against the global node table at solve time.

**Intersection host-rank rule (PR #329 hardening).** A subtle bug
in the original `_plan_rank_constraints` used `if partition_rank in
node_owners[masters[0]]` to decide which rank owned the embedded
record. When `masters[0]` was a partition-boundary node owned by
multiple ranks, every owning rank's plan included the record and
the `embeddedNode` line emitted on each — duplicate stiffness
contribution at solve time. The fix: the unique owner is the HOST
ELEMENT's owning rank, NOT every rank in `node_owners[masters[0]]`.
A boundary-shared master appearing in multiple ranks' ownership sets
is correct (the node IS shared), but the embedded-record itself
binds to one host-rank only. Locked end-to-end by
[`test_emit_partitioned_embedded.py`](../../../../tests/opensees/integration/test_emit_partitioned_embedded.py).

**Off-host / mixed-host / bad-Rnode-count guards (PR #331).** Three
fail-loud build-time guards on the resolver / emit path catch
silent-wrong configurations the original pipeline would emit:

1. **Off-host extrapolation** — `EmbeddedDef.tolerance` is now
   enforced. The resolver computes barycentric coordinates and
   populates an `excess: float | None` field on
   `InterpolationRecord`; the emit-time guard raises `BridgeError`
   naming the offending slave node + its excess when
   `excess > tolerance`.
2. **Mixed-host silent drop** — `_collect_host_elems` previously
   kept only Gmsh element types 2 (tri3) and 4 (tet4); quad / hex /
   prism / pyramid / tri6 / tet10 were silently dropped. A
   transition-region mesh (hex + a few tets) would project embedded
   nodes in the hex region onto distant tets (extrapolation again).
   The collector now retains all supported host types; mixed-host
   configurations the C++ `ASDEmbeddedNodeElement` parser cannot
   consume raise at build time, not at OpenSees runtime.
3. **Bad Rnode count** — the C++ parser only accepts 3 (tri host)
   or 4 (tet host) Rnodes; 5+ are misread as flag positions, 2
   aborts in `setDomain`. A build-time guard raises `BridgeError`
   naming the offending record's Rnode count.

## Regions interaction (ADR 0024)

Per ADR 0024 `emitter.region(tag, -node n1 n2 ..., -ele e1 e2 ...)`
groups elements and nodes for MPCO output filtering. Under
partitioning:

- A region's `node_ids` and `element_ids` are intersected with the
  per-rank ownership tables before emission on each rank.
- A rank with an **empty intersection** does **not** emit the region
  on that rank — there is no bare `region $tag` line and no
  empty-payload guard trip (the empty-PG guard from ADR 0024 INV-6
  remains a build-time check against the full FEM, not a per-rank
  check).
- The region `tag` is the **same** across all ranks that emit the
  region. MPCO post-processing stitches per-rank `.mpco` files into a
  single logical region by tag identity. A region that fans into
  three ranks produces three per-rank `.mpco` files all carrying the
  same `-R $tag` filter.

The MPCO recorder's `_region_tag` field (ADR 0024 §"Extend the MPCO
recorder primitive") is set once by the build pipeline before the
per-rank fan-out begins; it is the same scalar on every rank.

## Tag determinism

All rank-emitted blocks must agree on every numeric tag:

- node tags (including foreign-declared nodes via `node(...)`),
- element tags (**including ASDEmbeddedNodeElement tags** — see
  note below),
- phantom-node tags (per §"Phantom-node policy" above),
- region tags (per §"Regions interaction" above),
- integration tags,
- geomTransf tags,
- material / section tags,
- pattern / time-series / recorder tags.

**Embedded-element tag namespace (PR #329).** Embedded-element
tags share the canonical `TagAllocator`'s `"element"` counter — the
same per-kind counter that structural elements use. They are NOT
allocated from a separate reserved range. The retired
`_allocate_embedded_tag_base` helper returned a static `1_000_000`
base that `_emit_surface_couplings_for_rank` restarted on every
rank, so two distinct embedded records on different ranks both
received tag `1_000_000` — a determinism violation that produced
silent solver collisions. The fix threads the bridge's canonical
`TagAllocator` through `emit_mp_constraints` /
`emit_mp_constraints_partitioned` / `_emit_surface_couplings` /
`_emit_surface_couplings_for_rank` so every embedded element tag
comes from `tags.allocate("element")`. Locked by
[`test_emit_partitioned_embedded.py`](../../../../tests/opensees/integration/test_emit_partitioned_embedded.py).

`TagAllocator` already produces a single canonical numbering at build
time, not rank-local — there is exactly one producer per
`apeSees(fem).h5/.tcl/.py` call. **This remains true under partition
emission.** Partition is a labeling pass over an already-numbered
model, not a renumbering. P4 must not introduce any rank-local tag
counter; if a foreign node tag appears on rank K's block, that exact
integer was already allocated by the single canonical allocator and
is the same integer that appears on the natively-owning rank's block.

## Constraint handler interaction

Per ADR 0022 Phase 8, `Plain` auto-upgrades to `Transformation` when
MP constraints are present and the user has not explicitly picked a
constraint handler.

Under partitioning the auto-upgrade rule extends to the numberer and
the system of equations:

- `Transformation` remains the auto-default for the constraint
  handler. OpenSeesMP supports it; no change to the ADR 0022 rule.
- When `len(fem.partitions) > 1` and the user has **not** explicitly
  set a numberer, auto-emit `numberer ParallelPlain`. This replaces
  the serial `Plain` / `RCM` defaults; serial numberers are not
  valid under OpenSeesMP.
- When `len(fem.partitions) > 1` and the user has **not** explicitly
  set a system of equations, auto-emit `system Mumps`. Mumps is the
  expected parallel sparse-direct solver for OpenSeesMP.
- When the user **has** explicitly set a numberer or system that is
  detectably MP-incompatible (e.g. user-picked `system BandSPD` with
  `len(fem.partitions) > 1`), emit the user's choice verbatim but
  also emit a `UserWarning` at build time describing the
  incompatibility. The bridge does not silently override the user's
  choice — fail-loud at the warning layer; the deck still emits.

The auto-upgrade is gated on `len(fem.partitions) > 1`. Single-
partition models (no partitioning declared, or `g.mesh.partitioning`
producing a single trivial partition) follow the ADR 0022 rule
unchanged.

## Invariants

- **INV-1.** A cross-partition MP constraint emits **byte-identical**
  text inside every owning rank's `partition_open(rank)` /
  `partition_close()` block. The argument order, the trailing
  whitespace, the line terminator, and any preceding
  `mp_constraint_comment` line all match across ranks.

- **INV-2.** A foreign node tag referenced by a replicated constraint
  is declared (via `node(tag, *xyz, ndf=...)`) on the foreign-side
  rank's emit block, **before** the constraint line. The `ndf` value
  comes from the broker's `NodeRecord` for that tag and is identical
  to the value used on the natively-owning rank.

- **INV-3.** Phantom-node tags are broker-derived and identical
  across ranks; phantom-node coordinates are identical across ranks
  to within float-bit precision (no per-rank rounding; the broker's
  `phantom_coords` array is the single source).

- **INV-4.** A region's `element_ids` and `node_ids` are intersected
  with per-rank ownership before emission; empty intersection ⇒ no
  `region` line emitted on that rank; the region `tag` is the **same**
  scalar across every rank that does emit, so MPCO post-processing
  can stitch per-rank `.mpco` files by tag identity.

- **INV-5.** When `len(fem.partitions) > 1` and the user has not
  explicitly set a numberer / system, the deck emits a
  **runtime-conditional fallback** — primary choice
  (`ParallelPlain` / `Mumps`) under OpenSeesMP, automatic fallback
  (`RCM` / `UmfPack`) under single-process OpenSees. The same deck
  runs in both runtimes. Existing user choices are preserved
  verbatim with `UserWarning` on detected incompatibility (see
  **Amendment 2026-05-23** below for the rationale and the exact
  Tcl / Py forms).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Hard-error at build time on any cross-partition MP constraint** | Blocks the common floor-diaphragm-across-ranks pattern (straddle scenario #2) that arises naturally from any column-distributed partition. A user who declared `g.constraints.rigid_diaphragm(name="floor_3", ...)` on a 4-rank partition would be forced to either redo the partition by hand or split the diaphragm into per-rank fragments — both undo the value of the declarative API. |
| **Warn-and-drop the straddling constraint** | Mechanics silently diverge from the single-process deck — a model that converges and matches an analytical result on a single rank produces different (wrong) answers under partitioning, with only a warning. Foot-gun; rejected on the same fail-loud principle ADR 0025 invoked against silent capability fallbacks. |
| **Master-owner-only emission** (emit the constraint only on the rank that owns the master node) | Works for `equalDOF` and `rigidLink` (two-node constraints) but fails for `rigidDiaphragm` — OpenSeesMP's `Transformation` constraint handler needs the constraint declared on every rank that owns any slave so the local handler can resolve the slave-side DOF condensation. Slave-side ranks would see the slave node free, the master node absent, and produce a wrong stiffness contribution. Rejected for the grouped-constraint case; partial adoption (master-owner-only for two-node, replicate-on-both for grouped) was rejected for adding a second policy where one suffices. |
| **Reroute every cross-partition MP constraint through `embeddedNode`** | ASDEmbeddedNodeElement is mechanically different from `equalDOF` / `rigidLink` / `rigidDiaphragm` — it introduces a stiffness coupling, not a kinematic condensation. Substituting one for the other would change the answer. Rejected as a mechanics change, not a transport change. |

## Consequences

- **Test surface (P4 + P5).** P4 needs per-emitter unit tests for the
  cross-partition cases of all five MP-constraint Protocol methods
  (`equalDOF`, `rigidLink`, `rigidDiaphragm`, `embeddedNode`,
  `mp_constraint_comment`) plus phantom-node replication. P5 needs
  an integration test for a frame with a floor diaphragm
  partitioned across 4 ranks, asserting the emitted Tcl/Py decks
  run under OpenSeesMP and the modal periods match a single-rank
  baseline.

- **Runtime cost.** Constraint redundancy across ranks is a known
  property of OpenSeesMP — the constraint handler dedupes equivalent
  declarations at solve time, so emitting the same `equalDOF` on
  both owning ranks is not a correctness or performance problem. The
  cost is in the emitted deck size (~O(cross-partition-constraint-
  count) extra lines per rank) and is bounded by the partition
  topology.

- **Debugging.** Per-rank emission can be hard to reason about when
  the deck is large. P4 will expose
  `apeSees(fem, debug_partitions=True)` as an opt-in that logs
  per-rank emission counts (node-decl count, element count, MP-
  constraint count, region count) at build time. Off by default.

- **Schema.** Per ADR 0023, the partition emission feature is an
  **additive** change to the OPENSEES zone (new groups under
  `/opensees/partitions/`; existing groups unchanged). Minor bump:
  opensees `2.9.0 → 2.10.0`. The two-version reader window from
  ADR 0023 means 2.9.x readers still open 2.10.x files (they ignore
  the new partition groups; the resulting `OpenSeesModel.from_h5(...)`
  flow continues to produce an unpartitioned deck on re-emit).

- **No new Protocol methods this ADR.** The two emission scoping
  methods (`partition_open(rank)`, `partition_close()`) are added by
  P4 and will be documented in the P4 implementation notes; they are
  scoped under the same "architecture event" ceremony as the ADR
  0022 / 0024 / 0025 widenings but are intentionally narrow (rank
  brackets, not new typed primitives) and do not warrant a separate
  ADR. The constraint-emission policy itself — the load-bearing
  decision — is captured here.

## Implementation pointer

P4 (task #4 in the current work stream) implements this ADR:

- New emitter Protocol methods `partition_open(rank: int)` and
  `partition_close()` on `opensees/emitter/base.py`. Implementations
  on `TclEmitter` / `PyEmitter` write the `if {[getPID] == K} { ... }`
  / closing brace pair on Tcl, the `if pid == K:` / dedent pair on Py.
  `LiveOpsEmitter` no-ops both (live cannot drive OpenSeesMP).
  `H5Emitter` writes a `PartitionScopeRecord` to a per-rank sub-group
  under `/opensees/partitions/`. `RecordingEmitter` records both
  method+args.
- `_internal/build.py::emit_mp_constraints` becomes partition-aware.
  When `len(fem.partitions) > 1` the helper switches from the
  unpartitioned single-pass walk to a per-rank fan-out that applies
  the replication rules defined in §"Decision" above.
- Phantom-node emission (ADR 0022 INV-3) extends to per-rank
  replication per §"Phantom-node policy" above. Phantom tags continue
  to come from the broker snapshot — no rank-local allocation.
- Region emission (ADR 0024) intersects with per-rank ownership per
  §"Regions interaction" above.
- Constraint-handler / numberer / system auto-emit extends per
  §"Constraint handler interaction" above.
- Schema bump opensees `2.9.0 → 2.10.0` per ADR 0023 (additive minor;
  two-version reader window).

P5 (task #5) ships the end-to-end test: mesh partition → FEMData →
OpenSeesMP-runnable deck → modal periods match single-rank baseline.

## Cross-references

- [ADR 0022](0022-mp-constraint-emission-fanout.md) — the MP-fan-out
  pattern this ADR extends; the five Protocol methods (`equalDOF`,
  `rigidLink`, `rigidDiaphragm`, `embeddedNode`,
  `mp_constraint_comment`); INV-3 phantom-node-first ordering.
- [ADR 0023](0023-per-zone-schema-versioning.md) — additive-minor
  rule and two-version reader window. The opensees `2.9.0 → 2.10.0`
  bump driven by P4 follows this policy.
- [ADR 0024](0024-emitter-protocol-widen-region.md) — `region()`
  Protocol method and the empty-PG guard; the per-rank intersection
  rule in §"Regions interaction" above sits on top of ADR 0024's
  filtering machinery.
- [ADR 0025](0025-emitter-protocol-widen-eigen.md) — most recent
  Protocol-widening precedent. The ceremony shape, fail-loud policy
  for capability mismatches, and the "do not pre-build deferred
  surfaces" stance are reaffirmed here.
- `opensees/_internal/build.py:1284` — `emit_mp_constraints` helper
  that P4 makes partition-aware.
- `opensees/emitter/base.py` — Protocol header documenting that
  widenings are architecture events.

## Amendment 2026-05-23 — runtime-conditional numberer / system

The original INV-5 wording mandated **unconditional** emission of
`numberer ParallelPlain` and `system Mumps` at the top level of every
partitioned deck. P4-B implemented that literally. Field testing
exposed a contradiction with P4-A's partition-open shim:

* P4-A's `partition_open(rank)` emits a single-process fallback shim
  — Tcl declares `proc getPID {} { return 0 }` if not already
  registered, and Py wraps `from openseespy.opensees import getPID`
  in a `try / except ImportError` with a `def getPID(): return 0`
  fallback. The shim lets `if {[getPID] == K}` blocks safely no-op
  (for K != 0) under single-process OpenSees so the per-rank emit
  blocks "skip themselves" instead of raising.
* P4-B's INV-5 auto-emit of `numberer ParallelPlain` / `system Mumps`
  is **outside** any per-rank block — it is global. Regular
  OpenSees has no `ParallelPlain` numberer and no `Mumps` system; it
  errors out before the deck even reaches the per-rank `if`-blocks:
  `WARNING No Numberer type exists (Plain, RCM only)`.

The shim was therefore illusory cover: the deck *looked* portable
but only ran under OpenSeesMP. Two single-process accommodations
contradicted each other.

The amendment replaces the unconditional emit with a
**runtime-conditional** fallback that mirrors the shape of the
`partition_open` shim. The deck still names `ParallelPlain` /
`Mumps` as the intended primary (so OpenSeesMP-aware readers see the
intent), but wraps the call in a Tcl `catch` / Python `try / except`
so a single-process OpenSees swallows the parse error and falls back
to a serial choice (`RCM` / `UmfPack`) that does exist in every
build.

**Emitted Tcl (when the user has not explicitly declared a
numberer / system):**

```tcl
# ParallelPlain only exists in OpenSeesMP; fall back to RCM under single-process OpenSees.
if {[catch {numberer ParallelPlain} _err]} { numberer RCM }
# Mumps requires OpenSeesMP + MPI; fall back to UmfPack under single-process OpenSees.
if {[catch {system Mumps} _err]} { system UmfPack }
```

**Emitted Python:**

```python
# ParallelPlain only exists in OpenSeesMP; fall back to RCM under single-process OpenSees.
try:
    ops.numberer('ParallelPlain')
except Exception:
    ops.numberer('RCM')
# Mumps requires OpenSeesMP + MPI; fall back to UmfPack under single-process OpenSees.
try:
    ops.system('Mumps')
except Exception:
    ops.system('UmfPack')
```

The amendment introduces two new Emitter Protocol methods —
`parallel_runtime_fallback_numberer(primary, fallback)` and
`parallel_runtime_fallback_system(primary, fallback)` — alongside
the existing `numberer` / `system` ones. The build pipeline's
`_maybe_auto_emit_parallel_numberer` /
`_maybe_auto_emit_parallel_system` paths now call the new
runtime-conditional methods instead of `emitter.numberer(...)` /
`emitter.system(...)`. Per-emitter behaviour:

| Emitter | Runtime conditional |
|---|---|
| Tcl | Emits the `catch` wrapper shown above. |
| Py | Emits the `try / except` block shown above. |
| LiveOps | Actually calls `self._ops.numberer(primary)` in-process; on `Exception` falls back to `self._ops.numberer(fallback)` and fires a one-shot `UserWarning`. Same for system. |
| H5 | Records the primary as the canonical `numberer` / `system` attribute under `/opensees/analysis/`, plus a new `numberer_runtime_fallback` / `system_runtime_fallback` attribute carrying the fallback string. Re-emit can reconstruct the conditional from the pair. |
| Recording | Captures `("parallel_runtime_fallback_numberer", (primary, fallback), {})` and the system equivalent. |

User-override paths are **unchanged**: a user who declares a
specific numberer or system gets that choice emitted verbatim
(plus the existing MP-incompatibility `UserWarning` when the choice
is detectably serial). The runtime conditional only fires on the
"no user declaration" branch.

Schema impact: zero. The H5 emitter adds two optional attributes
under `/opensees/analysis/`; the schema 2.10.0 envelope (introduced
by P4 for the new `/opensees/partitions/` group) already covers
attribute additions per ADR 0023's additive-minor rule. Readers
that ignore the new attributes degrade gracefully — they reconstruct
the canonical numberer / system without the runtime fallback wrapper
(matching pre-2.10.0 behaviour).
