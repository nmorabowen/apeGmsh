# ADR 0049 — Decoupled nodes (analysis auxiliary nodes)

**Status:** Proposed (2026-05-31, revised). Sibling to
[ADR 0048](0048-infer-per-node-ndf-from-elements.md) (per-node `ndf`
inference). The two together make **one** statement: *DOF semantics live on
the bridge; the session and the neutral broker are DOF-free.* 0048 covers the
`ndf` of nodes an element touches (inferred); this ADR adds **decoupled
nodes** — analysis nodes that are not gmsh vertices — and routes *their* `ndf`
to the bridge too. Builds on the broker-chain invariants of
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) /
[ADR 0019](0019-opensees-model-read-side-broker.md) /
[ADR 0020](0020-results-carries-opensees-model.md), generalizes the
phantom-node mechanism of [ADR 0022](0022-mp-constraint-emission-fanout.md),
and reuses the `∩ ndf_ok` gate of
[ADR 0046](0046-shell-on-solid-node-sharing-guard.md).

> **Revision note (2026-05-31).** The first draft of this ADR put `ndf` *on
> the creation call* (`g.nodes.add(x, y, z, ndf=3, ...)`) and added an
> optional `ndf` field to the neutral broker. That re-coupled the session to
> OpenSees DOF semantics — the exact coupling 0048 removed for mesh nodes —
> and left the broker carrying DOFs after 0048 worked to evict them. This
> revision **moves `ndf` out of node creation and onto the bridge**
> (`ops.ndf(...)`), so the session declares node *identity* and the bridge
> assigns node *DOF*. The composite is renamed `decouple_node` to name what it
> does (declare a node decoupled from the mesh/dedup pipeline), not what it is.

## Dependencies — sequenced after the 0048 clean break

This ADR assumes the world *after*
[ADR 0048](0048-infer-per-node-ndf-from-elements.md)'s breaking change has
landed, and **must ship after it**. As of this writing that clean break is
**not yet implemented**: `g.node_ndf` / `NodeNDFComposite`, the broker `_ndf`
array (still folded into `fem_hash`), the `ndf=` kwarg on `ops.model`, and
`validate_envelope_covers_broker_ndf` all still exist in `src/`. This ADR's
premises — *broker is DOF-free*, *mesh-node `ndf` is inferred*, *`ops.model`
takes `ndm` only* — describe the proposed end-state, not current code.
Everything this ADR adds (`g.decouple_node`, the node `provenance` flag,
`ops.ndf`, the `/opensees/nodes_ndf` store, the G1–G3 gates) is **greenfield**.
Building `ops.ndf` *before* `ops.model(ndf=)` is removed would leave two
competing `ndf` envelopes; the sequencing is not optional.

## Context

apeGmsh is mesh-centric: every emitted node is a gmsh vertex carried on the
`FEMData` broker (`fem.nodes`). The only non-mesh nodes today are the internal
phantoms ([ADR 0022](0022-mp-constraint-emission-fanout.md)), synthesized for
`node_to_surface` constraints and never user-visible.

Real OpenSees models routinely need **auxiliary nodes that are not mesh
vertices**:

- **Spring / dashpot supports** — every `zeroLength`-to-ground needs a fixed
  partner node at (usually) the *same location* as a boundary mesh node. This
  is the spine of soil-structure interaction (Lysmer dashpots, soil springs,
  p-y / t-z / q-z BNWF). It is the dominant case for this project's staged-SSI
  work.
- **Constraint master / control nodes** — a `rigidDiaphragm` retained node at
  a floor's center of mass, a control node for a displacement-driven pattern,
  an `equalDOF` reference node. These are the legitimately *element-less*
  nodes [ADR 0048](0048-infer-per-node-ndf-from-elements.md) cannot infer.
- **Point-load / lumped-mass anchors** not at a mesh vertex.

These nodes raise **two separable questions**, and conflating them is what the
first draft got wrong:

1. **Where does the node come from?** It is not a gmsh vertex, so the
   mesher will not produce it, and a coincident one must survive node dedup.
   This is a *geometry / topology* question → the **session**.
2. **How many DOFs does it carry?** For a spring ground that equals the
   structural side; for a diaphragm master, whatever the constraint needs.
   This is an *analysis* question, the same kind 0048 already answered for
   mesh nodes → the **bridge**.

Keeping these separate is the whole design. The session answers (1) without
ever touching DOFs; the bridge answers (2) for *every* node — inferred for
mesh nodes (0048), stated for element-less decoupled nodes (here).

Two candidate homes for the node itself were considered and rejected:

1. **Bridge-only node origin** (rejected) — a node created *outside* the
   lineage-hashed broker. `fem_hash` (computed over the broker node set) would
   never see it, yet `model_hash = blake2b(fem_hash ‖ opensees-zone)` would
   fold in `/opensees/` elements referencing its tag → the chain no longer
   describes the model. The viewer (reads neutral `/model`) and
   `Results.model.fem` would have dangling node references. Breaks the
   broker-chain invariant that **the broker is the single source of truth for
   node existence**. (Note: this rejection is about node *existence*, not
   `ndf` — `ndf` legitimately lives on the bridge; see Decision.)
2. **Reuse a geometry point as the node** (rejected as the *spine*; kept as a
   *location* source) — a geometry point is a CAD entity subject to boolean
   ops, healing, and **node dedup**. `remove_duplicate_nodes` merges a
   point-node into a coincident mesh node — which destroys exactly the
   spring-support case (the ground node *must* be a second, distinct node at
   the boundary node's location). Geometry points also require an
   embed / dim-0-PG step to materialize a node at all. Fine as a *location*
   source for a free-standing node, fatal as the *identity* of a coincident
   one.

## Decision

**A decoupled node is a first-class broker node created session-side through a
`decouple_node` composite — carrying coordinates, a label, and a provenance
flag, but *no* DOF information. Its `ndf`, like every other node's, is
assigned on the bridge.**

### Session surface — identity only, DOF-free

```python
# Spine — a dedup-immune analysis node at explicit coords (the SSI case):
gnd  = g.decouple_node(coords=(x, y, z), label="pile_tip_ground")

# Convenience — location taken from an EXISTING geometry point
# (free-standing nodes: control / master / anchor):
ctrl = g.decouple_node(point="floor_cm", label="diaphragm_master")
```

`decouple_node` takes a **location** (explicit `coords=` or an existing
geometry `point=`) and a **label**. It does **not** take `ndf` or any DOF
argument. It returns a **handle** referenced everywhere downstream (no raw
tags, per the no-raw-tags rule) — by `g.constraints` / `g.loads` pre-mesh and
by the bridge (`ops.element.zeroLength(soil_node, gnd, ...)`,
`ops.fix(gnd, ...)`, `ops.ndf(gnd, ...)`). The handle resolves through the
same label tier as everything else.

The verb is `decouple_node` because the node is **decoupled from the gmsh
mesh/dedup pipeline** (it is not a vertex, and a coincident copy is not
merged) while staying **coupled to the broker** (it is a real node the
lineage chain, persistence, and viewer all see).

### Broker representation — identity + provenance, no `ndf`

`fem.nodes` gains **one** new per-node field: a **provenance** discriminator
(`mesh` | `decoupled`; phantoms remain broker-internal and are not exposed
here). It gains **no** `ndf` field — neither for mesh nodes (0048 evicted it)
nor for decoupled nodes (their `ndf` lives on the bridge, below). Decoupled
nodes fold into `fem_hash` and round-trip through the neutral `/model` zone
exactly like mesh nodes — so the viewer and `Results.model.fem` see them for
free. **That round-trip is the entire point of making them broker nodes.**

### `ndf` — bridge-side for every node, no exceptions

There is **one** per-node `ndf` map and it is **bridge-resident**, produced at
build time against the frozen `fem` snapshot:

- **Mesh node, or decoupled node an element attaches to** → `ndf` is
  **inferred** from the incident element classes
  ([ADR 0048](0048-infer-per-node-ndf-from-elements.md)). A decoupled node is
  not special to inference; if an element touches it, it is resolved like any
  node. (The `zeroLength` family is adaptive — floor 1, `ndf_ok = {1..6}` — so
  it never inflates the inferred value; the structural side of a spring
  supplies the real count.)
- **Element-less decoupled node** (the diaphragm master, the control node, and
  the ground end of an adaptive spring) → its `ndf` is **stated on the
  bridge** with a new directive:

  ```python
  ops.ndf(gnd, 3)                 # by handle
  ops.ndf(label="diaphragm_master", ndf=6)
  ops.ndf(pg="pile_grounds", ndf=3)   # a whole decoupled-node set at once
  ```

  `ops.ndf` resolves its target (handle / label / PG) to decoupled-node ids
  against the frozen snapshot and writes those ids into the bridge `ndf` map. PG
  resolution reuses the existing `expand_pg_to_nodes` fan-out that `ops.fix` /
  `ops.load` already use at emit; the **handle tier** (a `decouple_node` handle
  → its reserved-range node id) and label-to-decoupled resolution are
  **net-new** — the bridge directives take `pg` / `nodes` today, not a label
  tier, so this is added machinery, not an extension. `ops.ndf` is the **only**
  explicit `ndf`
  channel anywhere in apeGmsh, and it exists **only** for element-less
  decoupled nodes. It cannot target a mesh node (those are inferred; an attempt
  fails loud, preserving 0048's no-two-headed-model guarantee).

- **Element-less node with no `ops.ndf` statement** → **fail loud** (0048's
  existing backstop), with a message naming `ops.ndf(<handle>, ...)`. A node
  carrying DOFs no element stiffens and no directive sizes is a
  singular-matrix modeling error, never a silent default.
- **Validation is the `∩ ndf_ok` gate _plus_ three bridge consistency
  checks** — the `∩` gate alone is structurally blind to the dominant
  decoupled-node case. See the next subsection; this is the load-bearing
  correctness mechanism of this ADR.

### Validation — the `∩ ndf_ok` gate is necessary but **not** sufficient

The shell-on-solid `∩ ndf_ok` gate
([ADR 0046](0046-shell-on-solid-node-sharing-guard.md)) is reused unchanged
for the **element-attached** case: a node shared by two elements with disjoint
`ndf_ok` fails loud. But it **cannot** catch the failure mode decoupled nodes
introduce, for two compounding reasons:

1. **The adaptive 2-node-link family defeats the set-intersection.** A
   `zeroLength` / `zeroLengthSection` / `twoNodeLink` / `dashpot` carries
   `ndf_ok = {1..6}` (it tolerates any endpoint DOF count). It therefore
   intersects every other family and can **never** produce the empty `∩` the
   0046 gate fires on. The gate is a **no-op precisely on the spring grounds
   this ADR exists to serve**. The gate also keys on a *tolerance set*, while
   the real failure is an *equality of resolved values* between two specific
   nodes — a different predicate it does not model.
2. **OpenSees fails these cases _silently_, not loudly.** `ZeroLength::setDomain`
   (`ZeroLength.cpp:614-619`; same in `TwoNodeLink.cpp`, `ZeroLengthSection.cpp`,
   `dashpot`) prints a `warning` and **`return`s** when its two endpoints have
   differing `ndf` — no exception, no abort. The element contributes **zero
   stiffness**; a batch run swallows the stderr line. `RigidDiaphragm`
   (`RigidDiaphragm.cpp:95-100`) likewise warns-and-returns unless its retained
   node is **exactly** `ndf=6` (3D) / `3` (2D). So a wrong `ops.ndf` does not
   crash — it produces a silently half-built model. **This is the exact
   silent-equilibrium-violation class ADR 0046 was created to eliminate,
   re-entering through the decoupled-node door.**

The fix is **three bridge-side consistency checks**, each evaluated at build
time over the already-resolved per-node `ndf` map (`/opensees/nodes_ndf`) plus
the collected bridge directives. **None of them touches the session or broker —
they consume resolved DOF + bridge records only, so the DOF-free-session
principle holds:**

- **G1 — equal-endpoint gate (2-node-link family).** For every `zeroLength` /
  `zeroLengthSection` / `twoNodeLink` / `dashpot`, assert
  `ndf[end_i] == ndf[end_j]`, mirroring OpenSees' `dofNd1 == dofNd2`
  requirement. This is the load-bearing fix for the SSI spring ground: a
  `ops.ndf(gnd, 6)` against a `ndf=3` structural endpoint now fails loud at
  build instead of dying silently in `setDomain`. (Today only `zeroLength` /
  `zeroLengthSection` are bridge elements; `twoNodeLink` / `dashpot` are
  greenfield and, when added, must carry a registry entry or G1 cannot classify
  them — the same false-negative as `element_class_ndf_ok`'s unregistered-class
  `None` path. G1 keys on the OpenSees equal-endpoint *contract*, not on the
  current registry, so the rule is stated by family, not by today's coverage.)
- **G2 — constraint-endpoint required `ndf` (master *and* constrained DOFs).**
  `rigidDiaphragm` / `rigidLink` / `equalDOF` publish the `ndf` floor each
  endpoint they touch requires — the retained / master node (diaphragm: 6 in
  3D, 3 in 2D) *and* every DOF referenced on a constrained node — into the same
  check. A diaphragm master or a constrained decoupled node is element-less, so
  the `∩` gate never runs on it; without G2 a wrong `ops.ndf(master, 3)`
  silently kills the diaphragm (`RigidDiaphragm.cpp:95-100` warn-and-return).
- **G3 — node `ndf` consistent with every referenced record.** The resolved
  `ndf` of any node must be DOF-consistent with each `ops.fix` / `ops.load` /
  **`ops.mass`** / `sp` / pattern record on it: the node must carry the DOFs the
  record addresses (vector records — fix mask, nodal-load / mass vector — by
  their component count; `sp` by its 1-based DOF index ≤ `ndf`). OpenSees does
  **not** partially apply a mismatched record — `Node::addUnbalancedLoad` /
  `setMass` warn-and-return and drop the whole load / mass, so the deficiency is
  silent. Catches a control / **mass anchor** (a named decoupled-node use case)
  stated with too few DOFs for the load or mass it carries. (Whether the bound
  is `==` or `≥` depends on the emit-time vector padding the clean break
  inherits — apeGmsh pads a short vector up to the node's `ndf`, so the live
  failure is a record *longer* than `ndf`; pin this predicate against the final
  padding behavior when implementing G3.)

G1–G3 **generalize** the 0046 guard from "per-node `ndf_ok` set-intersection"
to "the full set of node-DOF contracts the bridge can see (element, link
endpoint, constraint master, referenced load DOF)." The `∩` gate remains the
element-sharing half; G1–G3 add the link / constraint / load halves that the
decoupled-node surface makes reachable.

### Persistence — one unified bridge `ndf` store (closes ADR 0048 OQ-1)

The resolved per-node `ndf` (inferred mesh values **and** stated decoupled
values) is written to a single bridge dataset `/opensees/nodes_ndf`. The
neutral `/model` node zone stays DOF-free; it carries coordinates,
connectivity, PGs, labels, and the new provenance flag. The read-side replay
and viewer source per-node `ndf` from `/opensees/nodes_ndf`, never from
`/model`.

This **resolves [ADR 0048](0048-infer-per-node-ndf-from-elements.md)'s Open
Question 1** (persist vs. re-derive) in favor of **persist**. 0048 could lean
either way because inferred mesh `ndf` is re-derivable from `elements + ndm`.
But a **stated** decoupled-node `ndf` is *not* re-derivable — an element-less
ground / master node has no element to re-infer from — so once this ADR lands,
the "re-derive at read" option is off the table for that half of the map. A
single persisted store is the only design that covers both halves with one read
path. (This dataset is greenfield; it ships with the 0048 clean break, on which
this ADR depends — see Dependencies.)

### Dedup immunity — by construction, not by gating a gmsh pass

Coincidence with a mesh node is the *intended* state for spring/dashpot
grounds, so a decoupled node must never be merged away. Crucially, this is
**automatic**: `g.mesh.editing.remove_duplicate_nodes` calls
`gmsh.model.mesh.removeDuplicateNodes()` on the **live gmsh session**, which
runs *upstream* of the `FEMData` broker — and a decoupled node is **never a
gmsh vertex** (it is created session-side into the broker, not meshed). It
therefore never reaches that merge at all; there is no broker-side provenance
flag to "gate," because the gmsh-level pass cannot see broker nodes.

The invariant to hold is narrower and forward-looking: **no future broker-side
merge / weld pass may collapse a `decoupled` node**, even one coincident with a
mesh node. The provenance flag exists for *that* guard (and for dedup, viewer,
and partitioning to recognize these nodes) — not to influence the existing
gmsh-level dedup, which is already blind to them.

### Tag allocation + broker injection (net-new machinery)

Decoupled-node tags occupy a reserved range disjoint from mesh tags and phantom
tags, allocated **deterministically and MP-safe** (per-rank consistent) so a
decoupled node shared across partitions gets the same tag on every rank. This
borrows only the `> max(broker_tag)` *discipline* from the phantom-tag
allocator — **not** its implementation: the phantom counter lives inside the
constraint resolver (the wrong layer for a session-authored node) and is
MP-divergent per the cross-partition-counter caveat, so the decoupled-node
allocator is net-new and its MP-determinism must be designed, not inherited.

Likewise, **injecting a non-gmsh node into the frozen post-mesh `fem.nodes`
snapshot** — so its `coords`, `ids`, the new `provenance` field, and the
`fem_hash` fold all agree — has no existing mechanism (today every broker node
originates from a gmsh vertex). This is the key net-new primitive the ADR needs
and the **first thing to design when implementing** (see Open Question 7).

## Rationale

**The session must not know about DOFs.** The neutral broker and the session
are solver-agnostic: they carry geometry, mesh topology, PG membership, and
labels — not the fact that PG "Frame" will be a `forceBeamColumn` carrying 6
DOFs. [ADR 0048](0048-infer-per-node-ndf-from-elements.md) made this explicit
for mesh nodes (inference runs on the bridge, after element declarations).
Putting `ndf` on `g.decouple_node(ndf=)` would punch a DOF concept straight
back through that boundary for one class of node — and would do it at the
*worst* time (pre-mesh session authoring), forcing the user to know a node's
DOF count before any element is declared. Routing it through `ops.ndf` on the
bridge keeps the boundary intact: **the session says where the node is and
what it is called; the bridge says how many DOFs it has.**

**Node existence is a broker fact; node DOF is a bridge fact.** The two
rejected "homes" in Context conflate these. Bridge-only node *existence* breaks
the lineage chain (the broker stops being the single source of truth for what
nodes exist). But bridge-side node *DOF* is correct and is exactly what 0048
established. So the decoupled node lives in the broker (existence) and its
`ndf` lives on the bridge (DOF). Each fact sits in the layer that owns it.

**Geometry points are CAD, not nodes.** They are the right *location* source
for a free node (hence `g.decouple_node(point=...)`), but cannot be the
analysis node itself: dedup would merge the coincident spring ground node, and
CAD ops (fragment, heal) have no business mutating an analysis node.
`decouple_node` reuses the point for location without dragging the analysis
node through the CAD/mesh pipeline.

**`ops.ndf` is not a relapse to explicit-`ndf`-everywhere.**
[ADR 0048](0048-infer-per-node-ndf-from-elements.md) deleted explicit `ndf`
for *mesh* nodes because it *competed* with element inference — two sources
for one fact, the two-headed model. `ops.ndf` competes with nothing: it
applies *only* to element-less decoupled nodes, where there is no inference to
contradict. The moment an element attaches, validation takes over — the
`∩ ndf_ok` gate for element-sharing, and G1–G3 (§Validation) for the link /
constraint / load contracts the `∩` gate is structurally blind to (notably the
adaptive `zeroLength` spring ground, whose `ndf_ok = {1..6}` makes the `∩` gate
a no-op). Explicit `ndf` lives exactly where DOFs are otherwise unknowable — an
element-less node the user deliberately created — and nowhere else.

**MP consistency by determinism.** Every OpenSeesMP rank runs the same
inference over the same broker + element declarations, and resolves the same
`ops.ndf` directives over the same decoupled-node set, so shared boundary
nodes and shared decoupled nodes resolve identically without cross-rank
communication — the property [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md)
previously got from folding `_ndf` into `fem_hash`.

## Consequences

- **Closes the element-less-node gap** left open by
  [ADR 0048](0048-infer-per-node-ndf-from-elements.md): a `rigidDiaphragm`
  master / control node is now a first-class decoupled node whose `ndf` is a
  one-line bridge statement, not a synthesized or unconditionally-fail-loud
  case.
- **Makes 0048's "`ndf` leaves the broker entirely" hold unconditionally.**
  The first draft of this ADR re-added a broker `ndf` field and forced 0048 to
  carry a caveat; this revision removes that. The neutral broker carries
  **zero** DOF data. The only new broker field is the provenance flag (pure
  topology). 0048's Consequences §"Mesh-node `ndf` is derived, not stored" is
  amended accordingly (its 0049 caveat is withdrawn).
- **One `ndf` store, one read path.** Inferred (mesh) and stated (decoupled)
  per-node `ndf` share `/opensees/nodes_ndf`; the reader/viewer/replay have a
  single source. No second mechanism, no neutral-zone DOF leak.
- **`decouple_node` is a new session composite**, sibling to `g.constraints` /
  `g.loads` / `g.masses`. Authoring side; resolves node *identity* into
  `fem.nodes`. It has no DOF surface at all.
- **`ops.ndf` is a new bridge directive**, sibling to `ops.fix` / `ops.mass`.
  It is the sole explicit-`ndf` channel and is restricted to element-less
  decoupled-node targets.
- **Three new bridge-side validation gates (G1–G3)** are introduced and are
  non-optional: without them the decoupled-node surface silently reopens the
  half-load / zero-stiffness failure ADR 0046 closed (the adaptive
  `zeroLength.ndf_ok = {1..6}` makes the `∩` gate a no-op for spring grounds).
  They live entirely on the bridge over resolved `ndf` + bridge records —
  generalizing 0046's guard, not duplicating it.
- **Dedup, partitioning, and the viewer** must all respect the provenance flag
  (no-merge, MP-deterministic tags, render lone nodes).
- **Schema bumps.** *This ADR's* additions are additive per
  [ADR 0023](0023-per-zone-schema-versioning.md): the neutral-zone provenance
  flag (DOF-free) and decoupled-node rows in `/opensees/nodes_ndf`. **But** the
  0048 clean break it depends on *removes* the broker `/model/nodes/ndf`
  dataset, which is a **layout break of the neutral zone → a major bump** (old
  H5 files carry that dataset; the versioned stub cache absorbs it). That major
  bump is 0048's, sequenced before these additive ones. `/opensees/nodes_ndf`
  lives in the opensees zone and folds into `model_hash` (not `fem_hash`).

## Open questions

1. **Element-less `ndf` source: state it, or propagate it?** Note this is now
   an **ergonomics** question, *not* a correctness one — the **G1 equal-endpoint
   gate makes both options safe** (a wrong or omitted value fails loud at build
   either way). The choice is only "must the user type the spring ground's
   `ndf`, or is it inferred from its twin?":
   - **(a) Explicit `ops.ndf(gnd, K)`** — the user states the ground's `ndf`.
     Dead simple, covers all use-cases (spring, diaphragm master, control node,
     anchor) uniformly. **The required baseline:** diaphragm masters / control
     nodes / anchors have no twin to inherit from, so (a) must exist regardless.
   - **(b) `zeroLength` endpoint-`ndf` propagation** — the ground end *inherits*
     the structural end's resolved `ndf`, so the user states nothing for that
     sub-case. Ergonomic sugar for springs **layered on top of (a)**, gated by
     G1: propagation must still defer to G1 when a 2-node-link bridges two
     *real* nodes of legitimately differing `ndf` (a solid-face / shell-edge
     interface link) — there is no single "structural side" to copy, and G1
     correctly fails it loud rather than silently picking a winner. Propagation
     is likewise well-defined only for a node reached by a **single** link: a
     ground shared by two springs whose structural sides differ in `ndf` has no
     unique source → it must fall back to (a), and G1 fires on the resulting
     equality conflict if it doesn't. **Lean: (a) now; (b) as a later
     convenience, never a replacement for G1.**
2. **`ops.ndf` target grammar.** Handle-only, or also `label=` / `pg=` for a
   whole decoupled-node set? Lean: accept a handle, a label, or a PG of
   decoupled nodes — all resolving through the standard contract — but **only**
   decoupled-node targets (a mesh-node target fails loud).
3. **Composite / directive naming — DECIDED (2026-05-31).** Session verb is
   **`g.decouple_node(...)`** (the verb names the action); bridge directive is
   **`ops.ndf(...)`** (shortest unambiguous verb). Because the 0048 clean break
   removes `ops.model(ndf=)` *before* `ops.ndf` ships (see Dependencies), there
   is no live `ndf` directive to collide with — the only `ndf` the user ever
   types is `ops.ndf` on an element-less decoupled node.
4. **MP master eligibility.** Can a decoupled node be the *master* of an
   MP-constraint that spans ranks? Tag determinism (above) is necessary;
   whether the cross-partition replication
   ([ADR 0027](0027-cross-partition-mp-constraints.md)) handles a decoupled-node
   master needs a dedicated check.
5. **`g.decouple_node(point=...)` materialization.** Does it copy the point's
   coords at call time (snapshot) or track the point through later transforms?
   Lean: snapshot at resolution time (a decoupled node is not a CAD entity and
   should not follow boolean re-geometry).
6. **Viewer rendering** of lone decoupled nodes (glyph + label) — reuse the
   existing point-glyph path or a dedicated overlay.
7. **Broker-injection primitive + decoupled-tag allocator (the critical net-new
   piece).** How does `g.decouple_node` add a non-gmsh node to the frozen
   post-mesh `fem.nodes` (coords / ids / provenance / `fem_hash` all consistent),
   and where does the MP-deterministic reserved-range tag counter live (not the
   phantom resolver — wrong layer, MP-divergent)? This is the largest greenfield
   gap and must be designed before coding the rest of 0049. Implementation
   spike, not a prose question.

## Related

- [ADR 0048](0048-infer-per-node-ndf-from-elements.md) — `ndf` inference for
  element-touched nodes; this ADR is its sibling and routes element-*less*
  node `ndf` to the same bridge layer (`ops.ndf` → `/opensees/nodes_ndf`).
- [ADR 0046](0046-shell-on-solid-node-sharing-guard.md) — the `∩ ndf_ok`
  gate, reused to validate a stated decoupled-node `ndf` against any attaching
  element.
- [ADR 0022](0022-mp-constraint-emission-fanout.md) — phantom nodes; the
  existence mechanism this generalizes (but decoupled nodes live in
  `fem.nodes` and are user-visible, phantoms do not and are not).
- [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) /
  [ADR 0019](0019-opensees-model-read-side-broker.md) /
  [ADR 0020](0020-results-carries-opensees-model.md) — the broker-chain
  invariants decoupled-node *existence* preserves.
- `g.mesh.editing.remove_duplicate_nodes` — runs on the live gmsh session and
  never reaches decoupled nodes (they are not vertices); the no-merge invariant
  binds any *future broker-side* weld pass, not this one.
