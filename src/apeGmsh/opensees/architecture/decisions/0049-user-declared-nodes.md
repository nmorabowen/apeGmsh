# ADR 0049 — User-declared nodes (analysis auxiliary nodes)

**Status:** Proposed (2026-05-31). Sibling to
[ADR 0048](0048-infer-per-node-ndf-from-elements.md) (per-node `ndf`
inference): 0048 says *mesh*-node `ndf` is inferred; this ADR adds
*user*-declared nodes whose `ndf` is **stated at creation** — the single
legitimate home for an explicit `ndf`. Builds on the broker-chain invariants
of [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) /
[ADR 0019](0019-opensees-model-read-side-broker.md) /
[ADR 0020](0020-results-carries-opensees-model.md) and generalizes the
phantom-node mechanism of [ADR 0022](0022-mp-constraint-emission-fanout.md).

## Context

apeGmsh is mesh-centric: every emitted node is a gmsh vertex carried on the
`FEMData` broker (`fem.nodes`). The only non-mesh nodes today are the
internal phantoms ([ADR 0022](0022-mp-constraint-emission-fanout.md)),
synthesized for `node_to_surface` constraints and never user-visible.

Real OpenSees models routinely need **auxiliary nodes that are not mesh
vertices**:

- **Spring / dashpot supports** — every `zeroLength`-to-ground needs a fixed
  partner node at (usually) the *same location* as a boundary mesh node.
  This is the spine of soil-structure interaction (Lysmer dashpots, soil
  springs, p-y / t-z / q-z BNWF). It is the dominant case for this project's
  staged-SSI work.
- **Constraint master / control nodes** — a `rigidDiaphragm` retained node at
  a floor's center of mass, a control node for a displacement-driven pattern,
  an `equalDOF` reference node. These are the legitimately *element-less*
  nodes [ADR 0048](0048-infer-per-node-ndf-from-elements.md) cannot infer.
- **Point-load / lumped-mass anchors** not at a mesh vertex.

Two candidate homes were considered and rejected before B:

1. **Bridge-only `ops.node(...)`** (rejected) — a node origin *outside* the
   lineage-hashed broker. `fem_hash` (computed over the broker node set) would
   never see these nodes, yet `model_hash = blake2b(fem_hash ‖ opensees-zone)`
   would fold in `/opensees/` elements referencing their tags → the chain no
   longer describes the model. The viewer (reads neutral `/model`) and
   `Results.model.fem` would have dangling node references. Breaks the
   broker-chain invariant that **the broker is the single source of truth for
   nodes**.
2. **Reuse a geometry point** (rejected as the *spine*; kept as a convenience)
   — a geometry point is a CAD entity subject to boolean ops, healing, and
   **node dedup**. `remove_duplicate_nodes` merges a point-node into a
   coincident mesh node — which destroys exactly the spring-support case (the
   ground node *must* be a second, distinct node at the boundary node's
   location). Geometry points also require an embed / dim-0-PG step to
   materialize a node at all. Fine for a *free-standing* node, fatal for a
   *coincident* one.

## Decision

**User-declared nodes are first-class broker nodes (option B), authored
through a new `g.nodes` session composite, dedup-immune, with `ndf` stated at
creation.**

### API surface

```python
# Spine — a dedup-immune analysis node at explicit coords (the SSI case):
gnd = g.nodes.add(x, y, z, *, ndf=3, label="pile_tip_ground")

# Convenience — source location + label from an EXISTING geometry point
# (free-standing nodes: control / master / anchor):
ctrl = g.nodes.at(point="floor_cm", *, ndf=3, label="diaphragm_master")
```

Both return a **handle** referenced everywhere downstream (no raw tags, per
the no-raw-tags rule) — by `g.constraints` / `g.loads` pre-mesh and by the
bridge (`ops.element.zeroLength(soil_node, gnd, ...)`, `ops.fix(gnd, ...)`)
post-session. The handle resolves through the same label tier as everything
else.

### Broker representation

`fem.nodes` gains:
- a **provenance** discriminator per node (`mesh` | `user`);
- an **optional `ndf`** field, populated **only** for `user` nodes that
  stated one (mesh nodes never carry it — they are inferred per
  [ADR 0048](0048-infer-per-node-ndf-from-elements.md)).

User nodes fold into `fem_hash` and round-trip through the neutral `/model`
zone exactly like mesh nodes — so the viewer and `Results.model.fem` see them
for free. **This is the entire point of B.**

### `ndf` rule (unifies with ADR 0048)

One rule for all nodes, provenance choosing the source:

- `ndf` is **inferred from incident elements** for every node — mesh or user.
  A user ground node with a `zeroLength` attached infers from the zeroLength;
  no declaration needed.
- A **user node may state an explicit `ndf` at creation**, used when *no*
  element attaches (the free control / master node). This is the **only**
  place an explicit `ndf` is accepted anywhere in apeGmsh.
- Validation is the **same `∩ ndf_ok` gate** ([ADR 0046](0046-shell-on-solid-node-sharing-guard.md)):
  if a user states `ndf` and later attaches an element that disagrees → fail
  loud. The stated value never silently wins over a contradicting element, so
  no two-headed model is reintroduced.
- **Element-less node with no `ndf`** → fail loud. For a mesh node that is a
  modeling error; for a user node the message names `g.nodes.add(..., ndf=)`.

### Dedup immunity

`g.mesh.editing.remove_duplicate_nodes` (and any merge pass) **must not**
merge a `user` node — even one coincident with a mesh node. Coincidence is
the *intended* state for spring/dashpot grounds. The provenance flag gates
the merge.

### Tag allocation

User-node tags occupy a reserved range disjoint from mesh tags and phantom
tags. Allocation must be **deterministic and MP-safe** (per-rank consistent),
so a user node shared across partitions resolves to the same tag on every
rank — mirroring the phantom-tag allocator's `> max(broker_tag)` discipline
and the cross-partition counter caveat.

## Rationale

**B is the only option that preserves the broker chain.** The rejected
bridge-only path creates a second node origin the lineage hash never sees;
B keeps the broker as the single source of truth, so every invariant in
ADRs 0019/0020/0021 holds unchanged.

**Geometry points are CAD, not DOFs.** They are the right *location* source
for a free node (hence `g.nodes.at(point=...)`), but cannot be the analysis
node itself: dedup would merge the coincident spring ground node, and CAD ops
(fragment, heal) have no business mutating a DOF carrier. `g.nodes` reuses the
point for location without dragging the analysis node through the CAD/mesh
pipeline.

**Stated-at-creation `ndf` is principled, not a relapse.** [ADR 0048](0048-infer-per-node-ndf-from-elements.md)
deleted explicit `ndf` for *mesh* nodes because it competed with element
inference (the two-headed model). A user node has no element inference
competing with it until an element attaches — at which point the `∩ ndf_ok`
gate validates rather than silently overrides. Explicit `ndf` lives exactly
where the node is *born by the user*, and nowhere else.

## Consequences

- **Closes the element-less-node gap** left open by
  [ADR 0048](0048-infer-per-node-ndf-from-elements.md): a `rigidDiaphragm`
  master / control node is now a first-class user node with a stated `ndf`,
  not a synthesized or fail-loud case.
- **Amends [ADR 0048](0048-infer-per-node-ndf-from-elements.md)'s "ndf leaves
  the broker entirely."** The *old* `_ndf` (explicit-for-mesh + envelope
  fallback) is gone; this ADR adds a **provenance-scoped** user-node `ndf`
  field with different semantics (authored, only on user nodes, only
  meaningful when element-less). 0048's wording is updated to forward-
  reference this.
- **`g.nodes` is a new session composite**, sibling to `g.constraints` /
  `g.loads` / `g.masses`. Authoring side; resolves into `fem.nodes` (which
  already aggregates resolved analysis data), mirroring the
  `g.loads → fem.nodes.loads` pattern.
- **Dedup, partitioning, and the viewer** must all respect the provenance
  flag (no-merge, MP-deterministic tags, render lone nodes).
- **Schema bump** for the neutral node zone (provenance + optional user-node
  `ndf`), additive per [ADR 0023](0023-per-zone-schema-versioning.md).

## Open questions

1. **Composite name.** `g.nodes` reads naturally but risks confusion with
   `fem.nodes` (broker) and the deleted `g.node_ndf`. Alternatives: `g.points`
   (collides with geometry points), `g.aux_nodes`. Lean: `g.nodes`, documented
   as the *authoring* surface (like `g.loads` vs `fem.nodes.loads`).
2. **MP master eligibility.** Can a user node be the *master* of an
   MP-constraint that spans ranks? Tag determinism (above) is necessary;
   whether the cross-partition replication
   ([ADR 0027](0027-cross-partition-mp-constraints.md)) handles a user-node
   master needs a dedicated check.
3. **`g.nodes.at(point=...)` materialization.** Does it copy the point's
   coords at call time (snapshot) or track the point through later transforms?
   Lean: snapshot at resolution time (a user node is not a CAD entity and
   should not follow boolean re-geometry).
4. **Viewer rendering** of lone user nodes (glyph + label) — reuse the
   existing point-glyph path or a dedicated overlay.

## Related

- [ADR 0048](0048-infer-per-node-ndf-from-elements.md) — `ndf` inference;
  this ADR is its sibling and the home for the only explicit `ndf`.
- [ADR 0046](0046-shell-on-solid-node-sharing-guard.md) — the `∩ ndf_ok`
  gate, reused to validate a user node's stated `ndf` against any attaching
  element.
- [ADR 0022](0022-mp-constraint-emission-fanout.md) — phantom nodes; the
  mechanism this generalizes (but user nodes live in `fem.nodes`, phantoms
  do not).
- [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) /
  [ADR 0019](0019-opensees-model-read-side-broker.md) /
  [ADR 0020](0020-results-carries-opensees-model.md) — the broker-chain
  invariants B preserves.
- `g.mesh.editing.remove_duplicate_nodes` — must skip `user`-provenance nodes.
