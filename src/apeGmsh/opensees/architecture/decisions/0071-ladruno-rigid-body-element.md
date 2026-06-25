# ADR 0071 — `rigid_body(as_element=True)` → `element LadrunoRigidBody`

**Status:** Accepted (2026-06-24). Third "Ladruno constraints coverage"
cluster (B2). Adds an opt-in element-backed emission to the existing
`g.constraints.rigid_body`. Rides the `emitter.element(...)` path — no
`Emitter` Protocol change.

## Context

`g.constraints.rigid_body(master_label, slave_label, …)` resolves to a
`NodeGroupRecord(kind="rigid_body", master_node, slave_nodes)` and is
emitted as a **chain of `rigidLink "beam"`** constraints (master → each
slave). That captures the kinematics but nothing else: there is no body
mass, no centre-of-mass node, and no explicit-dynamics support — the
rigidLinks are pure MP constraints.

The Ladruno fork ships a dedicated `element LadrunoRigidBody` (class tag
33015, `OPS_LadrunoRigidBody.cpp`):

```
element LadrunoRigidBody $tag $N $s1..$sN [-mass $m] [-internalNode $tag]
```

It builds the whole node set into one 6-DOF rigid body with a **private
internal CoM node** and **condensed mass** (`-mass`, or condensed from the
slaves' nodal mass), and is built for explicit dynamics (ballistic
translation now, SO(3) rotation at P2). Crucially it has **no external
master** — the CoM is internal — so it is not a drop-in re-plumb of the
master/slave `rigid_body`; the natural mapping is "the body = the node set
`{master, *slaves}`".

## Decision

Expose it as an **opt-in flag on the existing method**, not a new method:

* `g.constraints.rigid_body(..., as_element=False, mass=None)`. With
  `as_element=True` the body `{master, *slaves}` is emitted as
  `element LadrunoRigidBody $tag $N $nodes… [-mass]`; `as_element=False`
  (default) keeps the rigidLink chain byte-for-byte. `mass` is valid only
  with `as_element` (validated at construction; `mass >= 0`).
* `RigidBodyDef` / `NodeGroupRecord` gain `as_element: bool = False` and
  `mass: float | None = None` (additive, default off ⇒ byte-stable). The
  resolver carries them onto the record.
* **Emission** — a new `build._emit_rigid_body_elements` (a tags-aware
  pass, run right after the rigidLink pass in all four emit paths —
  global, staged, and the two partitioned ranks). It emits one element
  over `{master, *slaves}`; `_emit_rigid_links` skips `as_element` rigid
  bodies so they never double-emit. `-internalNode` is omitted so the fork
  auto-assigns the CoM tag (`9000000 + eleTag`, collision-safe).
* **Fork gating** — `LadrunoRigidBody` added to the live emitter's
  `_FORK_ONLY_ELEMENTS`, so a stock-OpenSees live run fails loud; deck
  emission (`.tcl()`/`.py()`) works on any build.
* **Persistence** — `node_group_payload_dtype` gains `as_element` (uint8)
  + `mass` (float64, NaN ⇒ condensed) columns (**neutral schema 2.19.0**,
  additive; column names match the record fields per the parity contract;
  pre-2.19.0 files presence-probe and decode the chain default).

## Scope / deferred

`ndm` is checked by the fork at runtime (3D only, v1). The `-omega`
initial-angular-velocity knob (P2 explicit IC) and an explicit
`internalNode` tag override are not exposed — niche; tracked in
`plan_ladruno_constraints_coverage.md`. The standalone "rigid body over a
node set with no master" spelling is intentionally not added — the
`{master, *slaves}` mapping covers it with zero new API surface.

## Consequences

* `rigid_body` gains a real element backend (CoM, body mass, explicit
  support) behind an opt-in flag; existing rigid-body emission is
  unchanged. Schema 2.18.0 → 2.19.0 (additive minor; two-version reader
  window per ADR 0023).
