# ADR 0022 — MP constraint emission via `/neutral/` fan-out — closes the §3.3 deferral

**Status:** Accepted (Phase 7b of the major architectural refactor,
May 2026). Closes the §3.3 deferral in the Wave-1 audit; widens the
`Emitter` Protocol (the explicit "architecture event" the Protocol's
header documents).

## Context

Multi-point (MP) constraints are declared on `g.constraints` and
resolved into the FEMData snapshot. The kinds in scope:

- `equal_dof`
- `rigid_link`
- `rigid_diaphragm`
- `node_to_surface` (uses phantom-node interpolation)
- `tie`, `tied_contact`, `mortar`, `embedded` (surface couplings,
  emitted via ASDEmbeddedNodeElement)

The broker (`FEMData`) holds these in
`fem.nodes.constraints.{rigid_link_groups, equal_dofs,
rigid_diaphragms, ...}` and `fem.constraints.surface_couplings`.
The neutral zone in `model.h5` persists them under
`/neutral/constraints/{kind}` per the symmetric compound contract
(ADR 0014; see `mesh/_record_h5.py`).

**They are never emitted into runnable Tcl / Py.**

The current `Emitter` Protocol (`opensees/emitter/base.py`) has only
the bridge-typed surface: `uniaxialMaterial(...)`, `section(...)`,
`geomTransf(...)`, `element(...)`, `pattern(...)`, `timeSeries(...)`,
`recorder(...)`, `analysis(...)`. There is **no** `equalDOF` method,
**no** `rigidLink` method, **no** `rigidDiaphragm` method, **no**
ASDEmbeddedNodeElement support. Users today who declare
`g.constraints.rigid_diaphragm(...)` on a session and then call
`apeSees(fem).tcl(p)` produce a Tcl deck that is **missing the
constraint** — the deck is silently incorrect.

The header of `opensees/emitter/base.py` documents that widening the
Protocol is *"an architecture event"*: every concrete emitter
(`TclEmitter`, `PyEmitter`, `LiveOpsEmitter`, `H5Emitter`,
`RecordingEmitter`, and any future emitter) must grow new methods,
and the H5 schema must gain new groups. This ADR is that event.

The project owner has explicitly authorized closing **all** MP
constraint kinds in this work — not the conservative "close the 3
simple kinds and defer surface couplings" path.

## Decision

### Widen the `Emitter` Protocol — five new methods

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None: ...
    def rigidLink(self, kind: Literal["beam", "bar"], master: int, slave: int) -> None: ...
    def rigidDiaphragm(self, perp_dir: int, master: int, *slaves: int) -> None: ...
    def embeddedNode(self, ele_tag: int, embedding_ele: int, *args: int | float) -> None: ...
    def mp_constraint_comment(self, name: str) -> None: ...
```

Implementation on all five concrete emitters:

| Emitter | `equalDOF` / `rigidLink` / `rigidDiaphragm` | `embeddedNode` | `mp_constraint_comment` |
|---|---|---|---|
| `TclEmitter` | Append matching Tcl command string | Append `element ASDEmbeddedNodeElement ...` | Append `# {name}` |
| `PyEmitter` | Append matching `ops.<method>(...)` line | Append `ops.element('ASDEmbeddedNodeElement', ...)` | Append `# {name}` |
| `LiveOpsEmitter` | Call `ops.equalDOF` / `ops.rigidLink` / `ops.rigidDiaphragm` directly | Call `ops.element('ASDEmbeddedNodeElement', ...)` | No-op (live can't carry comments) |
| `H5Emitter` | Append to `/opensees/constraints/{kind}/` group | Append to `/opensees/constraints/embeddedNode/` group | Captured as a `name` attribute on the record row |
| `RecordingEmitter` | Record method+args | Record method+args | Record method+args |

`mp_constraint_comment` is unusual on the Protocol — comments are not
normally first-class. It is on the Protocol because the user's
declaration label (the `name` field that B1 closed in Phase 2) is a
load-bearing piece of provenance: a Tcl deck written from a model
that contained `g.constraints.rigid_diaphragm(name="floor_1", ...)`
should produce a line `# floor_1` immediately before the
`rigidDiaphragm 3 101 201 202 203` command. Without the Protocol
method, the comment would have to be written by the build-time
fan-out reaching around the emitter's encapsulation — exactly the
"using `_internal` side-channels" anti-pattern ADR 0018 §Alt 2
rejected for `ModelData`.

### Build-time fan-out — Kahn topological sort

New helper `emit_mp_constraints(emitter, fem)` in
`opensees/_internal/build.py`. The pass walks the broker's constraint
collections in dependency order:

```
1. Phantom-node creation FIRST.
   NodeToSurfaceRecord rows carry `phantom_nodes` + `phantom_coords`.
   For each: emitter.node(phantom_id, *xyz)  before any constraint
   references the phantom.

2. rigid_link_groups()    → emitter.rigidLink(kind, master, slave)
3. equal_dofs()           → emitter.equalDOF(master, slave, *dofs)
4. rigid_diaphragms()     → emitter.rigidDiaphragm(perp, master, *slaves)

5. Surface couplings:
     tie, tied_contact, mortar, embedded
     → emitter.embeddedNode(...) per the SurfaceCouplingRecord shape.
```

Each call is preceded by `emitter.mp_constraint_comment(name)` if the
record carries a non-empty `name`. The pass is **deterministic**:
order within each kind follows the broker's stored row order, which is
deterministic per ADR 0021's canonical-bytes property.

### Wiring into `BuiltModel.emit`

`emit_mp_constraints(emitter, fem)` is called by `BuiltModel.emit`
**between element emission and pattern emission**. The ordering
matters:

- **Elements before constraints** — constraint references must point
  at existing element tags (especially for ASDEmbeddedNodeElement,
  which references the embedding element).
- **Constraints before patterns** — patterns reference DOFs that
  constraints may consolidate (`equalDOF` collapses two DOFs into
  one); the pattern's `ops.load(node, ...)` calls must come after
  the constraint topology is locked.

### ASDEmbeddedNodeElement — its own Protocol method

ASDEmbeddedNodeElement is technically an OpenSees element. The
alternative was to ride it through the existing `emitter.element(...)`
channel. Rejected because:

1. ASDEmbeddedNodeElement is the **surface coupling primitive** —
   semantically it is a constraint, not a structural element. Routing
   it through `element(...)` puts it in the wrong conceptual bucket.
2. The H5 schema is cleaner: `/opensees/constraints/embeddedNode/`
   sits alongside `/opensees/constraints/equalDOF/` etc, instead of
   leaking into `/opensees/element_meta/`.
3. The build-time emit pass is cohesive — *all* MP constraints fan
   out through one pass, in one helper.

Trade-off accepted: one more Protocol method (5 instead of 4), but
the semantic clarity is worth it.

### Invariants

**INV-1.** `apeSees(fem).tcl(p)` produces a **runnable** OpenSees
deck for any model with `g.constraints.*` declarations. "Runnable"
means `analyze` returns non-zero (i.e. it converges on a fixture; not
just syntactically valid). This is enforced by a new integration
test that runs OpenSees on a tied-contact fixture and asserts a
non-trivial `analyze` result. Without this gate, "syntactically
valid" decks that silently drop constraints would still pass.

**INV-2.** The `name` field on every MP constraint declaration
round-trips into the emitted deck via
`mp_constraint_comment(name)`. The user's declaration label survives
from `g.constraints.rigid_diaphragm(name="floor_1", ...)` all the way
to a `# floor_1` line in the Tcl output. Without INV-2, the B1 fix
from Phase 2 (which preserved the name through the FEM round-trip) is
half-finished — preserved in the broker, lost at the emitter.

**INV-3.** Phantom-node creation precedes constraint emission. The
build pass is responsible for issuing `emitter.node(phantom_id, *xyz)`
before any constraint references the phantom tag. This is enforced
by the build-pass implementation; not by the Protocol shape (the
Protocol can't express ordering invariants).

**INV-4.** Every concrete emitter implements every Protocol method.
`RecordingEmitter` records all five; `LiveOpsEmitter` no-ops
`mp_constraint_comment` (live can't carry comments) but implements
the four constraint methods; `H5Emitter` writes all five into the
schema. Partial implementations are not allowed; the `Protocol` shape
is the contract.

**INV-5.** The MP constraint pass runs **between** element emission
and pattern emission inside `BuiltModel.emit`. Re-ordering the pass
(e.g. moving it before element emission) requires repealing this ADR.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Keep §3.3 deferred indefinitely** | Structural embarrassment per Wave-1 RED #9 — `apeSees(fem).tcl(p)` can emit element loads but cannot emit a rigid link the user declared on the same session, on the same FEM. The owner explicitly chose to close, not defer. |
| **Ride `equalDOF` / `rigidLink` / `rigidDiaphragm` via the existing `emitter.element(...)` channel** | Only ASDEmbeddedNodeElement is an OpenSees element. The other three are **domain-level openseespy commands** (`ops.equalDOF`, `ops.rigidLink`, `ops.rigidDiaphragm`); they cannot ride `element(...)` without faking type-tokens and breaking the bridge's typed-primitive vocabulary. |
| **Implement only the 3 simple kinds (`equalDOF`, `rigidLink`, `rigidDiaphragm`); defer surface couplings** | Owner explicitly chose all kinds. ASDEmbeddedNodeElement covers `tie` / `mortar` / `tied_contact` / `embedded`; deferring it leaves `g.constraints.tied_contact(...)` permanently broken on the Tcl/Py path. The whole point of the §3.3 closure is to finish the MP story. |
| **User keeps hand-iterating `fem.nodes.constraints.*` into raw openseespy** | This is the status quo. Defeats the purpose of `apeSees(fem).tcl(p)` producing a runnable deck. Charter P1 (*"declarative model → runnable solver input"*) is half-fiction without this fan-out. |
| **Implement `mp_constraint_comment` outside the Protocol** (the build pass writes comments via direct emitter-state access) | Reaches around the emitter's encapsulation; the build pass would need to know whether each emitter supports comments (Tcl: yes, live: no, H5: as attr). The Protocol method puts the polymorphism where it belongs. |
| **Combine `equalDOF` / `rigidLink` / `rigidDiaphragm` into a single `mp_constraint(kind, ...)` Protocol method** | Saves two Protocol methods at the cost of stringly-typed dispatch inside each emitter. The three commands have different shapes (`equalDOF` has variable `*dofs`, `rigidLink` has `kind`, `rigidDiaphragm` has `perp_dir`); a single dispatch method would need a tagged-union argument that none of the emitters actually want. The five-method shape matches OpenSees vocabulary. |

## Consequences

**Positive:**

- Closes a long-standing structural gap. `apeSees(fem).tcl(p)` for a
  model with `g.constraints.rigid_diaphragm(...)` now produces a
  runnable OpenSees deck (INV-1).
- The `name` field round-trips into the emitted deck (INV-2), citing
  the B1 fix from Phase 2.
- Future emitters (Code_Aster, Abaqus, …) have a clear contract: the
  Protocol shape is locked at five constraint-related methods. Adding
  a sixth requires another ADR.
- The neutral zone's `/neutral/constraints/{kind}` and the bridge
  zone's `/opensees/constraints/{kind}` agree on shape, on
  representation, and on lifecycle. The symmetric compound contract
  (ADR 0014; `mesh/_record_h5.py`) covers both.

**Negative:**

- Emitter Protocol widens. Every concrete emitter (current + future)
  must implement five more methods. The widening is the cost of
  closing §3.3, and the Protocol header's "architecture event"
  language documented this would happen.
- Phantom-node-first ordering adds a new pass to the build pipeline.
  ~200 LOC in `opensees/_internal/build.py`; isolated and reviewable.
- `mp_constraint_comment` on the Protocol is unusual (comments-as-
  Protocol-methods). Accepted because the alternative — silently
  losing the user's declaration label, undoing the B1 fix — is worse.
- One integration test must run an actual OpenSees `analyze` on a
  tied-contact fixture to prove the emitted deck is runnable (INV-1).
  This is a new CI cost; one fixture, small mesh, fast. Accepted.
- Future schema bump (per ADR 0023): adding the five constraint
  groups to `/opensees/constraints/` is a minor bump on the
  `opensees_schema_version` (`2.7.0`-ish; precise number is Phase 7a's
  decision).

## Open questions

- **Q4 — phantom-node emission timing.** Resolved (implementation
  step): before Phase 7 starts, audit
  `src/apeGmsh/_kernel/resolvers/_constraint_resolver/_resolver.py` to
  confirm phantoms are NOT already baked into `fem.nodes` at resolve
  time. If they are, drop the phantom-emit pass in INV-3 (the nodes
  emit through the normal element-emission pass); if not, ship the
  phantom-emit pass as scoped in INV-3. Either outcome preserves
  INV-3 (phantoms exist before constraints reference them); the
  implementation differs only in which layer issues the
  `emitter.node(...)` call.

- **Q5 — `mp_constraint_comment` Protocol smell.** Resolved (this
  ADR): ship as scoped. The five-method shape is the locked
  Protocol. If code review during Phase 7b surfaces a cleaner
  alternative (e.g. emitters carrying a comment-stream out-of-band),
  document and migrate in a follow-up ADR.

## References

- [decisions/0008-three-emit-targets.md](0008-three-emit-targets.md)
  — the original Emitter Protocol shape this ADR widens.
- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — the H5 emitter; the fifth Protocol method
  `mp_constraint_comment` is the unusual case this ADR documents.
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the symmetric compound-row contract for record sets; the MP
  constraint groups follow it.
- [decisions/0019-opensees-model-read-side-broker.md](0019-opensees-model-read-side-broker.md)
  — `OpenSeesModel.constraints()` will read the
  `/opensees/constraints/{kind}` groups this ADR writes.
- [phase-8-untangle.md](../phase-8-untangle.md) §7 closure — the MP
  constraint emission gap this ADR closes.
- `opensees/emitter/base.py` header — the "architecture event"
  documentation; this ADR is that event.
- B1 fix in Phase 2 — preserved the `name` field through FEM
  round-trip; this ADR's INV-2 carries it the rest of the way.
