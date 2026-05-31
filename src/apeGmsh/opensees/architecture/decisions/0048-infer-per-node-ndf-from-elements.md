# ADR 0048 — Infer per-node `ndf` from declared element classes

**Status:** Proposed (2026-05-31). **Supersedes** both
[ADR 0032](0032-explicit-only-per-node-ndf.md) (explicit-only `g.node_ndf`)
and [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) (override-only emit
envelope) — this is a **clean break, no compatibility shim**. Builds on the
node-sharing guard of [ADR 0046](0046-shell-on-solid-node-sharing-guard.md)
and the element registry it shares. Related to
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) (lineage) and
[ADR 0022](0022-mp-constraint-emission-fanout.md) (phantom nodes).

> **No straddling.** The explicit-only API (`g.node_ndf`, the `ndf=` kwarg on
> `ops.model`, the broker `_ndf` array, `ndf_for`,
> `validate_envelope_covers_broker_ndf`) is **deleted**, not deprecated.
> Keeping both an inference path and the old composite would leave a
> two-headed model no one can reason about. Inference is the *only* path.

## Context

[ADR 0032](0032-explicit-only-per-node-ndf.md) made per-node `ndf` an
explicit user contract (`g.node_ndf.set(...)` / `.set_default(...)`,
fail-loud `ndf_for`), and [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md)
wired it into emit with **override-only** semantics: the OpenSees model
envelope (`ops.model(ndm, ndf)`) is the default, per-node `-ndf` is emitted
only where the broker carries a non-sentinel value.

Two problems surfaced in use:

1. **`ops.model(ndf=K)` leaks OpenSees' stateful global.** Because the
   envelope-coverage validator (`validate_envelope_covers_broker_ndf`)
   requires `envelope >= max(per-node ndf)`, the envelope is a **ceiling**
   the user must pin to the most-DOF-hungry region — then pull the bulk of
   the model *down* with `g.node_ndf.set_default`. "Set the default to the
   maximum and override everyone else downward" is exactly backwards from
   how the word *default* reads. The abstraction needs a paragraph of
   correction to use safely.

2. **Per-region `ndf` duplicates the element declaration.** Once the user
   writes `ops.element.ShellMITC4(pg="Shell")`, the bridge already knows the
   Shell PG needs `ndf=6` — it is in `_ELEM_REGISTRY[...].ndf_ok`. Making the
   user *also* declare `ndf=6` restates element-determined metadata and
   introduces a way for the two to drift.

[ADR 0032](0032-explicit-only-per-node-ndf.md) explicitly rejected
inference. But every *technical* failure it cites is a failure of
**dim-keyed** inference — the `{line:6, surface:6, volume:3}` table PR #307
shipped, which cannot tell a truss from a beam (both line-dim) and cannot
decide a mixed shared node. **Element-class** inference, keyed on the actual
declared element's `ndf_ok`, has none of those failures: a 3D truss resolves
to 3 (not 6), and a shared node is decided by the rule below. The remaining
0032 objection — "explicit matches the loads/masses/constraints precedent" —
is a category error: a load carries information not derivable from the mesh,
whereas `ndf` is fully determined by the element you already declared.

## Decision

**`ndf` is inferred per node from the declared element classes. `ndm` stays
an explicit declaration with a new compatibility guard.**

### `ndm` — explicit, guarded

`ops.model(ndm=K)` remains required. `ndm` is the coordinate-space dimension
— a single global fact that cannot be regional or mixed, and that all-truss
/ all-beam models leave genuinely ambiguous (so it cannot be fully
inferred). A new guard asserts `K ∈ ⋂ ndm_ok` over all declared elements;
an empty intersection (e.g. a `quad` and a `stdBrick` in one model) raises
`BridgeError` naming the offending pair — turning "you can't mix 2D and 3D
elements" from a rule the user must remember into a detected contradiction.

### `ndf` — inferred, validity-gated

At build time, **on the bridge** (element→PG assignment lives on the bridge,
not the neutral broker), for each mesh node touched by ≥1 declared element:

```
candidate = max over attached elements of required_floor(elem, ndm)
VALID iff  candidate ∈ ndf_ok(elem)  for EVERY attached element
empty / incompatible  →  BridgeError:
    "<elemA> and <elemB> cannot share node <nid> (ndf_ok disjoint);
     use separate coincident nodes + equalDOF on the shared DOFs."
```

This is **not** plain max-wins. Per the verified `FE_Element::setID` law
(see Rationale), an element requires each node's `ndf` to **exactly** equal
its per-node DOF expectation; it does not ignore surplus DOFs. So a shared
node's `ndf` must lie in the **intersection** of every attached element's
`ndf_ok`. The candidate (max of required floors) is then checked against
that intersection; an empty/incompatible result means the elements must not
share a node at all (the classic separate-node + `equalDOF` idiom — the same
resolution [ADR 0046](0046-shell-on-solid-node-sharing-guard.md) prescribes
for shell-on-solid). The validity gate **is** the existing
`validate_node_ndf_element_compat`.

### `required_floor` — a new registry field, distinct from `ndf_ok`

`_ElemSpec` gains `required_floor(ndm) -> int`, the element's **minimum**
DOF-per-node at a given `ndm`, kept separate from the `ndf_ok` **tolerance**
set. Single-`ndm` elements are trivial (`quad`→2, `stdBrick`→3, shells→6);
multi-`ndm` elements need the explicit map because the set alone cannot be
collapsed (`elasticBeamColumn` `ndf_ok={3,6}` → floor 3 at ndm=2, **6** at
ndm=3 — not derivable by "smallest ≥ ndm", which would wrongly give 3;
`truss` `ndf_ok={2,3,6}` → 2 at ndm=2, 3 at ndm=3).

### Coupled-field DOFs (pressure, thermal) — still inferred, but **per node position**

Pressure (`u-p`) and similar coupled-field elements are the stress-test, and
they confirm two things. First, the pressure DOF is **still element-
determined** — declaring a `u-p` element on a PG *is* the declaration that
those nodes carry a pore-pressure DOF. There is no user `ndf` override for
pressure; it falls out of the element exactly like a rotational DOF does.

Second, they break the "one element → one floor" simplification. The
mixed-order `u-p` elements assign **different ndf to different node positions
within a single element**, verified in source:

- `Nine_Four_Node_QuadUP` (`Nine_Four_Node_QuadUP.cpp:392-396`): first
  `nenp=4` corner nodes must be `ndf=3` (2 disp + p); the other 5 `ndf=2`.
- `Twenty_Eight_Node_BrickUP` (`Twenty_Eight_Node_BrickUP.cpp:487-489`):
  8 corner nodes `ndf=4` (3 disp + p); 12 mid-side `ndf=3`.

Therefore the required-ndf abstraction is keyed by **(element class, `ndm`,
local node index)**, not by element class alone:
`required_floor(elem, ndm, local_index) -> int`. For every element currently
in `_ELEM_REGISTRY` this is position-independent (all nodes equal), so the
common path is unchanged — but the **seam must be position-aware from day
one** so the `u-p` family slots in without a redesign. `node_reorder`
already gives the gmsh→OpenSees local index needed to apply the rule.

Consequences that fall straight out of the existing rules:

- **`u-p` ↔ `u` interface** (e.g. `brickUP` ndf=4 meeting `stdBrick` ndf=3 at
  a shared node): `∩ ndf_ok = ∅` → the validity gate fails loud → separate
  coincident nodes + `equalDOF` on the **displacement** DOFs (the pressure
  DOF stays independent). Identical pattern to shell-on-solid; no new rule.
- **Drainage / pressure boundary** (`p = 0` at a free-draining face) is a
  single-point constraint on the pressure DOF —
  `ops.fix(pg=..., dofs=(0,0,0,1))` — **not** an `ndf` concern. Orthogonal,
  already handled.
- The valid `ndf` range must come from the **registry's `ndf_ok` union**
  (`u-p` needs 4; three-phase `u-p-U` would need 7), never a hardcoded `1..6`
  (the deleted `NodeNDFComposite` capped at 6).

> The `u-p` family is **not yet in `_ELEM_REGISTRY`** — adding it (with
> per-position ndf) is separate future work. ADR 0048 only commits to the
> position-aware *seam* so that work is additive.

### No user-facing `ndf` declaration — at all

There is **no** `g.node_ndf` and **no** `ndf=` anywhere. The user declares
`ndm` and their elements; `ndf` is entirely derived. The only nodes
inference cannot see are:

- **Phantom nodes** ([ADR 0022](0022-mp-constraint-emission-fanout.md)) —
  keep their hardcoded `ndf=6` carveout (broker-internal, never user-facing).
- **Element-less emitted nodes** — a mesh node carried into the deck but
  belonging to no declared OpenSees element. Under pure inference these
  **fail loud**: `"node <nid> belongs to no element — it would carry
  unconstrained DOFs; attach an element or remove it from emission."` We do
  **not** add an override hatch for them: a node with DOFs no element
  stiffens is a singular-matrix modeling error, not a feature. (If a genuine
  element-less-but-constrained case ever appears — e.g. a stand-alone
  rigidDiaphragm master — we add *constraint-based* inference then, not a
  parallel user declaration.)

The "deliberate spare DOF" case the override used to serve does not exist:
if a node legitimately needs a DOF, some attached element needs it, so
inference already produces it; if no element needs it, it is unconstrained
and wrong.

### Envelope — auto-computed, never stated

`ops.model(ndm=K)` takes **only** `ndm`. The OpenSees builder envelope is set
automatically to `max(resolved per-node ndf)`. The emitter elides `-ndf` on
nodes whose value equals the envelope — not for back-compat, but because
emitting a redundant `-ndf 3` on every node of a uniform model is noise.
(Homogeneous decks happen to come out identical to the pre-0048 output; that
is a side effect, not a goal.)

## Rationale

**The node-sharing law is verified, not assumed.** `FE_Element::setID`
(`SRC/analysis/fe_ele/FE_Element.cpp:274-281`) packs each node's *entire*
DOF list (`theDOFid.Size()`) sequentially into `myID`, sized to the
element's own `numDOF`. `FourNodeQuad::getNumDOF()` returns a hardcoded `8`
(`:440-443`) and `setDomain` bails (silently — error message commented out)
if any node's `ndf != 2` (`:479-484`). So surplus node DOFs are never
ignored as a prefix: they overflow/scramble the equation map, or the element
silently fails to assemble. Hence intersection-of-`ndf_ok`, not max-wins,
and hence the empty-intersection case is a real "cannot share" — exactly
[ADR 0046](0046-shell-on-solid-node-sharing-guard.md).

**Inference belongs on the bridge.** The neutral `FEMData` broker is
solver-agnostic — it carries gmsh element types and PG membership, not the
fact that PG "Frame" will be a `forceBeamColumn`. That mapping is established
by `ops.element.X(pg=...)`. So element-class inference can only run on the
bridge, at build time, after element declarations are collected and before
node emission. This also vindicates the original instinct that `ndf` is
analysis intent (bridge-side), not mesh metadata.

**MP consistency by determinism.** Every OpenSeesMP rank runs the same
inference over the same broker + same element declarations, so shared
boundary nodes resolve identically without cross-rank communication — the
property [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) previously got from
folding `_ndf` into `fem_hash`.

## Consequences

- **Breaking API change, accepted deliberately.** `ops.model(ndm, ndf)` →
  `ops.model(ndm)`; `g.node_ndf` and the broker `_ndf` / `ndf_for` surface
  are deleted. Every existing `apeSees(fem).model(ndm=K, ndf=N)` call site
  and example must be updated to drop `ndf=`. We are not at a
  compatibility-promise stage; the clean model is worth the mechanical churn.
- **Nothing for the user to learn or mis-set.** Declare `ndm` + elements;
  `ndf` is automatic. The only ndf-related surface the user ever sees is a
  fail-loud message — on an incompatible shared node, or an element-less
  emitted node. No ceiling, no default, no override.
- **The `quad + beam` / frame-on-plane-strain interface fails loud.** If the
  user fragments those into a shared node, the validity gate raises and
  points to separate-node + `equalDOF` — the correct model.
- **Emitted decks for homogeneous models are unchanged** (auto-envelope = the
  uniform value, `-ndf` elided) — a side effect of doing it right, not a
  compatibility target.
- **Mesh-node `ndf` is derived, not stored.** The *old* `_ndf` array
  (explicit-for-mesh + envelope fallback) and its `fem_hash` fold are removed.
  The bridge computes the resolved *mesh*-node map at build; persist-vs-
  re-derive for the viewer / replay is OPEN (see below). **Caveat (ADR 0049):**
  the broker is *not* left with zero `ndf` — [ADR 0049](0049-user-declared-nodes.md)
  adds a **provenance-scoped** user-node `ndf` field (authored at creation,
  only on user-declared nodes, only meaningful when element-less). That is a
  different field with different semantics from the deleted `_ndf`; it folds
  into `fem_hash` because user nodes are first-class broker nodes.

## Open questions

1. **Persist vs. re-derive the resolved per-node `ndf`.** The viewer
   (ADR 0014/0026 — reads only via `emitter.h5_reader`) and
   `OpenSeesModel.build` replay need the per-node `ndf`. Two clean options,
   both leaving the neutral broker untouched:
   (a) **persist** a bridge-written `/opensees/nodes_ndf` dataset the reader
   surfaces; (b) **re-derive** — since `ndf = f(elements, ndm)` and both are
   persisted, the reader re-runs inference. Lean: **(a) persist** — simplest
   reader boundary, avoids putting the registry behind `h5_reader`.
2. **Element-less emitted nodes — fail loud vs. silently skip.** A node in
   `fem.nodes` used by no declared element is currently still emitted. Under
   0048 it has no inferable `ndf`. Lean: **fail loud** (it is a modeling
   error). Confirm there is no legitimate flow that emits such nodes (other
   than phantoms, which are carved out).

## Related

- [ADR 0032](0032-explicit-only-per-node-ndf.md) — superseded.
- [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) — amended (override-only
  emit, envelope-coverage validator reused as the supplied-envelope check).
- [ADR 0046](0046-shell-on-solid-node-sharing-guard.md) — the node-sharing
  guard, reused verbatim as the inference validity gate.
- `opensees/_element_capabilities.py` — `_ELEM_REGISTRY`, `ndf_ok`,
  `element_class_ndf_ok`; gains `required_floor(ndm)`.
- `opensees/_internal/build.py` — `validate_node_ndf_element_compat`
  (validity gate), `validate_envelope_covers_broker_ndf` (supplied-envelope
  check), `_emit_node_with_broker_ndf` (elide-on-equal).
- OpenSees source: `SRC/analysis/fe_ele/FE_Element.cpp:274-281` (setID),
  `SRC/element/fourNodeQuad/FourNodeQuad.cpp:440-443,479-484`.
