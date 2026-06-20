# ADR 0068 — Constraint-based non-matching tie via `equationConstraint` (EQ_Constraint)

**Status:** Proposed (2026-06-20).  Extends ADR 0022 (MP-constraint
emission fan-out) with a **sixth** `Emitter` Protocol method and a second
*enforcement route* for the surface-coupling constraints (`tie`,
`tied_contact`).  Threads through ADR 0035/0036 (`ASDEmbeddedNodeElement`
exposure + host decomposition — the existing **penalty** route) and ADR
0041 (chain-phase routing for geometry-intensive constraints).  Pairs
with ADR 0066 (H5DRM authoring) — the constraint route is what makes a
*non-matching* soil-structure interface work under **explicit** DRM.  No
apeGmsh geometry change (the projection + shape-function machinery is
reused verbatim); the fork OpenSees side is **already shipped** (see
Context).

## Context

apeGmsh's `g.constraints.tie(...)` / `tied_contact(...)` already do the
hard geometric work.  `resolve_tie`
(`_kernel/resolvers/_constraint_resolver/_resolver.py`) Newton-projects
each slave node onto the closest master element face, evaluates the
face's shape functions at the projection, and produces one
`InterpolationRecord` per slave carrying the exact partition-of-unity
weights:

```
u_d(slave) = Σ_i  w_i · u_d(master_i)        w_i = N_i(ξ, η),  Σ w_i = 1
```

Today **every** such record is emitted as an `ASDEmbeddedNodeElement`
*penalty* element (`build.py::_emit_one_interpolation` →
`emitter.embeddedNode`, ADR 0022 §"Surface couplings", ADR 0035/0036).
That is one legitimate enforcement of the relation — a tunable penalty
spring — but it is the *only* one apeGmsh can emit.

The relation above is exactly what OpenSees `EQ_Constraint` expresses
(`equationConstraint cNode cDOF cCoef  rNode1 rDOF1 rCoef1 …`, the form
`cCoef·cDOF + Σ rCoef_i·rDOF_i = 0`).  **apeGmsh emits
`equationConstraint` nowhere.**  This ADR closes that gap.

**Why the constraint route matters (it is not redundant with penalty):**

- **Exact** enforcement — no penalty stiffness to tune, no conditioning
  penalty/no spurious compliance at the interface.
- **Explicit-safe** — via the fork `constraints LadrunoProjection`
  handler (ADR-30), enforcement is a *projection*, not a stiffness
  addition: momentum-conserving and **Δt-neutral**.  A penalty tie
  shrinks the critical time step; an equation tie under projection does
  not.  This is precisely the explicit-DRM / SSI context of ADR 0066.
- **Tie-force recovery** — the active projection handler exposes
  `ladrunoProjectionTieForce` (ADR-30 P3), the analogue of LS-DYNA's
  `*DATABASE_NCFORC` interface-force output (a follow-on, §8 here).

This maps the apeGmsh tie surface onto LS-DYNA
`*CONTACT_TIED_SURFACE_TO_SURFACE`: the LS-DYNA default *is* a kinematic
constraint (slave interpolated from the master segment), with penalty
("offset") variants.  apeGmsh's penalty route already covers the offset
variant; this ADR adds the kinematic/constrained one.

**Provenance (load-bearing — do NOT fork-extend the wrong thing):**
`EQ_Constraint` / `equationConstraint` is **upstream OpenSees**
(contributed by Yuli Huang, merged ~May 2025).  It is enforced by the
upstream `Lagrange` / `Penalty` handlers and the **fork**
`LadrunoProjection` handler (ADR-30; `buildGroups()` iterates
`getEQs()`).  It is **not** enforced by `Transformation`.  Therefore the
enforcement strategy (penalty / Lagrange / projection) is a property of
the **handler**, never of the constraint object — and we add nothing to
`EQ_Constraint` itself.

## Decision

### §1 — `enforce=` taxonomy: three coupling targets, one geometry

`tie` and `tied_contact` gain an `enforce=` selector.  All three targets
consume the **same** `InterpolationRecord` produced by `resolve_tie`;
only the emission differs.

| `enforce=` | Emits | Layer | Properties |
|---|---|---|---|
| `"penalty"` (default) | `ASDEmbeddedNodeElement` (217) | element | tunable `K`/`KP`; no handler dependency; current behaviour, **unchanged** |
| `"penalty_al"` | `LadrunoEmbeddedNode` (33006) | element | penalty **+ augmented-Lagrange + bipenalty** (the fork penalty richness) |
| `"equation"` | `equationConstraint` (EQ_Constraint) | constraint + handler | **exact**; handler = `Lagrange` (implicit) / `LadrunoProjection` (explicit) |

Default stays `"penalty"` so no existing model changes behaviour.

**`enforce="equation"` carries no penalty/AL/bipenalty knobs** — by
design.  AL exists to recover exactness from a penalty; bipenalty exists
to stop a penalty stiffness from crushing the explicit Δt.  Lagrange and
projection are exact and (projection) Δt-neutral, so both patches are
moot.  A `stiffness=`/`bipenalty=` passed with `enforce="equation"` is a
fail-loud error (§7 INV-3).

`"penalty_al"` (→ `LadrunoEmbeddedNode`) is the wiring that already
exists for `g.reinforce`/embedded; exposing it here is a small fan-out
and is *fork-only* (gated like the other `_FORK_ONLY_ELEMENTS`).  It may
ship after `"equation"` — the `"equation"` route is the ADR's reason for
being.

### §2 — Sixth `Emitter` Protocol method: `equationConstraint`

ADR 0022 locked the constraint surface at five methods (`equalDOF`,
`rigidLink`, `rigidDiaphragm`, `embeddedNode`, `mp_constraint_comment`).
This ADR is the next "architecture event": one new method.

```python
class Emitter(Protocol):
    # ... ADR 0022's five unchanged ...
    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None: ...
```

`retained` is an ordered list of `(rnode, rdof, rcoef)` triples.  Every
concrete emitter implements it (INV-2, mirroring ADR 0022 INV-4):

| Emitter | `equationConstraint` |
|---|---|
| `TclEmitter` | append `equationConstraint $cnode $cdof $ccoef  $rn $rd $rc …` |
| `PyEmitter` | append `ops.equationConstraint(cnode, cdof, ccoef, rn, rd, rc, …)` |
| `LiveOpsEmitter` | call `ops.equationConstraint(cnode, cdof, ccoef, *flat)` directly |
| `H5Emitter` | append a row to `/opensees/constraints/equationConstraint/` |
| `RecordingEmitter` | record `("equationConstraint", args)` |

Unlike `embeddedNode`, this is **not** an `element` — it is a
domain-level command (exactly like `equalDOF`), so it rides its own
Protocol method, not `emitter.element(...)` (same reasoning as ADR 0022
§"its own Protocol method").

### §3 — Expansion: one `equationConstraint` per tied DOF

In `build.py::_emit_one_interpolation`, an `InterpolationRecord` with
`enforce == "equation"` expands to one `equationConstraint` per
translational DOF in `rec.dofs`:

```python
for d in rec.dofs:                                  # [1,2,3]
    retained = [(int(m), d, -float(w))
                for m, w in zip(rec.master_nodes, rec.weights)]
    emitter.equationConstraint(int(rec.slave_node), d, 1.0, retained)
```

This is `1·u_d(slave) − Σ w_i·u_d(m_i) = 0  ⇔  u_d(slave) = Σ w_i·u_d(m_i)`.
The same `w_i` apply to each translational component `d`.  No element tag
is allocated (it is not an element).  `enforce in {"penalty",
"penalty_al"}` keep the existing `embeddedNode` / `LadrunoEmbeddedNode`
paths.

The `_check_embedded_rnode_count` 3/4-Rnode guard does **not** apply to
the equation route — an equation tie accepts any face arity the shape
functions support (tri3/quad4/tri6/quad8), so the master-node count is
`len(weights)`, not constrained to 3 or 4.

### §4 — Handler selection becomes EQ-aware

`apeSees._maybe_auto_emit_constraint_handler` today auto-emits
`Transformation` when MP constraints are present and the user declared no
handler.  **Transformation does not enforce `EQ_Constraint`** — so when
any `enforce="equation"` tie is present the auto-emit must upgrade:

| Model | Auto-emitted handler |
|---|---|
| MP only (equalDOF/rigidLink/diaphragm/penalty-tie) | `Transformation` (unchanged) |
| any **equation** tie, implicit analysis | `Lagrange` |
| any **equation** tie, explicit analysis | `LadrunoProjection` (fork) |

`Lagrange` / `LadrunoProjection` also enforce the equalDOF-family, so a
mixed model upgrades the *single* handler line consistently.  The
implicit/explicit split keys off the integrator the bridge is emitting;
when it cannot be determined, an explicit `tie_handler=` bridge kwarg
disambiguates and a fail-loud guard rejects a contradictory user handler
(e.g. user declared `Transformation` **and** an equation tie → raise with
an actionable message, not a silent wrong deck).

A user who explicitly declares `Lagrange` / `Penalty` /
`LadrunoProjection` is respected (no auto-emit), exactly as today.

### §5 — `LadrunoProjection` handler primitive (fork)

apeGmsh's `opensees/analysis/constraint_handler.py` has `Penalty` /
`Transformation` / `Lagrange` / `Auto`.  Add a fork `LadrunoProjection`
handler:

```python
@dataclass(frozen=True)
class LadrunoProjection(ConstraintHandler):
    """constraints LadrunoProjection <-verbose> <-projectICs> <-icTol $tol>"""
    verbose: bool = False
    project_ics: bool = False
    ic_tol: float | None = None
    def _emit(self, emitter, tag=None):
        args = []
        if self.verbose:      args.append("-verbose")
        if self.project_ics:  args.append("-projectICs")
        if self.ic_tol is not None: args += ["-icTol", self.ic_tol]
        emitter.constraints("LadrunoProjection", *args)
```

Fork-only (like `LadrunoKinematicCoupling`): gated so a stock-OpenSees
target fails loud rather than emitting an unknown handler.

### §6 — Data model: one field, threaded through

- `TieDef` / `TiedContactDef` (`_kernel/defs/constraints.py`): add
  `enforce: str = "penalty"`.
- `InterpolationRecord` (`_kernel/records/_constraints.py`): add
  `enforce: str = "penalty"`; included in `tag_rewrite_spec`'s
  non-tag passthrough so compose preserves it.
- `resolve_tie` / `resolve_tied_contact`: copy `defn.enforce` onto every
  emitted `InterpolationRecord`.

No new record type; the existing `InterpolationRecord` already carries
`slave_node`, `master_nodes`, `weights`, `dofs`.

### §7 — Invariants

- **INV-1.** `apeSees(fem).tcl/py(...)` for a model with an
  `enforce="equation"` tie produces a **runnable** deck that enforces the
  interpolation: under `Lagrange` a rigid master-face motion is
  reproduced at the slave to ~1e-12 (a new integration test, mirroring
  ADR 0022 INV-1).
- **INV-2.** Every concrete emitter implements `equationConstraint`
  (Protocol contract; partial implementations forbidden).
- **INV-3.** `enforce="equation"` with any penalty-only kwarg
  (`stiffness`, `stiffness_p`, `bipenalty`, …) raises at declaration —
  those knobs are meaningless for an exact constraint.
- **INV-4.** When an `enforce="equation"` tie is present, the emitted
  handler is never `Transformation` (it cannot enforce EQ).  Auto-emit
  upgrades; a contradictory explicit user handler fails loud.
- **INV-5.** The equation expansion is deterministic: rows emitted in
  `(slave_node, dof)` order (ADR 0021 canonical-bytes property).

## Alternatives rejected

- **Add bipenalty/AL knobs to `EQ_Constraint`.**  Rejected — wrong layer
  (enforcement belongs to the handler) *and* `EQ_Constraint` is upstream
  code (fork-extending it is ledger/merge debt).  The penalty richness
  already exists as the `"penalty_al"` element route
  (`LadrunoEmbeddedNode`).
- **Replace the penalty route with the equation route.**  Rejected — the
  penalty element is robust, handler-independent, and the right tool when
  a compliant interface is wanted.  The two are complementary (mirrors
  LS-DYNA's constrained vs offset-penalty tied contact).  `enforce=`
  selects.
- **Ride `equationConstraint` through `emitter.element(...)`.**  Rejected
  for the same reason ADR 0022 rejected it for `equalDOF`: it is a
  domain-level command, not an element; faking a type-token breaks the
  typed-primitive vocabulary.
- **Auto-pick the handler silently without a fail-loud on contradiction.**
  Rejected — a user `Transformation` + an equation tie is a wrong deck
  OpenSees would run (silently dropping the EQ); INV-4 raises instead.

## Consequences

**Positive**
- apeGmsh gains the exact, explicit-safe, force-recoverable tied
  interface that matches LS-DYNA's constrained tied contact — reusing the
  penalty route's geometry wholesale (zero new geometry code).
- Unblocks a **non-matching** soil-structure interface under **explicit**
  DRM (ADR 0066) — the penalty route's Δt penalty made that impractical.
- One new Protocol method; the constraint fan-out (ADR 0022) absorbs it
  cleanly.

**Negative (acknowledged)**
- `Emitter` Protocol widens to six constraint methods; every concrete +
  future emitter implements `equationConstraint`.
- Handler auto-selection gains an implicit/explicit branch (the one piece
  of real logic); a contradictory explicit handler now raises.
- `LadrunoProjection` and `"penalty_al"` are fork-only — a stock target
  fails loud on those (consistent with existing fork-element gating).

**Neutral**
- H5 schema gains `/opensees/constraints/equationConstraint/` (minor bump
  per ADR 0023), symmetric with the other constraint groups (ADR 0014).
- `enforce` is one additive field on two defs + one record; no new record
  type.

## Open items

1. **Implicit/explicit detection for handler auto-emit** — key off the
   emitted integrator; define the `tie_handler=` override + the
   fail-loud contradiction guard precisely during impl.
2. **Cross-partition EQ groups** under `LadrunoProjection` — ADR-30 built
   cross-rank equalDOF/diaphragm; verify EQ groups specifically (newest
   P3 surface, fork #312).  A parallel test, not an assumption.
3. **Tie-force recovery** (`ladrunoProjectionTieForce`) — expose a
   `results`/recorder helper (LS-DYNA `*DATABASE_NCFORC` analogue).
   Phase 2; not required for the tie itself.
4. **H5 round-trip** of `enforce` + the `equationConstraint` group —
   schema 2.x bump, forward-only.

## Phasing / implementation plan

- **P0** — `enforce=` field on `TieDef`/`TiedContactDef` +
  `InterpolationRecord`; threaded through `resolve_tie`.  Pure data;
  unit-tested off-session.
- **P1** — `Emitter.equationConstraint` Protocol method + all five
  concrete emitters (INV-2).  The "architecture event".
- **P2** — `_emit_one_interpolation` equation branch (§3) +
  `enforce="equation"` on `tie`/`tied_contact` + INV-3 guard.  Tcl/py
  deck test.
- **P3** — EQ-aware handler auto-emit (§4) + `LadrunoProjection` handler
  primitive (§5) + INV-4 fail-loud.  Implicit `Lagrange` runnable test
  (INV-1).
- **P4** — `"penalty_al"` → `LadrunoEmbeddedNode` fan-out (fork-only).
- **P5** — explicit `LadrunoProjection` + DRM integration test
  (non-matching soil/structure under explicit DRM, ADR 0066) +
  tie-force helper (Open item 3); cross-partition EQ (Open item 2).

## References

- `src/apeGmsh/_kernel/resolvers/_constraint_resolver/_resolver.py` —
  `resolve_tie` (projection + weights, reused unchanged).
- `src/apeGmsh/_kernel/records/_constraints.py` — `InterpolationRecord`.
- `src/apeGmsh/opensees/_internal/build.py` —
  `_emit_one_interpolation` / `_emit_surface_couplings`,
  `emit_mp_constraints`.
- `src/apeGmsh/opensees/emitter/base.py` — the `Emitter` Protocol (ADR
  0022's five constraint methods; this ADR's sixth).
- `src/apeGmsh/opensees/analysis/constraint_handler.py` — `Penalty` /
  `Transformation` / `Lagrange` / `Auto` (+ the new `LadrunoProjection`).
- `src/apeGmsh/opensees/apesees.py` —
  `_maybe_auto_emit_constraint_handler` (made EQ-aware, §4).
- OpenSees fork: `SRC/domain/constraints/EQ_Constraint.{h,cpp}` (upstream
  Huang 5/2025); `SRC/analysis/handler/LadrunoProjectionHandler.cpp` +
  `LagrangeEQ_FE` / `PenaltyEQ_FE`; `equationConstraint` Tcl/Py wrappers
  (`TclWrapper.cpp:518`, `PythonWrapper.cpp:871`);
  `OPS_LadrunoProjectionTieForce` (ADR-30 P3).
- ADR 0022 (MP-constraint emission fan-out — the five-method surface this
  widens), 0035/0036 (`ASDEmbeddedNodeElement` penalty route), 0041
  (chain-phase routing), 0066 (H5DRM authoring — the explicit-DRM
  consumer).
