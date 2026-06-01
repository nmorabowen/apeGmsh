# ADR 0052 — Staged reference-position contract for constraints and elements

**Status:** Proposed (2026-06-01). Extends the SSI ADR set
([0028](0028-initial-stress-via-parameter-ramping.md) /
[0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0034](0034-stage-bound-bcs-and-recorders.md)). Grounded in the fork
design note `Ladruno_implementation/constraints_reference_position.md`
and two source cross-checks against the fork `SRC/` tree (below). No
H5 schema bump. May widen the `Emitter` Protocol for the HOLD-support
constant pattern (TBD at implementation).

## Context

When a model object is added *after* the domain has already deformed
(the staged-construction case ADR 0029/0030 enable), **which
configuration it acts in is decided per object class, not globally.**
The fork note `constraints_reference_position.md` establishes the
asymmetry, and because the bridge *authors the deck*, the bridge is
exactly the layer that chooses which side of each asymmetry we land
on. The relevant facts (all verified against the fork `SRC/`, not only
source-read):

- **Single-point (`fix`, `sp`) are absolute / reference-frame.** A
  mid-stage `fix` drives the DOF back to its `t = 0` position (`fix`
  is only a homogeneous SP at value 0;
  `OPS_HomogeneousBC → SP_Constraint(node, dof, 0.0, true)`,
  `SP_Constraint.cpp:80`). `fix` *cannot* express "hold this DOF where
  it is now" — it has no value lever.
- **Multi-point (`equalDOF`, `rigid*`) are incremental /
  deformed-frame.** Added mid-stage they capture `Uc0`/`Ur0` and tie
  only future increments → the standing offset is preserved with zero
  initial force, no flag required
  (`MP_Constraint.cpp:294-313`; handlers enforce `Uc - Uc0 = C·(Ur -
  Ur0)`). **This is already the staged-construction behavior we want.**
- **Line elements** (truss, beam-column) are born **stress-free at the
  deformed shape** by default: `Truss` captures `initialDisp` when
  `useInitialDisp` (default true) and the committed disp is nonzero
  (`Truss.cpp:92,407-419`); beam-columns do the same one layer down in
  the `CrdTransf` (`LinearCrdTransf3d.cpp:211-227`).
- **Continuum elements** (`FourNodeQuad`, `stdBrick`, tets) have **no
  such hook** — `ε = B·u` is computed straight from the node pointers
  against `t = 0` (`FourNodeQuad.cpp:578-581,1647-1650`), so an element
  appended at `u = d` is born carrying the full accumulated strain as
  spurious stress.

So the bridge needs **one lever for constraints** (the SP value) and
**two routes for elements** (line vs continuum). MP needs nothing.

### Source cross-check (the two findings that shaped the decision)

Two runtime-mechanics claims were verified in the fork `SRC/` before
committing to an API; both came back decisive.

**(1) `sp` requires a load pattern; `-const` pins; openseespy cannot
subtract-init.**

- A bare `ops.sp(...)` with no active pattern and no `-pattern` flag
  **fails outright** — `OPS_SP` errors "no current pattern is set"
  (`interpreter/OpenSeesPatternCommands.cpp:1081-1089`). `sp` always
  lives *inside* a `LoadPattern`
  (`Domain.cpp:928-970`, two-arg overload); only `fix` attaches to the
  domain directly (`Domain.cpp:566`).
- `-const` sets `isConstant`, which skips the
  `valueC = loadFactor*valueR` scaling so the prescribed value never
  moves with pseudo-time (`SP_Constraint.cpp:330-338`). A non-const
  `sp` ramps with the pattern's `timeSeries` factor
  (`LoadPattern.cpp:405-427`).
- `retZeroInitValue` defaults `true` for both `fix` and `sp`
  (`SP_Constraint.h:50`), and the **openseespy `sp -subtractInit` flag
  is a no-op** — it sets `retZeroInitValue = true` (the default)
  instead of `false` (`OpenSeesPatternCommands.cpp:1065-1067`),
  diverging from both Tcl parsers
  (`TclModelBuilder.cpp:3766-3768`, `constraint.cpp:392-393`). So there
  is **no increment path in openseespy**; a `sp` value is always
  treated as an absolute total. That is exactly right for "pin at the
  current displacement `d`": read `d = nodeDisp(...)` and pass the
  absolute value.

**(2) `InitialStateAnalysisWrapper` cannot be scoped to a single
mid-stage element — it corrupts already-equilibrated elements.**

- `InitialStateAnalysis off` calls `theDomain->revertToStart()`
  *before* flipping the global flag back
  (`interpreter/OpenSeesMiscCommands.cpp:1314-1316`), and
  `Domain::revertToStart()` walks **every element in the domain**
  (`Domain.cpp:2214-2230`). Each wrapper instance then runs
  `mEpsilon_o += mStrain` while the global `ops_InitialStateAnalysis`
  is still true (`InitialStateAnalysisWrapper.cpp:169-177`). There is
  no tag filter anywhere on this path, and the update *accumulates*.
- The toggle is a single process-global bool
  (`OPS_Globals.h:75`; `Domain.cpp:93`;
  `InitialStateParameter.cpp:57-66`). Used mid-stage to neutralize one
  newly-born element, it **re-zeros the reference config of every
  already-equilibrated element** — the standard geotech idiom only at
  the global `t = 0` initial-state phase, never for a mid-stage subset.
- `InitStrainNDMaterial` is the safe per-instance alternative
  (`material/nD/InitStrainNDMaterial.cpp:156-163`, never touched by
  domain-wide `revertToStart`), **but** its offset `epsInit` is a
  fixed construction-time constant (one per material tag, carried by
  `getCopy` to all GPs), whereas the birth strain `ε₀ = B·u₀` is
  runtime-dependent and per-GP. So InitStrain is exact only for
  constant-strain (single-GP / CST) elements *and* still needs an
  emit-time-unknown offset. There is **no in-tree, per-element-scoped,
  runtime-capturing stress-free-birth mechanism for general
  continuum.**

## Decision

### 1. Two distinct verbs: `s.support` (HOLD, emits `sp`) and `s.fix` (ANCHOR, emits `fix`)

A stage-bound single-point support has two distinct intents, and they
get two distinct methods rather than one method with a mode flag (a
method called `fix` that emits `sp` and does *not* return to zero is
surprising):

- **`s.support(...)` — HOLD.** Freeze the DOF at its current deformed
  value, zero initial force. The staged-construction case (install a
  support to hold what's there). Emits `sp`.
- **`s.fix(...)` — ANCHOR.** Genuinely return the DOF to its `t = 0`
  position, with a physical restoring force. Emits `fix`. **Unchanged
  from ADR 0034** — no behavior inversion for existing callers.

`fix` can only express ANCHOR (it is a homogeneous SP at value 0).
HOLD needs the SP value lever, so `s.support` emits:

```python
# emitted into the stage's dedicated constant HOLD pattern:
d = ops.nodeDisp(n, dof)
ops.sp(n, dof, d, '-const')     # absolute value, isConstant → pinned, zero initial force
```

- **HOLD requires a constant pattern.** Per finding (1) a bare `sp`
  fails. The bridge emits, per stage, **one dedicated `Plain` pattern**
  referencing **one shared global `Constant` `timeSeries` (factor
  1.0)** to carry that stage's HOLD supports — created lazily, only
  when the stage has ≥1 `s.support` (see Resolved decision §3). Reuses
  the stage-scoped pattern machinery + deterministic tag allocator from
  ADR 0051; whether it needs a new `Emitter` method or composes over
  existing ones is an implementation detail.
- **Absolute value, by design.** We rely on `retZeroInitValue=true`
  (openseespy has no `-subtractInit`) and pass the absolute `d` read at
  runtime. The `nodeDisp` resolves in the live deck after the prior
  stage's `analyze` + `loadConst` — this is the idiomatic edge: the
  deck is executable, so it captures runtime state inline; we never
  need the displacement at generation time.
- **Base build (`t = 0`) stays `fix`.** At `t = 0`, `u = 0`, so HOLD
  and ANCHOR are identical and `fix` is the clean, pattern-free,
  byte-stable idiom. The `sp` policy is **stage-bound only** — it does
  not touch the global pre-stage block.
- **MP is untouched.** `equalDOF` / `rigidLink` / `rigidDiaphragm`
  already hold the deformed offset with zero force (finding above); the
  "always sp" rule is **single-point only** and must not leak into the
  MP fan-out (ADR 0022/0027).
- **Transient caveat (momentum-kill, not value-jump).** A HOLD support
  installed in a transient stage injects no *displacement-jump* impulse
  (the value equals the current position), but rigidly pinning a moving
  DOF cuts its velocity `u̇⁻ → 0` in one step → the reaction absorbs
  `m·u̇⁻` and `½m u̇²` is removed discontinuously. This is **not**
  fixable by ramping the value (there is no value trajectory); the only
  genuine mitigations are physical (install when quiescent, or model a
  stiff spring + dashpot). The `s.support` docstring documents this and
  points dynamic users at soft-support modeling; the bridge offers no
  "ramped HOLD" knob because for a constant-value pin it would be a
  misleading no-op (see Resolved decision §2). Imposed-motion ramping
  (the value-jump case) already lives in `imposed_displacement` (ADR
  0031).

### 2. Frame element birth (Route A) — `useInitialDisp` + commit-first

Stage-activated line elements (truss, beam-column) are born stress-free
at the deformed shape for free. The bridge's only obligations:

- **Commit-first** is already satisfied by the existing
  `stage_close → loadConst -time 0 → domain_change` sequencing (ADR
  0029/0030) — the deformation is committed before the next stage's
  elements are appended.
- Emit `Truss -useInitialDisp 1` **explicitly** for stage-activated
  trusses (deterministic, immune to an upstream default flip);
  beam-columns get it automatically via the `CrdTransf`. Offer
  `-useInitialDisp 0` as the rare "born referenced to `t = 0`" opt-out.

### 3. Continuum element birth (Route B) — deferred to the gradient wrapper; fail loud

Per finding (2) there is no safe in-tree mechanism for general
mid-stage continuum stress-free birth: `InitialStateAnalysisWrapper` is
globally destructive, and `InitStrain` cannot capture the runtime
per-GP birth strain. The architecturally correct seam is the
**per-GP `F₀`-at-`setDomain` gradient-displacement wrapper**
(`F_rel = F·F₀⁻¹`, the fork's `[[staged_deformation_gradiend]]` note),
which captures per-instance at birth with no global flag — and is under
development.

Therefore:

- Continuum stress-free birth is **unsupported in the bridge** until
  that wrapper lands in the fork.
- The bridge **fails loud** if a stress-free continuum activation is
  requested, pointing at this ADR. It must **never** emit a global
  `InitialStateAnalysis on/off` bracket to achieve it — that path
  silently corrupts every already-equilibrated element in the domain.
- When the gradient wrapper ships, Route B wires the activated
  continuum element to the wrapped material; a small-strain variant is
  a wrapper-side concern, not a bridge-side InitStrain hack.

## Docstring plan

The reference-position semantics live in the method docstrings
(consistent with the DOC-2 docstring sweep), and **land with each
implementation slice** — never ahead of the behavior, so a docstring
never documents an emit the code doesn't yet produce.

- **`s.support` (new, HOLD)** — document HOLD semantics, the `-const`
  constant-pattern emission, the absolute-value/`nodeDisp` runtime
  capture, and the transient momentum-kill caveat (with the
  soft-support pointer). Lands with slice 1.
- **`s.fix` (ANCHOR, unchanged)** — one line clarifying it returns the
  DOF to `t = 0` with physical force, and contrasting it with
  `s.support`. Lands with slice 1.
- **`s.activate`** — document Route A (frames: stress-free at deformed
  shape, commit-first, `useInitialDisp`) and the Route B fail-loud
  refusal for stress-free continuum, citing this ADR. Lands with
  slices 2–3.
- **`apeSees.fix` (global)** — one line clarifying base-build `fix`
  stays absolute/`t = 0` and the `sp` policy is stage-bound only.

## Alternatives considered

- **"Always `sp`" everywhere (base + staged).** Rejected — at `t = 0`,
  `fix` and `sp(value=0)` are identical in effect; rewriting base-build
  `fix` to `sp` churns the deck, pulls every support into a pattern,
  and breaks byte-stability for zero behavioral gain. The value lever
  only matters once the model has deformed, so the policy is
  stage-bound.
- **`InitialStateAnalysisWrapper` for mid-stage continuum birth.**
  Rejected on source: the `off` path is domain-wide and accumulating
  (finding 2), so it corrupts already-equilibrated elements. Only valid
  at the global `t = 0` initial-state phase.
- **`InitStrainNDMaterial` narrow path now (constant-strain elements +
  runtime offset injection).** Rejected as scope — exact only for
  single-GP elements, the offset is emit-time-unknown, and
  `setParameter` is per-element-broadcast not per-GP. A partial path
  that tempts misuse on multi-GP/finite-strain elements; the gradient
  wrapper is the correct seam.
- **Inferring HOLD vs ANCHOR from context.** Rejected — intent is not
  derivable from the model state. Made explicit instead via two
  distinct verbs (`s.support` / `s.fix`, Resolved decision §1).
- **One method with a `hold=` mode flag.** Rejected — a method named
  `fix` that emits `sp` and does not return the DOF to zero is
  surprising; two verbs read truer at the call site (Resolved decision
  §1).

## Consequences

- **New emission:** a per-stage dedicated `Plain` HOLD pattern + one
  shared global `Constant` series carrying HOLD `sp` lines, installed
  after the stage's topology and prior committed state are valid.
  Possible `Emitter` widening (TBD).
- **No behavior change for existing callers.** `s.fix` keeps its ADR
  0034 ANCHOR meaning; HOLD is the *new* `s.support` verb. Nothing that
  works today changes.
- **Fail-loud guard** for stress-free continuum activation; a registry
  flag distinguishing finite-strain/continuum element classes feeds
  both this guard and the future wrapper wiring.
- **No H5 schema bump.** Constraints and element-birth policy are
  emit-side only.
- **openseespy `-subtractInit` no-op** (finding 1) is a fork/upstream
  inconsistency we depend on *not* needing — worth an upstream issue,
  but the absolute-value design sidesteps it.

## Resolved decisions

1. **Naming (resolved 2026-06-01).** Two verbs, not a mode flag:
   **`s.support`** = HOLD (emits `sp`), **`s.fix`** = ANCHOR (emits
   `fix`, unchanged from 0034). Avoids the surprise of a `fix` that
   doesn't return to zero and keeps existing callers intact.
2. **Transient install (resolved 2026-06-01): document-only, no
   built-in ramp.** A HOLD pin has no value trajectory, so the
   transient impulse is a momentum-kill, not a value-jump — ramping the
   value would be a misleading no-op. Document the caveat on
   `s.support` and point dynamic users at soft-support modeling (stiff
   spring + dashpot) or installing at a quiescent instant. The
   value-jump ramp (imposed motion) already lives in
   `imposed_displacement` (ADR 0031).
3. **HOLD pattern granularity (resolved 2026-06-01): one dedicated
   `Plain` pattern per stage + one shared global `Constant` series.**
   Per-stage (not per-group, not one-global) because nothing keys off
   HOLD pattern identity — removal is `remove sp $node $dof`
   (`s.remove_sp`), all HOLD `sp` are identical-in-kind, and they
   persist across stages for free (`loadConst` freezes,
   `wipeAnalysis` spares the domain). Per-stage matches the ADR 0051
   stage-scoped pattern model + deterministic (INV-1 byte-parity)
   allocator and references that stage's freshly-captured `nodeDisp`.
   Created lazily (only when a stage has ≥1 `s.support`), and kept
   **separate** from the user's `s.pattern` load patterns so support
   and load lifetimes don't couple.

## Open questions

None blocking. Implementation slicing (slice 1 = `s.support` HOLD path
+ constant pattern + docstrings; slices 2–3 = frame `useInitialDisp` +
continuum fail-loud guard) to be planned when work starts.
