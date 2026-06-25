# Deferred items

Things we've discussed and consciously held for later. Not bugs, not
backlog — concrete capability ideas that earn their own scope when a
real model needs them.

## Node aggregator capabilities (v1 ships lean)

`Node` ships:

- `.coords`, `.tag`
- `.fix(dofs=...)` (model-level)
- `.mass(values=...)` (model-level)
- `.load(forces=...)` inside a pattern context
- `.region(name)` — assign the node to a named OpenSees Region;
  ships on top of the `Emitter.region()` Protocol method from
  ADR 0024.

Held for later, in priority order:

1. **`.disp_history()` / `.element.disp_history()`** — pull recorder
   output for a node or element after analysis.  Requires a
   registered :class:`Recorder` matching the query; the typed
   query layer composes over the existing Recorders system.
2. **`.get_reaction()`** — query post-analysis reaction forces.
3. **`.coupled_dofs()`** — list MP_Constraints that touch this node
   (rigid links, equal_dof, etc.).
4. **`.partition`** — owning partition tag for parallel runs.

## Asymmetric-section warning at `geomTransf` build

Today's CS resolver in `_orientation.py::resolve_vecxz` emits the
correct vecxz, but the leg sign-flip on arches is inherent. For
symmetric sections (W12, RHS, circular HSS) the sign flip is
harmless. For asymmetric sections (channels, angles, T-sections)
it produces inconsistent physical orientation across the legs.

Deferred capability:

- At build time, detect when an element using an asymmetric section
  hits the degenerate branch (the `else` arm at
  `_orientation.py:462-465`).
- Emit a warning naming the affected PG and suggesting `roll_deg`
  on one of the legs.

Lives in `_internal/build.py::emit_transform_specs` (around the
`compute_vecxz_for_element` call at line 601) once we get there.

**Load-bearing blocker — needs a section asymmetry predicate.**
The detection requires knowing "is the section attached to this
element asymmetric." Today the section primitives (`Fiber`,
`Elastic`, `LayeredShell`, ...) carry no `is_asymmetric` field and
class name alone is not a signal — a `Fiber` section can hold a W14
patch (symmetric) or a channel/angle patch (asymmetric). The clean
trigger for this work is one of:

- A future typed `Channel` / `Angle` / `Tee` section primitive
  that carries the discriminator natively.
- apeSteel-section metadata being plumbed into the bridge so the
  `apeSteel.SingleAngleSection` / `ChannelSection` / `TeeSection`
  classes propagate `is_asymmetric=True` through to the bridge.
- A geometric-fiber-layout audit on `Fiber` sections that infers
  asymmetry from the patch / fiber positions (more fragile).

Don't implement against the current section types — a class-name
whitelist would either be too noisy (warn on every `Fiber`) or
miss cases (only typed primitives, none of which exist yet).

## Custom convergence / retry recipes

`apeSees` ships recipes for the common cases:

- `Static.linear(steps=...)`
- `Static.load_control(...)`
- `Static.disp_control(...)`
- `Transient.newmark(...)`
- `Transient.hht(...)`

Held for later:

- A `RetryStrategy` primitive that lets users compose convergence
  recovery (line search → reduce step → switch algorithm). For now,
  users who need this drop to live mode and write the loop
  themselves: `bm = ops.build(); bm.run_live(analysis=None)`.

## Multi-pattern aggregation

`Pattern` instances aggregate their loads after the `with` block
closes. A future capability is cross-pattern queries on a Node:
"show me every load this node has received across all patterns."
Defer until users ask.

## ANSYS / Code_Aster / JSON emit targets

The `Emitter` Protocol is designed to support more targets (P8).
None planned for v1. The first non-OpenSees target is the test of
whether the abstraction was right; we'll know when we get there.

## Code-generated namespace methods

The signature duplication between typed dataclass and namespace
method (ADR 0003) is hand-written for v1. If it becomes painful as
the type catalog grows, generate the namespace from the typed
classes via introspection.

## Staged-analysis follow-ups

The SSI feature set ([ADR
0028](decisions/0028-initial-stress-via-parameter-ramping.md) /
[0029](decisions/0029-staged-analysis-context-manager.md) /
[0030](decisions/0030-stage-bound-topology-activation.md) /
[0031](decisions/0031-ssi-convenience-helpers.md) /
[0034](decisions/0034-stage-bound-bcs-and-recorders.md)) ships the
declarative `ops.stage(name)` / `s.activate(pgs=)` /
`s.fix(...)` / `s.mass(...)` / `s.region(...)` / `s.recorder(...)` /
`ops.initial_stress(...)` surface, the runnable Tcl / Py text
emit, the combined partitioned + staged emit (Phase SSI-2.C), and
the four-validator ownership-tier surface (Phase SSI-2.D).
Follow-ups still explicitly deferred:

### Live execution of staged models

`apeSees.analyze` and `apeSees.eigen` refuse staged models with
`NotImplementedError`. `LiveOpsEmitter.stage_open` /
`stage_close` raise. Lifting requires staging the analysis-chain
re-binding, per-stage `analyze` loops, `loadConst` / `wipeAnalysis`
interleaving, and hook-list clearing inside the live emitter. The
contract is documented (ADR 0029 §"Stage-close cleanup contract");
the implementation is the missing piece.

The workaround is `ops.tcl(p, run=True)` / `ops.py(p, run=True)` —
the OpenSees subprocess runs every stage's analyze loop and
inter-stage cleanup as part of executing the deck. The Cerro
Lindo migration uses this; live execution is the ergonomic gap.

Lives in `emitter/live.py::stage_open` / `stage_close` (currently
raise); `apesees.py::analyze` / `eigen` (currently refuse).

### H5 archival of staged structure + initial-stress

`H5Emitter.addToParameter` / `step_hook_ramp` / `stage_open` /
`stage_close` / `domain_change` are no-ops — staged structure and
the in-situ stress ramp are not persisted by the per-zone
`/opensees/` schema today. Because a silent-drop H5 round-trip
would produce a non-staged flat model that no longer matches the
declared one, `apeSees.h5(path)` is **guarded**: it raises
`NotImplementedError` (#313) when `self._stage_records` or
`self._initial_stress_records` is non-empty, pointing the user at
`ops.tcl(path)` / `ops.py(path)` instead. The H5 emitter-side
no-ops remain reachable from direct `H5Emitter` unit tests outside
the bridge; the guard is at the user-facing `apeSees.h5`.

A future schema bump (per [ADR
0023](decisions/0023-per-zone-schema-versioning.md)) bringing
`opensees_schema_version` from `2.11.0` → `2.12.0` would persist
per-stage primitive lists and initial-stress records under
`/opensees/stages/` and `/opensees/initial_stress/`, lift the
guard, and restore round-trip parity. Open design questions
before that lands:

- Persistence shape for the per-step ramp proc. Three plausible
  readings: serialise the `(name, targets, n_steps_to_full,
  phase)` tuple (lossless re-emit on read); persist the rendered
  Tcl/Py body bytes (only useful for textual re-emit, not for
  live replay); persist the `InitialStressRecord` pre-resolve (the
  cleanest — re-runs `emit_initial_stress_global` on read).
- How a viewer would render staged state. `Results.viewer()`
  currently shows a single time-history slab; staged decks have a
  per-stage analyze loop with reset pseudo-time. Likely needs a
  `Stage` discriminator on the slab.

Lives in `apesees.py::h5` (the bridge-side guard, #313) and
`emitter/h5.py::addToParameter` / `step_hook_ramp` / `stage_open`
/ `stage_close` / `domain_change` (the schema-side no-ops).

### ✅ `remove sp` / `remove element` / mass overwrite / time-state mutators (Phase SSI-2.E, SHIPPED 2026-05)

Phase SSI-2.E lifted the append-only restriction with five new
`_StageBuilder` verbs:

- `s.remove_sp(*, pg=, nodes=, dofs=)` — emits `remove sp $node
  $dof`; releases prior-tier SP constraints.
- `s.remove_element(*, pg=, elements=)` — emits `remove element
  $tag`; drops elements from the Domain mid-analysis.
  `elements=` is FEM eids per the recorder.Element convention,
  translated through `fem_eid_to_ops_tag` at emit time.
- `s.mass(..., overwrite=True)` — relaxes validator V2 when the
  user opts in to mid-run mass overwrite (acknowledging the
  `Domain::setMass` silent overwrite).
- `s.set_time(t)` / `s.set_creep(on)` — emit `setTime` /
  `setCreep 1|0` right after `stage_open`; overrides the prior
  stage_close's `loadConst -time 0.0` reset / toggles creep for
  time-dependent concrete materials.
- `s.reset()` — emits the bare OpenSees `reset` between the
  stage's recorder declarations and its analyze loop.

Validators V5 (remove_sp targets must reference a prior-tier SP)
and V6 (remove_element targets must reference a prior-tier
element) gate the new verbs at build time. V2 was widened to
subtract `s.remove_sp` targets from the fix alive set so the
atomic-replace pattern (release prior + re-fix in the same
stage) passes both validators.

The "removal BEFORE new BCs" design question landed on **BEFORE**
(the conservative reading): within a stage block, removals emit
between topology and the new fix/mass/region/MP block. Atomic
replace works by construction.

Lives in `_internal/build.py` (`SPRemovalRecord`,
`ElementRemovalRecord`, `MassRecord.overwrite`,
`StageRecord.{remove_sp_records, remove_element_records, set_time,
set_creep_on, pre_analyze_reset}`), `apesees.py`
(`_StageBuilder.{remove_sp, remove_element, set_time, set_creep,
reset}`; V5 / V6 validators), and per-emitter implementations of
the five new Protocol methods (`set_time` / `set_creep` /
`reset` / `remove_sp` / `remove_element`). Tests in
`tests/opensees/unit/test_stage_ssi_2e_mutators.py`.

Live execution of staged models that use these verbs is still
deferred (the live emitter raises at `stage_open`); the verbs
work for the Tcl / Py emit paths and capture into the per-stage
H5 buckets (ADR 0055 Phase 2, schema 2.18.0 — the old
"staged-H5 stays fail-loud" remark here predates the lift;
partitioned staged archives followed at Phase 5 / 2.19.0).

### Filtered recorders under stages emit one global region (CORRECT — deck-size only)

Stage-bound MPCO/Ladruno recorders claim through `s.recorder(spec)`
and fall through `emit_recorder_spec`'s materialize path, emitting
one whole-resolved `region` (single shared tag) in global scope
rather than the per-rank INV-4 intersection used by the non-staged
path.

**This is correct, not a bug** (investigated 2026-06-18). OpenSees
`MeshRegion` silently filters region members to each rank's local
domain — `setElements` / `setNodes` guard every tag with
`if (theEle != 0)` / `if (theNode != 0)`, always return success,
and emit no warning (`MeshRegion.cpp:182-233`, `:79-144`;
source- and run-verified). So under SPMD each rank's region becomes
its owned intersection, and the per-rank `.ladruno` / `.mpco` files
merge by the single shared tag — identical results to the per-rank
path. The non-staged INV-4 per-rank emission is a **deck-size
optimization** (keeps each rank's region line O(model/np) vs. an
O(model) global line every rank parses — the ADR 0061 resident-text
concern), NOT a correctness requirement.

Optional lifting (deck-size only): route stage filtered recorders
through `_plan_partitioned_mpco_recorders` so the staged path gets
the same O(model/np) per-rank win.

Trigger this work ONLY if O(model) staged region text becomes a
measured problem under MP. Results are already correct. Today's
call sites use whole-model MPCO
(no filter) or filtered MPCO at global scope only.

### `s.tied_contact` / `s.mortar` stage-bound claim

The Phase SSI-2.D extension (ADR 0034 §5a) ships nine claim-by-
name methods on `_StageBuilder` (`s.embedded`, `s.equal_dof`,
`s.rigid_link`, `s.rigid_diaphragm`, `s.kinematic_coupling`,
`s.tie`, `s.distributing`, `s.node_to_surface`,
`s.node_to_surface_spring`) — but `s.tied_contact` and `s.mortar`
are intentionally omitted:

- **`s.tied_contact`** — `tied_contact` records resolve to a
  `SurfaceCouplingRecord` whose nested `slave_records: list[
  InterpolationRecord]` is what actually emits via the
  `interpolations()` iterator at global emit time. The
  `_ExcludeClaimedConstraints` filter operates on outer-record
  identity (`id(rec)` of the `SurfaceCouplingRecord`); the nested
  slaves have distinct ids and slip through the global exclusion
  filter. Result: claiming a `tied_contact` by id would leave the
  slave interpolations emitting in BOTH the global pre-stage pass
  AND the stage block — double emission, which crashes OpenSees
  with duplicate element tag.
- **`s.mortar`** — as of ADR 0073 `g.constraints.mortar(...)` is a deprecated
  alias delegating to the fork contact-tie, so it now resolves to a
  `ContactRecord` on `fem.elements.contacts` (a serial-only subsystem emitted
  by `emit_contacts`), NOT a claimable MP `SurfaceCouplingRecord`. The
  stage-bound claim version stays deferred — the claim model targets MP
  constraints, and there is no stage-claimable contact record.

Lifting `s.tied_contact`: extend `_ExcludeClaimedConstraints.
interpolations()` to also filter nested slaves when their parent
`SurfaceCouplingRecord` is claimed (probably by carrying a
parent-id map alongside the claim set), or claim individual
slave InterpolationRecord ids directly (requires the user to
name the slaves, which isn't ergonomic).

Trigger this work when an SSI deck legitimately needs to stage-
bind a `tied_contact` interface — most lining/excavation models
use `embedded` (volume host) or `tie` (surface host) instead.

### Implicit promotion of `g.constraints.*` records to stages (Path A)

The Phase SSI-2.D extension shipped CLAIM-by-name (Path D2 from
the scoping conversation) rather than implicit derivation in
`compute_stage_ownership` (Path A). The forgotten-claim failure
mode — user adds a new embed at apeGmsh time, forgets to claim
it inside the appropriate stage block, deck routes it to the
global pre-stage pass and crashes when stage-bound nodes don't
exist yet at parse time — was the principal critique against the
shipped approach. Today the V1-style ownership-tier validator
catches the resulting "stage N node referenced by global record"
failure with a clear offender list, but the user still has to
edit the stage block to fix it.

Lifting via implicit promotion would extend
`compute_stage_ownership` to walk constraint records and promote
them to a stage when ALL referenced nodes resolve to that stage's
node ownership (and fail loud on cross-stage spans). Architectural
concerns flagged in the scoping critique: (a) it's a "third
pattern" relative to ADR 0034's PUSH/PULL/CLAIM trichotomy;
(b) PG is the authoring spine — implicit promotion arguably
matches the existing pattern (materials/sections/loads/masses
all derive from PG ownership). The CLAIM-by-name shipping
decision was driven by the wish to keep the architecture surface
narrow (and to ship sooner for the Cerro Lindo SSI V5 forcing
function).

Trigger this work if the forgotten-claim failure becomes a real
authoring footgun across SSI decks (more than the occasional
"oops, forgot `name=`"). Likely won't lift soon — CLAIM-by-name
covers the canonical SSI workflow ergonomically, and Path A's
"third pattern" concern from the architecture critics still
holds.

Lives in `_internal/build.py::compute_stage_ownership` (would
gain constraint-record promotion logic) + `apesees.py::
_run_staged_bc_validators` (a new V6 for cross-stage spans).

### `s.remove_mp(name=)` — mid-stage MP constraint removal

Phase SSI-2.E shipped `s.remove_sp` and `s.remove_element` but
deferred the symmetric `s.remove_mp` (release of `equal_dof` /
`rigid_link` / `rigid_diaphragm` / `embedded` constraints
between stages). Deferral was decided after a four-agent
architecture / API / OpenSees-semantics / use-case critique pass
(May 2026); the verdict was unanimous: ship nothing, document
the trigger.

OpenSees support is present and unambiguous — the Tcl parser at
`SRC/tcl/commands.cpp:6223-6247` exposes both `remove mp -tag
$mpTag` (single MP by tag) and `remove mp $constrainedNodeTag`
(cascade — removes every MP whose constrained node matches).
The C++ Domain methods are `Domain::removeMP_Constraint(int
tag)` at Domain.cpp:1265 and `Domain::removeMP_Constraints(int
nodeTag)` at Domain.cpp:1286. Neither cascades to other
constraints or to the constrained / retained nodes themselves
(consequence: `s.remove_element` on a host element does NOT
release MP constraints on its nodes — those must be removed
explicitly).

The apeGmsh-side blockers are two:

1. **No per-record MP tag tracking in emit today.** The fan-out
   helpers in `_internal/build.py::emit_mp_constraints`
   (`_emit_rigid_links`, `_emit_equal_dofs`,
   `_emit_rigid_diaphragms`, `_emit_kinematic_couplings`) emit
   pure OpenSees commands with no `tags.allocate("MPconstraint")`
   call. Only `_emit_surface_couplings` (ASDEmbeddedNodeElement)
   allocates an element tag — so the only MP kind that has a
   per-record handle today is the embedded one. To support
   `s.remove_mp -tag` for the other kinds, the emit helpers
   would need to thread an allocator through and persist the
   resulting `name → tag` map for the stage emit pass to read
   back. See the "MP per-record tag tracking" follow-up entry
   below for the enabling-refactor shape.

2. **Architecture surface concern — UN-CLAIM is a fourth
   pattern.** ADR 0034 ships PUSH / PULL / CLAIM. CLAIM-by-name
   acquires a globally-resolved constraint into a stage's pool
   so the global emit skips it. UN-CLAIM would mean "a stage
   claims a CLAIMED record from an earlier stage's pool to emit
   `remove mp` against it" — that's a fourth routing rule
   adjacent to CLAIM, with its own validator surface (V7?). The
   `_DEFERRED.md` "Implicit promotion" entry already documents
   the cost of adding a third pattern; adding a fourth without a
   forcing function would erode the architecture-critique floor
   ADR 0034 §"Alternatives considered" sets.

**Trigger:** a real SSI deck with reusable temporary shoring —
an `equal_dof` / `rigid_link` installed for one excavation lift
and released before the next — that cannot be served by
removing the host element via `s.remove_element` on the temp-
strut PG. The Cerro Lindo SSI V5 deck does not have this
pattern (the cimbra-rock embed is permanent). Until a deck
forces it, the deferral holds.

**Available workarounds today:**

- Remove the host elements via `s.remove_element` on the
  temp-PG. OpenSees does NOT auto-cascade, but if the
  constrained/retained nodes are themselves owned only by the
  removed elements, the MP becomes inert at the next analysis
  chain bind (post `wipeAnalysis`).
- Drive raw `remove mp` lines via a custom Tcl postamble per
  stage, outside the apeGmsh emit pass.
- Issue `wipeAnalysis` + re-emit-without via a fresh `apeSees`
  instance per stage segment (heavy; loses the cross-stage tag
  identity guarantees).

**File map for the eventual implementation:**

- `_internal/build.py` — extend `MassRecord`-style records with
  a frozen `MPConstraintRemovalRecord(name=, kind=)` per kind;
  the per-kind emit helpers would also need per-record name +
  tag tracking on the way out.
- `apesees.py::_StageBuilder` — add `s.remove_embedded(name=)` /
  `s.remove_equal_dof(name=)` / `s.remove_rigid_link(name=)` /
  `s.remove_rigid_diaphragm(name=)` / `s.remove_kinematic_coupling(name=)`
  (per-kind verbs; a single `s.remove_mp(name=)` is ambiguous
  because `name=` is not unique across kinds — see ADR 0034
  §5a `_claim_constraints_by_name` which filters by
  `(name, kind, scope)` for exactly this reason).
- `apesees.py::_run_staged_bc_validators` — V7 "remove_mp target
  must reference a claimed-or-global MP from a strictly-earlier
  scope."
- Per-emitter Protocol method `remove_mp_constraint(tag=)` (or
  the cascade-by-node form); plus matching no-op H5 / capture
  Recording / live forward.

### `s.add_node(...)` / `s.add_element(...)` — runtime topology authoring

Deferred indefinitely after the May 2026 four-agent critique
pass. Three of four critics returned the same verdict:
DO-NOT-SHIP. The fourth (OpenSees-semantics auditor) confirmed
the mechanics work — `Domain::addNode` at Domain.cpp:498 calls
`setDomain(this)` + `domainChange()`; `Domain::addElement` at
Domain.cpp:444 calls `setDomain(this)`, `update()`, and
`domainChange()` — `update()` at line 474 captures the
element's initial state at the CURRENT deformed configuration,
not zero stress. So mid-run topology IS legal at the OpenSees
level.

The deferral is architectural, not mechanical.

**Invariants at risk:**

- **FEMData-is-the-snapshot.** The bridge's read surface is
  `self._bridge._fem` everywhere. `_claim_constraints_by_name`,
  `compute_stage_ownership`, `allocate_element_tags`,
  `_recorder_node_targets`, `_recorder_element_targets`, V1-V6
  ownership maps — all consume FEMData. A `s.add_node` /
  `s.add_element` verb would have to maintain a parallel
  "stage-local broker" that participates in every one of those
  consumers. That's not a verb; it's a second authoring axis.
- **ADR 0027 cross-rank tag determinism.** Element tag
  allocation runs once globally via `allocate_element_tags`
  before any partition fan-out. Mid-stage allocation would
  either require user-supplied tags (a regression from the
  bridge's tag-management model) or rank-divergent allocator
  state (a silent ADR 0027 INV-1 break).
- **ADR 0021 lineage chain.** `model_hash = blake2b(fem_hash ||
  canonical_opensees_zone_bytes)`. Stage-local nodes / elements
  not in FEMData would either widen the canonical opensees
  zone (the `/opensees/stages/` schema — unshipped when this
  was written, since landed as ADR 0055 Phase 2 / 2.18.0) or
  silently lose chain-of-custody on a class of deck mutations.
- **ADR 0014 viewer-pure-H5-consumer.** At decision time the
  viewer could not open staged-model H5 at all (the since-
  lifted `apeSees.h5(path)` fail-loud guard, #313 → ADR 0055).
  Adding stage-local topology pushes the now-shipped
  `/opensees/stages/` schema and the matching
  `ViewerData.from_h5` surface even wider.

**The use-case skeptic found ZERO consumers in the codebase**:
no example, no test, no TODO, no `_DEFERRED.md` entry, no
Cerro Lindo SSI workflow needs either verb. The existing
declarative-upfront + `s.activate(pgs=...)` path (Phase SSI-2.B,
ADR 0030) covers every staged-topology workflow that has
surfaced. Cohesive-zone modelling (`examples/02_cohesive_zones.ipynb`)
pre-builds duplicated interface nodes upfront and does not
propagate at runtime.

**Trigger:** a real workflow where new node coordinates depend
on **RUNTIME state** — XFEM crack-tip nodes whose position
follows the propagation front, contact-detected nodes whose
coordinates come from runtime collision pairs, adaptive
remeshing — AND the team explicitly accepts the months-long
refactor of the FEMData-is-snapshot invariant (touching
ADR 0014 / 0019 / 0021 / 0027 simultaneously). Neither
trigger has been observed in five SSI phases; the bet that
neither will surface in the foreseeable future is reasonable.

**Available workarounds today:**

- **Part fragment + `s.activate(pgs=...)`** — declare the new
  topology in a separate apeGmsh Part, fragment it into the
  assembly so it shares boundary nodes, give it its own PG,
  and birth-activate via `s.activate`. Strictly stronger than
  `s.add_*`: PG-rooted, typed, participates in every validator,
  archivable in H5, partitionable. This is what Cerro Lindo
  SSI V5 does for the cimbra (ADR 0034 §5c).
- **Pre-declared "control" nodes** — for the single-node use
  case (a control node for a multi-point spring, a beam-on-
  elastic-foundation reaction node), author the node in a
  synthetic `g.parts.Part` upfront. It lands in FEMData; every
  downstream consumer sees it. No `s.add_node` needed.

**Verdict:** indefinite deferral. If the FEMData immutability
invariant ever does need to relax, that's an ADR-level decision
on the broker side first, with the staged verbs following as a
downstream consequence — not the other way round.

### MP per-record tag tracking — enabling refactor for `s.remove_mp` and future swap verbs

Independent of any immediate consumer, the per-record tag-
identity gap in the MP emit fan-out is a worthwhile target for
opportunistic landing.

Today, `_internal/build.py::emit_mp_constraints` and its
per-kind helpers (`_emit_rigid_links`, `_emit_equal_dofs`,
`_emit_rigid_diaphragms`, `_emit_kinematic_couplings`) write
pure OpenSees commands with no allocator interaction — the
emitted `equalDOF $m $s 1 2 3` line is anonymous. Only the
surface-coupling helper allocates an element tag (because
ASDEmbeddedNodeElement is parsed by the Tcl `element` parser,
not `equalDOF` / `rigidLink`). Consequence: an apeGmsh-side
record has no stable on-deck identity for any of the four
unbound MP kinds.

This blocks every future "mutate an MP after the fact" verb,
not just `s.remove_mp` — also any hypothetical `s.swap_mp` or
`s.update_mp_stiffness` (relevant for damper / spring
substitutions in nonlinear time history).

**Refactor shape:** open. A prior sketch (apeGmsh-side
`TagAllocator` thread; schema bump opensees 2.12.0 → 2.13.0
per ADR 0023) did not survive review. OpenSees auto-allocates
MP tags inside `MP_Constraint`'s constructor (static `nextTag++`
at `SRC/domain/constraints/MP_Constraint.cpp:238`), and the Tcl
/ Py / Live dialects expose the resulting tag asymmetrically:
`TclCommand_addEqualDOF_MP` returns it via `Tcl_SetObjResult`
(`constraint.cpp:500`), but `TclCommand_RigidLink` and
`TclCommand_RigidDiaphragm` (`rigid_links.cpp`) do NOT, and
`ops.equalDOF(...)` returns `None` because `OPS_EqualDOF`
(`MP_Constraint.cpp:49-101`) never populates the Python
wrapper's `currentResult`. Any future design must re-ground
against per-dialect tag-capture semantics — likely a
`getMPtags()` snapshot-diff in Live and a raw `MP_Constraint`
emit bypassing the convenience wrappers in Tcl/Py — before any
PR lands.

**Trigger:** any first consumer (`s.remove_mp`, `s.swap_mp`,
H5 readback of MP identity).

**Cost estimate (informational):** lift touches
`_internal/build.py::emit_mp_constraints`,
`emitter/h5.py::_write_mp_constraints`, and per-dialect
tag-capture work in every emitter (Tcl / Py / Live / Recording
— shape TBD per "Refactor shape" note above; the Emitter
Protocol classifies any new method as an architecture event per
`emitter/base.py`).

## Cylindrical / Spherical in 2-D models

`Cylindrical(axis=(0,0,1))` for a 2-D model would be meaningful
(in-plane radial / circumferential axes — both lie in the
xy-plane when the axis is perpendicular to it).  `Spherical` is
intrinsically 3-D and stays out of scope.

Today the build step raises :class:`BridgeError` when
``orientation=`` is supplied with ``ndm=2`` (see
`_internal/build.py::emit_transform_specs`).  This is the
defensive landing — the path used to silently produce an invalid
deck (``geomTransf Linear $tag $x $y $z`` with a 3-component
vecxz tail, which OpenSees rejects at parse time).  Refusing
loudly is correct until the lift lands.

To lift the restriction:

1. Decide what `orientation=Cylindrical(axis=(0,0,1))` *means*
   in OpenSees 2-D, given that 2-D `geomTransf` takes no vecxz
   argument.  Two plausible readings:
   - **Silently drop the orientation** (emit the bare 2-D form).
     Cheap; arguably surprising because the user supplied
     orientation explicitly.
   - **Use the orientation for downstream metadata** (e.g. the
     viewer's local-axis overlay, or a future curved-beam
     section orientation) but still emit the bare form.  Needs
     a downstream consumer to justify the work.
3. Add a 2-D + `Cylindrical(axis=(0,0,1))` end-to-end test
   exercising whichever semantics land (no test exists today —
   the existing 2-D tests at
   `tests/opensees/integration/test_full_emit_recording.py::test_2d_geomtransf_*`
   only cover the bare path and the new raise).
4. Drop the raise in `emit_transform_specs`.

Don't implement until at least one consumer needs in-plane
orientation metadata — the silent-drop interpretation is
indistinguishable from no orientation at all.

## Higher-order embedded coupling via `HostProjector` (RFC)

`g.constraints.embedded(...)` today couples each embedded node to
3 or 4 corner nodes of a sub-tri / sub-tet via linear barycentric
shape functions (ADR 0036 — the "linear-over-corners" contract).
The decomposition makes non-simplex / higher-order hosts
*structurally* embeddable, but the embedded node never feels the
host's native interpolation richness:

- An LST plate's quadratic curvature is discarded.
- A hex8's bilinear twist mode is discarded.
- A quad9 / hex20's biquadratic / triquadratic field is discarded.

For most rebar-in-concrete cases this is fine — the embed is so
much stiffer than the host that the global kinematic link
dominates. For cases where the host's higher-order field is doing
real work (LST plate intentionally chosen for bending; fibre in a
quad9 stress-concentration patch), the linear projection throws
away the accuracy gain.

The `host_coupling=` keyword on
[`EmbeddedDef`][apeGmsh._kernel.defs.constraints.EmbeddedDef] is
reserved for this: only `"linear"` is accepted today, but the
field, factory keyword, and `__post_init__` guard are all in
place so a future `"trilinear"` / `"biquadratic"` option can
land without breaking old models.

### What "doing it properly" requires

The clean architectural fix is a `HostProjector` strategy
interface — each gmsh etype gets its own projector that owns:

1. **Point location** in the host's natural coordinates
   (trilinear inverse-map for hex8; biquadratic inverse for
   quad9; etc.). Today's `_barycentric_tri3` /
   `_barycentric_tet4` only cover the simplex case.
2. **Coupling weights** evaluated against the host's native
   shape functions at the located parametric point. Today the
   weights are linear barycentric over 3 / 4 corners; the
   higher-fidelity branch would produce 8 weights for hex8
   trilinear coupling, 9 for quad9 biquadratic, etc.

### What it requires on the OpenSees side

Native N-node retained-set coupling needs new element classes:

- **`ASDEmbeddedHex8Element`** — 1 constrained + 8 retained =
  9 nodes total. `getTangentStiff` builds the penalty matrix
  using trilinear shape functions evaluated at the embedded
  node's local (ξ, η, ζ) coordinates. Mirrors the existing
  `TET_3D_U` / `TET_3D_UR` / `TET_3D_UP` triplet but on a
  9-node `m_nodes` vector.
- **`ASDEmbeddedQuad4Element`** — 1 + 4 = 5 nodes; bilinear
  shape functions. The 2D analogue.
- (Optional) `ASDEmbeddedHex20Element` / `ASDEmbeddedQuad9Element`
  for true higher-order coupling. Lower priority — most users
  who need higher-fidelity embed will be on hex8 / quad4 hosts.

Element-registration burden per new class:

- `classTags.h` entry.
- `OPS_ASDEmbeddedHex8` factory in `elementAPI.h`.
- `FEM_ObjectBroker::getNewElement` dispatch.
- `sendSelf` / `recvSelf` for parallel runs (mirroring the
  existing `ASDEmbeddedNodeElement::sendSelf` at
  [cpp:649-714](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)).
- CMake registration (`SRC/element/CEqElement/CMakeLists.txt`).

Coordinate-management with ASDEA (Massimo Petracca authored the
original `ASDEmbeddedNodeElement`) before opening upstream PRs.

### What it requires on the apeGmsh side

Once the OpenSees side has the new element classes:

- Accept `host_coupling="trilinear"` / `"biquadratic"` on
  `EmbeddedDef.__post_init__` (drop the current
  `host_coupling != "linear"` raise).
- In `_collect_host_subelements`, skip decomposition for hosts
  whose coupling mode is `"trilinear"` (return the native hex8
  rows of shape `(n, 8)`) or `"biquadratic"` (return native
  quad9 rows of shape `(n, 9)`).
- Widen the resolver's `npe in (3, 4)` check to `npe in (3, 4,
  8, 9)` with dispatch to new `_inverse_map_hex8` /
  `_inverse_map_quad9` functions. The inverse maps are
  iterative (Newton on the host shape functions); not trivial
  but well-trodden territory — see Hughes §3.7 or any FEM text.
- Emitter side (`opensees/emitter/*.py`): route
  `InterpolationRecord` with `n_masters in (8, 9)` to the new
  element class instead of `ASDEmbeddedNodeElement`.

### Trigger conditions

Don't start until one of these lands:

- A real user model where the linear-over-corners projection is
  measurably wrong (mesh-refinement study shows the embed
  results don't converge to the higher-order host's reference
  solution). Saying "this is theoretically lossy" is not enough
  — Cerro Lindo-style tunnel models with rebar in rock have
  shipped on the linear coupling and converged.
- A research project that explicitly needs the higher-order
  coupling (e.g., a paper on fibre-reinforced composite
  modelling where the matrix's quadratic field matters for
  the fibre's stress state).
- ASDEA volunteers to maintain a trilinear coupling element
  upstream (would change the cost calculus by removing the
  coordination burden).

The reserved `host_coupling=` keyword is the no-cost hook that
makes this future PR a non-breaking change. The work itself is
~2-3 months of coordinated OpenSees + apeGmsh changes plus a
real test suite (convergence study against a known
trilinear-coupled reference) — only worth it when a real model
needs it.

### See also

- ADR 0036 — Embedded-host decomposition: linear coupling over
  corner nodes.
- ADR 0022 — MP-constraint emission fan-out (the single primitive
  this work would replace for hex/quad hosts).
- ADR 0035 — `ASDEmbeddedNodeElement` C++ optionals exposure
  (the pattern this work follows for the new element classes).

