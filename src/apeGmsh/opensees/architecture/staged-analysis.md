# Staged-analysis emission model

The user-facing surface (`ops.stage(name)`,
`s.add(initial_stress_record)`, `s.activate(pgs=[...])`,
`s.analysis(...)`, `s.run(n_increments=, dt=)`) is documented in
[api-design.md](api-design.md) §"Staged analysis". This doc covers
the **internals** — how `BuiltModel.emit` lays out the deck when
stages are present, how topology ownership is computed, how the
hook dispatcher interacts with per-stage analyze loops, and the
cleanup contract that `stage_close` must satisfy for the next stage
to converge.

Read [api-design.md](api-design.md) first; the surface verbs and
caveats are not repeated here.

## Four phases, one builder

The staged-analysis work ships in four phases:

- **Phase SSI-2.A — staged analysis chain.** Adds `ops.stage(name)`
  + the `_StageBuilder` context manager + the `StageRecord`
  dataclass. Per stage emits its own analysis chain + analyze loop +
  `stage_open` / `stage_close` bracket.
- **Phase SSI-2.B — stage-bound topology activation.** Adds
  `s.activate(pgs=[...])`. Elements whose PG is in the activation
  set (and the nodes referenced exclusively by those elements) emit
  inside the stage block, between `stage_open` and `domain_change`.
- **Phase SSI-2.C — MP partitioned + staged.** Lifts the prior
  (stages + partitions) gate; combining stages with MP partitions
  builds without error. Per-rank fan-out under partitions while
  preserving cross-stage / cross-rank tag identity.
- **Phase SSI-2.D — stage-bound BCs + recorder.** Adds
  `s.fix(...)` / `s.mass(...)` / `s.region(...)` / `s.recorder(...)`.
  BCs emit inside the stage block alongside topology (between
  topology and `domain_change`); recorders emit after the chain and
  before `analyze` so they capture the stage's analyze steps. Ships
  with four ownership-tier validators (V1-V4) plus a unified
  `domain_change` gate.
- **Phase SSI-2.D extension — stage-bound MP constraints +
  `s.initial_stress` PUSH.** Adds nine CLAIM-by-name builder methods
  (`s.embedded`, `s.equal_dof`, `s.rigid_link`, `s.rigid_diaphragm`,
  `s.kinematic_coupling`, `s.tie`, `s.distributing`,
  `s.node_to_surface`, `s.node_to_surface_spring`) that route resolved
  MP-constraint records into a per-stage pool; the records emit inside
  the stage block (after regions, before `domain_change`) rather than
  in the global pre-stage MP-constraint pass. Adds `s.initial_stress`
  as a PUSH mirror of `ops.initial_stress(...)` alongside the existing
  `s.add(InitialStressRecord)` PULL path. CLAIM-by-name (rather than
  direct-create) because the kernel constraint resolver needs a live
  `gmsh` model + parts registry that are typically gone by bridge time
  ([ADR 0034](decisions/0034-stage-bound-bcs-and-recorders.md) §5a/5b).
  Forcing function: the Cerro Lindo SSI V5 cimbra-installation deck
  where embedded constraints leaking into the global pre-stage block
  caused Newton divergence at step 2 (ADR 0034 §5c).
- **Phase SSI-2.E — between-stage Domain mutators.** Lifts the
  append-only restriction on stage-bound BCs with five new verbs:
  `s.remove_sp(*, pg=, nodes=, dofs=)` (emits `remove sp $node $dof`),
  `s.remove_element(*, pg=, elements=)` (emits `remove element $tag`;
  `elements=` is FEM eids per the recorder.Element convention),
  `s.mass(..., overwrite=True)` (relaxes V2 for the intentional-
  overwrite case), and the imperative time-state mutators
  `s.set_time(t)` / `s.set_creep(on)` / `s.reset()`.  Removals emit
  BEFORE the stage's new fix / mass / region / MP-constraint block
  (the conservative reading); same-stage release-then-re-fix becomes
  the canonical atomic-replace pattern.  Widens the `Emitter`
  Protocol with five methods (`set_time`, `set_creep`, `reset`,
  `remove_sp`, `remove_element`); H5 archival deferred (the existing
  `apeSees.h5(...)` fail-loud guard on staged models still covers).
  Adds validators V5 (`s.remove_sp` target must reference a prior-
  tier SP) and V6 (`s.remove_element` target must reference a prior-
  tier element); V2 was extended to subtract `s.remove_sp` targets
  from the fix alive set so the atomic-replace pattern passes both
  validators.

The user surface is unified — `_StageBuilder` carries `activate`,
`fix`, `mass`, `region`, `recorder`, `add`, `initial_stress`,
`embedded` / `equal_dof` / `rigid_link` / `rigid_diaphragm` /
`kinematic_coupling` / `tie` / `distributing` / `node_to_surface` /
`node_to_surface_spring`, `remove_sp`, `remove_element`, `set_time`,
`set_creep`, `reset`, `analysis`, and `run` as verbs in the same
context manager. The deck layout below shows the combined effect.

## Deck layout

For a model with N stages, the emit pipeline produces:

```
# === Global pre-stage block ===
model 3 3
node 1 ...   # nodes NOT bound to any stage's activation
node 2 ...
...
nDMaterial Elastic ...   # materials, sections, time series
geomTransf Linear ...    # transforms (all global, fan-out shared)
element FourNodeTetrahedron 1 ...   # elements whose PG is NOT activated
                                    # by any stage
fix 5 1 1 1               # only on globally-emitted nodes (validated)
mass 7 100.0 100.0 100.0
region 1 -node ...
# MP constraints (Phase 7b / ADR 0022)
# Auto-emit Transformation handler when MP constraints present
# Initial stress for any records still in the bridge's global pool
#   (records ``.add()``'d to a stage are NOT emitted here)
pattern Plain 1 1 { load ... }
recorder Node ...

# === Stage 1 ===
# === Stage: insitu ===
# (no stage-bound topology in this stage)
parameter 100             # global side of stage's initial_stress
parameter 101
parameter 102
proc rock_insitu {...}    # per-step ramp body
lappend _apesees_before_step_hooks rock_insitu
addToParameter 100 element 1 commitStressIncrementXX   # one per
addToParameter 100 element 2 commitStressIncrementXX   # owned element
...
constraints Plain         # the seven analysis-chain primitives,
numberer RCM              # emitted per stage so OpenSees sees a
system UmfPack            # fresh chain when wipeAnalysis fires
test NormDispIncr 1e-4 150
algorithm Newton
integrator LoadControl 0.1
analysis Static
for {set _apesees_i 0} {$_apesees_i < 10} {incr _apesees_i} {
    _apesees_call_before_step
    analyze 1 0.1
    _apesees_call_after_step
}
loadConst -time 0.0
wipeAnalysis
set _apesees_before_step_hooks {}     # cleared between stages
set _apesees_after_step_hooks {}

# === Stage 2 ===
# === Stage: excavate ===
node 451 ...              # stage-bound nodes (owned ONLY by stage 2)
node 452 ...
element FourNodeTetrahedron 87 ...   # stage-activated elements
element FourNodeTetrahedron 88 ...
fix 451 1 1 1             # SSI-2.D: stage-bound fix on stage-bound node
mass 452 100.0 100.0 100.0 # SSI-2.D: stage-bound mass
region 17 -node 451 452   # SSI-2.D: stage-bound region (per-stage tag)
element ASDEmbeddedNodeElement 200 999 451 452 453 -K 1e+08
                          # SSI-2.D ext: stage-bound MP constraint
                          # claimed via s.embedded(name="lining_embed")
domainChange              # tell OpenSees to rebuild the DOF map
parameter 200 ...         # stage's initial_stress (PR-A)
proc excavate_stress {...}
addToParameter 200 element 87 commitStressIncrementXX
...                       # analysis chain
constraints Plain
numberer RCM
...                       # SSI-2.D: stage-bound recorder (after chain,
recorder Element -file lining.out -ele 87 88 globalForce
                          # before analyze so it captures the loop)
for {set _apesees_i 0} {$_apesees_i < 5} {incr _apesees_i} {
    _apesees_call_before_step
    analyze 1 0.1
    _apesees_call_after_step
}
loadConst -time 0.0
wipeAnalysis
```

The line-by-line emission order is:

1. **Pre-stage global** — `BuiltModel._emit_flat` body, minus the
   per-stage analysis-chain primitives and minus stage-bound nodes
   / elements / `initial_stress` records.
2. **Per stage** (in registration order — `BuiltModel.stage_records`
   tuple, which mirrors the order the `with ops.stage(...)` blocks
   exited):
   - `stage_open(name)` — `# === Stage: <name> ===` banner.
   - Stage-bound nodes, sorted by FEM id for stable diffs.
   - Stage-bound elements, in the same per-spec order the global
     plan uses (tags pre-allocated upfront — see "Tag determinism"
     below).
   - **Stage-bound `fix` / `mass` / `region`** (Phase SSI-2.D PR-B/C)
     — per-record fan-out via `_resolve_node_target`; regions group
     by name within this stage, one tag per name allocated from the
     shared `TagAllocator`. V3 guarantees no cross-scope name
     collision so tags stay disjoint across stages and globals.
   - **Stage-bound MP constraints** (Phase SSI-2.D extension) — the
     stage's `stage_constraint_records` pool, claimed earlier via
     `s.embedded(name=...)` / `s.equal_dof(name=...)` /
     `s.rigid_link(name=...)` / `s.rigid_diaphragm(name=...)` /
     `s.kinematic_coupling(name=...)` / `s.tie(name=...)` /
     `s.distributing(name=...)` / `s.node_to_surface(name=...)` /
     `s.node_to_surface_spring(name=...)`. The records are wrapped in
     a `_StageConstraintAdapter` and the six per-kind emit helpers
     (`_emit_phantom_nodes`, `_emit_rigid_links`, `_emit_equal_dofs`,
     `_emit_rigid_diaphragms`, `_emit_kinematic_couplings`,
     `_emit_surface_couplings`) are reused unchanged. The bridge's
     `_claimed_constraint_ids()` is passed to the global emit
     orchestrators so claimed records are excluded from the pre-stage
     pass via the `_ExcludeClaimedConstraints` adapter — no double
     emission.
   - `domain_change()` — **unified gate (Phase SSI-2.D)**: fires if
     the stage added ANY topology (nodes / elements) OR any stage-
     bound BC (fix / mass / region) OR any stage-bound MP constraint.
     Single barrier per stage.
   - Stage's `initial_stress` records — `parameter` declarations +
     `step_hook_ramp` proc bodies + `addToParameter` calls. Same
     shape as the Phase SSI-1 non-staged global emit, scoped here.
   - Stage's analysis chain — each of `constraints / numberer /
     system / test / algorithm / integrator / analysis` emitted via
     its registered primitive's `_emit`.
   - **Stage-scoped patterns** (ADR 0051 BL-3) — emit AFTER the chain
     and BEFORE `analyze` so the pattern's `load` / `sp` / `from_model`
     lines drive this stage's analyze loop and are frozen by the
     stage's `stage_close` `loadConst`. Patterns are created via
     `s.pattern(series=)` (a stage-owned `Plain` context manager,
     registered with the bridge so it gets a tag); the bridge tracks
     claimed pattern ids in `_stage_claimed_pattern_ids` so the global
     post-element pattern pass skips them (no double emission). The
     flat path reuses `emit_pattern_spec` (so PG fan-out + `from_model`
     expansion match the non-staged path); the partitioned path reuses
     the per-rank `_emit_one_pattern_partitioned` helper, opening a
     `partition_open(rank)` bracket only for ranks that own pattern
     content (an empty bracket is a Py-emitter `SyntaxError`).
   - **Stage-bound recorders** (Phase SSI-2.D PR-C) — emit AFTER the
     chain (so the recorder sees the bound analysis chain) and
     BEFORE `analyze` (so the recorder captures the stage's analyze
     steps). Recorder specs are PULLed via `s.recorder(spec)` from
     globally-registered primitives; the bridge tracks claimed
     recorder ids in `_stage_claimed_recorder_ids` so the global
     post-element emit loop skips them.
   - `analyze(steps=stage.n_increments, dt=stage.dt)` — the emitter
     wraps this in a for-loop with hook-dispatcher calls between
     steps if any `step_hook_ramp` was registered (the emitter
     tracks `_step_hooks_registered` internally).
   - `stage_close()` — `loadConst -time 0.0` + `wipeAnalysis` + (if
     hooks are registered) reset of the dispatcher lists.

The relevant entry points are
`BuiltModel.emit → _emit_flat → _emit_stages_flat` for
single-partition models and `BuiltModel.emit → _emit_partitioned →
_emit_stages_partitioned` for MP-partitioned models (per Phase
SSI-2.C; see "MP partitioned + stages" below). Both branches live
in [apesees.py](../apesees.py). Per-stage helpers live in
`opensees/_internal/build.py::compute_stage_ownership` /
`emit_initial_stress_global` /
`emit_initial_stress_addtoparameter`.

## Ownership computation (Phase SSI-2.B)

`compute_stage_ownership(stage_records, elements, fem)` in
[`_internal/build.py:1830-1902`](../_internal/build.py) returns two
maps:

- `element_owner: dict[id(spec), stage_index]` — element-primitive
  identity → owning stage. Primitives not in any stage's activation
  set are absent (global emit).
- `node_owner: dict[fem_node_id, stage_index]` — FEM node id →
  owning stage. A node referenced by **any** globally-emitted
  element stays global, even if a stage's elements also touch it.
  A node referenced **only** by stage-bound elements is owned by the
  **lowest** stage index that references it.

Rules:

1. **PG-level activation is exclusive.** Activating the same PG in
   two stages raises `BridgeError` at build time. The implicit
   "first-write wins" semantics would silently misroute elements; if
   two stages truly need to share a PG the user must split the PG
   first.
2. **Global wins for shared nodes.** A node referenced by **any**
   globally-emitted element stays in the global pre-stage block.
   That node exists in OpenSees before any stage opens — so it can
   carry global `fix` / `mass` / `region` directives.
3. **Lowest-index wins for stage-shared nodes.** If a node is
   referenced only by stage-bound elements (no global element
   touches it), it belongs to the lowest-index stage that activates
   it. The deck then emits the node once, before the stage's
   `element` lines that reference it.
4. **Tag determinism survives staged emit.** Element tags are
   pre-allocated **once** by `allocate_element_tags(elements, fem,
   tags)` BEFORE any stage emits. The same `fem_eid → ops_tag` map
   is shared across the global block and every stage's block. This
   matters because Phase SSI-1's `addToParameter` calls and #314's
   `Element` recorder fan-out both index by this map; cross-stage
   tag drift would silently misroute recorder targets and ramp
   commitments.

## Stage-close cleanup contract

`emitter.stage_close()` must emit, in order:

```
loadConst -time 0.0
wipeAnalysis
# only if step hooks are registered:
set _apesees_before_step_hooks {}
set _apesees_after_step_hooks {}
```

Each line is load-bearing:

- **`loadConst -time 0.0`** — OpenSees needs to freeze the
  accumulated loads from this stage as the new permanent baseline
  and reset its pseudo-time. Without it, the next stage's analyze
  steps double-apply the prior stage's loads.
- **`wipeAnalysis`** — drops the previous stage's analysis-chain
  binding so the next stage's `constraints / numberer / system /
  test / algorithm / integrator / analysis` lines take effect.
  Without it, the second stage's chain is silently shadowed by the
  first.
- **Hook-list reset** — clears the dispatcher's `lappend`
  registrations so the next stage's `analyze` loop does not re-fire
  the previous stage's ramp procs. The proc *definitions* persist
  (they remain in the Tcl namespace), but become unreachable unless
  a later stage explicitly registers them again. The Tcl emitter
  also flips `_step_hooks_registered = False` so the next stage's
  bare `analyze` emits a flat `analyze N` line, not a for-loop
  wrapper — unless that stage itself registers a new ramp BEFORE
  the analyze.

The Py emitter mirrors this contract verbatim. The Live emitter
**raises `NotImplementedError` on both `stage_open` and
`stage_close`** — staged live execution is deferred (see "Deferred
work" below).

## Hook dispatcher (Phase SSI-1)

The dispatcher is the seam between `apeSees.initial_stress(...)`'s
per-step linear ramp and the emitter's `analyze` loop. Once any
`step_hook_ramp` has run on an emitter, that emitter's `analyze`
**must** wrap its analyze call with hook-dispatcher invocations.

The Tcl dispatcher boilerplate (emitted once across the deck's
lifetime, on the first `step_hook_ramp` call):

```tcl
# apeSees per-step hook dispatcher (Phase SSI-1)
set _apesees_before_step_hooks {}
set _apesees_after_step_hooks {}
proc _apesees_call_before_step {} {
    global _apesees_before_step_hooks
    foreach _f $_apesees_before_step_hooks { $_f }
}
proc _apesees_call_after_step {} {
    global _apesees_after_step_hooks
    foreach _f $_apesees_after_step_hooks { $_f }
}
```

Then the per-ramp proc + registration:

```tcl
parameter 100
parameter 101
parameter 102
proc rock_insitu {} {
    global rock_insitu_state
    if {![info exists rock_insitu_state(count)]} {
        set rock_insitu_state(count) 0
        set rock_insitu_state(cum_100) 0.0
        set rock_insitu_state(cum_101) 0.0
        set rock_insitu_state(cum_102) 0.0
    }
    set rock_insitu_state(count) [expr {$rock_insitu_state(count) + 1}]
    set _factor [expr {$rock_insitu_state(count) / 10.0}]
    if {$_factor > 1.0} { set _factor 1.0 }
    set _cur [expr {-6300.0 * $_factor}]
    set _delta [expr {$_cur - $rock_insitu_state(cum_100)}]
    updateParameter 100 $_delta
    set rock_insitu_state(cum_100) $_cur
    ...                                  # YY, ZZ axes
}
lappend _apesees_before_step_hooks rock_insitu
```

And the hook-wrapped `analyze` loop:

```tcl
for {set _apesees_i 0} {$_apesees_i < 10} {incr _apesees_i} {
    _apesees_call_before_step
    analyze 1 0.1
    _apesees_call_after_step
}
```

The Py emitter mirrors the same shape (`_apesees_call_before_step`
becomes a Python function, the proc body becomes a closure over a
`state` dict, `lappend` becomes `list.append`). The Live emitter
captures Python closures directly into per-instance
`_before_step_hooks` / `_after_step_hooks` lists and drives the
analyze loop in-process (no `for-loop` text emit).

The naming choices intentionally differ from STKO's
`STKO_VAR_OnBeforeAnalyze_CustomFunctions` / `_stressCtrl_<N>` so a
hand-written STKO block dropped into the same deck does not collide
with apeSees-emitted procs.

### Algorithmic match with STKO

The per-step math is byte-identical to STKO's `_stressCtrl_<N>`
proc body:

| Step | STKO | apeSees |
|---|---|---|
| Advance counter | `set _stressCtrl_N(count) [expr {…+ 1}]` | `set <name>_state(count) [expr {… + 1}]` |
| Capped factor | `set _stressCtrl_factor [expr {count / divisor}]; if {> 1.0} cap` | `set _factor [expr {count / n_steps_to_full}]; if {> 1.0} cap` |
| Per-axis delta | `set _stressCtrl_current [expr target * factor]; set _stressCtrl_incr [expr current - cum]; updateParameter tag incr` | `set _cur [expr target * _factor]; set _delta [expr $_cur - cum]; updateParameter tag _delta` |
| Persist cumulative | `set _stressCtrl_N(<XX|YY|ZZ>) $current` | `set <name>_state(cum_<tag>) $_cur` |

The acceptance test
[`tests/opensees/subprocess/test_initial_stress_acceptance.py`](../../../../tests/opensees/subprocess/test_initial_stress_acceptance.py)
locks the FIXED ramp values against
`C:\Users\nmora\opensees_runs\cerro_lindo\ssi_test_stressctrl\result_fixed.csv`
within ±0.5 kPa per step. The discriminating step is step 5:
correct emit produces σxx ≈ -3024 kPa (linear interpolation 0 →
target); the historical STKO single-step-jump bug produced σxx ≈
-5981 kPa.

### Per-record divergence from STKO

STKO's `_stressCtrl_11` (convergence-confinement at
`SSI/Interaccion/analysis_steps.tcl:19753-19767`) allocates only
**one** `parameter` when only the XX component is non-zero. apeSees
always allocates **three** (`parameter <xx>` + `parameter <yy>` +
`parameter <zz>`) and emits three `updateParameter` lines per step,
even when YY / ZZ targets are zero — the deltas are 0.0 and
constitute no-ops, but the parameter slots are reserved. This is a
documented divergence in
[`api-design.md`](api-design.md) §"Initial-stress injection"; the
target stress values still match STKO byte-for-byte.

## MP partitioned + initial_stress + stages (Phase SSI-2.C)

When the FEM carries >1 partition, `BuiltModel._emit_partitioned`
takes over. Phase SSI-2.C (PR #315) lifted the prior gate that
refused the (stages + MP partitions) combo, so a model can be
**both** staged AND partitioned. The initial-stress emit splits
across the `partition_open(rank)` boundary the same way as the
non-staged case; staged builds add a per-stage block after the
per-rank loop.

**Initial-stress emit (with or without stages):**

- **Global side** (outside any `partition_open`) — `parameter <tag>`
  declarations + the `step_hook_ramp` boilerplate / proc /
  `lappend`. Per OpenSeesMP semantics every rank parses the deck,
  so each rank ends up with the same `parameter` slots and proc
  bodies in its local Tcl namespace.
- **Per-rank inside `partition_open(K)`** — only the
  `addToParameter <tag> element <ele> commitStressIncrement<axis>`
  calls for elements owned by rank `K`. The fan-out checks
  `element_owner[fem_eid] == K` before emitting; non-owned elements
  silently skip. The owning rank issues the call exactly once.

**Partitioned staged emit** (`_emit_partitioned` dispatches to
`_emit_stages_partitioned` when `stage_records` is non-empty):

1. **Pre-stage global pass** — materials, sections, time series,
   transforms, **non-stage-bound** elements (filtered via
   `compute_stage_ownership`), per-rank fix / mass / region /
   loads / MP constraints. Analysis-chain primitives are SKIPPED
   here (each stage carries its own complete chain — same rule as
   the flat staged path).
2. **The `_maybe_auto_emit_*` constraint-handler / numberer /
   system upgrades are gated on `not staged`** — staged decks
   carry the chain per-stage and validate it at
   `_StageBuilder.__exit__`, so the auto-upgrades that fire on
   the global chain would emit twice if not gated.
3. **Per stage**, in registration order:
   - `stage_open(name)` (global, not inside `partition_open`).
   - **Per-rank loop** over the stage's owned topology — for each
     rank K with stage-bound nodes / elements, `partition_open(K)`
     + per-rank node + element emit + `partition_close`. Tags
     come from the global pre-allocated `element_plan` so cross-
     rank tag identity holds verbatim.
   - **Global `domain_change()`** (outside any `partition_open`)
     after the per-rank loop, only if the stage added topology.
   - **Per-rank `addToParameter` loop** for the stage's
     `initial_stress` records — `partition_open(K)` + filtered
     `addToParameter` calls for elements owned by rank K +
     `partition_close`. The `parameter` / `proc` /
     `lappend` globals emit once outside any `partition_open`.
   - **Global analysis-chain primitives** (each stage's bound
     `constraints / numberer / system / test / algorithm /
     integrator / analysis` via `_emit`).
   - **Global `analyze`** (hook-wrapped if the stage registered a
     ramp).
   - **`stage_close()`** (global).

The 4-quad 2-partition 2-PG fixture at
[`tests/opensees/integration/test_emit_partitioned_staged.py`](../../../../tests/opensees/integration/test_emit_partitioned_staged.py)
locks every assertion above — rank-K-owned nodes only appear in
that rank's `partition_open(K)` block; `domain_change` lands once
globally after the per-rank topology loop; `addToParameter` lines
appear only inside the right rank's block; tags hold identity
between the global element fan-out and the per-stage element
fan-out.

## Validation surface

Seven guard rails ship with the staged path, all orchestrated by
`BuiltModel._run_staged_bc_validators` (called from both
`_emit_flat` and `_emit_partitioned` when `stage_records` is
non-empty). H1 / H2 / M4 land in #312; V1-V4 land in Phase SSI-2.D
(#323 / #326) and follow the shared `_collect_ownership_offenders`
+ `_render_offender_line` helpers so error messages stay
consistent across rules.

One further build-time guard, the **two-mode no-mixing rule** (ADR
0051 §5, BL-4), is *not* part of `_run_staged_bc_validators` — it runs
in `BuiltModel.emit` (`_validate_two_mode_no_mixing`) before any emit
path is chosen: a staged model that also registers a **global**
`ops.pattern.*` (one whose id is not in `_claimed_pattern_ids`, i.e.
not created via `s.pattern(...)`) raises `BridgeError`. A global
pattern fires in every stage's analyze loop (ADR 0031) and would
silently double-apply its loads across the staged `loadConst`
boundaries. (An earlier `WarnUnconsumedModelLoads` reconciliation
warning + `ops.ignore_model_loads` silencer shipped alongside this
guard but were **removed** — with loads opt-in the deck is
authoritative; see ADR 0051 §7. The no-mixing guard stays.)

- **H1 — global fix/mass/region on stage-bound nodes** (PR #312,
  refactored in #323): `_validate_no_stage_bound_node_targets`
  raises `BridgeError` with an offender list naming each
  `(kind, target, node, stage)` tuple. Without this, the
  `fix N 1 1` line emits in the pre-stage block, references a node
  that only comes into being in stage 2, and OpenSees errors at
  parse time. **#323 bug fix:** the partitioned path previously
  skipped H1 entirely — a global `fix` on a stage-bound node would
  slip through under MP. Now invoked from both emit paths.
- **H2 — duplicate `initial_stress` name across stages** (PR #312):
  validated in `BuiltModel.emit`. Raises `BridgeError` naming both
  owners. The second `proc <name>` definition would otherwise
  override the first while reading uninitialised state.
- **M4 — nested `with ops.stage(...)` blocks** (PR #312):
  `apeSees.stage` raises `RuntimeError`.
- **V1 — stage N's BC targets nodes owned by stage M > N** (PR
  #323): `_validate_stage_bound_node_targets`. Each stage's BC
  block emits AFTER that stage's topology + `domain_change`, so a
  target owned by a later stage doesn't exist at parse time.
  Globally-emitted nodes and earlier-stage-owned nodes are legal.
- **V2 — cross-tier duplicate `(node, DOF)` fix or duplicate
  `(node)` mass** (PR #323):
  `_validate_no_duplicate_fix_mass_across_tiers`. Refuses both
  duplicate-fix (OpenSees `Domain::addSP_Constraint` rejects with
  an error — Domain.cpp:589-605) and duplicate-mass (`setMass`
  silently overwrites — Domain.cpp:3876-3883). Error message names
  the offending scopes.
- **V3 — region `name=` collision across scopes** (PR #323):
  `_validate_region_scope_invariants`. OpenSees `Domain::addRegion`
  silently appends on duplicate tag (Domain.cpp:2679-2697) and
  `getRegion` returns only the first match — silent data loss.
  Mangle the name (`lining_rayleigh_stage2`) to make scope
  explicit.
- **V4 — stage N's recorder targets owned by stage M > N** (PR
  #326): `_validate_stage_bound_recorder_targets`. Recorder lines
  (`recorder Node ...` / `Element ...` / `mpco ...`) parse at deck-
  read time and bind to the topology that exists at that point.
  Covers both node targets (`Node.pg/nodes`, `MPCO.nodes_pg/nodes`)
  and element targets (`Element.pg/elements`,
  `MPCO.elements_pg/elements`). Note: apeGmsh recorders do NOT
  reference regions by name — V4 is V1 extended to recorder
  pg/nodes/elements selectors, not the original "recorder-references-
  region-by-name" framing from the Red critique (which is moot for
  the current API).

## Deferred work

| Item | Where it would land |
|---|---|
| **Live execution of staged models** — `apeSees.analyze` / `apeSees.eigen` currently raise `NotImplementedError` when stages are present. Lifting requires staging the analysis-chain re-binding, per-stage analyze loops, `loadConst` / `wipeAnalysis` interleaving, and hook-list clearing inside `LiveOpsEmitter`. Tcl + Py text emit are the supported execution paths. | `emitter/live.py::stage_open` / `stage_close` (currently raise); `apesees.py::analyze` / `eigen` (currently refuse). |
| **H5 archival of staged structure + initial_stress + stage-bound BCs/recorders** — `H5Emitter` no-ops on `addToParameter` / `step_hook_ramp` / `stage_open` / `stage_close` / `domain_change`. Because that silent-drop would round-trip into a non-staged flat model, `apeSees.h5(path)` is **guarded** (#313) — it raises `NotImplementedError` when `self._stage_records` or `self._initial_stress_records` is non-empty, pointing the user at `ops.tcl(path)` / `ops.py(path)`. A future schema bump (per [ADR 0023](decisions/0023-per-zone-schema-versioning.md)) from `opensees_schema_version` `2.11.0` → `2.12.0` would persist per-stage primitive lists + BC pools + recorder claims under `/opensees/stages/`, lift the guard, and restore round-trip parity. | `apesees.py::h5` (bridge-side guard); `emitter/h5.py::addToParameter / step_hook_ramp / stage_open / stage_close / domain_change` (schema-side no-ops). |
| **~~`remove sp` / mass-zero-out across stages~~** — **SHIPPED Phase SSI-2.E (2026-05).** `s.remove_sp(...)` + `s.remove_element(...)` + `s.mass(..., overwrite=True)` are the live verbs; removals emit BEFORE same-stage fix / mass / region / MP-constraint blocks. Atomic-replace pattern (release prior + re-fix in same stage) is the canonical usage. Validators V5 / V6 gate at build time; V2 subtracts `s.remove_sp` targets from the fix alive set. See [_DEFERRED.md](_DEFERRED.md) §"`remove sp` / `remove element` / mass overwrite / time-state mutators (Phase SSI-2.E)". |
| **MPCO recorders with filters under stages** — stage-bound MPCO recorders DO claim through `s.recorder(spec)` but the per-rank filter-region planning (`_plan_partitioned_mpco_recorders`) currently only runs in the global emit pass. A stage-bound MPCO with a filter would fall through `emit_recorder_spec`'s materialize path and emit the filter region INSIDE the stage block instead of pre-allocated — works but doesn't reuse the cross-rank tag-identity infra. | Pre-allocate stage MPCO filter regions alongside the per-stage region tag cache; thread `_region_tag` into the materialised spec the same way the global path does. |
| **Cross-stage region union** — V3 refuses same region `name=` across scopes. A region whose conceptual identity spans multiple stages currently requires explicit per-stage mangling (`lining_r_stage2`, `lining_r_stage3`) and client-side aggregation of the per-stage recorder outputs. Lifting V3 to allow opt-in name continuity would either need OpenSees `region` extension (doesn't exist; `MeshRegion` membership is immutable post-construction — MeshRegion.cpp:82-85) or deferred emit of the unified region to the LAST contributing stage (loses recorder coverage for earlier stages). | Likely won't lift — the workaround is correct OpenSees usage and the alternatives degrade behavior. |
| **`s.tied_contact` / `s.mortar` stage-bound claim** — Phase SSI-2.D extension ships nine MP-constraint claim methods on `_StageBuilder` but `s.tied_contact` is omitted: `tied_contact` resolves to a `SurfaceCouplingRecord` whose nested `slave_records: list[InterpolationRecord]` is what emits via the `interpolations()` iterator. `_ExcludeClaimedConstraints` filters on outer-record identity; the nested slaves slip through and would double-emit. `s.mortar` is omitted because `g.constraints.mortar(...)` is kernel-NIY. | Extend `_ExcludeClaimedConstraints.interpolations()` to filter nested slaves when their parent `SurfaceCouplingRecord` is claimed, or accept per-slave naming as the user contract. See [_DEFERRED.md](_DEFERRED.md) §"`s.tied_contact` / `s.mortar` stage-bound claim". |
| **Implicit promotion of `g.constraints.*` to stages (Path A)** — Phase SSI-2.D extension shipped CLAIM-by-name rather than implicit derivation in `compute_stage_ownership`. The forgotten-claim failure mode (user adds an embed at apeGmsh time, forgets to claim inside a stage, deck crashes at parse time because stage-bound nodes don't exist yet) is caught by the V1-style validator with a clear offender list but still requires the user to edit the stage block. | `compute_stage_ownership` gains constraint-record promotion logic + a V6 validator for cross-stage spans. Likely won't lift soon — CLAIM-by-name covers the canonical SSI workflow ergonomically. See [_DEFERRED.md](_DEFERRED.md) §"Implicit promotion of `g.constraints.*` records to stages". |

## File map

| Concern | Source |
|---|---|
| User surface (`ops.stage`, `_StageBuilder`) | [`apesees.py`](../apesees.py) `class _StageBuilder` |
| `StageRecord` dataclass (+ SSI-2.D `fix_records` / `mass_records` / `region_records` / `recorder_specs` + SSI-2.D ext `stage_constraint_records`) | [`_internal/build.py`](../_internal/build.py) `class StageRecord` |
| `InitialStressRecord` dataclass | [`_internal/build.py`](../_internal/build.py) `class InitialStressRecord` |
| Per-stage emit pipeline (single-partition) | [`apesees.py`](../apesees.py) `BuiltModel._emit_stages_flat` |
| Per-stage emit pipeline (MP — Phase SSI-2.C / SSI-2.D) | [`apesees.py`](../apesees.py) `BuiltModel._emit_stages_partitioned` |
| Per-stage region emit helpers (Phase SSI-2.D PR-C) | [`apesees.py`](../apesees.py) `_emit_stage_regions` / `_emit_stage_regions_partitioned` |
| Recorder claiming + skip in global emit | [`apesees.py`](../apesees.py) `apeSees._stage_claimed_recorder_ids`, `BuiltModel._claimed_recorder_ids` |
| Stage-scoped pattern builder + claim + skip (ADR 0051 BL-3) | [`apesees.py`](../apesees.py) `_StageBuilder.pattern`, `apeSees._stage_claimed_pattern_ids`, `BuiltModel._claimed_pattern_ids`; per-rank `_emit_one_pattern_partitioned` / `_stage_pattern_specs_have_owned_content`; `StageRecord.pattern_specs` in [`_internal/build.py`](../_internal/build.py) |
| Stage-bound MP-constraint claim + skip in global emit (SSI-2.D ext) | [`apesees.py`](../apesees.py) `apeSees._stage_claimed_constraint_ids`, `BuiltModel._claimed_constraint_ids`, `_StageBuilder._claim_constraints_by_name` |
| Stage-bound MP-constraint emit orchestrators (SSI-2.D ext) | [`_internal/build.py`](../_internal/build.py) `emit_stage_mp_constraints`, `emit_stage_mp_constraints_partitioned`, `_StageConstraintAdapter`, `_ExcludeClaimedConstraints` |
| Stage-bound MP-constraint builder methods (SSI-2.D ext) | [`apesees.py`](../apesees.py) `_StageBuilder.embedded` / `.tie` / `.distributing` / `.equal_dof` / `.rigid_link` / `.rigid_diaphragm` / `.kinematic_coupling` / `.node_to_surface` / `.node_to_surface_spring` |
| `s.initial_stress(...)` PUSH builder + shared validation helper (SSI-2.D ext) | [`apesees.py`](../apesees.py) `_StageBuilder.initial_stress`, `_build_initial_stress_record` |
| `apeSees.h5` fail-loud guard (#313) | [`apesees.py`](../apesees.py) `apeSees.h5` |
| Ownership computation | [`_internal/build.py`](../_internal/build.py) `compute_stage_ownership` |
| Tag pre-allocation | [`_internal/build.py`](../_internal/build.py) `allocate_element_tags` |
| Initial-stress global emit | [`_internal/build.py`](../_internal/build.py) `emit_initial_stress_global` |
| Initial-stress `addToParameter` fan-out | [`_internal/build.py`](../_internal/build.py) `emit_initial_stress_addtoparameter` |
| Tcl emitter SSI methods | [`emitter/tcl.py`](../emitter/tcl.py) |
| Py emitter SSI methods | [`emitter/py.py`](../emitter/py.py) |
| Live emitter SSI methods + raises | [`emitter/live.py`](../emitter/live.py) |
| H5 emitter no-ops (deferred archival) | [`emitter/h5.py`](../emitter/h5.py) |
| Recording emitter capture | [`emitter/recording.py`](../emitter/recording.py) |
| Build-time validators (PR #312 + Phase SSI-2.D) — orchestrator | [`apesees.py`](../apesees.py) `BuiltModel._run_staged_bc_validators` (H1 / V1 / V2 / V3 / V4) |
| Ownership-tier offender helpers | [`apesees.py`](../apesees.py) `_collect_ownership_offenders`, `_render_offender_line`, `_records_as_targets` |
| Recorder target resolvers (V4) | [`apesees.py`](../apesees.py) `_recorder_node_targets`, `_recorder_element_targets`, `_build_fem_eid_owner_stage_map` |
| Bridge introspection (Phase SSI-2.D Red #19) | [`apesees.py`](../apesees.py) `apeSees.all_fix_records` / `all_mass_records` / `all_region_records` / `all_recorder_specs` |

## Test map

| Suite | Coverage |
|---|---|
| [`tests/opensees/unit/test_stages.py`](../../../../tests/opensees/unit/test_stages.py) | `_StageBuilder` lifecycle, `StageRecord` shape, `BuiltModel.emit` per-stage analysis-chain re-emit. |
| [`tests/opensees/unit/test_stage_activation.py`](../../../../tests/opensees/unit/test_stage_activation.py) | `s.activate(pgs=)` ownership computation, node + element routing, `domain_change` emission, duplicate-PG and global-shared-node rules. |
| [`tests/opensees/unit/test_phase3_helpers.py`](../../../../tests/opensees/unit/test_phase3_helpers.py) | `convergence_confinement` and `imposed_displacement` validations + emitted-pattern shape. |
| [`tests/opensees/unit/test_ssi_post_merge_cleanup.py`](../../../../tests/opensees/unit/test_ssi_post_merge_cleanup.py) | Red-team H1/H2/H3/M4 hardening — the build-time validators added in #312. |
| [`tests/opensees/unit/test_emitter_initial_stress.py`](../../../../tests/opensees/unit/test_emitter_initial_stress.py) | Per-emitter `addToParameter` / `step_hook_ramp` shapes + hook-wrapped `analyze`. |
| [`tests/opensees/unit/test_initial_stress_integration.py`](../../../../tests/opensees/unit/test_initial_stress_integration.py) | End-to-end build pipeline: `InitialStressRecord` → `parameter` decls → ramp proc → `addToParameter` per element. |
| [`tests/opensees/unit/test_asd_plastic_material_3d.py`](../../../../tests/opensees/unit/test_asd_plastic_material_3d.py) | `ASDPlasticMaterial3D` + `MohrCoulombSoil` + `PlaneStrain` primitives. |
| [`tests/opensees/subprocess/test_stages_subprocess.py`](../../../../tests/opensees/subprocess/test_stages_subprocess.py) | Tcl + Py subprocess smoke — multi-stage deck runs end-to-end on `OpenSees` / `python -m openseespy`. |
| [`tests/opensees/subprocess/test_stage_activation_subprocess.py`](../../../../tests/opensees/subprocess/test_stage_activation_subprocess.py) | Subprocess smoke for the topology-activation path. |
| [`tests/opensees/subprocess/test_phase3_subprocess.py`](../../../../tests/opensees/subprocess/test_phase3_subprocess.py) | Subprocess smoke for `convergence_confinement` + `imposed_displacement`. |
| [`tests/opensees/subprocess/test_initial_stress_smoke.py`](../../../../tests/opensees/subprocess/test_initial_stress_smoke.py) | Subprocess smoke for the SSI-1 ramp end-to-end on `OpenSees`. |
| [`tests/opensees/subprocess/test_initial_stress_acceptance.py`](../../../../tests/opensees/subprocess/test_initial_stress_acceptance.py) | Empirical acceptance — locks the FIXED ramp values against `result_fixed.csv` within ±0.5 kPa per step; gated on the reference CSV and the Ladruno OpenSees binary being available. |
| [`tests/opensees/h5/test_h5_staged_fail_loud.py`](../../../../tests/opensees/h5/test_h5_staged_fail_loud.py) | `apeSees.h5` fail-loud guard (#313) — staged build + global `initial_stress` both raise `NotImplementedError`; vanilla non-staged build still writes successfully (guard is precise, no regression). |
| [`tests/opensees/integration/test_emit_partitioned_staged.py`](../../../../tests/opensees/integration/test_emit_partitioned_staged.py) | Phase SSI-2.C — 4-quad 2-PG 2-partition fixture; locks per-rank topology routing, global `domain_change` after the per-rank loop, `addToParameter` inside `partition_open(K)` only, cross-stage tag identity. |
| [`tests/opensees/unit/test_stage_bound_validators.py`](../../../../tests/opensees/unit/test_stage_bound_validators.py) | Phase SSI-2.D PR-A — V1 / V2 / V3 ownership-tier + duplicate-fix + region-name validators; StageRecord shape lock; orchestrator H1-before-V1 ordering. |
| [`tests/opensees/unit/test_stage_bound_fix_mass.py`](../../../../tests/opensees/unit/test_stage_bound_fix_mass.py) | Phase SSI-2.D PR-B — `s.fix` / `s.mass` builder positive + negative + XOR; `__slots__` assertion; `bridge.all_fix_records` / `all_mass_records` introspection; flat emit shape + slot ordering; unified `domain_change` gate (BC-only stage). |
| [`tests/opensees/integration/test_emit_partitioned_stage_bound_bcs.py`](../../../../tests/opensees/integration/test_emit_partitioned_stage_bound_bcs.py) | Phase SSI-2.D PR-B — per-rank fix/mass routing on owning rank, zero leak into global scope, single global `domain_change`, empty-bracket skip on non-contributing rank, BC-only stage drives per-rank loop. |
| [`tests/opensees/unit/test_stage_bound_region_recorder.py`](../../../../tests/opensees/unit/test_stage_bound_region_recorder.py) | Phase SSI-2.D PR-C — `s.region` / `s.recorder` builder positive + negative; recorder type / membership / double-claim checks; global-emit skip on claimed; slot ordering (chain → recorder → analyze; region → `domain_change`); V4 positive + negative; `all_region_records` / `all_recorder_specs` introspection. |
| [`tests/opensees/integration/test_emit_partitioned_stage_bound_regions.py`](../../../../tests/opensees/integration/test_emit_partitioned_stage_bound_regions.py) | Phase SSI-2.D PR-C — cross-rank region shares tag (per-stage tag cache), single-rank region skips non-contributing ranks (INV-4), per-stage tag cache scoping across stages, global+stage tag disjointness. |
| [`tests/opensees/unit/test_stage_initial_stress_push.py`](../../../../tests/opensees/unit/test_stage_initial_stress_push.py) | Phase SSI-2.D extension — `s.initial_stress(...)` PUSH ↔ `s.add(record)` PULL byte-identical deck parity; return-record contract; stage-scoped error-message prefix. |
| [`tests/opensees/unit/test_stage_embedded_claim.py`](../../../../tests/opensees/unit/test_stage_embedded_claim.py) | Phase SSI-2.D extension — `s.embedded(name=...)` claim semantics: pool population, double-claim refusal, missing-name fail-loud, empty-name fail-loud, global-emit skip on claimed, unclaimed-record passthrough. |
| [`tests/opensees/unit/test_stage_constraint_siblings.py`](../../../../tests/opensees/unit/test_stage_constraint_siblings.py) | Phase SSI-2.D extension — smoke tests for `s.equal_dof` / `s.rigid_link` / `s.rigid_diaphragm` / `s.kinematic_coupling` / `s.tie` / `s.distributing` claim-and-route into stage block. |
| [`tests/opensees/unit/test_stage_constraint_e2e_2stage.py`](../../../../tests/opensees/unit/test_stage_constraint_e2e_2stage.py) | Phase SSI-2.D extension — Cerro Lindo SSI V5 forcing-function fix: 2-stage SSI deck structure with cimbra + embed inside stage 2, `domainChange` AFTER constraint emit and BEFORE analysis chain, no leak into stage 1 or pre-stage block. |
| [`tests/opensees/unit/test_stage_patterns.py`](../../../../tests/opensees/unit/test_stage_patterns.py) | ADR 0051 BL-3 — `s.pattern(series=)` records + claims a stage-owned `Plain`; load/sp lands inside the stage block (after chain, before `loadConst`); two stages independent; `from_model(case)` import inside a stage pattern (loads + prescribed-sp-only); global pattern unaffected; partitioned per-rank routing + empty-rank bracket skip. |

## Cross-references

- [api-design.md](api-design.md) — user surface for `ops.stage(...)`,
  `s.activate(...)`, `s.fix(...)` / `s.mass(...)` / `s.region(...)` /
  `s.recorder(...)` (SSI-2.D), `ops.initial_stress(...)`,
  `ops.convergence_confinement(...)`, `ops.imposed_displacement(...)`.
- [emitter.md](emitter.md) §"Phase SSI-1 analyze hook-wrapping" —
  the seven Protocol methods that ship in the staged emit
  (`addToParameter`, `step_hook_ramp`, `stage_open`, `stage_close`,
  `domain_change`) plus the `analyze` behaviour change.
- [decisions/0028-initial-stress-via-parameter-ramping.md](decisions/0028-initial-stress-via-parameter-ramping.md)
  — Phase SSI-1 design decision (parameter ramping over hand-rolled
  fiber prestress).
- [decisions/0029-staged-analysis-context-manager.md](decisions/0029-staged-analysis-context-manager.md)
  — Phase SSI-2.A design decision (context manager over
  declarative stage tuple).
- [decisions/0030-stage-bound-topology-activation.md](decisions/0030-stage-bound-topology-activation.md)
  — Phase SSI-2.B design decision (PG-level activation + lowest-
  index node ownership).
- [decisions/0031-ssi-convenience-helpers.md](decisions/0031-ssi-convenience-helpers.md)
  — Phase SSI-3 design decision (typed helpers
  `convergence_confinement` / `imposed_displacement` vs. raw
  composition).
- [decisions/0034-stage-bound-bcs-and-recorders.md](decisions/0034-stage-bound-bcs-and-recorders.md)
  — Phase SSI-2.D design decision (four-validator surface, PUSH-vs-
  PULL builder asymmetry, recorder claiming, per-stage region tag
  cache, empty-bracket skip + unified `domain_change` gate).
- ADR [0023](decisions/0023-per-zone-schema-versioning.md) — the
  per-zone schema policy any future archival of stages /
  initial-stress will bump.
- [_DEFERRED.md](_DEFERRED.md) §"Staged-analysis follow-ups" — the
  open items above with deferral rationale.
