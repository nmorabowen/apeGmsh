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

## Two phases, one builder

The staged-analysis work ships in two phases:

- **Phase SSI-2.A — staged analysis chain.** Adds `ops.stage(name)`
  + the `_StageBuilder` context manager + the `StageRecord`
  dataclass. Per stage emits its own analysis chain + analyze loop +
  `stage_open` / `stage_close` bracket.
- **Phase SSI-2.B — stage-bound topology activation.** Adds
  `s.activate(pgs=[...])`. Elements whose PG is in the activation
  set (and the nodes referenced exclusively by those elements) emit
  inside the stage block, between `stage_open` and `domain_change`.

The user surface is unified — `_StageBuilder` carries `activate` as
one verb in the same context manager. The deck layout below shows
the combined effect.

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
domainChange              # tell OpenSees to rebuild the DOF map
...                       # stage's initial_stress + analysis chain
                          # + analyze loop + stage_close
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
   - `domain_change()` — only if the stage added any topology.
   - Stage's `initial_stress` records — `parameter` declarations +
     `step_hook_ramp` proc bodies + `addToParameter` calls. Same
     shape as the Phase SSI-1 non-staged global emit, scoped here.
   - Stage's analysis chain — each of `constraints / numberer /
     system / test / algorithm / integrator / analysis` emitted via
     its registered primitive's `_emit`.
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

## Validation surface (post-merge hardening — PR #312)

Three guard rails ship with the staged path:

- **H1 — fix/mass/region on stage-bound nodes**: validated in
  `BuiltModel._validate_no_stage_bound_node_targets`
  ([apesees.py:1084-1140](../apesees.py)). Raises `BridgeError`
  with an offender list naming each `(kind, target, node,
  stage)` tuple. Without this, the `fix N 1 1` line emits in the
  pre-stage block, references a node that only comes into being in
  stage 2, and OpenSees errors at parse time with a less-helpful
  message.
- **H2 — duplicate `initial_stress` name across stages**: validated
  in `BuiltModel.emit` ([apesees.py:390-419](../apesees.py)). Raises
  `BridgeError` naming both owners. Without this, the second
  `proc <name>` definition overrides the first, but each was built
  with different parameter tags + cumulative-state keys, so the
  surviving proc would reference an uninitialised
  `${name}_state(cum_<tag>)` array element and crash at the first
  analyze step of the later stage.
- **M4 — nested `with ops.stage(...)` blocks**: validated in
  `apeSees.stage` ([apesees.py:2296-2306](../apesees.py)). Raises
  `RuntimeError`. Without this, the inner builder's `__exit__`
  fires first and registers the inner stage **before** the outer
  one in `_stage_records` — reverse of lexical order, which is the
  opposite of what readers expect.

## Deferred work

| Item | Where it would land |
|---|---|
| **Live execution of staged models** — `apeSees.analyze` / `apeSees.eigen` currently raise `NotImplementedError` when stages are present. Lifting requires staging the analysis-chain re-binding, per-stage analyze loops, `loadConst` / `wipeAnalysis` interleaving, and hook-list clearing inside `LiveOpsEmitter`. Tcl + Py text emit are the supported execution paths. | `emitter/live.py::stage_open` / `stage_close` (currently raise); `apesees.py::analyze` / `eigen` (currently refuse). |
| **H5 archival of staged structure + initial_stress** — `H5Emitter` no-ops on `addToParameter` / `step_hook_ramp` / `stage_open` / `stage_close` / `domain_change`. Because that silent-drop would round-trip into a non-staged flat model, `apeSees.h5(path)` is **guarded** (#313) — it raises `NotImplementedError` when `self._stage_records` or `self._initial_stress_records` is non-empty, pointing the user at `ops.tcl(path)` / `ops.py(path)`. A future schema bump (per [ADR 0023](decisions/0023-per-zone-schema-versioning.md)) from `opensees_schema_version` `2.11.0` → `2.12.0` would persist per-stage primitive lists + initial-stress records under `/opensees/stages/` and `/opensees/initial_stress/`, lift the guard, and restore round-trip parity. | `apesees.py::h5` (bridge-side guard); `emitter/h5.py::addToParameter / step_hook_ramp / stage_open / stage_close / domain_change` (schema-side no-ops). |
| **Stage-bound `fix` / `mass` / `region` directives** — currently refused at build time (H1 validator). A future phase would let the user attach BCs to stage-bound nodes that emit inside the stage block after `domain_change`. Today the workaround is to keep the BC on a globally-emitted node. | `apesees.py::_validate_no_stage_bound_node_targets` (currently raises); needs a `StageRecord.fix_records` / `.mass_records` field + per-stage emit pass. |

## File map

| Concern | Source |
|---|---|
| User surface (`ops.stage`, `_StageBuilder`) | [`apesees.py:2125-2760`](../apesees.py) |
| `StageRecord` dataclass | [`_internal/build.py:161-212`](../_internal/build.py) |
| `InitialStressRecord` dataclass | [`_internal/build.py:215-266`](../_internal/build.py) |
| Per-stage emit pipeline (single-partition) | [`apesees.py::_emit_stages_flat`](../apesees.py) |
| Per-stage emit pipeline (MP — Phase SSI-2.C) | [`apesees.py::_emit_stages_partitioned`](../apesees.py) |
| `apeSees.h5` fail-loud guard (#313) | [`apesees.py::h5`](../apesees.py) |
| Ownership computation | [`_internal/build.py::compute_stage_ownership`](../_internal/build.py) |
| Tag pre-allocation | [`_internal/build.py::allocate_element_tags`](../_internal/build.py) |
| Initial-stress global emit | [`_internal/build.py::emit_initial_stress_global`](../_internal/build.py) |
| Initial-stress `addToParameter` fan-out | [`_internal/build.py::emit_initial_stress_addtoparameter`](../_internal/build.py) |
| Tcl emitter SSI methods | [`emitter/tcl.py:350-485`](../emitter/tcl.py) |
| Py emitter SSI methods | [`emitter/py.py:322-423`](../emitter/py.py) |
| Live emitter SSI methods + raises | [`emitter/live.py:296-521`](../emitter/live.py) |
| H5 emitter no-ops (deferred archival) | [`emitter/h5.py:1152-1202`](../emitter/h5.py) |
| Recording emitter capture | [`emitter/recording.py:252-288`](../emitter/recording.py) |
| Build-time validators (post-merge hardening) | [`apesees.py::_validate_no_stage_bound_node_targets`](../apesees.py), [`apesees.py::emit:390-419`](../apesees.py) |

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

## Cross-references

- [api-design.md](api-design.md) — user surface for `ops.stage(...)`,
  `s.activate(...)`, `ops.initial_stress(...)`,
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
- ADR [0023](decisions/0023-per-zone-schema-versioning.md) — the
  per-zone schema policy any future archival of stages /
  initial-stress will bump.
- [_DEFERRED.md](_DEFERRED.md) §"Staged-analysis follow-ups" — the
  open items above with deferral rationale.
