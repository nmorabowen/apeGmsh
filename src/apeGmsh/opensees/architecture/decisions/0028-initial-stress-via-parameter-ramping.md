# ADR 0028 — Initial-stress injection via parameter ramping (Phase SSI-1)

**Status:** Accepted (Phase SSI-1, May 2026). First of a four-ADR
SSI feature set ([0028](0028-initial-stress-via-parameter-ramping.md)
/ [0029](0029-staged-analysis-context-manager.md) /
[0030](0030-stage-bound-topology-activation.md) /
[0031](0031-ssi-convenience-helpers.md)) driven by the Cerro Lindo
tunnel migration off STKO. Widens the `Emitter` Protocol — the
explicit "architecture event" the Protocol's header documents.

## Context

In-situ stress is the boundary condition for every geotechnical
simulation: a tunnel excavation only makes physical sense against
a rock mass that already carries the lithostatic / tectonic stress
field it acquired over geologic time. The standard STKO workflow
for installing this state on `ASDPlasticMaterial3D` elements is to
**ramp the committed stress tensor** in over `N` analyze steps
using OpenSees's parameter / addToParameter / updateParameter
mechanism, attached to the material's
`commitStressIncrement<XX|YY|ZZ|XY|YZ|XZ>` responses defined at
[`ASDPlasticMaterial3D.h:888-905`](../../../../../OpenSees_Compile/OpenSees/SRC/material/nD/ASDPlasticMaterial3D/ASDPlasticMaterial3D.h).

The STKO emit for one stress-control region (reference deck at
[`SSI/Interaccion/analysis_steps.tcl:358-507`](file:///C:/Users/nmora/seadrive_root/nmb/My%20Libraries/Cerro%20Lindo/Modelos/SSI/Interaccion/analysis_steps.tcl)):

```tcl
parameter 1                                       # one per axis
parameter 2
parameter 3
if {$STKO_VAR_process_id == 6} {
    foreach _e $_stressCtrl_6_elems_6 {
        addToParameter 1 element $_e "commitStressIncrementXX"
    }
}
# ... repeated per process_id, per axis ...

proc _stressCtrl_6 {} {
    global _stressCtrl_6
    if {![info exists _stressCtrl_6(count)]} {
        set _stressCtrl_6(count) 0
        set _stressCtrl_6(XX) 0.0
        set _stressCtrl_6(YY) 0.0
        set _stressCtrl_6(ZZ) 0.0
    }
    set _stressCtrl_6(count) [expr {$_stressCtrl_6(count) + 1}]
    set _stressCtrl_factor [expr {$_stressCtrl_6(count) / 1.0}]
    if {$_stressCtrl_factor > 1.0} { set _stressCtrl_factor 1.0 }
    set _stressCtrl_current [expr -6300.0 * $_stressCtrl_factor]
    set _stressCtrl_incr [expr $_stressCtrl_current - $_stressCtrl_6(XX)]
    updateParameter 1 $_stressCtrl_incr
    set _stressCtrl_6(XX) $_stressCtrl_current
    # ... YY, ZZ axes ...
}
lappend STKO_VAR_OnBeforeAnalyze_CustomFunctions _stressCtrl_6
```

Pre-SSI-1, an apeGmsh user driving the SSI tunnel migration had
three painful options:

1. Hand-write the parameter / addToParameter / proc Tcl after
   `apeSees(fem).tcl(p)` and concatenate. Fragile — any change to
   the emit format breaks the concatenation point.
2. Build the openseespy equivalent inline: `ops.parameter(...)` +
   `ops.addToParameter(...)` + a Python closure registered with
   `apeSees.run()` and a hand-rolled analyze loop. Loses the typed
   surface and the live / Tcl / Py / H5 emitter parity.
3. Pre-stress via a body-load gravity step. Wrong for elastic
   loading paths — gravity also activates the lining, which is the
   thing we're trying to test the SSI of.

None of these compose with the bridge's "declare once, emit to any
target" contract.

## Decision

### A new bridge method `apeSees.initial_stress(...)`

```python
def initial_stress(
    self, *,
    name: str,
    pg: str | None = None,
    elements: Iterable[int] | None = None,
    sigma_xx: float,
    sigma_yy: float,
    sigma_zz: float,
    ramp_steps: int,
    lambda_install: float = 1.0,
) -> InitialStressRecord: ...
```

Validation (build-time loud):

| Condition | Outcome |
|---|---|
| `name` empty | `ValueError` |
| `name` not a Tcl identifier (alphanumeric + `_`, can't start with a digit) | `ValueError` — name becomes a Tcl proc name |
| `(pg is None) == (elements is None)` | `ValueError` — XOR required |
| `ramp_steps < 1` | `ValueError` |
| `lambda_install ∉ (0, 1]` | `ValueError` |
| Duplicate `name` across the global pool + every stage's pool | `BridgeError` at `BuiltModel.emit` (red-team H2, see ADR 0029 §"Post-merge hardening") |

The method appends a frozen `InitialStressRecord` to the bridge and
returns it so the caller can pass the record to a stage block via
`s.add(record)` (see [ADR
0029](0029-staged-analysis-context-manager.md)). Non-staged
callers can ignore the return value — the record is already
registered for the flat emit.

### Widen the `Emitter` Protocol — two methods plus an `analyze` behaviour change

```python
class Emitter(Protocol):
    # ... existing methods unchanged ...

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None: ...

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None: ...

    # Behaviour change — once any step_hook_ramp has run on this
    # emitter, analyze MUST wrap its analyze loop with per-step
    # hook-dispatcher calls.
    def analyze(self, *, steps: int, dt: float | None = None) -> int: ...
```

The two emit-level helpers split the responsibilities cleanly:

- `addToParameter(tag, ele_tag, response)` is a **single-line
  per-rank-scoped** directive. Wrapped inside `partition_open(K)`
  blocks for MP-partitioned models so each rank emits only the
  `addToParameter` calls for elements it owns. The `"element"`
  token between `tag` and `ele_tag` in the OpenSees command is
  inserted by the emitter, not the caller — apeSees primitives
  never see Tcl vocabulary.
- `step_hook_ramp(name, targets, n_steps_to_full, phase)` is the
  **multi-line bundle**: dispatcher boilerplate (emitted once, on
  the first call across the emitter's lifetime), one `parameter
  <tag>` declaration per `(tag, target_value)` in `targets`, the
  per-step proc body, and the `lappend` registration. The
  `lambda_install` parameter of the user-facing
  `apeSees.initial_stress` is baked into each `target_value`
  (`target_value = sigma * lambda_install`), so the proc body
  always ramps `factor` 0 → 1.0.

The `analyze` behaviour change is what makes the ramp actually
advance. Without hook-wrapping, the registered closures / procs
would never fire and the parameter ramp would stay at zero forever.

### Concrete emitter matrix

| Emitter | `addToParameter` | `step_hook_ramp` | Hook-wrapped `analyze` |
|---|---|---|---|
| `TclEmitter` | `addToParameter $tag element $ele_tag $response` | One-shot dispatcher procs + `parameter` decls + `proc <name> {...}` body + `lappend _apesees_before_step_hooks <name>` | `for {set _apesees_i 0} {$_apesees_i < N} {incr _apesees_i} { _apesees_call_before_step; analyze 1 [Δt]; _apesees_call_after_step }` |
| `PyEmitter` | `ops.addToParameter(tag, "element", ele_tag, response)` | Same shape but as Python: closure body + `_apesees_before_step_hooks.append(...)` | `for _apesees_i in range(N): _apesees_call_before_step(); ops.analyze(1, Δt); _apesees_call_after_step()` |
| `LiveOpsEmitter` | Forward to `ops.addToParameter` in-process | Build a Python closure with captured per-hook state (count + cumulative per parameter) and append to the emitter's `_before_step_hooks` / `_after_step_hooks` lists | In-process loop: per step, fire `before` closures, `ops.analyze(1, Δt)`, fire `after` closures. Break on first non-zero return (matches `ops.analyze(N)` short-circuit). |
| `H5Emitter` | No-op — H5 archival deferred (see "Consequences") | No-op — deferred | Records `(N, Δt)` into the analyze attribute; no hook archival |
| `RecordingEmitter` | Capture as `("addToParameter", (tag, ele_tag, response), {})` | Capture as `("step_hook_ramp", (name,), {"targets":..., "n_steps_to_full":..., "phase":...})` | Capture as `("analyze", (), {"steps":N, "dt":Δt})` — observe high-level intent, not loop expansion |

### Build-time fan-out

Two helpers in [`_internal/build.py`](../_internal/build.py):

- `emit_initial_stress_global(records, emitter, tags)`: per record,
  allocate **three** parameter tags (XX/YY/ZZ) from the bridge
  allocator, build the `targets` tuple as
  `((xx_tag, sigma_xx * λ), (yy_tag, sigma_yy * λ), (zz_tag,
  sigma_zz * λ))`, then call `emitter.step_hook_ramp(...)`. Return
  the mapping `{record_name: (xx_tag, yy_tag, zz_tag)}` so the
  per-rank `addToParameter` fan-out can reach the same tags without
  re-allocating.
- `emit_initial_stress_addtoparameter(records, emitter, fem,
  name_to_param_tags, fem_eid_to_ops_tag, element_owner=None,
  partition_rank=None)`: per record, per element under the record's
  `pg` / `elements`, emit three `addToParameter` lines (one per
  axis, response strings `commitStressIncrementXX/YY/ZZ`). Honors
  the global `fem_eid_to_ops_tag` map (shared with the element
  fan-out and with #314's recorder pg= translation) so the
  `addToParameter` `ele_tag` argument lands on the right OpenSees
  element. Elements missing from the map silently skip (rank-foreign
  in MP, or user-supplied element list with stray ids).

The build pipeline wires the global helper into both `_emit_flat`
(single-partition) and `_emit_partitioned` (MP). In MP, the global
side runs once outside any `partition_open` and the
`addToParameter` fan-out runs inside each `partition_open(K)`
block, filtered by `element_owner == K`.

## Invariants

- **INV-1.** `Emitter.addToParameter` and `Emitter.step_hook_ramp`
  are on the Protocol; every existing and future emitter implements
  them. H5 / Recording / Live implement them as no-ops / capture /
  in-process closure respectively.
- **INV-2.** The `analyze` hook-wrap is **mandatory** once any
  `step_hook_ramp` has run on an emitter. The contract is: every
  ramp registered via `step_hook_ramp` MUST fire between every
  pair of `analyze` substeps until `stage_close` resets the flag.
  Without INV-2, the ramp would stay at zero and the deck would
  silently produce a constant-stress analysis instead of a ramped
  one — exactly the bug the acceptance test discriminates against
  at step 5.
- **INV-3.** Always **three** `parameter` declarations per record
  (XX/YY/ZZ), even when only one component is non-zero. The
  emitted `updateParameter` deltas for zero-target axes are 0.0 —
  harmless no-ops, but the parameter slots are reserved. This is a
  documented divergence from STKO's per-record `_stressCtrl_<N>`
  which allocates fewer parameters when fewer components are
  populated. Accepted because the bridge's typed surface promises a
  full stress tensor; reading "ramp σ_xx only" off the tuple of
  three would require ad-hoc compaction the user can't introspect.
- **INV-4.** `lambda_install` ramps the **target stress**, not the
  number of steps. The proc body always ramps `factor` 0 → 1.0
  over `ramp_steps`; the asymptote is `sigma_* × lambda_install`.
  Partial-install (convergence-confinement intermediate; see
  [ADR 0031](0031-ssi-convenience-helpers.md)) is therefore
  expressible without changing the proc shape.
- **INV-5.** The hook dispatcher state names (`_apesees_before_step_hooks`,
  `_apesees_after_step_hooks`, `_apesees_call_before_step`,
  `_apesees_call_after_step`) deliberately differ from STKO's
  `STKO_VAR_OnBeforeAnalyze_CustomFunctions`. A user mixing
  hand-written STKO blocks with apeSees-emitted decks must not see
  silent collisions; the prefix is the seam.
- **INV-6.** Per-record proc names are user-supplied via
  `name=`. The validation that `name` is a Tcl identifier
  (alphanumeric + `_`, not leading digit) is enforced at the bridge
  surface — the emitter trusts the validation. INV-6 is the
  reason `name=` is required, not auto-generated; users need control
  over the proc name to grep their decks.
- **INV-7.** H5 archival is **deferred and fail-loud**.
  `H5Emitter.addToParameter` / `step_hook_ramp` are no-ops. Because
  a silent-drop H5 round-trip would yield a non-staged flat model
  that no longer matches the declared one, `apeSees.h5(path)` is
  **guarded** (#313): it raises `NotImplementedError` when
  `self._initial_stress_records` is non-empty (or any stage is
  declared per [ADR 0029](0029-staged-analysis-context-manager.md)),
  pointing the user at `ops.tcl(path)` / `ops.py(path)` instead.
  A future `opensees_schema_version` bump (`2.11.0` → `2.12.0` per
  [ADR 0023](0023-per-zone-schema-versioning.md)) would persist
  initial-stress records under `/opensees/initial_stress/`, lift
  the guard, and restore round-trip parity.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Auto-generate proc names** (e.g. `_apesees_ramp_<region_hash>`) | Removes user control over proc names. Users grep their decks; opaque hashes would defeat that. The validation cost (Tcl-identifier check at one call site) is trivial; the visibility win is large. |
| **One Protocol method `initial_stress(rec)` instead of two helpers** | Conflates "register a global ramp proc" with "tag this element's commitStressIncrement". The two have different scopes — global (declarations + proc, emitted once) vs. per-rank (addToParameter calls, filtered by ownership). Two methods preserve the scope distinction at the Protocol level. |
| **Reuse STKO's identifier prefixes** (`STKO_VAR_*`) | Silent collisions when a user pastes a hand-written STKO block into an apeSees-emitted deck. INV-5 forbids. |
| **Make hook-wrapping opt-in** (separate `analyze_with_hooks` Protocol method) | Doubles the API surface for a behaviour that's MANDATORY once a ramp is registered (INV-2). Users who don't use `initial_stress` are unaffected — the flag stays `False` and `analyze` emits a bare line. The single-method surface is simpler. |
| **Allocate fewer parameters when a component is zero** (match STKO byte-for-byte) | Saves a few `parameter` declarations and three `updateParameter` no-ops per step. Adds branching to the proc body, breaks the documented "always three" invariant (INV-3), and would require a different code path for the convergence-confinement helper that legitimately drives only one axis. Accepted as a documented divergence instead. |
| **H5 archival of the ramp** (eager schema bump) | Would require designing the persistence shape (parameter group? proc body bytes? closure replay?) before any consumer exists. Deferred per INV-7 until a Results-side viewer or a model.h5 round-trip needs it. |

## Consequences

**Positive:**

- Closes the SSI prerequisite for the Cerro Lindo tunnel migration
  off STKO. The pre-SSI workarounds (hand-write Tcl tails;
  body-load gravity; in-process Python closures) are all eliminated.
- The bridge's typed surface extends naturally to a geotechnical
  use case it wasn't designed for. `apeSees(fem).tcl(p)` now
  produces a complete in-situ stress install for an
  `ASDPlasticMaterial3D` model — same `apeSees(fem)` contract
  whether the user is modelling a steel frame or a rock tunnel.
- The acceptance test
  [`tests/opensees/subprocess/test_initial_stress_acceptance.py`](../../../../tests/opensees/subprocess/test_initial_stress_acceptance.py)
  locks the FIXED ramp values within ±0.5 kPa per step against the
  empirical reference `result_fixed.csv` from the same Ladruno
  build hash (288f6d0f). The step-5 discriminator catches the
  historical STKO single-step-jump bug if it ever regresses.
- The per-record `name=` becomes part of the deck's grep-vocabulary.
  A `addToParameter $tag element $e commitStressIncrementXX` line
  in a 1000-line Tcl deck is easy to trace back to its
  `ops.initial_stress(name="rock_insitu", ...)` declaration in the
  Python deck-builder.

**Negative:**

- Emitter Protocol widens by two methods + one behaviour change.
  Every concrete emitter (current + future) must implement them.
  The Protocol's header documents this as expected.
- H5 archival is deferred (INV-7). `ops.h5(path)` on an SSI model
  raises `NotImplementedError` (#313) instead of writing an
  incomplete archive; users must keep their Python deck-builder
  around to reproduce the ramp. Documented in
  [staged-analysis.md](../staged-analysis.md) §"Deferred work" and
  in [api-design.md](../api-design.md) §"Initial-stress injection"
  caveats.
- The acceptance test is gated on the Ladruno binary + the
  reference CSV being on the local filesystem. CI under a fresh
  checkout skips by design — fine, because the discriminator is
  binary-specific and we don't want a CI gate that depends on a
  particular `OpenSees` build hash silently drifting under us.

## Cross-references

- ADR [0023](0023-per-zone-schema-versioning.md) — the per-zone
  schema policy any future H5 archival of the SSI-1 ramp will bump.
- ADR [0025](0025-emitter-protocol-widen-eigen.md) — the immediate
  precedent for a Protocol widening with the same lightweight
  ceremony. Eigen had one method + asymmetric return; SSI-1 has two
  methods + a behaviour change, same shape.
- ADR [0029](0029-staged-analysis-context-manager.md) — the
  staged-analysis ADR that consumes `addToParameter` /
  `step_hook_ramp` from inside per-stage blocks.
- [staged-analysis.md](../staged-analysis.md) §"Hook dispatcher" —
  the per-step Tcl + Py + Live emit walkthrough.
- [emitter.md](../emitter.md) §"Phase SSI-1 analyze hook-wrap
  behaviour" — the per-emitter matrix this ADR drives.
- [api-design.md](../api-design.md) §"Initial-stress injection" —
  the user-surface walkthrough this ADR backs.
- `ASDPlasticMaterial3D.h:888-905` — the OpenSees source for the
  `commitStressIncrementXX/YY/ZZ` responses the ramp wires to.
- `SSI/Interaccion/analysis_steps.tcl:358-507` — the STKO reference
  deck this ADR replicates the structure of (with the divergences
  documented in INV-3 / INV-5).
- `tests/opensees/subprocess/test_initial_stress_acceptance.py` —
  the empirical baseline lock for the ramped output.
