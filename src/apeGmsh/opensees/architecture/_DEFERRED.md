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
[0031](decisions/0031-ssi-convenience-helpers.md)) ships the
declarative `ops.stage(name)` / `s.activate(pgs=)` /
`ops.initial_stress(...)` surface, the runnable Tcl / Py text
emit, and (per Phase SSI-2.C) the combined partitioned + staged
emit. Three follow-ups are still explicitly deferred:

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

### Stage-bound `fix` / `mass` / `region` directives

Currently refused at build time by
`_validate_no_stage_bound_node_targets` (red-team H1 hardening —
[ADR 0029 §"Build-time fan-out"](decisions/0029-staged-analysis-context-manager.md)).
The pre-stage global emit fires before any `stage_open`; a `fix N
1 1` line referencing a node that only emits in stage 2 would
reference a non-existent OpenSees node and crash at parse time.

The workaround is to keep the BC on a globally-emitted node — for
geotechnical models, the rock-mass boundary nodes are typically
global so this is usually fine. A future phase would add
stage-bound BCs by extending `StageRecord` with `fix_records` /
`mass_records` / `region_records` fields and a per-stage emit pass
that fires after the stage's `domain_change` (so the BC targets
exist).

Lives in `apesees.py::_validate_no_stage_bound_node_targets`
(currently raises with offender list); would need additional
fields on `StageRecord` at `_internal/build.py:161-212`.

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

