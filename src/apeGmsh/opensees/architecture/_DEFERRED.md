# Deferred items

Things we've discussed and consciously held for later. Not bugs, not
backlog — concrete capability ideas that earn their own scope when a
real model needs them.

## Node aggregator capabilities (v1 ships lean)

`Node` for v1 exposes:

- `.coords`, `.tag`
- `.fix(dofs=...)` (model-level)
- `.mass(values=...)` (model-level)
- `.load(forces=...)` inside a pattern context

Held for later, in priority order:

1. **`.region(name)`** — assign the node to an OpenSees Region
   for damping, recorders, etc.  The `Emitter.region()` Protocol
   method shipped with ADR 0024; the typed-primitive surface
   re-uses it.
2. **`.disp_history()` / `.element.disp_history()`** — pull recorder
   output for a node or element after analysis.  Requires a
   registered :class:`Recorder` matching the query; the typed
   query layer composes over the existing Recorders system.
3. **`.get_reaction()`** — query post-analysis reaction forces.
4. **`.coupled_dofs()`** — list MP_Constraints that touch this node
   (rigid links, equal_dof, etc.).
5. **`.partition`** — owning partition tag for parallel runs.

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

## Recorder Materializer protocol — deferred until trigger

`emit_recorder_spec` in `_internal/build.py` currently has 4 isinstance
branches (RecorderDeclaration / Node+pg / Element+pg / MPCO+selectors)
plus the `_region_tag` underscore-prefix-on-frozen-dataclass channel
used by `dataclasses.replace` to pass build-time state through MPCO.
When the recorder dispatch grows to a 5th branch (e.g. a Drift
recorder, an EnvelopeNode recorder, or any new `pg=`-style selector
on an existing recorder), refactor to a `Recorder.materialize(fem,
tags) -> Recorder` hook on each primitive.  `emit_recorder_spec`
then collapses to a 3-line dispatcher:

    def emit_recorder_spec(spec, emitter, tag, fem, *, tags=None):
        if isinstance(spec, RecorderDeclaration):
            return _emit_recorder_declaration(spec, emitter, fem)
        spec.materialize(fem, tags)._emit(emitter, tag)

Side benefit: the `_region_tag` channel becomes a normal field set
inside `MPCORecorder.materialize()` — co-located with the only code
that writes it, no more `dataclasses.replace` indirection.

Estimated cost when triggered: ~80 LOC across recorder primitives +
build pipeline. Risk: low.  DO NOT refactor speculatively — wait for
the trigger.
