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
   for damping, recorders, etc.
2. **`.disp_history()`** — pull recorder output for this node
   after analysis.
3. **`.get_reaction()`** — query post-analysis reaction forces.
4. **`.coupled_dofs()`** — list MP_Constraints that touch this node
   (rigid links, equal_dof, etc.).
5. **`.partition`** — owning partition tag for parallel runs.

## Asymmetric-section warning at `geomTransf` build

Today's CS resolver emits the correct vecxz, but the leg sign-flip
on arches is inherent. For symmetric sections (W12, RHS, circular)
the sign flip is harmless. For asymmetric sections (channels, angles,
T-sections) it produces inconsistent physical orientation across the
legs.

Deferred capability:

- At build time, detect when an element using an asymmetric section
  hits the degenerate branch of the CS resolver.
- Emit a warning naming the affected PG and suggesting `roll_deg`
  on one of the legs.

Lives in `_internal/build.py` once we get there.

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

## Nodal capability `.disp_history` and `.element.disp_history`

Pulls back recorder output. Requires a registered `Recorder`
matching the query. Lives in the existing Recorders system; we
expose a typed query layer once the bridge is shipped.

## Multi-pattern aggregation

`Pattern` instances aggregate their loads after the `with` block
closes. A future capability is cross-pattern queries on a Node:
"show me every load this node has received across all patterns."
Defer until users ask.

## `H5Emitter` implementation

The schema and viewer contract are settled
([h5-schema.md](h5-schema.md), [viewer-integration.md](viewer-integration.md),
[ADR 0011](decisions/0011-h5-as-fourth-emit-target.md)). The
implementation rides with the other emitters when the bridge
skeletons land:

- Implement `apeGmsh.opensees.emitter.h5.H5Emitter` against the
  Protocol in `emitter/base.py`.
- `apeSees.h5(path)` convenience method on the bridge.
- Test fixtures listed in `viewer-integration.md` § "Test fixtures."
- Reference reader (validation, schema-version check) shipped in
  the bridge package for the viewer team to borrow.

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
(in-plane radial / circumferential axes). Today the build step
raises if `orientation=` is supplied with `ndm=2`. Lift the restriction
when we add 2-D-specific tests.

