# Plan B: `Instance.entities` as a computed label-backed view

**Status:** Deferred (2026-04-23). Not in v1.0.3.

## Context

`Instance.entities` is currently a cached `dict[int, list[int]]` populated
at registration and rewritten in place by every boolean result-map walk.
This caches a view of geometry state; the cache can drift out of sync
when entities change under the Instance's feet — typically when a user
calls `g.model.boolean.fragment/cut/fuse/intersect` directly on tags that
happen to belong to a tracked Instance, bypassing the Parts-level methods
that do their own remap.

v1.0.2 fixed the symptom by adding an umbrella label per Instance (the
`label_names[-1]` entry) so callers that need live geometry for an
Instance can already query `g.labels.entities(inst.label)`. That gave us
a label-backed source of truth independent of the cache.

v1.0.3 (this PR) fixes the footgun by wiring every `_bool_op` call
through `PartsRegistry._remap_from_result`, so the cache now stays
consistent whether the user hits `g.parts.fragment_*` or
`g.model.boolean.*`. Plan B (replace the cache with a computed property)
was the more structurally honest alternative.

## Plan B (what was considered)

Redefine `Instance.entities` as a `@property` that resolves the Instance's
umbrella label through `g.labels.entities(self.label)` each time it's
read, partitioning by dim. The cache goes away entirely; the Instance is
a thin bookkeeping record around a label.

### Upside

- One source of truth. No snapshot/remap dance. The cache cannot lie.
- Every boolean that updates physical-group entities (which includes
  `_label:*` PGs) is automatically reflected in `inst.entities` with
  zero extra plumbing.
- Deleting the cache removes ~15 lines of remap logic from the
  fragmentation mixin.

### Downside (why it was deferred)

- Every reader of `inst.entities` pays a Gmsh query. The API has many
  readers (viewer, constraints resolver, FEM broker partitioning, tests)
  and some iterate in hot loops. Cache-free reads are measurably slower
  on CAD-heavy models.
- A property cannot be mutated in place. A lot of existing code writes
  `inst.entities[dim] = [...]` directly — the `_PartsRegistryMixin` does
  it, and so do some tests. Every one of those sites would need an audit
  and a rewrite to go through a setter helper, or the mutation contract
  changes silently.
- The umbrella label is created at the top dim only (see
  `_import_cad` → `labels_comp.add(top_dim, entities[top_dim], name=label)`).
  Instances that track entities at multiple dims (e.g. a solid and its
  named boundary faces) would need label coverage at every dim, which
  means extending `_import_cad` and `register()` to register per-dim
  labels, not just one umbrella.
- Non-imported instances (`g.parts.part("x"):` context manager,
  `g.parts.register(...)`) currently do not create an umbrella label.
  They would have to, or the property would return empty dicts for
  them — a silent behavior change.

The caller weighed these and chose the conservative path for v1.0.3: fix
the footgun via the remap helper, keep the cache.

## Signals that would trigger revisiting Plan B

Revisit when **any** of these is true:

1. A second bug surfaces that is clearly caused by stale
   `inst.entities` (i.e. the remap helper missed a call site). Two
   incidents in the same shape is a structural problem, not a coverage
   gap.
2. The remap logic accumulates branches for absorbed-into-result,
   mixed-dim inputs, or cut semantics — at the point where the helper
   itself becomes a maintenance burden bigger than the property rewrite
   would have been.
3. We add a third boolean entry point that doesn't go through `_bool_op`
   (e.g. a gmsh-native call path inside a new importer). Three places
   needing to remember to call the remap helper is enough duplication
   to warrant removing the cache.
4. A measurable performance gap makes the property-based access viable
   without regression — e.g. if we cache umbrella-label PG lookups at
   the Labels layer, or if Gmsh 4.x batches `getEntitiesForPhysicalGroup`
   cheaply enough that the per-call cost stops mattering.

When any of (1)–(4) hits, the revisit is:

- [ ] Add per-dim umbrella labels in `_import_cad`, `register()`, and
      the `part()` context manager so every Instance has a label at every
      dim it owns.
- [ ] Replace `Instance.entities` with a `@property` that partitions
      `labels_comp.entities(self.label, dim=d)` for each d in 0..3.
- [ ] Audit `inst.entities[dim] = ...` mutation sites — route through a
      setter that updates the underlying label membership.
- [ ] Delete `_remap_from_result` and the `_bool_op` wiring added in
      v1.0.3.
- [ ] Benchmark viewer + constraint resolver on a large CAD session.

Until then: this PR's remap helper is the contract.
