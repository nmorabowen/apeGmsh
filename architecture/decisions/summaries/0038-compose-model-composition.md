# ADR 0038 — `g.compose()` model composition: flat-after-merge with namespaced tag-offset import (summary)

**Status:** Accepted (2026-05-26, Phase 3 of the model-composition work stream). […]

## Decision

- **Signature.** `g.compose(source, *, label, translate=(0,0,0), rotate=None, anchor=None,
  partition_rank=None, properties=None) -> ComposedModule`. `source` is an H5 path only (a
  prior `g.save()`). `label` is required and validated (`ComposeLabelError`); `rotate` is
  axis-angle `(x, y, z, theta)`; `anchor` (a host PG centroid) and a non-zero `translate` are
  exclusive (`ComposeAnchorError`). Helpers: `g.compose_inspect(path)`, `g.compose_list()`;
  `g.uncompose` / `g.recompose` are deferred.
- **Merge verdicts per record kind.** IMPORT (tag-offset + namespace): nodes, elements, PGs,
  labels, mesh selections, parts sub-records, constraints, loads, masses, materials, sections,
  integration rules, per-element-type assignments. DISCARD (re-derived): regions, cuts, sweeps
  (`MODEL_HASH_EXCLUDED_CHILDREN`), the module's `PartitionSet`. FILTER + `UserWarning`:
  stages, time-series, load patterns. FILTER silently: recorders, analysis settings,
  `/results/`.
- **Namespace rule.** Every string-keyed record is prefixed `{label}.`; a module may not reach
  into the host or another module. Nested compose: `max_compose_depth=3`
  (`ComposeDepthExceededError`), `.` ↔ `/` separator alternation at depth boundaries
  (`bldg_1.frame/beam_A.end`), provenance grafted; detection by `H5Lexists('/fem/composed_from')`.
  `ComposeNestedError` is removed; `ComposeNamespaceCollisionError` is added.
- **Tag offsets.** Per-module window `size = ceil(span / GRANULARITY) * GRANULARITY`
  (`GRANULARITY = 1_000_000`, override `compose_size_per_module=N`), `base` after the host's
  max tag, then cumulative; `new_tag = old_tag + (base - source_min_tag)` for every
  tag-bearing kind; span from `/fem/@tag_span_max`. The cover set (`tag_rewrite_spec`) lists
  every integer tag field, reference fields such as `mat_tag` included; a new record kind
  updates it in lockstep.
- **Tag-collision verifier**, five checks: imported tags above the host range; module ranges
  disjoint; constraint references inside their module's window; no PG-name collision
  (`PartTagCollisionError` for 1–4); span fits the reservation (`ComposeCapacityError`).
- **Rank model, 3 layers, eager populator:** host on rank 0 and one rank per module;
  `partition_rank=K` hint; an explicit `g.mesh.partitioning.partition(N)` overwrites. Last
  write to `fem.partitions` wins, and every overwrite warns.
- **Lineage.** `compose_hash()` sorts module records by `module_label` (compose order does not
  change it); `/composed_from/{label}/` is informational; drift uses `LineageStaleWarning`.
- **v1 scope gate.** A cross-rank-constraint cost benchmark (10k × 4 thresholds) decides full
  feature, `WARN_INTERFACE_SIZE = 50_000`, or a mesh-cache-only fallback where any
  cross-module MP constraint raises `ComposeUnsupportedError`.

Invariants: **INV-1** the post-compose `FEMData` is a flat canonical broker; composed and
directly assembled models give the same `compose_hash()`. **INV-2** tag uniqueness; reference
fields are explicit rewrite targets; sibling cross-module references are forbidden.
**INV-3** namespacing is total, with `/` at depth boundaries; `g.parts.add()` dotted names
are leaf labels. **INV-4** rank assignment is deterministic given the operation sequence.
**INV-5** provenance is informational; sources are never re-fetched. **INV-6** analysis-time
settings are never inherited (conditional under the scope-gate fallback). **INV-7** neutral
schema 2.8.0 → 2.9.0, additive-minor (ADR 0023).

## Amendments

- 2026-05-27, flat graft instead of tree graft (PR #369): nested provenance lands as flat
  top-level `/fem/composed_from/{joined_label}/` groups. The hash (`_hash_composed_from`) is
  the one-way door and must stay a flat fold over sorted joined labels; `g.compose_tree()`
  (PR #370) is the derived tree view, which makes separator alternation load-bearing
  (`ComposeLabelError` forbids `.` and `/` in labels).
- 2026-10-10, public entry points removed (ADR 0117 D7, AS5-c): `g.compose`,
  `apeGmsh.compose`, `FEMData.compose` and the v1 `Assembly.add / couple / materialize` are
  gone; the engine stays behind the private `mesh._compose._compose_module`, which
  `Assembly.bridge` calls. Host asymmetry retired (every instance namespaced); separator
  alternation stands; the materials row corrected (they travel by `/opensees` rehydration,
  ADR 0117 D4, never through this engine).

Full text: [../0038-compose-model-composition.md](../0038-compose-model-composition.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
