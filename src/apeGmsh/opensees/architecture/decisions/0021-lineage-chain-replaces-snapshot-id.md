# ADR 0021 — Lineage chain replaces `snapshot_id` binding; warn-not-raise

**Status:** Accepted (Phase 6 of the major architectural refactor,
May 2026). Ratifies and extends the May 2026 `project_bind_contract`
decision (no procedural `BindError` enforcement). Complements
[ADR 0019](0019-opensees-model-read-side-broker.md) and
[ADR 0020](0020-results-carries-opensees-model.md).

## Context

`snapshot_id` is a deterministic blake2b content-hash over `(nodes,
elements, PGs)` computed by `mesh/_femdata_hash.py` and stored on
every `FEMData` instance and on every emitted file's `/meta` attrs.
It was originally designed to **enforce** pairing on
`Results.bind(fem)`: if `fem.snapshot_id != results._meta_snapshot_id`,
raise `BindError` and refuse the bind.

That enforcement was withdrawn in May 2026 (user memory
`project_bind_contract`). The withdrawal was correct: legitimate
workflows (re-meshing with the same model intent, label refreshes,
PG-name changes that don't perturb geometry) trip the hash-equality
check and produce false negatives. The procedural `Results.bind(fem)`
flow accepted whatever the user passed; `BindError` survived as an
exported stub that nothing raises.

The result is structurally inconsistent:

- A user pairing a stale FEMData with a fresh results file gets **no
  diagnostic** — there is no signal at all that the pairing is wrong.
- The `snapshot_id` is **computed on every write** but **enforced
  nowhere**. It exists in the file as documentation of intent, not as
  a contract.
- The Phase 4 work ([ADR 0020](0020-results-carries-opensees-model.md))
  flips pairing from procedural (`bind(fem)`) to structural
  (`Results.model`). Structural pairing reduces the surface area of
  the mispairing bug, but doesn't address *whether the
  structurally-paired model and results agree on content*.

A user emails a peer `results.h5` (Composed-file per ADR 0020) and a
sibling `model.h5`. The peer opens the model.h5, edits a recorder,
saves over the original. The next time they open the results, the
embedded model in `results.h5` is now older than the standalone
`model.h5` — and there is no signal of the divergence.

The correct mental model is git's commit DAG: each layer's hash
depends on its parent's hash. You **cannot lie about history**; you
**can amend it freely**. Staleness is *visible* in the chain; never
*blocks* the workflow.

## Decision

### Three chained hashes — the lineage triple

Three deterministic hashes, stored under `/meta/lineage` in every
emitted file:

```
fem_hash      = blake2b-128(canonical_neutral_zone_bytes)
model_hash    = blake2b-128(fem_hash || canonical_opensees_zone_bytes)
results_hash  = blake2b-128(model_hash || canonical_run_zone_bytes)
```

`fem_hash` is the existing `snapshot_id` semantics unchanged — bytes
over the neutral zone in name-sorted, attribute-canonical order.

`model_hash` chains `fem_hash`: changing the FEM mechanically changes
the model hash, even if the OpenSees zone bytes are byte-equal. This
is the git-parent property that makes the DAG meaningful — you cannot
have a `model_hash` that doesn't trace back to its FEM.

`results_hash` chains `model_hash`: same property at the next layer.

### Canonical bytes — name-sorted, h5py-version-stable

The "canonical bytes" for each zone walk `h5py.Group` children in
name-sorted order, hashing:

1. Each dataset's raw bytes (`np.asarray(ds).tobytes()` post-
   sort-by-name).
2. Each attribute's bytes (sort attribute names; serialize each
   value with a stable representation — strings as UTF-8, ints/floats
   as IEEE-754 bytes, arrays as raw bytes).
3. Recurse into subgroups in name-sorted order.

Raw `h5py` file-level bytes are not stable across versions or write
orderings (creation timestamps, btree balancing, chunk layout
heuristics). The canonical walk is. Documented in
`_internal/lineage.py` with a unit test that asserts two `to_h5` calls
of the same model produce equal hashes across:

- different h5py minor versions (within the two-version reader window
  of ADR 0023),
- different write orderings (the same `OpenSeesModel.to_h5` called
  twice should produce identical bytes),
- chunked vs contiguous dataset layouts.

### `model_hash` scope — opensees zone minus cuts/sweeps

`model_hash` covers `/opensees/{materials, sections, transforms,
beam_integration, patterns, recorders, analysis, element_meta,
constraints}`.

It does **not** include `/opensees/cuts` or `/opensees/sweeps`. Those
are user-attached post-hoc artifacts that change without the model
definition changing — a user adds a cut to inspect a result; the model
hash is invariant. Coupling them would produce false-positive
lineage warnings on every cut edit.

If finer-grained tracking is needed later, a separate `cuts_hash`
attribute under `/meta/lineage` can be added as a non-chaining
sibling. Out of scope for this ADR; not chained into the triple.

### Surface — warn, not raise

`OpenSeesModel.lineage` exposes `(fem_hash, model_hash)` and
`lineage.warnings: list[str]`. `Results.lineage` exposes the full
triple `(fem_hash, model_hash, results_hash)` and its own warnings.

On every read (`from_h5`, `from_native`, `from_mpco`), the loader:

1. Reads `/meta/lineage` attrs into memory.
2. Recomputes the canonical bytes for the zones present in the file.
3. Compares stored vs recomputed. On mismatch, appends a warning to
   `lineage.warnings` with the prefix `"[lineage] "` and a human-
   readable description of the divergence.

The mismatch **does not raise**. The mismatch **does not block**.
The mismatch **is visible**. Users who want to fail loudly opt in via
`results.lineage.assert_clean()` (a one-liner that raises if any
warning was attached).

### Phase 8 prune — delete `BindError`

The inert `BindError` class (exported but never raised) is deleted in
the Phase 8 prune. No shim period. User memory `project_bind_contract`
is the ratification; this ADR is its structural implementation.

### Invariants

**INV-1.** `fem_hash` semantics are byte-identical to today's
`snapshot_id` for the neutral zone. Existing files written under
`schema_version` 2.x continue to validate; the migration is purely
additive (new `model_hash` / `results_hash` attrs).

**INV-2.** Lineage mismatches **never raise** from a constructor or
loader. They produce warnings only. Re-introducing
`raise BindError(...)` on lineage mismatch requires repealing this
ADR.

**INV-3.** The hash chain is one-directional and tamper-evident: a
`results_hash` is bound to its `model_hash`, which is bound to its
`fem_hash`. Changing any layer's bytes without updating the chain
produces a recompute-mismatch warning. The user cannot "edit the
model" and "keep the results valid" without re-running.

**INV-4.** `model_hash` does **not** include cuts or sweeps. A
cut-edit workflow produces zero lineage warnings. Future cut-tracking
is via a separate `cuts_hash` attribute, not in this chain.

**INV-5.** Canonical-bytes definition is version-stable across the
two-version reader window (per ADR 0023). The
`tests/test_lineage_determinism.py` test enforces this across h5py
minor versions in CI.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Re-enforce `snapshot_id`-equality on `bind`** | Already rejected May 2026 (user memory `project_bind_contract`). Blocks legitimate re-meshing workflows; produces false-positive failures on PG-name changes that don't perturb geometry. The withdrawal is correct; this ADR ratifies it. |
| **Hash only FEM (status quo — no `model_hash`)** | Loses the bridge-side staleness signal. The exact "peer edited the model.h5 between sends" scenario described in Context becomes silent again. |
| **No hashing at all — delete `snapshot_id`** | Loses the diagnostic *"are these two files from the same run?"* and forecloses any future lineage tooling. The user explicitly chose warn-not-raise; this is the dual extreme that warn-not-raise rejects. |
| **Include cuts in `model_hash`** | Cuts are user-attached post-hoc artifacts. Coupling them into the model identity means every cut edit produces a lineage warning on every viewer open — noise that masks real staleness. The separate `cuts_hash` (if needed) is the right scope. |
| **Hash on Python-level objects (record-graph hashes) instead of on canonical HDF5 bytes** | Two valid representations of the same model in memory (different traversal orderings, different cache states) could produce different hashes. Hashing on bytes is the canonical representation; it makes the hash equivalent to "did the on-disk file change." |
| **Strict-mode flag on `from_h5(strict=True)` that raises on mismatch** | Two ways to do it. The opt-in raise is already covered by `lineage.assert_clean()`. A flag is just procedural-bind under a different name. |

## Consequences

**Positive:**

- Git-style DAG: mismatch is **visible**, **queryable**, **never
  raises**. The user gets a signal where today there is silence.
- Replaces the inert `BindError` stub with something useful. The
  Phase 8 prune cleans up the unused export.
- Lineage tooling becomes buildable. `diff_lineage(model_a, model_b)
  → list[str]` is a small wrapper over the canonical-bytes walk; the
  ADR doesn't ship it but unblocks it.
- Composed-file (ADR 0020) and standalone `model.h5` both annotate
  with the same chain; the user can compare a peer's `results.h5`
  against their own `model.h5` and see whether they agree.

**Negative:**

- Hash computation cost on every write (~10-50ms for typical models;
  larger models are linear in HDF5 byte count). Acceptable —
  recompute happens at file open and at file write, both of which are
  already dominated by HDF5 I/O.
- "Canonical bytes" is non-trivial; needs careful unit-testing for
  determinism across h5py minor versions, write orderings, and dataset
  layout choices. INV-5 makes this a CI gate.
- Lineage warnings appear in tests that have always silently
  mispaired. Those tests must be updated to either fix the pairing or
  explicitly assert-and-suppress the warning. Expected pain: ~5-15
  legacy tests that bind a known-stale fixture. Migration is one-shot.
- The warn-not-raise design relies on the user reading
  `lineage.warnings`. Documented in the public API; suggested in the
  `OpenSeesModel.from_h5` and `Results.from_native` docstrings.
  Users who don't read it get the same silence they have today, plus
  an opt-in upgrade path (`lineage.assert_clean()`).

## Open questions

- **Q6 — `model_hash` scope.** Resolved (this ADR): excludes cuts and
  sweeps. `cuts_hash` is deferred to a future ADR if the use case
  arises; the chain itself is locked at three layers.
- **Q7 — two-version reader window edges.** Cross-reference:
  [ADR 0023](0023-per-zone-schema-versioning.md) owns the precise
  schema-version window semantics. This ADR's INV-5 requires lineage
  determinism *within* the window; ADR 0023 defines the window.

## References

- [decisions/0019-opensees-model-read-side-broker.md](0019-opensees-model-read-side-broker.md)
  — the `OpenSeesModel.lineage` surface this ADR specifies.
- [decisions/0020-results-carries-opensees-model.md](0020-results-carries-opensees-model.md)
  — the `Results.lineage` surface; the Composed-file annotated by
  this chain.
- [decisions/0023-per-zone-schema-versioning.md](0023-per-zone-schema-versioning.md)
  — the schema-version window within which canonical-bytes
  determinism is required.
- [phase-8-untangle.md](../phase-8-untangle.md) §7 closure — the
  lineage / `snapshot_id` question this ADR resolves.
- User memory `project_bind_contract` — the May 2026 ratification
  this ADR is the structural implementation of.
- `mesh/_femdata_hash.py` — today's `snapshot_id` implementation; the
  basis for `fem_hash`.
