# Phase 8.4 — `model.h5` zone reshuffle

**Status:** Scoping (May 2026).  Drafted after Phase 8.3b shipped
(PR #134 + #135).  This phase moves the bridge-written groups in
`model.h5` under a `/opensees/` namespace so the broker can own the
root and a future second solver can plug in without colliding.

This is the **breaking** schema change called out by the master
plan ([phase-8-untangle.md §5](phase-8-untangle.md)) — schema_version
goes from `1.1.0` to `2.0.0` and the reader's `EXPECTED_SCHEMA_MAJOR`
goes from `1` to `2`.

## 1. The change

### 1a. Groups that move

The H5 writer ([emitter/h5.py](../emitter/h5.py)) currently creates 11
groups at the root of `model.h5`.  After Phase 8.4, **9 of them**
move under `/opensees/`:

| Current path | Phase 8.4 path | Notes |
|---|---|---|
| `/meta` | `/meta` | Stays at root — solver-neutral metadata. |
| `/bcs` | `/opensees/bcs` | OpenSees-resolved `fix` / `mass` compound datasets. |
| `/materials` | `/opensees/materials` | `uniaxial` / `nd` sub-families. |
| `/sections` | `/opensees/sections` | |
| `/transforms` | `/opensees/transforms` | |
| `/beam_integration` | `/opensees/beam_integration` | |
| `/elements` | `/elements` | **Stays at root** — neutral-zone group per master plan §3.  Phase 8.5 will swap the writer from the bridge to the broker; for 8.4 the bridge keeps writing it at root. |
| `/time_series` | `/opensees/time_series` | |
| `/patterns` | `/opensees/patterns` | |
| `/recorders` | `/opensees/recorders` | |
| `/analysis` | `/opensees/analysis` | |

The `/opensees/tag_map` group called out by the master plan is a
**Phase 8.6** addition — out of scope here.  This phase only moves
existing groups; it does not add new ones.

### 1b. Source-side consumers

Surprisingly narrow — the audit found **zero** runtime consumers of
`model.h5` outside the emitter package itself:

| File | Role |
|---|---|
| `src/apeGmsh/opensees/emitter/h5.py` | Writer.  11 `f.create_group(...)` sites for the root-level groups; 9 need rewriting. |
| `src/apeGmsh/opensees/emitter/h5_reader.py` | Reference reader.  10 typed accessors (`materials()`, `sections()`, …, `analysis()`) + `validate()` walks `/materials` / `/sections` / `/patterns` for cross-ref checks. `EXPECTED_SCHEMA_MAJOR` constant. |

No `results/` or `viewers/` module imports `h5_reader` today — the
viewer-integration contract documented in
[viewer-integration.md](viewer-integration.md) is aspirational.  This
makes the blast radius of the breaking change much smaller than the
master plan's "breaks any external Phase-6 consumer" line implies.

### 1c. Test-side consumers

| File | Touches |
|---|---|
| `tests/opensees/h5/test_h5_emitter.py` | ~20 literal-path assertions: `f["bcs/fix"]`, `f["materials/uniaxial/Steel02_1"]`, `f["sections/ElasticMembranePlateSection_2"]`, `f["sections/Fiber_1"]`, `f["transforms/Linear_1"]`, `f["beam_integration/Lobatto_1"]`, `f["elements/forceBeamColumn"]`, `f["time_series/Linear_1"]`, `f["patterns/Plain_1"]`, `f["recorders/Node_0"]`, `f["analysis"]`, etc. |
| `tests/opensees/h5/fixtures.py` | 7 `FIXTURE_EXPECTATIONS["…"]["expected_groups"]` lists with root-level path strings (`"materials/uniaxial"`, `"sections"`, …), plus the `wrong_major` fixture which currently uses `schema_version="2.0.0"` to test rejection — this collides with the new major and needs renumbering. |
| `tests/opensees/h5/test_fixtures.py` | Asserts `schema_version_starts_with` from the expectation table. |
| `tests/opensees/h5/test_h5_schema_v1.py` | Tests the reader's major-version check; renames itself or its fixture for the new major. |
| `tests/opensees/h5/test_h5_end_to_end.py` | Round-trips fixtures through `h5_reader.open` — picked up automatically by the path changes above. |

That's the entire test surface.  No app-level or notebook test
opens `model.h5` directly.

### 1d. Architecture-doc consumers

| File | Drift / rewrite |
|---|---|
| `architecture/h5-schema.md` | Layout diagram + per-group sections need rewriting for the new paths.  Also has pre-existing drift: shows `/elements/{pg_name}` but the writer emits `/elements/{type}` (the writer changed at some point and the doc didn't follow).  Worth correcting in commit 1 as pre-flight cleanup. |
| `architecture/viewer-integration.md` | ~15 root-level path references (`/materials/*`, `/sections/*`, `/transforms/*/per_element_vecxz`, etc.) in the panel-by-panel guidance. |
| `architecture/parallel-execution.md`, `architecture/README.md`, `architecture/decisions/0011-h5-as-fourth-emit-target.md` | Mention paths in passing.  Scan for incidental references; usually a one-line touch. |

## 2. Design tension

The master plan flagged this as **medium-high risk** because of
external consumers, but the audit shows none exist on `main`.  That
collapses two of the three obvious flavors:

### Flavor A — Clean break, single atomic schema bump

Move every writer + reader path in one go.  Bump `SCHEMA_VERSION`
`1.1.0 → 2.0.0` and `EXPECTED_SCHEMA_MAJOR` `1 → 2` in the same
commit pair.  Update tests + docs in their own commits.  No dual-write,
no dual-read.

- **Pros:** Smallest diff.  No transitional code to clean up later.
  Matches the master plan's recommendation.
- **Cons:** Anyone running `h5dump` on a pre-8.4-produced
  `model.h5` blob still sees old paths.  The reader rejects them
  with `SchemaVersionError` — clear failure mode but no automatic
  upgrade.

### Flavor B — Transitional dual-write

Writer emits at both root and `/opensees/` for one release cycle;
reader walks `/opensees/` only.  External tools observing the file
get a deprecation window.

- **Pros:** Soft landing for any external `h5dump`-style consumer.
- **Cons:** File size roughly doubles for the duration.  Adds
  conditional logic to the writer that has to come back out.  No
  evidence anyone needs the soft landing.

### Flavor C — Re-derive on read

Reader transparently maps old paths to new (or vice versa) for files
whose schema_version starts with `1.`.  Writer goes straight to
`/opensees/`.

- **Pros:** Old files keep opening through the new reader.
- **Cons:** Complicated reader.  Same problem as B — no consumer
  pressure to justify the complexity.

### Recommendation

**Ship Flavor A.**  With zero runtime consumers outside the emitter
package, the dual-write / re-derive flavors only buy us soft-landing
for hypothetical external tools.  Picked the same way Phase 8.3b
chose Flavor 1 — minimum-viable change, defer the harder question
until we have evidence the harder answer is needed.

## 3. Phase 8.4 sub-commits (Flavor A)

If the maintainer picks Flavor A, the commit shape is small:

### Commit 1 — Pre-flight: fix `h5-schema.md` `/elements/{pg_name}` drift

The current writer emits `/elements/{type}` (e.g.
`/elements/forceBeamColumn`) but the schema doc shows
`/elements/{pg_name}`.  Correct the doc to match reality.  Pure
documentation; no code change.  Splitting this off means commit 4
(doc rewrite) only has the actual 8.4 work, not unrelated drift.

### Commit 2 — Writer + reader paths move together

Single atomic commit so the test suite never sees a half-migrated
state:

- `emitter/h5.py`: each of the 9 `f.create_group("…")` sites
  becomes `_ops_group(f).create_group("…")` or equivalent.
- `emitter/h5.py`: `SCHEMA_VERSION = "1.1.0"` → `"2.0.0"`.
- `emitter/h5_reader.py`: every typed accessor walks
  `/opensees/X` instead of `/X` (except `meta()` and the future
  `elements()` neutral-zone accessor).
- `emitter/h5_reader.py`: `EXPECTED_SCHEMA_MAJOR = 1` → `2`.
- `emitter/h5_reader.py:118` doctest example `'1.1.0'` → `'2.0.0'`.

### Commit 3 — Tests follow

- `tests/opensees/h5/test_h5_emitter.py`: ~20 literal-path
  assertions rewritten.
- `tests/opensees/h5/fixtures.py`: 7 `expected_groups` lists
  rewritten.  The `wrong_major` fixture moves from `"2.0.0"` to
  `"3.0.0"` so it still tests the reader's major-mismatch path.
- `tests/opensees/h5/test_h5_schema_v1.py`: rename / repurpose if
  the "v1" framing is no longer accurate (the file's purpose is
  testing the reader's schema-major check, which is forward-
  looking — the file probably wants to live as
  `test_h5_schema_compat.py` or similar.  Open question — see §6).

### Commit 4 — Rewrite `h5-schema.md` for the new layout

- New top-level layout diagram with `/opensees/` namespace.
- Per-group sections re-rooted.
- Add the "why the namespace" paragraph from the master plan's §3
  inline (it's a stable design rationale; readers shouldn't need to
  hop to the master plan to understand the schema).

### Commit 5 — Rewrite `viewer-integration.md` for the new paths

- Update every path reference in the panel-by-panel guidance.
- Refresh the "schema-version check" snippet at the top.
- Skim `parallel-execution.md`, `README.md`, and ADR-0011 for
  incidental references and update if needed.

(Commits 4 and 5 can ship together if the doc churn feels small in
practice; keep them separate if the rewrite is heavier than expected.)

## 4. Verification gates (per commit)

Same as previous Phase-8 PRs:

- `mypy --strict src/apeGmsh/`
- `ruff check src/apeGmsh/ tests/`
- `pytest -m "not live and not subprocess" --ignore=tests/acad --continue-on-collection-errors`

Each commit's verification: no new errors / regressions relative to
the pre-PR baseline.  Re-measure at PR-open time rather than hard-
coding numbers — the master-plan and Phase-8.3b experience shows
baselines drift fast between PRs.

Special checks for this phase:

- `tests/opensees/h5/` passes end-to-end after commit 3.
- `validate()` returns empty on every `FIXTURE_BUILDERS` fixture
  after the path rewrite.
- The `wrong_major` fixture still triggers `SchemaVersionError`.

## 5. Open questions for the implementing session

1. **`test_h5_schema_v1.py` naming.** The file's name encodes the
   old major.  After 8.4 the test still asserts "reader refuses
   wrong-major files" but the major it accepts is now 2.  Options:
   (a) rename to `test_h5_schema_compat.py` and adjust internal
   comments; (b) keep the v1 name but turn the file into the "v1
   files are rejected" test; (c) split into two — one "accepts
   current major" file and one "rejects other majors" file.
   Recommendation: (a) — the file's purpose is forward-looking
   schema-version handling, not v1-specific assertions.

2. **`SCHEMA_VERSION` location.** Constant currently lives in
   `emitter/h5.py` and is exported.  After the major bump, it
   probably wants a one-line `__version__` history comment in the
   file so future readers can see the bump cadence.

3. **`/opensees/tag_map` placeholder.** Master plan §3 puts it
   under `/opensees/`, but Phase 8.6 owns the actual content.
   Should 8.4 write an empty `/opensees/tag_map` group as a
   placeholder so 8.6 can populate it without rev-bumping schema
   again, or skip it entirely?  Recommendation: skip — adding
   later is additive (no major bump needed), and an empty group
   would clutter the file.

4. **`h5-schema.md` drift sweep.** The `/elements/{pg_name}` vs.
   `/elements/{type}` drift was found incidentally; might there
   be other drift the rewrite should catch?  A focused diff
   between the current doc and what the writer actually emits is
   cheap to do as part of commit 1.

5. **Old-file migration script.** If anyone has a Phase-6
   `model.h5` checked into a personal scratch directory and wants
   to keep it working, do we ship a one-off migrator?
   Recommendation: no — the schema is young enough that nobody is
   likely to have such a file, and the major-version check raises
   a clear `SchemaVersionError` if they do.

## 6. Out of scope (defer to later phases)

- **Broker-side neutral-zone writers** (`/nodes`, `/elements/{type}`
  rewrite, `/physical_groups`, `/labels`, `/constraints/{kind}`,
  `/loads/{kind}`, `/masses`, plus the symmetric-record-compound
  helper).  That's Phase 8.5, independent of this phase.
- **`/opensees/tag_map`.**  Phase 8.6.
- **Viewer migration off `FEMData` / `solvers`.**  Phase 8.7.
- **Architectural unification of `Recorders` with the typed
  primitives.**  Deferred indefinitely; needs its own scoping.

## 7. Risk assessment vs. master-plan estimate

The master plan ([phase-8-untangle.md §5](phase-8-untangle.md))
rated 8.4 as **medium-high risk** because of external Phase-6 file
consumers.  After the consumer audit on current `main`:

- Zero runtime consumers outside the emitter package.
- All tests rebuild fixtures on demand (no checked-in `.h5` blobs).
- No notebook or app opens `model.h5` directly.

Re-rated as **low–medium risk**: the schema break is real (any
external `h5dump`-style observer sees different paths after 8.4)
but the in-repo surface is small and deterministic.  Effort is
dominated by docs + test rewrites, not by source-side complexity.

## References

- [phase-8-untangle.md](phase-8-untangle.md) — master plan
- [h5-schema.md](h5-schema.md) — current schema (will be rewritten)
- [viewer-integration.md](viewer-integration.md) — viewer contract
  (will be rewritten)
- [decisions/0011-h5-as-fourth-emit-target.md](decisions/0011-h5-as-fourth-emit-target.md)
  — original ADR for `model.h5` as an emit target
- [phase-8.3b-scope.md](phase-8.3b-scope.md) — companion scope doc
  the 8.4 doc mirrors structurally
