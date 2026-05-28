# ADR 0023 — Per-zone schema versioning; two-version reader window

**Status:** Accepted (Phase 7a of the major architectural refactor,
May 2026). Replaces the single-envelope schema-version policy with a
per-zone scheme; defines the bounded reader-compatibility window for
all `model.h5` / `results.h5` consumers.

## Context

Today the `model.h5` file has **two** `schema_version` stamps,
written by **two** zones, and they race:

| Stamp | Writer | Today's value (May 2026) |
|---|---|---|
| `/meta/schema_version` (neutral, `NEUTRAL_SCHEMA_VERSION`) | `mesh/_femdata_h5_io.py` | `"2.4.0"` |
| `/meta/schema_version` (bridge, `SCHEMA_VERSION`) | `opensees/emitter/h5.py` | `"2.5.0"` |

Both stamp **the same attribute key**, in the same group. Whichever
writes last wins. Readers see a single version that doesn't
accurately reflect what is in the file.

The current read-side rule (`opensees/emitter/h5_reader.py:50`) is
binary:

```python
EXPECTED_SCHEMA_MAJOR = 2   # accept major 2 of any minor
```

Readers either accept the major or refuse. There is no graceful
evolution path — a minor bump on either side is invisible to readers,
and a major bump is a hard break.

Adding a third zone (the `/run/` zone for results, introduced by
ADR 0020's Composed-file pattern) compounds the race. The fix is
per-zone semver evolution with a bounded reader-compatibility window.

## Decision

### Three per-zone version stamps + one envelope

```
/meta/neutral_schema_version    = "2.5.0"   (neutral zone, post-Phase 2)
/meta/opensees_schema_version   = "2.8.0"   (opensees zone, post-Phase 7)
/meta/results_schema_version    = "1.0.0"   (results zone, NEW in Phase 4)

/meta/schema_version            = "2.x"     (envelope; back-compat for one-key readers)
```

The **envelope** `/meta/schema_version` bumps only when the partition
shape itself changes (a new zone added, the structure rearranged).
Today's envelope value reflects the bridge schema (because the bridge
wrote last); this is preserved as back-compat for readers that look
at one key. The envelope is **not** the per-zone version; readers
that need fine-grained version awareness must consult the per-zone
keys.

### Per-zone read validation — two-version reader window

Each zone reader validates **its own** zone's version, not the
envelope. The validation rule is the **two-version window**:

A reader at version `X.Y.Z` accepts:
- `X.Y.*`  (any patch of its own minor)
- `X.(Y-1).*`  (any patch of the previous minor)

It **refuses** with `SchemaVersionError`:
- `X.(Y-2).*` and earlier minors (too old; outside the window)
- `X.(Y+1).*` and later minors (newer than the reader understands; refusing is safer than silent tolerance)
- `(X-1).*.*` and earlier majors (breaking change)
- `(X+1).*.*` and later majors (breaking change)

`SchemaVersionError` text includes the explicit upgrade-path
recommendation: *"file written at `opensees_schema_version=2.5.0`; this
reader supports 2.7.x–2.8.x. Upgrade apeGmsh to read this archive, or
re-emit the model with the current version."*

### Single-stamp legacy files — back-compat mapping

Files written before this ADR have only `/meta/schema_version` (no
per-zone keys). At read time, the reader maps:

- `/meta/neutral_schema_version` ← `/meta/schema_version` (assumed)
- `/meta/opensees_schema_version` ← `/meta/schema_version` (assumed)
- `/meta/results_schema_version` is **absent** (no results zone in
  pre-Phase-4 files).

This back-compat covers every file written under `schema_version`
2.0.0 through 2.4.0 of the legacy single-envelope era. Files written
under 2.5.x with the bridge's value also have a single key (one stamp,
written by the bridge); the same mapping applies.

### Bump cadence — locked policy

| Bump | Trigger | Old readers |
|---|---|---|
| **Patch (Z)** | Fix-only changes; no schema-shape change. | Continue to parse identically. |
| **Minor (Y)** | Additive changes: new dataset, new attribute, new payload field. Old required fields remain. | Continue to parse, ignoring new content. The two-version window means the previous minor's readers can still open the file. |
| **Major (X)** | Breaking changes: removed field, renamed dataset, changed dtype. | Refuse with `SchemaVersionError`. Migration tooling (out of scope) is the only path. |

The two-version window is a **deliberate forcing function**. Users
must update the apeGmsh library at least once per minor cycle for the
zone they care about. Archived files that outlive the window must be
read through migration tooling (a `migrate_model_h5(path, *,
target_minor)` helper; not in scope for this ADR).

### Invariants

**INV-1.** The three per-zone keys are independent. A bridge-side
minor bump (e.g. `opensees_schema_version: 2.7.0 → 2.8.0`) does not
require a corresponding bump on `neutral_schema_version` or
`results_schema_version`. The race condition in today's
single-envelope design is structurally removed.

**INV-2.** The envelope `/meta/schema_version` is **back-compat
only**. New code must not branch on the envelope; new code reads the
per-zone keys. The envelope exists so that pre-Phase-7 readers (which
look at one key) keep working.

**INV-3.** The two-version window applies independently to each
zone. A reader at `opensees=2.7.0` and `neutral=2.5.0` accepts
`opensees ∈ {2.6.*, 2.7.*}` AND `neutral ∈ {2.4.*, 2.5.*}` — the two
windows are conjunctive but not coupled.

**INV-4.** A reader **refuses newer minors** (`X.(Y+1).*`). The
alternative — silent tolerance of unknown content — would let new
attributes be invisibly dropped, masking bugs. INV-4 is the dual of
the warn-not-raise principle in [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md):
lineage mismatches warn (data is recoverable); schema mismatches
refuse (data may be unrecoverable).

**INV-5.** Migration of archived files outside the window is the
user's responsibility, via a separate migration tool. This ADR does
not ship that tool; it documents that it is owed. Files older than
the window become read-only-after-migration; nothing in this ADR
auto-upgrades them.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Single envelope version covering all zones** | Doesn't scale beyond 2 zones; couples evolution (bridge can't ship a minor without coordinating with neutral); the race condition described in Context is the present-day proof. |
| **Strict version match (no window — only exact-match accepted)** | Every minor bump breaks every reader. Users would need to upgrade the library every release just to open last week's file. No graceful upgrade path. |
| **Open window (read all old versions back to the first ever schema)** | Unbounded compat burden. Each new minor adds a code path that must be maintained forever. The two-version window keeps the read-side surface bounded. |
| **Per-file version manifest (separate `manifest.json` next to .h5)** | Adds a second file the user must keep paired with the .h5. Defeats the one-file-rules-all benefit of HDF5 and undoes the Composed-file design from ADR 0020. The version stamps belong inside the file. |
| **Three-version window (`X.Y.*`, `X.(Y-1).*`, `X.(Y-2).*`)** | Wider window dilutes the forcing function. Users who never upgrade can stay three minors behind indefinitely; bug fixes that depend on schema updates won't reach them. Two-version is the minimum that allows a one-release-cycle upgrade rhythm. |
| **Refuse only newer majors; accept any older minor** | Asymmetric and weakens INV-4. The whole point of a window is bounded compat on both sides; newer minors might add fields the reader doesn't know about, and silently dropping them is a defect class. |

## Consequences

**Positive:**

- Each zone evolves independently. Bridge can ship `opensees=2.8.0`
  while neutral stays at `neutral=2.5.0`. The race condition in
  today's single-envelope design is gone (INV-1).
- The two-version window is **bounded** and **documented**. Users
  know exactly how long their archive is readable; library
  maintainers know exactly how many old code paths to keep.
- Future zones (Code_Aster, Abaqus, …) slot in with their own
  version key. The pattern scales — adding `aster_schema_version` is
  one new key, one new reader rule, no impact on existing keys.
- Archived files have an expiration horizon users can plan against.
  *"This file was written under `opensees=2.6.0`; this reader supports
  2.7.x–2.8.x"* is a precise message.
- Lineage determinism (per [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md)
  INV-5) is required only **within** the two-version window —
  bounding the canonical-bytes determinism requirement.

**Negative:**

- More meta keys. Readers need to know which zone they care about
  and consult the corresponding key. Mitigated by INV-2 (envelope
  stays as a one-key fallback for simple consumers).
- Two-version window means users must update at least once per minor
  cycle for any zone they read. Acceptable forcing function;
  documented in the public release notes.
- Migration tooling for archives older than the window is **deferred**
  (INV-5). The gap is documented; building a migration tool is owed
  but not in this ADR's scope.
- Single-stamp legacy file mapping (the back-compat rule above) is a
  small ongoing maintenance cost. Per-zone readers must check
  whether the per-zone key exists before falling back to the
  envelope. Encapsulated in one helper
  (`_internal/schema_version.py::read_zone_version(f, zone)`).

## Open questions

- **Q7 — precise window edges during the Phase 7 transition.**
  Resolved (this ADR): Phase 7's bridge reader at `opensees=2.7.0`
  accepts `opensees ∈ {2.6.x, 2.7.x}` and rejects `opensees ∈
  {2.5.x, 2.4.x, ...}`. The legacy single-envelope files (written
  under `/meta/schema_version=2.4.0` or `2.5.0`) are mapped to
  `opensees=2.4.0` or `opensees=2.5.0` respectively at read time, and
  therefore refuse under the two-version window. Users who need to
  open pre-2.6 archives use the migration tool (INV-5); Phase 7's
  bridge does not silently downgrade.

- **Migration tool — owed but deferred.** The migration helper
  `migrate_model_h5(path, *, target_minor)` is owed by INV-5 but is
  not in this ADR's scope. Tracked separately; expected to land
  before any zone reaches a third minor cycle (which would leave the
  first-minor files outside the window).

## References

- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — original schema-versioning policy referenced in
  [viewer-integration.md §"Versioning policy"](../viewer-integration.md).
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the viewer consumer affected by per-zone versioning; its
  `ViewerData.from_h5` must consult per-zone keys.
- [decisions/0020-results-carries-opensees-model.md](0020-results-carries-opensees-model.md)
  — introduces the `results_schema_version` zone this ADR defines.
- [decisions/0021-lineage-chain-replaces-snapshot-id.md](0021-lineage-chain-replaces-snapshot-id.md)
  — INV-5 (canonical-bytes determinism within the two-version
  window).
- [decisions/0022-mp-constraint-emission-fanout.md](0022-mp-constraint-emission-fanout.md)
  — the schema additions that drive the `opensees_schema_version`
  bump in Phase 7.
- [phase-8-untangle.md](../phase-8-untangle.md) §7 closure — the
  schema-version race this ADR resolves.
- `opensees/emitter/h5_reader.py:50` — today's binary
  `EXPECTED_SCHEMA_MAJOR` rule, replaced by the per-zone window
  validation.

## Amendment — 2026-05-28 — Window applies to additive-only changes (B2 / PR #398)

The B2 schema bump (`neutral_schema_version: 2.9.0 → 2.10.0`) closed
the snapshot_id drift bug by restructuring the on-disk H5 layout for
`/physical_groups/` and `/labels/` into node-side / element-side
sub-trees. Per the bump-cadence table above, a non-additive change to
a required field's layout is a **major** bump. We instead shipped it
as a minor bump and accepted that the 2.10 reader cannot parse 2.9
files even though they nominally fall within the two-version window.

The path of least resistance was reframing the window's scope rather
than bumping to `3.0.0`:

**The two-version window applies to ADDITIVE changes only.** When a
minor bump perturbs the canonical bytes of a required structure (the
classic example being the B2 layout restructure), the window does
not guarantee that the prior minor can be read by the current reader.
Rejecting `(Y-1).*` files in that case is the correct behaviour, not
a regression.

**Implications for the bump-cadence table**:

- A minor bump that is **purely additive** (new dataset, new
  attribute, new payload field; old required fields preserved) still
  benefits from the window. Two-minor compat is preserved.
- A minor bump that **restructures required content** (renamed group,
  changed dtype, layout split) effectively walks the window forward
  by one and locks the prior minor out — even though the version
  number remains in the window's range. This was previously
  ambiguous in the ADR; this amendment makes it explicit.

**INV-5 (ADR 0021) is reframed accordingly.** Canonical-bytes
determinism holds *within a single neutral minor version*, not
across a layout-affecting bump.

**For future bumps**: if a planned change perturbs canonical bytes of
a required structure, either (a) bump the major and refuse the prior
minor unambiguously, or (b) ship the change behind a side-attribute
that lets the prior layout coexist (the "additive-only" path). Choice
(a) is the cleaner signal; choice (b) preserves the window but
constrains the design.

The window mechanism itself is unchanged; the contract test
`test_pre_2_8_0_schema_rejected` (renamed to follow the prior minor
as the window slides) still locks the "below-window rejected"
guarantee.
