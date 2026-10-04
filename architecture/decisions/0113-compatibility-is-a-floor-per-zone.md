# ADR 0113 — Compatibility is a floor per zone

**Status:** Accepted (2026-10-03; ratified by the maintainer on #1303)

**Owner:** nmora

**Evidence:** two independent architect briefs (`prog-architect-opus`,
`prog-architect-fable`) grounded on `origin/main` 9f32526b plus PR #1300,
reconciled in #1303 and ratified "as recommended" on 2026-10-03. The
briefs are attached in full on the issue.

**Amends** [ADR 0023](0023-per-zone-schema-versioning.md): the
two-version reader window is retired for every zone, INV-5's migrator
gets its trigger, and the G1 "same refuse rule" for external readers
gains one app-only exception. The per-zone keys, the envelope, the
bump cadence, INV-1, INV-2 and INV-4 stand. INV-3 ("the two-version
window applies independently to each zone") is **restated**: the floor
applies independently to each zone; the per-zone checks stay
conjunctive and uncoupled. ADR 0023 carries a dated amendment pointing
here; its original text is unchanged.

**Program slice:** #1303, the first of six PRs listed under
[PR slices](#pr-slices).

## Context

### The window refuses files the readers can parse

`opensees/_internal/schema_version.py::validate_zone_version` accepts a
file only when `file.major == reader.major` and
`file.minor in {reader.minor - 1, reader.minor}`. It guards
`FEMData.from_h5` (`mesh/_femdata_h5_io.py`), `mesh/_compose.py`,
`OpenSeesModel.from_h5` (`opensees/emitter/h5_reader.py::open`) and
`results/readers/_native.py`, which also validates the `/model` and
`/opensees` zones embedded in a results file.

The neutral zone went from 2.10.0 (2026-05-28) to 2.33.0 (2026-09-15):
23 minors in 110 days, four of them on one day. Every one of those
minors is additive. The readers do not branch on them: they
presence-probe the newer groups and columns (about sixteen `in parent`
group checks and about twenty-five `in p.dtype.names` column checks).
There are exactly two version branches in the Python readers: the
`/nodes/ndf` fallback at `_femdata_h5_io.py` (`_fv < (2, 7, 0)`, below
the floor this ADR sets, so dead code once it lands), and #1300's
`h5_reader.py::read_spatial_ndm`, keyed on `META_NDM_IS_SPATIAL_FROM`. The opensees zone has one non-additive
minor, the 2.11.0 flip to 0-based runtime ranks. So the window check
alone refuses a 2.12 file that the reader body would parse correctly.

### The window's two reasons no longer hold

ADR 0023 gave the window two reasons.

1. **A forcing function to upgrade.** Refusing *newer* minors (INV-4)
   already delivers that: a reader older than the file refuses it, so
   the user upgrades the library. The window's *lower* edge forces
   something else, that users regenerate their files, and
   [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1 ("the
   files are the model") rules that out. The file is the durable
   artifact; the script that wrote it may no longer run.
2. **A bounded read surface.** ADR 0023 rejected the open window because
   "each new minor adds a code path that must be maintained forever".
   The code shows that surface is already paid: the presence probes
   *are* the compatibility code, and they exist whether the window
   admits the file or not.

The migrator that ADR 0023 INV-5 promised, "before any zone reaches a
third minor cycle", never shipped. The neutral zone is in its
twenty-fifth.

### What expires today

- `tests/fixtures/neutral_zone/box.h5`, the published external golden
  (now regenerated at 2.34.0 by #1300), and
  `apeGmshViewer/fixtures/shoebuckle.h5` (2.33.0) are real files, and
  each is refused within two neutral bumps of its stamp.
- A results file expires through its embedded `/model`:
  `results/writers/_native.py::NativeWriter.write_model` stamps the
  embedded `/model/meta` with the current neutral version, and the
  results reader validates it. The results zone itself has sat at 1.1.0
  since 2026-05-21. The architectural-strains assessment named this
  RS6, "the evolution policy taxes growth".
- Two semantic changes exist in the history after 2.10: patch 2.26.1
  split `/loads/sp/default` into one group per case, and 2.34.0 changed
  what `/meta/ndm` means (#1300, which ships its shim).

### The app already chose leniency

`apeGmshViewer/src/reader/read.ts::checkVersion` is hand-written
(ADR 0112 D4: nothing crosses the wall). It refuses other majors and
neutral minors below `NEUTRAL_MIN_MINOR = 10`, and only *warns* on any
other minor, newer ones included (`test/failclosed.test.ts`). That is
against ADR 0023's 2026-08-24 amendment, which holds external readers
"to the same refuse rule". Python users, meanwhile, are refused the
same file: ADR 0112 D1 splits by reader.

The expert panel (`internal_docs/plan_expert_panel_2026-09.md`, S7)
ratified "a compatibility floor plus the additive-column rule" as
build-now and deferred the migrator to a named trigger.

## Decision

### D1 — Compatibility is a per-zone floor, not a window

`validate_zone_version` accepts a file iff

```
file.major == reader.major  and  floor.minor <= file.minor <= reader.minor
```

The patch is ignored. The refusal names both ends: *"supports
2.10.x–2.34.x"*. The floors at ratification:

| Zone | Floor | Why there |
|---|---|---|
| neutral | **2.10.0** | the B2 layout split (ADR 0023, 2026-05-28 amendment); the app already hard-codes 10 |
| opensees | **2.11.0** | the 0-based rank flip, the zone's last non-additive bump |
| results | **1.0.0** | unchanged; the zone has never broken |
| geometry | **1.0.0** | its first version; in `_ZONE_KEY` since #1311 |
| provenance | **1.0.0** | its first version; in `_ZONE_KEY` since #1311 |

A new zone joins the table at its first version, so its floor is
`X.0.0` until its first major bump: `/geometry` and `/provenance`
(ADR 0112 D2) already sit in `_ZONE_KEY` at 1.0.0 (#1311), and
`/sequence` joins when it lands. Files below a floor are pre-release
(2026-05-12 to 2026-05-28 for the neutral zone) and stay refused.

### D2 — Python readers keep INV-4

A Python reader refuses a same-major *newer* minor, as ADR 0023 INV-4
requires: silently tolerating content the reader cannot understand
would drop new attributes invisibly. Nothing in this ADR adds a
`strict=False` or guessing mode to Python.

### D3 — Floors are writer-owned, evidence-gated, then fixed until a major

The floor constants sit beside the version constants they bound:
`NEUTRAL_SCHEMA_FLOOR` next to `NEUTRAL_SCHEMA_VERSION` in
`mesh/_femdata_h5_io.py`, `SCHEMA_FLOOR` next to `SCHEMA_VERSION` in
`opensees/emitter/h5.py`, `RESULTS_SCHEMA_FLOOR` next to
`RESULTS_SCHEMA_VERSION` in `results/schema/_versions.py`,
`GEOMETRY_SCHEMA_FLOOR` and `PROVENANCE_SCHEMA_FLOOR` beside
`GEOMETRY_SCHEMA_VERSION` and `PROVENANCE_SCHEMA_VERSION` in
`opensees/_internal/schema_version.py`. `reader_floor(zone)` mirrors
`reader_version(zone)`, so reader and writer cannot disagree.

**The initial floors are evidence-gated.** A floor stands only where
real corpus files (D8) prove every minor from it to the current one
opens through today's reader. An era whose frozen writer will not run
raises that zone's floor past it; the gap is recorded, never faked.

**After that, a floor moves only with a major bump**, to `X.0.0`, by
ADR. A floor only rises.

### D4 — A semantic change ships a reader shim keyed on a named `*_FROM` constant

A minor that changes the *meaning* of an existing field (not its
presence) ships a reader shim switched on a named constant, following
#1300's `META_NDM_IS_SPATIAL_FROM`. All shims are listed in one shim
ledger in `architecture/h5-schema.md`. Every ledger entry is above its
zone's floor, so raising a floor deletes the shims beneath it.

A quirk rule in `scripts/check_quirks.py` flags any bare
`SchemaVersion` tuple compare in a reader. Its proof site is #1300's
`read_spatial_ndm`, keyed on `META_NDM_IS_SPATIAL_FROM` (2.34.0): the
self-test passes the named constant and fails a bare-tuple rewrite of
the same compare. The only other version branch, the `/nodes/ndf`
fallback `_fv < (2, 7, 0)` in `_femdata_h5_io.py`, keys on a version
**below** the neutral floor; by this ledger's own rule it is dead code
once the floor lands, so slice 6 **deletes** it rather than naming it.
A **restructure** (renamed group, changed dtype, layout split) is a
**major** bump, never a shim. ADR 0023's 2026-05-28 amendment allowed
a layout-perturbing minor by walking the window forward; with no
window to walk, that path is closed.

**Shim ledger at ratification:**

| Zone | Constant | Below it | At or above it |
|---|---|---|---|
| neutral | `META_NDM_IS_SPATIAL_FROM` (2.34.0) | `/meta/ndm` is salvaged by `read_spatial_ndm` | the attribute is trusted |
| neutral | SP loads before 2.26.1 | every SP record is read as one `default` case (Q5, accepted; the per-case split is not reconstructed) | one group per case |

### D5 — Additive changes still bump the minor

The stamp stays INV-4's gate and the forensic record of which writer
produced the file. The panel's S7 corollary, "no minor bump for an
additive column", is **not** adopted: an older reader would open the
file and silently drop a deck-affecting column such as
`cpl_al_update`, then build a different analysis without a word. What
disappears is the reader-side cost of a bump: the `*_PRIOR_MINOR`
renames and the sliding `test_two_version_window_*` tests. The
remaining per-bump cost is one corpus file (D8).

### D6 — The migrator is deferred to the first major bump of any zone

Until a zone bumps its major, a migrator would duplicate the reader: a
`from_h5 → to_h5` step needs the old minor's reader, which D1 already
keeps. When the first major bump of any zone lands, the migrator ships
with it: the **old major's last reader** does `from_h5`, then `to_h5`
**into a new file beside the old one**, stamping `/meta/migrated_from`
with the source version. **Never in place, never on open.** A rewrite
changes `created_iso`, `snapshot_id` and `/meta/lineage`, which severs
the pairing with every `results.h5` that cites the old hashes; that is
the user's decision to take, not the reader's.

### D7 — The app carries the same floors, and a banner on newer files

apeGmshViewer reads the same floor table as Python: neutral 2.10,
opensees 2.11, results 1.0. A Python drift test parses `read.ts` as
text and compares the numbers to the writer constants; nothing imports
across the D4 wall. A same-major file at or above the floor and at or
below the app's target opens with no banner.

A same-major **newer** file opens **with exactly one banner** naming
the file's stamp and the app's target. This is `main`'s behaviour,
ratified as an **app-only deviation from INV-4** (Q2): installed apps
lag the library by design, the app only reads (ADR 0095 INV-2,
ADR 0112), and ADR 0112 D2 puts new data in its own zone, so a newer
neutral minor adds columns the app does not show rather than changing
what it shows. The app ships no shims for fields it does not read
(`ndm` among them).

### D8 — A committed corpus of real old files, from git's frozen writers

The old writers live in git. `scripts/build_schema_corpus.py` checks
out each schema-bump commit of a zone in a temporary worktree and runs
an era-stable generator under `PYTHONPATH=<worktree>/src`: the
deterministic box of `tests/fixtures/neutral_zone/_generate_fixtures.py`
for the neutral zone (needing only `FEMData.to_h5`, stable since
2.1.0), and a minimal frame through `apeSees.h5()` for the opensees
zone. The files, about twenty-five of them and about 1 MB, are
**committed** under `tests/fixtures/schema_corpus/` with a
`MANIFEST.json` naming each file's commit SHA (Q6: frozen writers give
frozen files, so there is nothing to regenerate nightly). Where an
era's API cannot run the generator, the script **records the gap**
(Q7) and the floor rises past it (D3).

**Oracle.** Each era's *own* reader writes a semantic dump beside its
file: `snapshot_id`, `ndm` and `ndf`, element counts per type,
physical-group and label names with their sizes, load and constraint
counts, and the declared primitives with their parameters. Today's
reader must reproduce that dump, except for the fields the shim ledger
names. Each era's own `build("tcl")` is stored beside the file for
chain K2's round-trip oracle; decks compare within one era only, since
emitter fixes such as #1282 change decks across eras.

Each future bump adds the outgoing minor's file. A paper floor, one set
to the current minor, must make the corpus test fail; the slice that
lands the corpus records that failure in its PR body, as #1300 did.

### D9 — Results files whose `/model` is below the floor still open their `/stages`

A results file carries its own zone at `/stages` and two embedded
zones, `/model` (neutral) and `/opensees`, each validated against its
own floor by `results/readers/_native.py::_validate_per_zone_versions`.
When the embedded `/model` is **below the neutral floor**, the results
reader opens `/stages` **read-only and flagged**: `Results.model` is
unavailable, the reader says so, and nothing is rewritten (Q3). Results
should outlive their model zone. The same ratified rule applies to the
other embedded zone: an embedded `/opensees` **below its floor**
(2.11.0) also opens `/stages` read-only and flagged, with the bridge
side of `Results.model` unavailable. An embedded zone *newer* than the
reader is not covered by Q3 and refuses as INV-4 requires (D2); the
results zone itself is validated as before.

### D10 — Stamp `/meta/apeGmsh_version`

Both writers leave `/meta/apeGmsh_version` empty today (both committed
fixtures show `''`), so neither the app nor a user can name the release
that wrote a file. Both writers stamp it from the installed package
version. The stamp is forensic only: no reader branches on it.

## Invariants

Each is testable and is held by a test in the slice that lands it.

1. **Edges.** For each zone in `_ZONE_KEY` and every `(file, reader)`
   pair, `validate_zone_version` accepts iff same major and
   `floor.minor <= file.minor <= reader.minor`; the patch is ignored.
   Grid-tested at both edges: where `floor.minor > 0`,
   `floor.minor - 1` refuses as "too old", naming the floor; where the
   floor is `X.0.0` (results, geometry, provenance) the lower edge is
   the previous major, `(X-1).*`, refused as a major mismatch;
   `reader.minor + 1` refuses as "newer than this reader" in Python.
2. **Writer-owned.** `reader_floor(zone) <= reader_version(zone)`, and
   both equal the writer constants.
3. **Registry.** The `architecture/h5-schema.md` version registry has a
   Floor column equal to the constants, held by the existing
   doc-registry test.
4. **App drift.** The floors in `apeGmshViewer/src/reader/read.ts`
   equal Python's; a Python test parses them as text. Nothing imports
   across the wall.
5. **Corpus opens.** Every corpus file opens through every applicable
   reader (`FEMData.from_h5`; `OpenSeesModel.from_h5` when it has
   `/opensees`; `NativeReader` on a composed results twin), and its
   semantic dump equals the one its era's reader recorded, except for
   fields the shim ledger names.
6. **Corpus complete.** One corpus file exists per minor from the floor
   to the current minor, each with its commit SHA in the manifest.
7. **Bytes unchanged.** Opening a corpus file never changes its bytes
   (sha256 before and after the session).
8. **Shims named.** Every reader version branch compares against a
   named `*_FROM` constant at or above its zone's floor, has a test on
   a corpus file below it, and is listed in the shim ledger. A branch
   keyed below the floor is dead code and is deleted (the `_fv < (2, 7,
   0)` fallback, slice 6). The quirk rule fails a bare tuple compare,
   proven on `read_spatial_ndm`.
9. **No laundering.** No path forwards an old zone's value under a
   newer stamp without its shim (the class #1300 found: a restamped
   file whose `/meta/ndm` still carried the old meaning). The one such
   path today is the results twin: `results/writers/_native.py::
   NativeWriter.write_model` restamps the embedded `/model/meta` with
   the current neutral version, and `NativeWriter.write_opensees_from`
   forwards the source's `ndm` onto it through
   `read_spatial_ndm(src_meta, src)`. Slice 4 tests it: a twin composed
   from a pre-2.34.0 source whose `/meta/ndm` carries the mesh
   dimension must read the spatial `ndm` the source's own reader
   resolves; a raw forward of the attribute fails the test.
10. **Tamper check kept.** The `snapshot_id` check still raises on a
    tampered corpus copy.
11. **Results floor.** A `results.h5` whose `/model/meta` carries the
    floor stamp opens through `NativeReader` and `Results.model`
    resolves; one whose embedded `/model` or `/opensees` is below its
    floor opens `/stages` read-only and flagged (D9); one whose
    embedded zone is newer than the reader refuses (D2).
12. **Paper floor fails.** With a zone's floor constant set to its
    current minor, the corpus test fails.

## PR slices

Each PR is opened with `--base main` and touches about ten files or
fewer. Order: 1, 2, then 3 in parallel with 4, then 5, then 6.

| # | Slice | Owns |
|---|---|---|
| 1 | **This ADR** and the ADR 0023 amendment | `architecture/decisions/` |
| 2 | **The floor**: `reader_floor`, the new `validate_zone_version` and its message, the five floor constants (`NEUTRAL_SCHEMA_FLOOR`, `SCHEMA_FLOOR`, `RESULTS_SCHEMA_FLOOR`, `GEOMETRY_SCHEMA_FLOOR`, `PROVENANCE_SCHEMA_FLOOR`), `tests/fixtures/schema.py` (`*_FLOOR`; `*_PRIOR_MINOR` kept for shim tests), boundary tests replacing `test_two_version_window_*` in `tests/opensees/h5/test_h5_schema_compat.py`, the `schema-literal` quirk text, a changelog fragment | `schema_version.py`, the three writer modules, the compat tests |
| 3 | **The corpus**: the builder script, `tests/fixtures/schema_corpus/` (about 25 files, about 1 MB, plus manifest, dumps and decks), the corpus test (INV 5, 6, 7, 10, 12) | `scripts/`, the corpus fixtures, one test module |
| 4 | **The app**: the floor table in `read.ts` with the banner text, `test/failclosed.test.ts`, the Python drift test (INV 4), the results-path test (INV 11, both embedded zones) and the results-twin laundering test on `write_opensees_from`'s `ndm` forward (INV 9) | `apeGmshViewer/src/reader/`, its tests, `tests/results/` |
| 5 | **Docs**: `architecture/h5-schema.md` "Versioning" with the Floor column and the shim ledger, `docs/design/model-h5-neutral-zone.md` "Version rule", the bridge-feature guide's bump checklist (add the outgoing minor's corpus file) | the three documents |
| 6 | **The quirk rule** for bare version compares, with a self-test proven on `read_spatial_ndm` / `META_NDM_IS_SPATIAL_FROM` (the rule passes the constant and fails a bare-tuple rewrite of that compare); deletes the dead `_fv < (2, 7, 0)` fallback in `_femdata_h5_io.py`, keyed below the floor (INV 8) | `scripts/check_quirks.py`, its test, one reader branch |

Slice 2 lands after #1300 (landed as 0d95c30b). D10, the
`apeGmsh_version` stamp, rides with slice 2 (both writers are in its
owned files).

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **A migrator now, with strict readers** (`python -m apeGmsh.migrate`) | Each step is `from_h5 → to_h5`, so it needs the old minor's reader anyway: 24 steps, 24 frozen readers, one more per bump forever. It must also rewrite `/opensees` and gigabyte results files. A rewrite changes `created_iso`, `snapshot_id` and `/meta/lineage`, severing every `results.h5` that cites the old hashes. The user runs a command before double-clicking, against ADR 0112 D1. Under D4 it never helps the app. It stays lossy until chain K2's xfail ledger empties. |
| **An N-version window** (three, five, …) | It still expires files; it only moves the date. ADR 0023 rejected the three-version window for diluting the forcing function; D1 shows the forcing function is INV-4's job, so no width of window has a reason left. |
| **A lenient Python reader** (`strict=False`, forward tolerance) | "Parse what I recognise and hope" is what INV-4 forbids and what the published neutral-zone page states as a requirement. A deck-affecting column silently dropped builds a different analysis. The app's banner (D7) is bounded by ADR 0112 D2, which Python readers are not. |
| **A per-minor shim module tree** | The presence probes *are* the shims; a tree beside them would encode the same knowledge twice and drift. Shims exist only for semantic changes, named in one ledger (D4). |
| **A lenient app only** (`main`'s state) | Python still refuses the file the agent is asked to edit, so D1 splits by reader. Leniency without a floor goes silent at the first semantic bump to a field the app reads. |
| **No minor bump for additive columns** (panel S7's corollary) | An older reader would open the file and silently drop the column (D5). Revisit only if ADR 0023's patch rule is rewritten. |
| **Floors tied to dates or releases**, or a window back to 2.0 | A floor is a claim about parseable layouts, proven by a corpus; a date proves nothing. Files below the floors are pre-release. |
| **A second version number for the app** | ADR 0023's G1 amendment forbids it: the published page is a view of `neutral_schema_version` and nothing else. |
| **Nightly corpus regeneration**, or building the corpus in a cached CI step | Frozen writers give frozen files, committed once. |
| **Any in-place or on-open resave** | A reader that writes is the defect class ADR 0095 INV-2 and ADR 0112 exclude; INV 7 holds the bytes. |

## Consequences

**Positive.**

- A `model.h5` written since 2026-05-28 opens in today's library and
  app, and will keep opening until the first major bump, which ships
  the migrator. ADR 0112 D1 holds for files whose script no longer
  runs.
- The per-bump cost drops to one corpus file from one command, plus a
  shim only for a semantic change (one of the 24 neutral minors since
  the floor needed one). No fixture expires for a version stamp alone,
  so no golden regeneration and no external redeploy for that reason.
- Results files stop expiring through their embedded `/model` (RS6).
- The compatibility claim is evidence: a floor is only as low as the
  corpus proves, and a paper floor turns the corpus test red.

**Negative.**

- Every shim lives until the next major of its zone. The ledger is the
  bound: it is short because restructures are majors, and it shrinks
  only with a floor raise.
- The forcing function on the lower edge is given up. Users are still
  forced to upgrade by INV-4 (Python) and told to by the banner (app).
- The app deviates from INV-4 on newer files. The deviation is bounded
  by ADR 0112 D2 and limited to the app; it is recorded in ADR 0023's
  amendment so a reader of the G1 text finds it.
- About 1 MB of binary fixtures joins the repository, growing by one
  small file per bump.
- Pre-2.26.1 SP loads read as one `default` case; the per-case split
  cannot be reconstructed from those files, and the ledger says so.

## Open questions

- **Opensees floor after the backfill.** 2.11.0 is ratified; if the
  frame generator will not run on an early era, the corpus records the
  gap and the floor rises to the first era that runs. The slice that
  lands the corpus reports the final floor in its PR body.
- **The migrator's shape at the first major.** D6 fixes its contract
  (old major's last reader, new file beside the old,
  `/meta/migrated_from`); its command surface is decided in the ADR that
  bumps the major.

## References

- [ADR 0023](0023-per-zone-schema-versioning.md), per-zone versioning
  and the two-version window this ADR retires; its 2026-05-28,
  2026-07-19 and 2026-08-24 notes; its dated amendment pointing here.
- [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md), lineage
  determinism, which the window bounded and the floor now bounds per
  major.
- [ADR 0020](0020-results-carries-opensees-model.md), the composed
  results file whose embedded `/model` D9 addresses.
- [ADR 0095](0095-apegmsh-studio.md) INV-2 and
  [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1, D2,
  D4: the files are the model, new data gets its own zone, the app
  never imports apeGmsh.
- `internal_docs/plan_expert_panel_2026-09.md` S7 and
  `internal_docs/plan_architectural_strains_2026-09.md` RS6.
- #1300, the `read_spatial_ndm` shim and `META_NDM_IS_SPATIAL_FROM`,
  the precedent for D4; #1303, the design and its ratification.
