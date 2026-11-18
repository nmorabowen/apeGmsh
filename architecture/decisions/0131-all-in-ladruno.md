# ADR 0131 — All in `.ladruno`: apeGmsh reads one results format

**Status:** Accepted (2026-11-03; accepted by the maintainer on #2756).
Merging that PR (`human-gate`) is the acceptance, so the pointers in ADRs
0122 and 0115 land with it, as ADR 0121 D5 says.

**Owner:** nmora

**Rulings:**

- The owner's rulings on #2655 (2026-11-02): R22b, a `Results` held only
  in memory, is "Ladruno only" (comment 6595544718), and "apeGmsh will
  allow mpco output to be given on the deck. But our viewer is ladruno
  only" (6595709683).
- The reconciled design on #2079 (6596348962). Two architect briefs, one
  by Opus and one by Fable, were written independently against
  `origin/main` e1014a6f8, and the VA orchestrator reconciled them.
- The owner's rulings on its five questions, Q1–Q5, on #2079
  (6596355058, 6596365155, 6596384120, 6596394731, 6596405233).

**Supersedes in part**
[ADR 0122](0122-ladruno-canonical-results-format.md):

- D2, for apeGmshViewer only (D2 here);
- D5's "`Results.from_mpco` stays a reader" (D5 here);
- D6, for footfall and element buckets (D6 here);
- D10, the rest of row 9 and all of row 10 (D10 here).

**Supersedes in part** [ADR 0115](0115-read-back-by-label.md): D3's first
route, "an `MPCO` recorder → `Results.from_mpco`" (D11 here).

**Leaves standing:** ADR 0122 D1 stands and gains one clause (D1 here).
ADR 0122 D3, D4, D7, D8, D9 and ruling 4 stand, and so does ADR 0020
INV-3. [ADR 0121](0121-charter-v2-b-with-amendments.md) D5 puts a changed
decision in a new ADR that names the clauses it supersedes, with the
pointers on the superseded ADRs landing in the PR that accepts it. ADR 0122
and ADR 0115 carry those pointers in this PR.

**Program slice:** #2755 (A0 of chain X, written by link VA).

## Context

ADR 0122 made `.ladruno` canonical. It kept native `results.h5` readable
until its slice 10, and kept `Results.from_mpco` as a reader beside the
MPCO → `.ladruno` converter (D5). On #2655 the owner then ruled that every
viewer route reads `.ladruno` only, while a deck may still emit MPCO. The
architect pair on #2079 worked out what that means for the rest of
apeGmsh, and the owner ruled the five points the briefs left open.

The reconciliation lists the points both briefs agreed on as proposed for
settlement. The owner settles them by accepting this ADR. Q1–Q5 settle the
rest.

## Decision

The end state: apeGmsh reads one results format, `.ladruno`. Decks may
still emit `.mpco` and `.out`, and the fork records `.ladruno`. The
following are deleted:

- native `results.h5`: `NativeWriter`, `Results.from_native` and
  `to_native`;
- `Results.from_mpco` and the slab `readers/_mpco*` stack;
- the viewer's native read path.

Ruling 4 stands: no native → `.ladruno` migrator is built, and old
`.results.h5` files stop opening at the native deletion. No reader
converts on open, and no re-export shim keeps a deleted name alive. Every
deletion is human-gated and lands only after its replacement has a green
oracle on `main`.

### D1 — Every viewer route reads `.ladruno` only

ADR 0122 D1 stands and gains one clause: apeGmshViewer, the v2 stills
([ADR 0128](0128-agent-views-in-apegmshviewer.md)) and the render of a
`Results` held only in memory read `.ladruno` only.

- The in-memory render repoints to the `.ladruno` file the `Results` was
  read from.
- Every other source is refused with a message that names `.ladruno`
  (R22b, as read in 6595544718). That covers `.mpco`, a native
  `.results.h5`, recorders, and `fem=` or units overrides held only in
  memory.

The Qt `ResultsSession` stays until the Qt sunset. It has no format code of
its own, so its sources narrow as the readers below are deleted.

### D2 — The viewer drops native before Python does

ADR 0122 D2's "Until then a `<stem>.results.h5` still opens wherever it
opens today" no longer holds for apeGmshViewer. Slice V1d deletes the
viewer's native read path: `apeGmshViewer/src/reader/results.ts` and the
`.results.h5` slot in `src/main/pairing.ts`. It runs once the writers that
produce native files today have moved to `.ladruno` or `.user.h5` (W1–W4
below). In Python, `Results.from_native` keeps reading until the last N
slice, as ADR 0122 D2 says.

### D5 — The converter stays; `from_mpco` goes

ADR 0122 D5's "The `.mpco` is kept, and `Results.from_mpco` stays a
reader" is replaced by the points below.

- **The converter stays** (Q1, 6596355058). `convert_mpco` is the stock
  on-ramp and apeGmsh's only consumer of an `.mpco`. It keeps the slim MPCO
  read core it uses in `results/readers/_ladruno_from_mpco.py`:
  - `_open_with_reader` checks its input with `MPCOReader` and
    `MPCOMultiPartitionReader`;
  - `_scan_stage` uses `_mpco_spring_io`;
  - `_holds_only_modes` and `_stage_step_counts` use `_mpco_translation`.

  The `.mpco` is kept and never modified. A deck may still emit
  `recorder mpco` (6595709683).
- **Slice DM2 deletes `Results.from_mpco` and the slab `_mpco*` readers**,
  and only after the shared code has moved out.
  - The `.ladruno` reader imports `_mpco*` code today:
    `readers/_ladruno.py`, `_ladruno_element_io.py` and `_ladruno_multi.py`
    import `_mpco`, `_mpco_nodal_io`, `_mpco_translation`,
    `_mpco_layer_io`, `_mpco_local_force_io`, `_mpco_refusals` and
    `_mpco_multi`.
  - That code moves out as pure moves, proven with
    `scripts/verify_move.py`: M1a, then M1b. Both land before any MPCO
    reader is deleted.
- **The stock in-process route converts, or refuses by name** (Q1; the
  reconciliation's point 5, Fable's D-C). Two calls read the `.mpco` with
  `from_mpco` today: `LiveDefault`'s stock MPCO route
  (`results/live/_default.py`, `LiveDefault.results`), and
  `read_deck_mpco` when conversion refuses. Both now convert when the
  stage kinds are facts, and otherwise refuse by name, with no default
  stage kind. A stage kind is a fact in either of two cases:
  - the program declares it in an `ops.stage` block;
  - for an unstaged deck, the deck-run record `<recorder file>.run.json`
    (#2210) passes the checks in the module docstring of
    `_ladruno_from_mpco.py`. `read_deck_mpco` already converts such a
    deck through those checks, and that route stays.

  An in-process run never writes a record, and its stage kind is not yet a
  fact (`_default.py`: its module docstring and the warning on the `mpco`
  route). So an unstaged in-process run refuses, and the converting branch
  for in-process runs waits on #2181.
- **The converter's offline oracle** is a golden `.ladruno` corpus
  converted from the committed `.mpco` fixtures, beside the live-fork
  oracle of ADR 0122 D8. The converter's tests stop reading through
  `from_mpco`.
- **#2454 lands before DM2** (Q5, 6596405233). It adds part-set reduction
  (`PARTITION_REDUCTION`) to the converter, so partitioned stock runs keep
  a route once `from_mpco` is gone.

### D6 — Footfall goes to `.user.h5`, and element buckets carry only known class tags

- **Footfall** writes `<stem>.user.h5` instead of a `.ladruno` bucket that
  apeGmsh would have to invent against ADR 0122 D4 (Q2, 6596365155).
  [ADR 0130](0130-states-quantities-and-the-user-results-file.md) D7
  already names footfall `R` a state-free user result. Footfall leaves the
  list of sources in ADR 0122 D6 that the Python writer serves. That is
  slice W3.
- **Element buckets** in a Python-written `.ladruno` carry only the class
  tags known as fact, which are the `ELE_TAG_*` constants in
  `opensees/_response_catalog.py` (Q3, as read in 6596384120).
  - Every other class is refused by name in `PROVENANCE@unsupported`.
  - The fix belongs at the source: a chain F slice has the fork export its
    element class tags. apeGmsh then takes them from the fork. It never
    guesses them and never keeps a table of its own.
  - The `.out` element coverage of slice W4b waits on that fork slice at
    `FORK_PIN`.

### D10 — The slice plan from here

ADR 0122's slices 0–8 have landed. So has the eigen and parallel-modal
writer from its row 9 (`EigenResult.to_ladruno`,
`ParallelModalResult.to_ladruno`). The table below replaces the rest of
row 9 and all of row 10.

- The stages run in order.
- Within a stage, slices may run in parallel unless an arrow orders them.
- DM1, the converter's deletion, is not carded (Q1).
- The slice list goes to T before any card is cut (6596405233). Each card
  names its own owned files and locks.

| Stage | Slice | Chain | Work | Needs |
|---|---|---|---|---|
| 0 | M1a | X | Shared-code moves out of `_mpco*`, first part (pure moves) | — |
| 0 | W0 | X | Writer moves | — |
| 0 | W1 | X | Modes default to `.ladruno`; only `to_native` still writes native modes | — |
| 0 | W2 | X | The demo writes `.ladruno` | — |
| 0 | V1a–c | V | Viewer test fixtures as `.ladruno` | — |
| 1 | A0 | X | This ADR, with its pointers in ADRs 0122 and 0115 | owner accepts |
| 2 | M1b | X | Shared-code moves, second part | M1a |
| 2 | W3 | X | Footfall → `.user.h5` (D6) | A0 |
| 2 | RT | X | The stock route (D5) and `ops.results` read-back (D11) | A0; #2181 for the in-process converting branch |
| 2 | W4a → W4b | X | `.out` → `.ladruno`: nodes, then element buckets (D6) | A0; W4b's full element coverage also needs the chain F class-tag slice |
| 3 | DM2 | X | Delete `Results.from_mpco` and the slab `_mpco*` readers | M1a, M1b, #2454; human gate |
| 4 | V1d | V | Delete the viewer's native read path (D2) | the VU viewer locks (`main.ts`, `types.ts`, `reduce.ts`, `app.ts`); human gate |
| 5 | N1–N6 | X | Migrate the tests that call `from_native`, by family; the last PR deletes `NativeWriter` and `from_native` | the Qt `tests/viewers` family and the last PR wait on V5, the Qt deletion (Q4); human gate; the changelog announces the cutoff (ruling 4) |

W4a, DM2 and the last N slice take `lock:Results.py` in turn.

On `origin/main`, 49 files under `tests/viewers` call `from_native`.

- **48 are Qt tests.** They are not migrated and go with the Qt viewers at
  V5 (Q4, 6596394731).
- **`tests/viewers/test_agent_manifest_oracle.py` is the exception.** It
  is ADR 0128's v2 oracle, not a Qt test, and it survives V5, so it
  migrates in N.

### D11 — `ops.results` read-back follows D5

ADR 0115 D3's first route, "an `MPCO` recorder → `Results.from_mpco`", is
replaced. For an `MPCO` recorder, `ops.results(path)` follows Q1's D-C
(6596355058; the reconciliation's point 5):

- **It converts and reads the `.ladruno`** when the stage kinds are facts
  in the sense of D5: either declared in `ops.stage` blocks, or, for an
  unstaged deck, proven by a deck-run record that passes the checks in
  `_ladruno_from_mpco.py`'s module docstring, as `read_deck_mpco` does.
- **Otherwise it refuses.** That covers an unstaged run with no record
  that passes those checks: an in-process run, which never writes a
  record, or a deck emitted without `analyze_steps`. The refusal names a
  remedy that works: declare the stages with `ops.stage(...)`, or run the
  program as a deck emitted with `analyze_steps`, which writes the record.
  It does not point at the public `convert_mpco`, which refuses an unstaged
  program. Only `read_deck_mpco` passes the private `_implicit_stage`
  flag.

A `Ladruno` recorder reads through `Results.from_ladruno`. Today
`opensees/_results_route.py::_unreadable` refuses it, as "not a history
ops.results reads".

The `recorder.declare(...)` and typed `recorder.Node` routes stand. They
are read by `Results.from_recorders`.

Three refusal messages in `_results_route.py` still recommend
`recorder.MPCO`: the catch-all (line 103) and the two staged and
partitioned text-record refusals (lines 366 and 372). Changing them is
slice RT's scope. ADR 0115 Invariant 7 holds: a deck may still emit
`recorder mpco` (6595709683), so decks stay byte-identical.

## Consequences — what each group of users loses

- **Fork users** keep reading their `.ladruno` as before. A `.mpco` they
  also record still opens in STKO and still converts, but it no longer
  reads through `from_mpco` after DM2.
- **Stock OpenSees users** lose `Results.from_mpco` at DM2, for old
  `.mpco` files too.
  - Their runs reach Python and the viewer only through `convert_mpco`, so
    its `unsupported` list (today `local_axes`) is their ceiling.
  - Partitioned runs depend on #2454 landing first.
  - An unstaged in-process run loses its default route and refuses by name
    until #2181 lands.
  - The converter's correctness rests on the golden corpus offline and on
    the non-required `live-fork` lane.
- **Pip openseespy users** lose nothing. `.out` gains a viewer route at W4.
  Its element results stay partial until the fork exports its class tags.
- **Viewer users** stop opening a `<stem>.results.h5` in apeGmshViewer at
  V1d. That includes a file `to_native` writes between V1d and the last N
  slice. The v2 stills refuse every source but `.ladruno`.
- **Footfall users** find their results in `<stem>.user.h5`.
- **Everyone** loses old native `.results.h5` files at the last N slice
  (ruling 4).

The price of Q1 is that apeGmsh keeps a slim MPCO read core inside the
converter, not just two helpers.

## Alternatives rejected

- **Dropping the converter** (the Opus brief), with stock and pip users
  going through `.out` (Q1). Through `.out`, stock users would lose
  layered-shell layers, Tri31 per-material strains and the `n_steps`
  cadence.
- **No route that reads MPCO at all** (the Opus brief's RT). It applied
  only if the converter went (the reconciliation's point 5).
- **Footfall as a `.ladruno` bucket** written by the Python writer (the
  Fable brief; Q2).
- **Migrating the Qt viewer tests** in five PRs (the Fable brief; Q4).
- **Deleting `from_mpco` before the shared-code moves** (the Fable brief).
  The `.ladruno` reader imports that code (the reconciliation's points 2
  and 9).
- **A dated amendment to ADR 0122** instead of a new ADR. ADR 0121 D5
  keeps dated amendments for errata; these rulings change decisions.

## References

- #2079, the tracking card. The reconciled design is comment 6596348962;
  the rulings Q1–Q5 are 6596355058, 6596365155, 6596384120, 6596394731 and
  6596405233.
- #2655, the VA6 card: the rulings in 6595544718 and 6595709683.
- Issues named in the plan: #2181, #2454.
- [ADR 0020](0020-results-carries-opensees-model.md),
  [ADR 0115](0115-read-back-by-label.md),
  [ADR 0121](0121-charter-v2-b-with-amendments.md),
  [ADR 0122](0122-ladruno-canonical-results-format.md),
  [ADR 0128](0128-agent-views-in-apegmshviewer.md),
  [ADR 0130](0130-states-quantities-and-the-user-results-file.md).
