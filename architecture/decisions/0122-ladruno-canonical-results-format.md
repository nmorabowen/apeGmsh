# ADR 0122 — `.ladruno` is the canonical results format

**Status:** Accepted (2026-10-15; accepted by the maintainer on #2086)

**Owner:** nmora

**Evidence:** two architect briefs written independently, one by Opus and
one by Fable; neither read the other. They are kept in T's notes folder as
`ladruno_canonical/brief_opus.md` and `ladruno_canonical/brief_fable.md`.
T, the program's orchestrator, reconciled them on #2079 (comment 6205108402). The
maintainer ratified the reconciled design as written (6205128487) and then
ruled its five open points one at a time (6205160429, 6205168893,
6205177638, 6205183430, 6205193494). This ADR records those decisions; it
does not reopen them.

**Amends** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1
(the results artifact) and clarifies
[ADR 0020](0020-results-carries-opensees-model.md) INV-3. **Supersedes in
part, at slice 10,** when native `results.h5` is deleted:

- ADR 0020's "Composed file" section, the native `results.h5` that carries
  the embedded `/opensees/` zone;
- [ADR 0113](0113-compatibility-is-a-floor-per-zone.md), where it governs
  the native results zone: D1's results row (floor 1.0.0); D3's
  `RESULTS_SCHEMA_FLOOR` constant, with invariant 2 for the results zone;
  D7's app results floor ("results 1.0", with invariant 4's results
  entry); D9 (native results files whose embedded `/model` or `/opensees`
  is below its floor), with the erratum's restatement of it (E2) and E3's
  reading of its example; invariant 1's results edge; invariant 3's results
  Floor row in the registry; invariant 5's `NativeReader` clause (a
  composed results twin opens); invariant 9's results-twin path
  (`NativeWriter.write_model` and `NativeWriter.write_opensees_from`); and
  invariant 11 (the results floor through `NativeReader`).

All three ADRs carry a dated amendment, added in the PR that accepts this
one, as [ADR 0121](0121-charter-v2-b-with-amendments.md) D5 requires.

**Program slice:** #2085 (link X, slice 0 of the `.ladruno`-canonical chain,
tracking card #2079; chain issue #1199).

## Context

apeGmsh results reach a reader through three formats today. Native
`results.h5` is apeGmsh's own: `results/writers/_native.py::NativeWriter`
writes it for DomainCapture, eigen and parallel-modal results, footfall and
the demo, and it is the only results file apeGmshViewer opens
(`apeGmshViewer/src/main/pairing.ts` pairs `<stem>.h5` with
`<stem>.results.h5`). MPCO is STKO's recorder format, written by stock
OpenSees and the fork alike, and read by `Results.from_mpco`. `.ladruno` is
the fork's own recorder format, read by `Results.from_ladruno` behind an
`INFO/GENERATOR="Ladruno"` and `FORMAT_VERSION` check
(`results/readers/_ladruno.py`).

So a fork run that records `.ladruno`, the path the program invests in,
has no viewer route at all. The question put to the two architects was
which format is canonical, and how the others reach it without losing what
they carry.

Both briefs answered `.ladruno`, and agreed on most of the shape. Where they
differed, T, the program's orchestrator, picked, and the maintainer
ratified the picks with the rest (see *Consequences*).

## Decision

### D1 — `.ladruno` is the canonical results format

apeGmshViewer and `Results.from_ladruno` read the fork's `.ladruno`
directly. Every other route ends in a `.ladruno` or stays a reader of its
own format; none of them ends in a new native file.

### D2 — Native `results.h5` is frozen, then deleted

Native `results.h5` gets no new writer. It stays readable until slice 10,
the human-gated native deletion, which runs after the test files that call
`from_native` have migrated; that slice's changelog announces the cutoff
(ruling 4). Until then a `<stem>.results.h5` still opens wherever it opens
today.

### D3 — Pairing is a stamp in `INFO/PROVENANCE`

The fork creates `INFO/PROVENANCE` empty. After the recorder closes,
apeGmsh writes attributes into it, and nothing else into the file. It may
do so only once the fork doc PR (slice 1, chain F, merged by the
maintainer) blesses consumer writes after close (ruling 2). The contract
below is the one recorded on #2085 (comment 6205346123), aligned with that
fork doc PR.

The case of an attribute's name says who owns it. Lowercase attributes in
`INFO/PROVENANCE` are consumer-owned, which means apeGmsh. UPPERCASE names
stay reserved for the fork, for the optional, later `-provenance`. The fork
owns the layout, so it is the slice-1 fork doc PR that sets this case rule
and withdraws the fork's `MODEL_HASH` slot. `RESULTS_HASH` stays reserved
for the fork.

Every attribute is a scalar, variable-length UTF-8 string attribute. The
attributes are:

- `model_hash` and `fem_hash`, copied byte-equal from `<stem>.h5`
  `/meta/lineage`;
- `stage_names`, a JSON string;
- `element_index_space`, which is `"fem"` or absent;
- `unsupported`, a JSON string listing the lossy categories a converter or
  writer could not carry;
- `writer`, which names the apeGmsh component that produced or stamped the
  file, and its version.

There is no sidecar file, and the fork needs no code change for pairing.
A fork option that writes `PROVENANCE` at record time (`-provenance`) is
optional and later, in chain F; it matters for MPI part files on clusters.
`Results.from_ladruno` gains stage-name binding (`_bind_stage_names`),
which today only `from_mpco` has.

### D4 — The fork owns the format; apeGmsh owns its stamp

The fork owns the `.ladruno` format: its layout, its buckets, its token
names, `FORMAT_VERSION` and `ladruno_format.py`. apeGmsh owns only the
`INFO/PROVENANCE` attributes it stamps (ruling 3). The slice-1 fork doc PR
also corrects the sentence in `03_ladruno_recorder.md` that says apeGmsh
owns the spec going forward.

### D5 — MPCO converts to `.ladruno` explicitly, one way, and keeps the `.mpco`

The MPCO → `.ladruno` converter works at the HDF5 level, bucket for
bucket. Both formats share `MODEL_STAGE[k]/MODEL` and the bucket keys
`<classTag>-<Class>[<rule>:…]`, so the converter never goes through the
slab API, which drops the class, rule and section tags. It is explicit,
one-way and atomic. It refuses an existing target, never modifies the
`.mpco`, and lists every lossy category in `PROVENANCE@unsupported`. The
converted file keeps `GENERATOR="Ladruno"`; `PROVENANCE@writer` marks the
converter.

Conversion runs when a user asks for it, or automatically at the end of a
resolver-driven stock run. The `.mpco` is kept, and `Results.from_mpco`
stays a reader. No reader converts on open: ADR 0020 INV-3 holds
(ruling 5).

### D6 — `.out` transcodes through the Python writer

Pip openseespy has neither MPCO nor `.ladruno`, so `.out` is its users'
only recorder. Their `.out` output transcodes into `.ladruno` through a
small Python `.ladruno` writer, with the MODEL synthesised from the bridge
(ruling 1). The same writer serves every source that has no recorder (pip
eigen and modal results, footfall, the demo) and replaces `to_native`.
`.out` → native retires with native.

### D7 — One token mapping

`RESPONSE_CATALOG` stays the only mapping from apeGmsh components to
recorder tokens. Bucket names and `COMP_NAMES` are read from the file and
copied verbatim. No token table is copied into apeGmsh, and the viewer gets
no TypeScript token table of its own.

### D8 — The conformance oracle lives on the live-fork lane

One model records both an `.mpco` and a `.ladruno` on the fork.
`convert(.mpco)` must equal the fork's `.ladruno` and must pass the fork's
`ladruno_format.validate()`, and mutations of the converted file (a swapped
column, a dropped bucket) must fail it. The oracle runs on the `live-fork`
lane at every `FORK_PIN` bump.

### D9 — Ordering

DomainCapture is deleted only after the viewer opens `.ladruno`. #2071
(merged as #2080) and #1995 (the layered-shell read) land first, and #1995
lands before any MPCO convert-then-open: MPCO reads layered shells today,
so converting first would regress them.

### D10 — The slice plan

| Slice | Chain | Work | Gate |
|---|---|---|---|
| 0 | X | This ADR, with the ADR 0112 D1 and ADR 0020 amendments | maintainer accepts |
| 1 | F | Fork doc PR: `INFO/PROVENANCE` attributes, consumer-written after close; the `03_ladruno_recorder.md` ownership fix | maintainer merges |
| 2 | B | #2071 (merged as #2080) and #1995, the layered-shell read | — |
| 3 | X | `stamp_provenance()`, and stage-name binding in `from_ladruno` | `lock:Results.py`; slice 1 merged |
| 4 | V | Viewer `reader/ladruno.ts` behind a `GENERATOR` + `FORMAT_VERSION` gate: nodes, modes, pairing, one file, a node oracle that runs the real `pairResults`. This restores a viewer route for fork runs | — |
| 5 | X | The #1950 resolver: the fork records `.ladruno` and stamps it at `results()`; stock records MPCO and converts | `lock:spec/_emit.py`; slice 2 (#1995), slice 6 |
| 6 | X | The MPCO → `.ladruno` converter and its conformance oracle (D8) | slice 2 |
| 7 | X | DomainCapture deletion (#1767), after the parity row | human gate; slices 2–6 |
| 8 | V | Viewer Gauss points via `COLUMN_MAP`, `part-N` sets, and a measurement row for the deflate and chunking cost | — |
| 9 | X | The Python `.ladruno` writer: eigen and parallel-modal first, then footfall and the demo | — |
| 10 | X | Native deletion once no writer remains, split by test family (108 test files call `from_native`) | human gate |

## Consequences

The viewer and the Python reader read the same file the fork wrote, so a
fork run is viewable without an apeGmsh-side copy, and the token tables
have one home. The price is that apeGmsh writes into a file it does not own.
D3 and D4 keep that write to six attributes in a group the fork reserves,
after close, and only once the fork has said so in writing.

Pip users keep a full results route through D6, at the cost of a Python
writer that has to synthesise a MODEL the recorder never wrote. Stock users
keep their `.mpco` untouched and get a `.ladruno` beside it. The converter's
correctness is proven only where the fork runs, on the non-required
`live-fork` lane: like every fork-side claim under ADR 0121 D3 and D4, it
is only as good as that lane verifies.

Old native files stop opening at slice 10. That is the maintainer's ruling
(point 4), announced in that slice's changelog; until then nothing that
opens today stops opening.

The briefs differed on four points, and T's picks stand as ratified:

- **Class, rule and section tags on `GaussSlab` and its siblings.** The
  Opus brief found them unnecessary, because the file carries them; the
  Fable brief added them in its first slice. *Deferred*: the converter
  works at the bucket level, so no consumer needs them yet.
- **Slice order.** The Opus brief put the viewer reader early and the
  converter after it; the Fable brief put the converter first and the viewer
  last. *Viewer first*: fork users are the canonical path and today have no
  viewer; stock conversion follows.
- **The fork writing `PROVENANCE` at open.** The Opus brief left it
  optional and later; the Fable brief scheduled it with
  `FORMAT_MINOR_VERSION=2`. *Optional and later*, in chain F, where MPI part
  files on clusters will need it.
- **How a converted file names itself.** The Opus brief used
  `PROVENANCE@CONVERTER`; the Fable brief kept `GENERATOR="Ladruno"` and
  added `@writer`. The two are the same idea; *the Fable spelling* is kept.

## Relation to other ADRs

**ADR 0121.** Nothing here changes the engine policy. The fork records
`.ladruno`; stock records MPCO or `.out` and reaches `.ladruno` through a
converter that refuses an existing target and names what it lost. The
oracle runs on `live-fork`, which stays non-required. No slice adds to a
registry ADR 0121 D4 freezes; a slice that would have to grow one stops
until an Accepted ADR allows it. The amendments to ADRs 0112, 0020 and
0113 land in the PR that accepts this ADR, as D5 there requires.

**ADR 0112.** D1's results artifact becomes `<stem>.ladruno` (its
amendment). D2 holds: the viewer reads `.ladruno` against the fork's format
documents and its version gate, and imports nothing from apeGmsh.

**ADR 0020.** A `PROVENANCE` stamp names the model by hash; it is not a
copy of the model zone, so it is neither the Composed-file pattern nor the
derived copy INV-3 forbids (its amendment). The Composed-file section
retires with native `results.h5` at slice 10.

**ADR 0113.** Its clauses for the native results zone, listed in the
header above, hold unchanged until slice 10. Slice 10 retires the zone
together with its reader rather than raising its floor, and those clauses
lapse with it (its amendment). The neutral, opensees, geometry and
provenance zones keep their floors and the clauses that govern them, and
ADR 0113's section-document amendment is not affected.

## Alternatives rejected

These are the options the maintainer turned down when ruling the five open
points.

- **For `.out`:** retiring it, which would leave pip users with nodes and
  modes only; or keeping `.out` → native through the read window, which
  keeps a native writer alive (ruling 1).
- **Waiting for the fork to write `PROVENANCE` itself** before apeGmsh
  writes into it. The fork doc PR is enough; the fork option stays later
  (ruling 2).
- **apeGmsh owning the `.ladruno` spec**, as the fork doc said. It would
  make apeGmsh a second owner of the token tables (ruling 3).
- **A one-release native read window.** The window lasts until the test
  files have migrated and slice 10 runs (ruling 4).
- **Convert-then-open for MPCO.** `from_mpco` stays a reader, and the
  `.mpco` is kept (ruling 5).

## References

- #2079, the tracking card: the reconciled design (comment 6205108402), the
  ratification (6205128487) and the five rulings (6205160429, 6205168893,
  6205177638, 6205183430, 6205193494). This slice: #2085, whose comment
  6205346123 records the `INFO/PROVENANCE` contract of D3. Chain issue
  #1199.
- The briefs: T's notes, `ladruno_canonical/brief_opus.md` and
  `ladruno_canonical/brief_fable.md`.
- Issues named in the plan: #2071 (PR #2080), #1995, #1950, #1767.
- [ADR 0020](0020-results-carries-opensees-model.md),
  [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md),
  [ADR 0113](0113-compatibility-is-a-floor-per-zone.md),
  [ADR 0121](0121-charter-v2-b-with-amendments.md).

## Erratum (2026-10-16)

D5 says the MPCO to `.ladruno` converter copies "bucket for bucket" and that the two formats "share the bucket keys". That holds for the result buckets only: their names, their tokens (`<tag>-<Class>[r:c:h]`) and their payloads are copied verbatim. The structure around them is not. The converter rewrites the INFO datasets as attributes, turns each flat MODEL/ELEMENTS connectivity dataset into a group holding CONNECTIVITY plus the BASIS attributes, and a QUADRATURE subgroup where the rule is tabulated, restacks ON_NODES into chunked DATA, turns META into COLUMN_MAP plus COMP_NAMES, and adds a KIND to every stage. Slice 6 (#2120, PR #2121) implements it that way, from the scoping table summarised on #2079. This erratum changes no decision, and the D5 text above stands as written.
