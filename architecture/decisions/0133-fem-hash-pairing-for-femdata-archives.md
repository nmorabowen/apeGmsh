# ADR 0133 — A `.ladruno` pairs to a bare FEMData archive by `fem_hash`

**Status:** Proposed (2026-11-03). Merging its PR (`human-gate`) is the
maintainer's acceptance. The pointer on ADR 0122 lands in that same PR, as
[ADR 0121](0121-charter-v2-b-with-amendments.md) D5 requires.

**Owner:** nmora

**Rulings:**

- The owner's ruling on #2768 (W1b), 2026-11-03: "Yes, fem_hash". Results
  recorded against a bare FEMData archive are stamped and paired by
  `fem_hash` alone. Requiring an `ops.h5` (bridge) archive is rejected.
  The T coordinator recorded the ruling on #2768 and #2233.
- T ruling (c) on #1771 (2026-10-19), carded as #2233. It is folded into
  W1b by the ruling above.

**Supersedes in part**
[ADR 0122](0122-ladruno-canonical-results-format.md) D3: its first
attribute item, "`model_hash` and `fem_hash`, copied byte-equal from
`<stem>.h5` `/meta/lineage`". As written, that item needs both hashes.
D1 here replaces it for archives that carry only `fem_hash`.

**Leaves standing:** every other clause of ADR 0122 D3, and D4 (the fork
owns the format; apeGmsh owns its stamp). For a bridge archive, D3 is
unchanged: it is stamped with both hashes.
[ADR 0131](0131-all-in-ladruno.md) stands as written.

**Program slice:** #2788 (W1b-A of chain X). The implementation is W1b,
#2768.

## Context

`capture_live_modes` (`results/live/_modes.py`) accepts a `FEMData` or a
`fem.to_h5` archive. That is the plain-openseespy route of #2342: a deck
run with openseespy alone, no `apeSees` bridge, still gets its modes back
into apeGmsh and the viewer.

ADR 0131 D10 moves those modes to `.ladruno` (slice W1). The Python writer
stamps every file it writes with `stamp_provenance`
(`results/readers/_ladruno_provenance.py`). It reads both hashes from the
archive's `/meta/lineage` and refuses an archive that lacks either one.

A FEMData archive carries `/meta/lineage@fem_hash`, equal to the mesh's
`snapshot_id`, and no `model_hash`: `model_hash` is the bridge's hash of
the OpenSees model, and a FEMData archive has no OpenSees model. A probe on
6bc7e491b raised `ProvenanceStampRefused` and wrote nothing. So, under
ADR 0122 D3 as written, the #2342 route cannot reach `.ladruno` at all.

The viewer's `pairResults` already pairs a results file to its model by
`fem_hash` alone. Its `lineageOf` (`apeGmshViewer/src/reader/ladruno.ts`)
reads both attributes from `INFO/PROVENANCE`.

## Decision

### D1 — A bare FEMData archive is stamped with `fem_hash` only

When the archive `stamp_provenance` is given carries `fem_hash` and no
`model_hash`, the stamp writes `fem_hash`, copied byte-equal as before,
and leaves `model_hash` absent. Every other attribute of ADR 0122 D3 is
written as for a bridge archive.

- An archive with both hashes is stamped with both, unchanged.
- An archive with neither hash, or with an empty `fem_hash`, still refuses
  by name. The stamp never writes `""` and never invents a hash.
- An archive with `model_hash` and no `fem_hash` still refuses.
- In a partitioned set, every part carries the same `fem_hash`.

### D2 — Pairing by `fem_hash` alone is a pairing, not a fallback

A `.ladruno` whose stamp carries only `fem_hash` pairs to the FEMData
archive whose `/meta/lineage@fem_hash` matches it. This holds in apeGmsh's
readers and in the viewer's `lineageOf` and `pairResults`.

- It raises no `DefaultRouteWarning` and no "unpaired" notice. The results
  are paired, against the only model the run had.
- A stamp with both hashes still pairs only to an archive that matches
  both.

### D3 — The fork's wording follows

The fork schema's §7b.1 describes the stamp and its partitioned-set rule,
and it names both hashes. Its wording should say that `fem_hash` alone is
a valid stamp. That is a chain F doc item, with no owner as of
2026-11-03. It does not block this ADR or W1b, because §7b.1 already
tolerates a stamp that carries a subset of the attributes (#2233).

## Consequences

- The #2342 plain-openseespy route reaches `.ladruno` for modes (W1b), so
  that route has no reason left to keep native files.
- A `fem_hash`-only file says less than a two-hash file. It proves the
  mesh, not the OpenSees model built on it. That is all a bare FEMData
  archive can prove. The stamp does not claim more.
- Readers that assumed `model_hash` is always present must treat it as
  optional. W1b owns that change in `lineageOf`.

## Alternatives rejected

- **Require an `ops.h5` (bridge) archive** for `capture_live_modes`. This
  breaks #2342's plain-openseespy route. The owner rejected it on #2768.
- **Keep native modes for this one route until slice N.** This conflicts
  with ADR 0131's end state, in which apeGmsh writes no native results.
- **Stamp a placeholder `model_hash`** (for example `fem_hash` again, or
  `"none"`). A placeholder would claim a pairing the file cannot prove,
  and ADR 0122 D3 copies hashes byte-equal and never derives them.

## References

- [ADR 0122](0122-ladruno-canonical-results-format.md) D3 and D4;
  [ADR 0131](0131-all-in-ladruno.md) D10;
  [ADR 0121](0121-charter-v2-b-with-amendments.md) D5.
- #2768 (W1b), #2233, #1771, #2342.
