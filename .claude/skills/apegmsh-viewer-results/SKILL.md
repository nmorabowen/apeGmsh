---
name: apegmsh-viewer-results
description: >
  Read before changing apeGmsh's results stack or what crosses into the
  viewers: `src/apeGmsh/results/` (readers, capture, derived quantities),
  code that reads model.h5 / MPCO / .ladruno files with h5py, the
  FEMData ↔ Results binding, or a viewer's imports of the model. The entry
  point for `viewers/` too: it routes viewer internals (dispatcher, docks,
  key bindings, Qt/VTK) to `apegmsh-viewers-change` and visual proof to
  `apegmsh-viewers-visual-check`. For changing apeGmsh only; *using* the
  results or viewers is the `apegmsh` skill, out of scope here.
---

# Read this before changing the results stack or a viewer

Each item points at where its lesson lives. Open that file and grep for
the quoted heading. `decisions/` is
`src/apeGmsh/opensees/architecture/decisions/`.

## Changing code under `viewers/`

- [ ] Read `.claude/skills/apegmsh-viewers-change/SKILL.md` before
      touching the dispatcher, pumps, a diagram kind, a dock or pane, key
      bindings, or a VTK/pyvista call. Its **[guard]** items are enforced
      by AST tests in `tests/viewers/`.
- [ ] Read `.claude/skills/apegmsh-viewers-visual-check/SKILL.md` before
      claiming a viewer change works: reference and candidate stills,
      inspected, because a green suite is not a picture.
- [ ] Both guides arrive with #1170. Until it merges, see that PR.

## The boundary between results and viewers

- [ ] Viewers are pure `model.h5` consumers (ADR 0014). They import
      `OpenSeesModel` only from the `apeGmsh.opensees` package top level,
      never a submodule. `tests/test_viewers_pure_h5_consumer.py` lives
      **outside** `tests/viewers/`, so running only `tests/viewers` misses
      it; that is how #757 shipped red (fixed #759).
- [ ] Read through the `H5ModelReader` Protocol (ADR 0026), and never
      import the Protocol class into `viewers/`.
- [ ] FEMData ↔ Results binding is a lineage chain that warns and does
      not raise (ADR 0021). Do not add `snapshot_id`-equality guards.

## Reading HDF5

- [ ] Probe an optional child with `name in group` (the readers' `_child`
      helper), never `group.get(...)`, which returns `None` both for a
      missing child and for a broken link. The MPCO reader needed a
      33-probe sweep (414ca711, #424). Its AST guard,
      `tests/test_results_mpco_get_hazard.py`, scans only `_mpco*.py`, so
      a new reader is unguarded.
- [ ] Versions are per zone with a two-version window (ADR 0023, "Per-zone
      read validation"). Tests take versions from `tests/fixtures/schema.py`;
      the `schema-literal` quirk rule holds this.
- [ ] A new result type is also a response-catalog row. The catalog and its
      tests' `expected` sets are shared literals (AGENTS.md "How work lands").
- [ ] A value that must match across processes (a tag, a colour index, a
      cache key) never comes from the builtin `hash()` of a string, which
      is salted per process. Use `zlib.crc32` or a real digest.

## Gates before the PR

- [ ] Run the touched results tests, then the curated `suite` lane for the
      boundary tests. qt-marked viewer files run one per process (AGENTS.md,
      "Build and test").
- [ ] `python scripts/check_quirks.py`, plus a CHANGELOG section per
      `internal_docs/changelog_workflow.md`.
