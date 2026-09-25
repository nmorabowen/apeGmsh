---
name: apegmsh-viewer-results
description: >
  Read before changing apeGmsh's viewers or results stack: anything in
  `src/apeGmsh/viewers/`, `src/apeGmsh/results/` (readers, capture,
  derived quantities), the Qt/VTK windows and docks, the web/trame
  viewer, or code that reads model.h5 / MPCO / .ladruno files with h5py.
  A checklist of the traps viewer and results PRs have hit. For changing
  apeGmsh only; *using* the viewers is the `apegmsh` skill, out of scope
  here. For window-layout design, the `viewer-ui-designer` agent in
  `.claude/agents/`.
---

# Read this before changing a viewer or the results stack

Each item points at where its lesson lives. Open that file and grep for
the quoted heading. `arch/` means `src/apeGmsh/opensees/architecture/`,
and `decisions/` is `arch/decisions/`.

## Boundaries

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

## Qt and VTK

- [ ] Real-window tests carry `@pytest.mark.qt` and run one **file** per
      process (`pyproject.toml`'s `addopts` comment; the `qt-window-tests`
      lane). A CLI `-m` replaces the default, so repeat `and not qt`. A
      `show()`-until-closed test once hung the suite for 6 h (ADR 0089).
- [ ] `QT_QPA_PLATFORM=offscreen` plus a pyvistaqt `QtInteractor`
      segfaults on Windows. A real-viewer test runs in a subprocess, or
      catches the error and skips.
- [ ] Never call `restoreDockWidget` on the navigation docks. It brought
      back the stuck Outline dock (d6e7aabc,
      `tests/viewers/test_dock_invariant.py`), and that test scans only
      the files it names, so a new host is unguarded.
- [ ] Keyboard shortcuts over a VTK interactor use a `QShortcut` with
      `Qt.ApplicationShortcut` context, or the interactor eats the key.
- [ ] A results-viewer catch site that should fail loudly is covered by the
      strict pump-failure fixture. Opt out only with
      `@pytest.mark.allow_pump_failures` (ADR 0084 D4).
- [ ] `results.viewer()` in a notebook must not block the kernel; the
      default is now auto (2fe994cf). Examples and docs follow it, and
      `tests/test_skill_docs_drift.py` catches a skill that still says
      otherwise (124b4e30).

## Gates before the PR

- [ ] Run the touched viewer tests (qt-marked files one per process), then
      the curated `suite` lane for the boundary tests.
- [ ] `python scripts/check_quirks.py`, plus a CHANGELOG section per
      `internal_docs/changelog_workflow.md`.
- [ ] If the window changed, look at it. Capture before-and-after
      screenshots and check them for clipping, overlap, stale state and
      focus. Offscreen GL and on-screen `show()` both work on the
      maintainer's machine.
