# ADR 0095 — `apeGmsh.studio`: agent + script + viewer habitat (summary)

**Status:** Accepted (2026-08-16 — S0–S4e, S5a–S5j, and F-status-1 shipped; Amendments 1–3
closed; contract 1.4.0. Proposed 2026-08-12; owner workshop the same day; host-projection /
INV-9 workshop 2026-08-13)

## Decision

Amends ADR 0094 only in the disposition of its deferred MCP wrapping; 0094 S0–S5 stand.

- **Part 1: a sidecar, not a composite** (the `apeGmsh.hpc` shape): no `_COMPOSITES` entry,
  no `g.studio()`, not re-exported from `apeGmsh/__init__.py`; door `python -m apeGmsh.studio`.
- **Part 2: three processes, one contract.** Cursor stays the IDE and author; a daemon owns
  replay/refresh, last-good tessellation, assess, render and the selection envelope, with the
  Gmsh singleton in a worker process; the host is the promoted Qt `ViewerWindow` with three
  phase modes (model / mesh / results), picking through ADR 0045 `SelectionState`.
- **Part 3: payload before MCP.** v0 writes a names-first `SelectionEnvelope` to
  `.apegmsh/selection.json`; MCP tools come once the envelope is stable.
- **Part 4: preview is refresh** (save / turn end / explicit), last good frame kept on
  failure. **Part 5:** v1 host is Qt; Electron is only a later skin. **Part 6:** read-only
  projections of the replayed script and a PG → apeSees binding graph; never an editor.

Invariants: **INV-1** sidecar (AST guard). **INV-2** the script owns geometry; no write-back
of OCC or `model.h5`. **INV-3** names first; tags are evidence. **INV-4** last good frame.
**INV-5** no shared kernel. **INV-6** ADR 0094 stands (no `viewer()` for diagnosis).
**INV-7** selection/highlight through owner mutators (ADR 0056). **INV-8** phase gate.
**INV-9** project, do not own.

## Amendments

- A1 (2026-08-14): MCP growth law (**INV-10**: MCP wraps habitat verbs only — see, judge,
  show, pin, emit, point — never geometry, materials, constraints or solvers); CLI/JSON
  transport; ReportBundle; `animate(kind=)` (**INV-11**); authored vs generated, `.apegmsh/`
  stays generated (**INV-12**).
- A2 (2026-08-14): HTML and canvas emit skins; markdown stays the archive.
- A3 (2026-08-16): explicit root and contract freeze: **INV-15** root resolution
  (`root=` → `APEGMSH_ROOT` → ancestor with `.apegmsh/` → cwd), **INV-16** atomic
  single-writer snapshots, **INV-17** versioned published contract (schemas + goldens;
  unknown major refused).
- A4 (2026-08-16): live refresh (S6): **INV-18** refresh never steals a busy claim,
  **INV-19** disk is the source; `contract_version` 1.5.0.
- A5 (2026-08-17): the assess verb writes `.apegmsh/assess.json`, which reaches the report.
- A6 (2026-08-17): the habitat template ships as package data (`apeGmsh/studio/template/**`):
  **INV-20** skills are personal, **INV-21** template voice.
- A7 (2026-08-17): an oracle-bearing example library (`src/apeGmsh/studio/examples/`):
  **INV-22** oracle-bearing, **INV-23** dogfood-sourced.
- A8 (2026-08-17): local git and the checkpoint contract: **INV-24** git writes are
  human/agent acts, **INV-25** forge-agnostic, **INV-26** blobs stay out of git, **INV-27**
  checkpoints are convention.
- A9 (2026-08-17): checkpoint clarifications, no new invariants. A10 (2026-08-17): the case
  runner, template script `scripts/run_case.py`.
- A11 (2026-08-24): `.apegmsh/progress.json`, run durations, `contract_version` stamped on
  disk; `CONTRACT_VERSION` 1.8.0; **INV-28** a sidecar never founds a habitat, **INV-29** a
  side-effect writer cannot fail its host.
- A12 (2026-08-27): INV-15 gains `APEGMSH_STUDIO_ROOT` after `APEGMSH_ROOT`; the template
  stamps both; the user-level server is unbound. Contract stays 1.8.0.

Full text: [../0095-apegmsh-studio.md](../0095-apegmsh-studio.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
