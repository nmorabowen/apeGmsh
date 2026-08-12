# Handoff — ADR 0094 implementation brief (S0 + S2)

> **Audience:** the implementing agent (Grok 4.6 experiment).
> **Scope:** slice **S0** (skill fixes), then slice **S2**
> (`apeGmsh.assess`). Two separate PRs, S0 first.
> **Out of scope — do not touch:** S1 (`viewers.render`), S3–S5,
> `RES.ZERO_U`, `CAD.*` findings, any new finding code, any refactor
> of adjacent code. If something seems to need one of these, stop and
> say so in the PR description instead of doing it.
> **Line numbers below are anchors, not gospel** — verify by search
> before editing; they drift.

## Read first, in order

1. `src/apeGmsh/opensees/architecture/decisions/0094-agent-assess-and-viewer-render.md`
   — the ADR. It is the contract; this brief only adds pointers.
2. `docs/design/principles.md` — sidecar rule, "notebooks are the
   habitat", chain ends at `FEMData`.
3. `CLAUDE.md` (repo root) — surgical-change rules. Every changed
   line must trace to the ADR.
4. `internal_docs/changelog_workflow.md` — how to write the
   CHANGELOG entry.

## Environment

- Interpreter: `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`
  — **always by explicit path**. Bare `python` is system 3.12 and its
  `ModuleNotFoundError`s look like real bugs.
- Sanity check before starting:
  `...\python.exe -c "import apeGmsh; print(apeGmsh.__file__)"` must
  point at your checkout (editable install), not another copy.
- Tests: `...\python.exe -m pytest -q <paths>`. Keep assess tests out
  of the qt lane (no `@pytest.mark.qt`, no Qt imports) — they must
  pass on a GL-less runner.
- `set LADRUNO_OPENSEES_QUIET=1` to silence the fork banner.
- There is a mypy ratchet in CI — type new code fully; do not grow
  the baseline.

## S0 — skill fixes (docs only, no library code)

The claim "`results.viewer()` defaults to `blocking=True` and crashes
Jupyter" is stale. Live truth: `Results.viewer` in
`src/apeGmsh/results/Results.py` (~line 1167) takes
`blocking: Optional[bool] = None` and auto-detects — `True` in
scripts/CLI, `False` (subprocess) in a Jupyter kernel, `show_web()`
fallback for in-memory Results in a notebook. Fix every occurrence:

| File | Anchors |
|---|---|
| `skills/apegmsh/references/results.md` | ~259 (heading), ~262–263 (fabricated signature), ~266, ~450 |
| `skills/apegmsh/references/workflows.md` | ~431–432, ~578–579 |
| `skills/apegmsh/SKILL.md` | ~280–281 |
| `skills/apegmsh/references/section-properties.md` | ~280 — **only the `results.viewer` half is stale.** `sec.viewer` genuinely defaults to `blocking=True` (`src/apeGmsh/sections/_analysis.py` ~449); leave that claim alone. |

Also per ADR S0: the post-solve happy path in the skill teaches the
**inspect surfaces** (`fem.inspect`, `results.inspect.summary()` /
`components()` / `diagnose()`, `results.lineage`) as the check after
solving — not `viewer()` / `show_web()`. Qt viewers are for humans
who asked to look.

Check for mirrored copies: if the edited files are duplicated under
`distributable_skills/` or `.claude/skills/`, keep the copies in sync
(search for the same stale sentences).

## S2 — `apeGmsh.assess` (the real test)

### Shape

- New package `src/apeGmsh/assess/` modeled on `src/apeGmsh/hpc/`
  (standalone sidecar: types + pure runners; **not** in
  `_COMPOSITES` (`src/apeGmsh/_core.py` ~44); **not** re-exported
  from `apeGmsh/__init__.py`).
- `Finding` / `AssessmentReport` frozen dataclasses exactly as the
  ADR sketches. Pattern to copy: `src/apeGmsh/cuts/_preflight.py`
  (~49 `PreflightIssue`, ~75 `PreflightReport`).
- Public doors: `fem.assess()` on `FEMData`, `results.assess()` on
  `Results` — thin methods that call into the package.
- `figures=` ships in S3 — in S2 the parameter may exist but only
  `False` is implemented (document that; do not stub render calls).

### v1 catalog — data-source anchors

| Code | Where the data lives |
|---|---|
| `MODEL.EMPTY` | node/element counts on `FEMData` |
| `MESH.INVERTED` | connectivity + coords; **linear cell types only**, corner-node signed volume. Quadratic/Bézier cells → listed as skipped (ADR 0091 control values are not nodal coords). |
| `MESH.COINCIDENT_UNBRIDGED` | `FEMData.inspect.find_coincident_node_pairs` (`src/apeGmsh/mesh/FEMData.py` ~1479). Returns `dict[(i, j), list[str]]`. **Finding = empty list. Branch on emptiness only; never parse the strings.** Call once; cap evidence ids. |
| `MESH.EMPTY_PG` | `PhysicalGroupSet` (`src/apeGmsh/mesh/_group_set.py` ~382, `summary()` ~337) — H5-safe, no gmsh import. Do not parse summary text; use the frame/API. |
| `RES.UNBOUND_FEM` | `results.fem is None` (`Results.py` ~694) |
| `RES.NO_STAGE` | `results.stages` (`Results.py` ~1003) |
| `RES.NAN` / `RES.INF` | component names via `results.inspect.components()` (`src/apeGmsh/results/_inspect.py` ~54); values at `time=-1` only. NaN sentinels (unvisited/fill) must not be failed — sentinel semantics are reader-specific: state the rule you chose per reader in the PR description. |
| `RES.LINEAGE` | `results.lineage.warnings` (`Results.py` ~817). The property never raises (ADR 0021 INV-2) — do not add try/except theater around it, and never call `assert_clean()`. |
| `RES.U_VS_DIAG` | ‖u‖ at `time=-1` vs model diagonal. **First move `model_diagonal` from `src/apeGmsh/results/plot/_arrows.py` (~13) to a new numpy-only `src/apeGmsh/results/_geometry.py`, re-export from `plot._arrows`.** Never severity `error`. Skip `kind="mode"` stages. |
| `RES.ENERGY_ERR` | `Results.energy()` (`Results.py` ~709). Raises `TypeError` on native/MPCO (~734) → catch that one exception type and list the check as skipped. |

**Not in S2:** `CAD.SLIVER_*` (live-kernel only — `ImportHealth`
reads gmsh/OCC directly; belongs to the S4 live-session slice),
`RES.ZERO_U` (deferred to `OpenSeesModel.assess()`), anything not in
the v1 table (INV-6: new codes are an ADR amendment).

### Cost traps

- `nodes.get(..., time=None)` loads **every step**
  (`src/apeGmsh/results/_time.py` ~9). Findings use `time=-1`.
- `node_envelope` (`Results.py` ~772) and `energy()` are
  Ladruno-only; both raise `TypeError` on native/MPCO and
  `node_envelope` also on partitioned-envelope merges. Fall back to
  `time=-1`, never to all-steps (INV-9).

### Tests — same PR, non-negotiable

1. **AST import guard**: no `.py` under `src/apeGmsh/assess/`
   imports `apeGmsh.viewers` or `gmsh` (INV-1). Model it on
   `tests/test_viewers_pure_h5_consumer.py` (walk + AST + allowlist);
   include a positive control (a deliberately bad snippet the guard
   catches), like `tests/opensees/h5/test_model_data_ast_guard.py`
   does.
2. **Acceptance fixtures** (reuse the smallest existing H5 fixtures
   in `tests/` before creating new ones): planted unbridged
   coincident pair → `MESH.COINCIDENT_UNBRIDGED`; dirty lineage →
   `RES.LINEAGE` without raising; injected NaN in a recorded
   component → `RES.NAN`; empty model → `MODEL.EMPTY`.
3. **Skip ≠ pass** (INV-3): a Results without an energy channel
   lists `RES.ENERGY_ERR` under skipped checks in `report.text`, not
   as OK.
4. **FAIL reserved** (INV-4): assert the severity of every emitted
   code matches the v1 table; `RES.U_VS_DIAG` never `error`.
5. All tests green with no GL / no Qt / no live gmsh session.

## Process

- Branch names: `grok/adr94-s0-skill`, `grok/adr94-s2-assess`.
- One CHANGELOG entry per PR, per `changelog_workflow.md`.
- Match surrounding style (frozen dataclasses, tuple fields, the
  repo's docstring voice). No drive-by cleanups.

## How the review will judge this (read before writing code)

The maintainers will review against, in order:

1. **Invariant tests exist and bite** — especially INV-1 (guard with
   positive control), INV-3, INV-4, INV-9.
2. **The three judgment points** get review-first attention:
   placement/wiring of the promoted `model_diagonal`; the per-reader
   NaN-sentinel rule; that coincident-pair logic branches on
   emptiness only.
3. **PR description honesty** — every ambiguity you resolved is
   written there. An undocumented judgment call found in the diff
   counts against the work; a documented one counts for it.
4. **Diff hygiene** — every changed line traces to the ADR; no scope
   creep into S1/S3+, no new codes, no `fix=` snippets, no viewer
   imports.
5. **`report.text` quality** — 40–80 lines, no table dumps, top-N
   extrema appendix only, skipped checks listed.
