# Plan — Docs architecture implementation (ADR 0079)

**Status:** executed 2026-07-18 (P0 #817 · P1 #819 · P2 #821/#824/#825 · P3 #827 · P5 #828; P4 = open follow-up task)
**Driver:** ADR 0079 (`src/apeGmsh/opensees/architecture/decisions/0079-documentation-architecture.md`) —
one published didactic site, four doc tiers, single-home rule, course-not-archive
order + voice. This plan is the PR-by-PR execution spec.

## Mechanics discovered (what actually publishes the mess)

- `docs/_hooks.py` `EXTERNAL_DIRS = ("internal_docs", "architecture")` injects
  **every** `*.md` in both folders into the build via `on_files`. Un-publishing
  a tier = removing its folder from `EXTERNAL_DIRS` once no nav entry points at
  it. `internal_docs` can leave only after P2 (Concepts still serve from it);
  `architecture` only after P3.
- The legacy notebook gallery is `on_pre_build` copy machinery in the same hook
  (`CURATED_NOTEBOOKS`, `examples/EOS Examples` → `docs/examples/notebooks/`)
  plus the `mkdocs-jupyter` plugin.
- `.github/workflows/docs.yml` deploys on push to `main` with **no strict flag
  and no PR-time build check** — a broken link ships silently.
- `docs/index.md` / `docs/changelog.md` are snippet wrappers over `README.md` /
  `CHANGELOG.md` (don't touch when editing the hero — edit `docs/index.md`
  directly; the wrapper note in `_hooks.py` refers to file registration only).

## Verification harness (every phase)

- Build: `C:\Users\nmb\venv\opensees_env\Scripts\python.exe -m mkdocs build`
  (install docs extras once: `pip install -e ".[plot,opensees,viewer,dxf,docs]"`).
  Non-strict until P2-C, `--strict` from then on.
- Eyeball gate: `mkdocs serve`, walk the nav top-to-bottom — reading order must
  make sense with no orphan/misplaced entries.
- Rewritten pages with runnable snippets carry `# verified: tests/...` or are
  run green against the venv (blueprint rule, unchanged).

---

## P0 — Safe nav surgery (1 PR, mechanical)

Nothing is deleted from the repo; only un-published or relocated.

1. `mkdocs.yml` nav: delete the **Planning** section (14 entries), the
   **Notebook gallery (legacy)** subsection (7 entries), and the
   **"The deep walkthrough"** line. Keep the `internal_docs/guide_*` Concepts
   entries (they fall in P2).
2. Retire the gallery machinery: remove `on_pre_build` + `CURATED_NOTEBOOKS`
   from `docs/_hooks.py`, the `mkdocs-jupyter` plugin block from `mkdocs.yml`,
   and `docs/examples/.gitignore` if it only covered `notebooks/`.
3. `git mv docs/plans internal_docs/plans_viewer` (all viewer working plans,
   incl. `future/`); drop the three `plans/*-comparison.md` lines and
   `internal_docs/plan_session_save.md` from `exclude_docs`.
4. `git mv internal_docs/MIGRATION_v1.md docs/migration.md`; nav Migration
   entry → `migration.md`; fix any intra-file relative links.
5. New PR-time check: `.github/workflows/docs-check.yml` — on pull_request
   touching docs paths, install docs extras + `mkdocs build` (non-strict for
   now). Same apt/Qt deps as `docs.yml`.

**Verify:** build green; site shows no Planning section, no legacy gallery,
no deep-walkthrough; migration page renders.

## P1 — Order + learning path (1 PR, mostly writing)

1. **`internal_docs/docs_style.md`** — the ADR 0079 D5 contract, one page:
   nav order = reading order; page opens with a one-sentence job statement,
   closes with a *next* link; prose-first (bullets/tables only for enumerable
   content); banned markers (symmetric filler, per-paragraph bold headlines,
   repeated summaries, module manifests); distillation = rewrite, never
   splice; ~60-page site cap; model registers = `tutorials/first-model.md`
   and `concepts/mental-model.md`.
2. **`docs/tutorials/learning-path.md`** — the printed staircase: ordered
   table of T1–T4 + the 12 built examples (rung · teaches · prereq ·
   verification check), sourced from the IA-blueprint curriculum table.
   Linked from the hero tip on `docs/index.md` and from
   `tutorials/index.md` / `examples/index.md`.
3. **Frame pass** over existing Tier A pages (4 tutorials, 14 how-tos,
   2 concepts, 12 examples, 5 index pages): add the opening job sentence and
   closing *next* link where missing. API reference pages are exempt
   (reference mode has no sequence).
4. **Nav re-sequence** where order is not yet pedagogical (how-to groups
   follow the workflow spine: Geometry/CAD → Build → Physics → Solve →
   Results — mostly true today; fix stragglers).

**Verify:** build green; click through every *next* link (no dead ends);
learning-path prereq column matches actual example prerequisites.

*Execution note (2026-07-18):* the rendered-output audit found five
pages shipping bare code — `tutorials/first-model.md`,
`tutorials/save-reload-view.md`, `examples/results-strategies.md`,
`examples/staged-gravity-ssi.md`, `examples/moment-tensor-source.md`.
Their stills are deferred to P5 item 5, whose showcase scripts run the
same models with the same off-screen machinery. Nav re-sequence needed
no changes — the P0 nav already reads in workflow order.

## P2 — Concepts consolidation (3 PRs, the heavy rewrite)

18 published guides (~9.3k lines) → 11 fresh `docs/concepts/` pages, each
≤ ~300 lines, authored in the site voice (D5.6: guides are source material,
never spliced). Nav swaps one-for-one as each page lands; each absorbed guide
gets a one-line header `> Absorbed into docs/concepts/<page>.md (ADR 0079)`.

| New page (spine order) | Sources (lines) |
|---|---|
| `concepts/session.md` | guide_basics (531) + guide_parts_vs_session (437) |
| `concepts/geometry-and-cad.md` | guide_cad_import (531) + guide_transforms (934) |
| `concepts/meshing.md` | guide_meshing (963) + guide_partitioning (853) |
| `concepts/selection.md` | guide_selection (448) + guide_queries (218) + guide_selection_chain (407, unpublished) |
| `concepts/parts-and-assembly.md` | guide_parts_assembly (429) |
| `concepts/sections.md` | guide_sections (355) |
| `concepts/constraints.md` | guide_constraints (504) |
| `concepts/loads-and-masses.md` | guide_loads (679) + guide_masses (278) |
| `concepts/fem-broker.md` | guide_fem_broker (631) |
| `concepts/opensees-bridge.md` | guide_opensees (883) |
| `concepts/results.md` | guide_results (319) + guide_obtaining_results (463) + guide_recorders_reference (399) + guide_results_filtering (744) |

Unpublished guides (`absorbing_boundary`, `rebar`, `sensitivity`) stay Tier C
untouched. A `concepts/persistence-and-compose.md` has no guide source
(how-tos + ADR 0038 cover it) — backlog, add only if the concepts spine feels
gapped after the 11 land.

- **P2-A** (session · geometry-and-cad · meshing · selection) — the front of
  the spine.
- **P2-B** (parts-and-assembly · sections · constraints · loads-and-masses ·
  fem-broker · opensees-bridge).
- **P2-C** (results — the worst 4-guide merge) + retire `first_steps.md` from
  nav (header: absorbed across concepts/tutorials) + remove `"internal_docs"`
  from `EXTERNAL_DIRS` + add `mkdocs-redirects` mappings
  (`internal_docs/guide_*` → new concepts URLs) + flip `--strict` on in
  `docs-check.yml` and `docs.yml`.

**Verify per PR:** build green; each new page passes the docs_style checklist
(read it aloud once — if it sounds like a generated inventory, rewrite);
snippets verified. P2-C additionally: `--strict` green, redirects resolve.

*Execution notes:* P2-A shipped 2026-07-18 (#821: session,
geometry-and-cad, meshing, selection). P2-B shipped 2026-07-18 (#824:
parts-and-assembly, sections, constraints, loads-and-masses,
fem-broker, opensees-bridge; also fixed the stale auto-emit claims in
`how-to/gravity.md`). P2-C additionally dropped the two
`internal_docs` entries from the Internals & Design nav
(`architecture.md`, `Results_architecture.md` — their content is P3's
concern) and swapped `mkdocs-jupyter` for `mkdocs-redirects` in the
docs extra.

## P3 — Dissolve `architecture/` (1 PR)

| Fate | Files |
|---|---|
| Rewrite into `docs/design/` (~5 curated explanation pages + short index; link Tier B ADRs by number, don't publish them) | `apeGmsh_architecture` + `internal_docs/architecture.md` → `design/architecture.md` · `apeGmsh_principles` → `design/principles.md` · `apeGmsh_broker` → `design/broker.md` · `apeGmsh_partInstanceAssemble` → `design/parts-assembly.md` · `apeGmsh_results_obtaining` + `apeGmsh_results_viewer` + `internal_docs/Results_architecture.md` → `design/results.md` |
| New short page | `concepts/gmsh-under-the-hood.md` (what apeGmsh delegates to Gmsh; points to official Gmsh docs) — replaces publishing the 8 study notes |
| Move to `internal_docs/architecture/` (Tier C) | `apeGmsh_aesthetic`, `apeGmsh_hinting`, `apeGmsh_navigation`, `apeGmsh_groundTruth`, `apeGmsh_constraints`, `apeGmsh_loads`, `apeGmsh_visualization`, all 8 `gmsh_*.md` |
| Delete | the root `architecture/` folder |

Also: replace the nav **Internals & Design** section with the slim **Design**
section; remove `"architecture"` from `EXTERNAL_DIRS` (the constant — and the
`on_files` hook if now empty); trim `docs.yml`/`docs-check.yml` path triggers
(`internal_docs/**`, `architecture/**` out); redirects for moved pages.

**Verify:** `--strict` green; repo root no longer has `architecture/`;
site nav = Home · Tutorials · How-to · Concepts · Examples · Design ·
API Reference · Migration · Changelog.

*Execution note (2026-07-18):* shipped as planned with two additions —
`docs/_hooks.py` deleted outright (EXTERNAL_DIRS emptied with P3, so
the injection mechanism itself is gone), and `concepts/meshing.md`'s
Next now routes through the new `gmsh-under-the-hood` page. The home
"What's new" cards are dated (five-strategy framing, v1.4–v1.5) —
flagged as a small follow-up, not P3 scope. Authored page count after
P3: ~57 (+16 autodoc API pages) — at the D5.3 cap.

## P4 — Skill re-derivation (existing plan)

Re-sync `skills/apegmsh/` (and the derived `.claude` copy via
`scripts/sync_skill.*`) against the consolidated Tier A per
`plan_docs_skill_reconciliation.md` — its harness, citation rules, and
canonical-skill taxonomy apply unchanged. Point skill references at the new
concepts/design URLs.

**Verify:** skill snippets green per the reconciliation harness.

## P5 — Showcase animations (1 PR; parallel — any time after P1)

ADR 0079 D4 "show, don't tell": 4–6 short muted loops that sell the library.
The capability already exists — `viewer.export_animation(path.mp4, fps=,
step_stride=)` (`src/apeGmsh/viewers/animation.py`) drives the results time
scrubber headlessly and encodes H.264 via the `apegmsh[animation]` extra
(imageio-ffmpeg). No new viewer code; this is scripting + curation.

**The set (one clip per flagship example, produced from its documented model):**

| Clip | Source example | Placement |
|---|---|---|
| Plane-wave SSI wave field rippling through the soil box | `examples/plane-wave-ssi.md` | **hero** on `docs/index.md` + GitHub `README.md`, and atop the example |
| Mode-shape sweep (modes morphing in sequence) | `examples/modal-analysis.md` | atop the example |
| Pushover: frame deforming as the capacity curve draws | `examples/pushover-steel-frame.md` | atop the example |
| Double-couple radiation in a solid | `examples/moment-tensor-source.md` | atop the example |
| (backlog, only if cheap) compose/assembly build-up | `examples/compose-modules.md` | atop the example |

**Mechanics:**

1. `scripts/render_showcase/` — one small script per clip: build/solve the
   example model (or load its committed `model.h5` + results), construct the
   off-screen viewer, `export_animation(...)`. Committed and re-runnable —
   a clip nobody can regenerate is banned (D4).
2. Encoding: MP4 H.264, ~8 s seamless loop, ≤ 720p, target ≤ 3 MB/clip,
   **≤ 15 MB total** in `docs/assets/anim/` (hard budget; GIFs banned).
   Tune with `fps`/`step_stride`; re-encode with ffmpeg CRF if over budget.
3. Embedding: `<video autoplay muted loop playsinline>` snippet (Material
   handles raw HTML); one-line caption per clip — model · physics · the
   line count of the script that made it ("38 lines" is the pitch).
   Hero also lands in `README.md` (the actual first touchpoint) — as a
   linked poster-frame image if GitHub video embedding fights back.
4. Style guide: the "animations" clause already lives in
   `internal_docs/docs_style.md` (curated set, displacement rule,
   caption format).
5. Stills owed from the P1 audit: while each showcase model is up,
   save one PNG for the five pages that ship bare code today
   (`tutorials/first-model.md`, `tutorials/save-reload-view.md`,
   `examples/results-strategies.md`, `examples/staged-gravity-ssi.md`,
   `examples/moment-tensor-source.md`) and embed them.

**Verify:** each script runs green in the venv and reproduces its clip;
total `docs/assets/anim/` size within budget; loops autoplay in the built
site (check `mkdocs serve` in a real browser, not just build-green).

---

## Page budget after P3 (the D5.3 cap)

index 1 · tutorials 6 (index + path + T1–T4) · how-to 15 · concepts 14
(index + mental-model + 11 + gmsh-under-the-hood) · examples 13 · design 6 ·
api 16 + atlas · migration + changelog 2 ≈ **74 rendered pages** (~57 outside
the auto-generated API reference) — at the cap; any future page displaces or
fills a named slot.

## Out of scope

- Rewriting tutorials/how-tos/examples content (already in the site voice or
  close; only the P1 frame pass touches them).
- `Assembly`/`couple` docs (branch-only feature — reconciliation plan rule).
- Publishing the ADR log (ADR 0079 open question 2).
