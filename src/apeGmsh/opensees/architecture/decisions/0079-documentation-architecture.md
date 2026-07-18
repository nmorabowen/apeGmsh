# ADR 0079 — Documentation architecture: one didactic site, four doc tiers, a single-home rule

**Status:** Proposed (2026-07-18). Ratifies and completes the 2026-05-30
docs program (`internal_docs/plan_docs_learnability.md` = the diagnosis,
`internal_docs/plan_docs_ia_blueprint.md` = the Diátaxis authoring spec,
`internal_docs/plan_docs_skill_reconciliation.md` = the AI-skill side).
Those plans fixed *what the learning content is*; this ADR fixes *where
documentation lives, what gets published, and who owns each fact*.
Implementation plan = `internal_docs/plan_docs_architecture.md`
(PR-by-PR spec for phases P0–P5).

## Context

### What already works

The mkdocs Material site (`docs/`, published at
`nmorabowen.github.io/apeGmsh`) went through the learnability overhaul:
it has a real Diátaxis spine — Tutorials T1–T4 (each ending in a
closed-form-verified number), ~14 verb-titled How-to recipes, a
`concepts/mental-model.md` orientation page, a 12-rung example ladder,
autodoc API reference, and the interactive API Flow Atlas. Install is in
the hero; the front-door cards mirror the Tutorial → How-to → Concepts →
Reference order. That spine is the didactic web page the project needs —
it does not need to be rebuilt.

### The mess: four documentation roots, all published, no ownership rule

1. **`docs/`** — the public site source (the Diátaxis content above).
2. **`architecture/`** (repo root, 21 files) — early design notes
   (13 `apeGmsh_*.md`) plus Gmsh study notes (8 `gmsh_*.md`), *all*
   published under "Internals & Design".
3. **`internal_docs/`** (~85 files) — 23 `guide_*.md` (18 published in
   the nav *as* the Concepts section), ~40 `plan_*` / `handoff_*` /
   `scope_*` working memos (**14 of them published** under a public
   "Planning" nav section), the 2,986-line `first_steps.md` (still in
   nav as "The deep walkthrough" — the exact liability the diagnosis
   condemned), `MIGRATION_v1.md`, and `architecture.md` (which
   duplicates `architecture/apeGmsh_architecture.md`).
4. **`src/apeGmsh/opensees/architecture/`** — 70+ ADRs plus design
   docs: the real, current design record. Unpublished.

Plus the AI-facing skills (`skills/apegmsh/`, `distributable_skills/`,
`.claude/skills/`), which restate the mental model and API surface.

### Concrete symptoms

- **Redundancy.** The results subsystem alone has ≥10 homes across
  three roots (`guide_results`, `guide_obtaining_results`,
  `guide_results_filtering`, `guide_recorders_reference`,
  `how-to/read-results`, `how-to/results-mpco`,
  `how-to/choose-results-strategy`, `examples/results-strategies`,
  `architecture/apeGmsh_results_obtaining`,
  `internal_docs/Results_architecture`). Compose, tie, and
  parts/assembly each have 3–4. Two competing "architecture overview"
  pages sit in different roots.
- **Staleness a learner trips on.** Working memos and superseded plans
  render on the public site with no "this is historical" marker; the
  documented contradictions (15-vs-12 constraint kinds, the stale
  "MP emission deferred" claim) arose precisely because a fact fixed in
  one home survived in another.
- **Unclear canon.** A new user cannot tell the taught path
  (`docs/tutorials/`) from a 2,986-line internal walkthrough, or a
  current recipe from a retired plan — the nav presents them as peers.
  The "retired" legacy notebook gallery is still published.
- **No build gate.** `mkdocs build --strict` is impossible because nav
  pages under `internal_docs/` link into `src/` ADRs outside `docs_dir`
  (noted in `mkdocs.yml`), so broken links and anchors only warn.
- **Maintenance cost.** Every shipped feature owes updates to N pages
  in three roots; in practice the extra N−1 rot.
- **No order, no voice.** Much of the corpus was produced in bulk
  agent waves, and it reads that way: dozens of parallel documents of
  similar shape (`guide_*`, `apeGmsh_*`), each exhaustively symmetric,
  bullet- and table-heavy, none saying "read me before/after that one."
  The result is an *archive* you search, not a *course* you follow —
  there is no reading order between pages, and no single human voice
  within them. This is the felt "AI-ish" quality, and it is a distinct
  defect from redundancy: deduplicating the pile would still leave a
  pile.

## Decision

### D1 — Four tiers, one publishing rule

Every documentation file belongs to exactly one tier. **Only Tier A is
published.**

| Tier | Location | Audience | Contract |
|---|---|---|---|
| **A — Learning site** | `docs/` | users (structural engineers new to apeGmsh) | Diátaxis-pure, current, verified, published via mkdocs. The *only* content in `mkdocs.yml` nav; `docs_dir` contains nothing else. |
| **B — Design record** | `src/apeGmsh/opensees/architecture/` | maintainers, reviewers, AI agents | ADRs append-only next to the code they govern; authoritative on *why*. Not published. |
| **C — Working memory** | `internal_docs/` | the active developer | plans, handoffs, scopes, studies. Allowed to go stale; never linked from Tier A; never in nav. |
| **D — AI skills** | `skills/apegmsh/` (canonical) → derived copies | Claude/agents | derived from Tier A + B per `plan_docs_skill_reconciliation.md`; snippets carry `# verified:` citations. |

The `architecture/` root folder is dissolved (see D3) — four roots
become three, with only one published.

### D2 — The single-home rule

**One fact, one home.** Each topic has at most one canonical page *per
Diátaxis mode* (one tutorial appearance, one how-to, one concepts page,
one API page). Every other mention links to the canonical page and never
restates its content. Redundancy across *modes* is intentional
(a tie shows up as recipe, example, and concept — each serving a
different question); redundancy *within* a mode is a defect.

Canonical-home map for the worst offender (results):

- *Do it*: `how-to/read-results.md`, `how-to/results-mpco.md`
- *Choose*: `how-to/choose-results-strategy.md` (the run×read grid)
- *See it whole*: `examples/results-strategies.md` (E8)
- *Understand*: one consolidated `docs/concepts/results.md` (merging
  the four `guide_results*` pages)
- *Why it's built this way*: ADRs 0014/0019/0020/0026 (Tier B)

### D3 — Consolidation moves (completes the blueprint's Wave 4)

1. **Un-publish Tier C.** Remove the "Planning" nav section, the
   `internal_docs/*` Concepts entries, "The deep walkthrough"
   (`first_steps.md`), and the legacy notebook gallery from
   `mkdocs.yml`. Move `docs/plans/` (viewer working plans that live
   inside `docs_dir` today) to `internal_docs/plans_viewer/`.
2. **Concepts become native.** Each published `guide_*.md` is distilled
   into a `docs/concepts/<topic>.md` (merging overlapping guides —
   e.g. the four results guides → one page; selection + queries +
   selection_chain → one page). The source guides stay in Tier C
   untouched until their content is absorbed, then gain a one-line
   "absorbed into docs/concepts/<topic>.md" header. `first_steps.md` is
   retired the same way (most of its prose was already dispersed).
3. **Dissolve `architecture/`.** The handful of pages a *user* benefits
   from (architecture overview, principles, broker, parts/assembly)
   are curated into a slim `docs/design/` section (~5 pages,
   explanation-mode, linking to Tier B ADRs by number without
   publishing them). Everything else — aesthetic, hinting, navigation,
   ground truth, the 8 Gmsh study notes — moves to
   `internal_docs/architecture/`. A single short
   `docs/concepts/gmsh-under-the-hood.md` replaces the published Gmsh
   study notes, pointing to the official Gmsh docs.
4. **`MIGRATION_v1.md`** moves into `docs/` (it is user-facing) as
   `docs/migration.md`.

### D4 — The didactic front: a visible learning path, and motion that sells

The site's teaching spine gets one addition: a **Learning path** page
(`docs/tutorials/learning-path.md`, linked from the hero) that renders
the T1–T4 → E1–E12 ladder as an ordered table — rung, what it teaches,
prerequisite, and the verification check — so a new user sees the whole
staircase and where they stand. Tutorials/examples keep the blueprint
invariants: typed `apeSees` bridge only, one strategy on the first-
success path, every rung verified against a known answer, committed
rendered outputs so the published page shows the plots and numbers.

**Show, don't tell.** The library's appeal is visual — mode shapes,
wave fields, deforming frames — and the site currently shows only
static plots. A small, curated set of **showcase animations** (4–6
short muted loops, each captioned with the model, the physics, and the
line count of the script that produced it) carries the selling:

- one **hero loop** on the landing page (and the GitHub README — the
  first touchpoint most visitors ever see);
- one short loop atop each flagship example page (modal sweep,
  pushover, SSI wave field).

Rules, so this stays curated and honest: every animation is produced
by a **committed, re-runnable script** from a real documented example
(never a one-off render); loops are web-encoded video (WebM/MP4,
autoplay-muted-loop), not GIFs; a hard **total size budget ≈ 15 MB**
in-repo; and the D5 cap applies — a new animation displaces one or
fills a named slot, no gallery sprawl.

### D5 — Order and voice: a course, not an archive

The site is read in an order, and it sounds like one author.

**Order.**

1. **Nav order = reading order.** Sections and pages are sequenced
   pedagogically, never alphabetically: Tutorials first, and within
   Concepts the pages follow the workflow spine (session → geometry →
   mesh → parts & assembly → physics → bridge → results →
   persistence/compose). The Learning-path page (D4) is the printed
   table of contents of that order.
2. **Every page knows its place.** Each Tier A page opens with one
   sentence stating its job, and closes with an explicit *next* link
   (tutorials/examples additionally state their prerequisite rung).
   A page that cannot name its slot in the sequence does not belong on
   the site.
3. **Curated, capped set.** Tier A is a bookshelf of one book, not a
   dumping ground: target ≈60 pages total. A new page must either fill
   a named empty slot in the order or displace a page it obsoletes —
   "add another guide" is no longer a move that exists.

**Voice.**

4. **One human voice, prose-first.** Pages are written as continuous
   explanation in the register of the existing T1 tutorial and
   mental-model page (the two pages that already sound right). Bullets
   and tables are reserved for genuinely enumerable content (options,
   API grids), never as the default texture of a page.
5. **Ban the generated-doc markers.** No exhaustively symmetric
   coverage for symmetry's sake, no per-paragraph bold headlines, no
   restating the same summary at top and bottom, no "Grounded in the
   current source:" module manifests, no comprehensive-table filler.
   Say the thing once, in order, and stop.
6. **Distillation = rewrite.** The D3 guide-to-concepts consolidation
   is authored fresh in the site voice using the guides as *source
   material* — never assembled by splicing guide sections together.
   Splicing preserves the archive texture this decision exists to kill.
7. **Style contract lives with the docs.** These rules are recorded in
   a short `internal_docs/docs_style.md` referenced by the docs PR
   checklist, so future (human or agent) authors are held to them.

### D6 — Enforcement

1. Once `docs_dir` contains only Tier A, CI runs `mkdocs build
   --strict` (broken nav/links/snippets fail the build) — the gate
   `mkdocs.yml` itself says is currently unreachable.
2. The docs PR checklist gains two questions: *"which canonical page
   owns this fact, and did anything else restate it?"* and *"does the
   page name its slot in the reading order and pass the
   `docs_style.md` voice rules?"*
3. Example/tutorial snippets keep the `# verified:` convention and run
   green against the opensees venv before merge (blueprint Wave rule).

## Alternatives considered

- **Status quo (publish all roots).** Rejected — it *is* the mess:
  no canon, published staleness, no strict build.
- **A second mkdocs site for internals.** Rejected — doubles build
  maintenance for an audience (the maintainer + AI agents) that reads
  markdown in the repo just fine.
- **Move ADRs into `docs/` and publish them.** Rejected for now — ADRs
  deliberately live next to the code they govern and are append-only;
  publishing invites polishing history. `docs/design/` linking to them
  by number gives users the why without moving the record. Revisit if
  external contributors need the ADR log browsable.
- **Delete `internal_docs/` history.** Rejected — working memory has
  real value for the maintainer and agents; the fix is un-publishing,
  not deleting.
- **Big-bang content rewrite.** Rejected — the diagnosis found the
  content good and the packaging broken; this is consolidation.

## Consequences

- One published surface, so "is this current?" has a structural answer:
  if it renders on the site, it is canon and gated; if it lives in
  `internal_docs/`, it is working memory with no freshness promise.
- Feature PRs update at most one page per Diátaxis mode plus the skill.
- `--strict` becomes the docs regression net.
- Cost: the guide→concepts distillation (D3.2) is real writing work
  (~10 consolidated pages); until it completes, the un-published guides
  are reachable only via the repo. Mitigated by phasing: nav surgery
  only removes a guide once its concepts page exists.
- The `architecture/` root disappears from the repo top level —
  historical links into it break (acceptable; it was never a stable
  public URL surface).

## Execution plan (each phase = one verified PR; gate = `mkdocs build --strict` green + nav review)

- **P0 — Ratify + safe nav surgery.** Accept this ADR. Remove from nav:
  "Planning" section, legacy notebook gallery, "The deep walkthrough".
  Move `docs/plans/` → `internal_docs/plans_viewer/`. Move
  `MIGRATION_v1.md` → `docs/migration.md`. (No guide is removed yet —
  strict stays off until P2.)
- **P1 — Order + learning path.** Write `internal_docs/docs_style.md`
  (the D5 contract). Add `tutorials/learning-path.md` + hero link;
  re-sequence the nav into reading order; add the opening job-sentence
  and closing *next* link to every existing Tier A page; commit
  rendered outputs for any tutorial/example page still showing bare
  code.
- **P2 — Concepts consolidation.** Rewrite the 18 published guides into
  ~10 `docs/concepts/` pages (results, selection, meshing, parts &
  assembly, constraints, loads & masses, sections, broker, opensees
  bridge, persistence/compose) — authored fresh in the site voice per
  D5.6, sequenced along the workflow spine; swap nav entries
  one-for-one as each lands; mark absorbed guides; retire
  `first_steps.md` from nav; turn on `--strict` in CI at the end of
  this phase.
- **P3 — Dissolve `architecture/`.** Curate `docs/design/` (~5 pages),
  add `concepts/gmsh-under-the-hood.md`, move the rest to
  `internal_docs/architecture/`, delete the root folder.
- **P4 — Skill re-derivation.** Re-sync Tier D from the consolidated
  Tier A per `plan_docs_skill_reconciliation.md` (its harness and
  citation rules apply unchanged).
- **P5 — Showcase animations** (parallel, any time after P1). The D4
  curated loop set via the existing `viewer.export_animation` +
  committed `scripts/render_showcase/` scripts; hero on the landing
  page + `README.md`, one loop per flagship example; ≤ 15 MB total.

## Open questions

1. **Gmsh study notes** (`gmsh_*.md`): D3.3 demotes them to Tier C with
   a one-page pointer. If users are found relying on them, the
   alternative is a curated "Gmsh background" appendix in Tier A —
   decide at P3 from analytics/issues.
2. **Publishing the ADR log** read-only (rejected above) — revisit when
   external contributors appear.
