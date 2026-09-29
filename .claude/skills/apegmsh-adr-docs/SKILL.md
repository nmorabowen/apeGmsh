---
name: apegmsh-adr-docs
description: >
  Read before writing or amending an apeGmsh ADR, a plan or handoff in
  `internal_docs/`, a CHANGELOG section, a page on the docs site
  (`docs/`, `mkdocs.yml`), an example, or the canonical skill
  (`skills/apegmsh/`). Nearly every apeGmsh PR carries one of these. A
  checklist of the traps they have hit: ADR numbers that collided,
  CHANGELOG sections a union merge mangled, docs and skill that lagged the
  code. For changing apeGmsh only; *using* apeGmsh is the `apegmsh` skill,
  out of scope here.
---

# Read this before writing an ADR, a CHANGELOG section, docs or the skill

Each item points at where its lesson lives. Open that file and grep for
the quoted heading. `decisions/` is
`src/apeGmsh/opensees/architecture/decisions/`.

## ADRs

- [ ] Take the next number from `decisions/` **on `origin/main`**
      (`git ls-tree --name-only origin/main <decisions>/`), not your
      worktree's copy. The index is **generated** from each ADR's H1 and
      Status line: never hand-edit `decisions/README.md`; run
      `python scripts/adr_index.py` in the same PR (`--check` fails a stale
      one). An unindexed ADR makes its number look free: a second 0065
      landed (#676 → #677), and 0072 and 0074 had to be renumbered (#741,
      #817). The `adr-number` quirk rule fails both mistakes. Re-check
      `origin/main` right before merging: `main` does not require an
      up-to-date branch, so two open PRs that took one number both stay
      green, and the rule only goes red on `main` afterwards.
- [ ] ADRs are append-only. Amend with a dated section or a superseding
      ADR, never by rewriting the decision (`decisions/README.md`, first
      paragraph).
- [ ] Status and shipped notes are claims about code. Verify them against
      the source before writing "shipped" (AGENTS.md, "What this repo is").

## CHANGELOG

- [ ] Add **one fragment file** `changelog.d/<slug>.md` holding one
      `### ` section (lowercase kebab-case slug, date-prefixed), and do not
      edit `CHANGELOG.md`
      (`internal_docs/changelog_workflow.md` "How to add an entry"). Run
      `python scripts/changelog.py --check` before pushing.
- [ ] Why: the union merge driver on `CHANGELOG.md` never conflicts, it
      mangles: blank lines dropped between sections (repaired inside #773
      and #783, three times on 2026-09-25, and in #1186, #1191 and #1217).
      Fragments are separate files, so nothing merges. A maintainer folds
      them in with `--assemble` at release time; never run it in a PR.
      Old branches that still carry a direct section at the anchor stay
      valid.

## The docs site

- [ ] Follow `internal_docs/docs_style.md`. The nav order is the reading
      order, and each page opens with its job and closes with the next
      page. The site holds about 60 authored pages, so a new page fills a
      named slot or replaces one.
- [ ] `mkdocs build --strict` passes (the `docs-check` workflow).
- [ ] Samples use labels and queries, never raw tags (`[1]`, edge-tag lists).
- [ ] Hand-written OpenSees examples use raw `ops.recorder('mpco', ...)`,
      never a private `_emit`.

## The skill

- [ ] Edit only `skills/apegmsh/`. Regenerate the mirror with
      `python scripts/sync_skill.py`, which CI checks with `--check`.
      After the merge, run `python scripts/refresh_user_skill.py`: the
      user-level copy is outside CI's reach.
- [ ] Test every claim against the live code, run with
      `PYTHONPATH=<worktree>/src` (the editable install is main's).
      `tests/test_skill_docs_drift.py` caught seven lines claiming a stale
      viewer default (124b4e30).

## Before the PR

- [ ] `python scripts/check_quirks.py`, then the lock tests the change
      touches (`tests/test_changelog_structure.py`,
      `tests/test_skill_docs_drift.py`).
- [ ] `gh pr create --base main --body-file -` (AGENTS.md "How work lands").
