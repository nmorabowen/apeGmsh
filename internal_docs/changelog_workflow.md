# CHANGELOG workflow — union merge driver + frozen header

**Status:** Active since 2026-06-12.
**Guards:** `tests/test_changelog_structure.py` (curated suite) ·
`.gitattributes` (`CHANGELOG.md merge=union`).

## How to add an entry (the whole workflow)

Insert **one contiguous section** directly below the anchor comment at
the top of `## Unreleased` (newest first):

```markdown
### ADDED — short highlight title (ADR/PR reference if any)

One paragraph (or a few) describing the change. The section title IS
the highlight — there is no separate header item to append anymore.
```

Rules:

1. **Insert only — never edit existing lines.** In particular the
   single-line `## Unreleased — item · item · …` ledger is **frozen**
   (entries up to 2026-06-12 live there; nothing is ever appended).
2. One section per PR, contiguous (title + body, no interleaved
   edits elsewhere in the file).
3. Keep a blank line before and after your section.

That's it. No fragments directory, no assembly step.

## Why

The old convention had every PR (a) append ` · **item**` to the one
giant `## Unreleased — …` line and (b) prepend a section. Two
consequences, both documented in
`~/.claude/.../memory/feedback_ci_merge_hazards.md` (hazard 1):

* every merge to main re-conflicted **every open PR** on CHANGELOG.md
  alone (PR #604 hit 3× on 2026-06-10; PR #636 needed **five**
  reconciliation rounds on 2026-06-12);
* manual resolutions kept silently **dropping other PRs' header
  items** (#634/#635 were dropped and restored three times in one
  day).

## How the fix works

`.gitattributes` registers git's built-in **union** merge driver for
CHANGELOG.md. When two branches insert different sections at the same
anchor, union keeps both insertions (each contiguous) instead of
conflicting. GitHub's merge machinery (merge-ort) honours in-tree
`.gitattributes` for built-in drivers, so PR mergeability checks stop
flagging changelog-only conflicts too.

Verified by simulation (2026-06-12, three scenarios):

| scenario | result |
|---|---|
| two PRs append to the **giant single line** under union | merges *without conflict* but produces **two duplicated header lines** — the disqualifying mangling mode, and why the ledger line is frozen |
| two PRs each add a bullet **and** a section a few lines apart | hunks overlap → union interleaves the blocks — why the convention is ONE contiguous section, not bullet+section |
| two PRs (plus a third on a stale base) each insert one section at the anchor | clean: no conflict, all sections contiguous and present |

## The caveat that pays for the convenience

`merge=union` **never reports a conflict** — it keeps both sides of
any overlapping edit. For an insert-only file that is exactly right;
for edits to existing lines it is silent corruption. Hence:

* the frozen-ledger rule and the insert-only rule above;
* `tests/test_changelog_structure.py` fails the suite when the known
  mangling signature appears (duplicated `## Unreleased` line,
  missing anchor, conflict markers, or a dropped `.gitattributes`
  entry), so a bad merge turns main red instead of shipping.

## Migrating an in-flight PR (opened before 2026-06-12)

If your branch still edits the giant header line, merging it under
union will duplicate the header (the structure test will then fail).
Before merging:

1. `git merge origin/main`, take main's CHANGELOG wholesale
   (`git checkout --theirs CHANGELOG.md`),
2. re-add your entry as a single `###` section at the anchor comment,
3. drop your header-line edit entirely.

## Rejected alternative

Changelog fragments (towncrier-style `changelog.d/` + CI assembly)
also eliminate the conflict class, but add tooling, an assembly step,
and a moment where CHANGELOG.md on main lags reality. The repo treats
the Unreleased block as a living ledger, so the zero-tooling union
convention fits better (CLAUDE.md: simplicity first). Revisit only if
section-insertion ordering under union ever becomes a real problem.
