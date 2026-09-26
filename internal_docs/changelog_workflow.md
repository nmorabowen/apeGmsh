# CHANGELOG workflow — union merge driver + frozen header

**Status:** Active since 2026-06-12.
**Guards:** `tests/test_changelog_structure.py` (curated suite) ·
`.gitattributes` (`CHANGELOG.md merge=union`).

## How to add an entry (the whole workflow)

Insert **one contiguous section** directly below the anchor comment
(`<!-- ⚓ NEW ENTRIES GO DIRECTLY BELOW THIS COMMENT … -->`). The anchor
is the first thing under the `## Unreleased` header, so directly below it
is the top of Unreleased, and the newest entry is always first. Never
insert between the header and the anchor:

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
4. The anchor stays the first non-blank line under the header, and
   there is exactly one (`test_entry_anchor_is_top_of_unreleased`).

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
conflicting. Local merges honour the driver immediately. GitHub-side:
observed during rollout (PR #644) that a **branch-side-only**
attribute did NOT influence GitHub's conflict detection (the PR
flagged DIRTY on a changelog-only overlap that merged cleanly
locally) — the attribute has to be on the merge **target** (main) to
affect PR mergeability. Verify on the first concurrent changelog
overlap after this lands; if GitHub still flags conflicts, the
fallback is unchanged (merge main locally — the driver resolves it —
and push), which is already a one-command fix instead of a manual
resolution.

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
  It also fails when anything sits between the header and the anchor,
  or when there are two anchors (see the next section).

## How the anchor drifted, and the move back (2026-09-25)

The anchor went in right under the header, so "directly below the
anchor" and "the top of Unreleased" named the same place. Nothing held
it there. PR #974 (2026-08-15) inserted its section between the header
and the anchor. From then on the two phrases in this file named
different places, and each PR picked one. By 2026-09-25, 33 sections
had gone above the anchor, pushing it down to line 868, and about 90 had
gone below it. #1171's section, merged 2026-09-25, sat about 850 lines
below the one #1166 had added on 2026-09-19.

The anchor now sits under the header again, and
`test_entry_anchor_is_top_of_unreleased` keeps it there. The move
deleted the nine comment lines and left both blank lines around them,
so the old position now reads as two blank lines. That is deliberate.
Under union, a branch that still inserts below the old position merges
cleanly only if an unchanged line separates its insertion from the
deletion. When the deletion took a blank line with it, the simulated
merges of #1172 and #1174 brought the old anchor back as a second copy.

If your branch was cut before the move, merge `origin/main` and check
where your section landed. If it went below the old anchor, it now
sits where the anchor was, after the two blank lines (about line 900
at the move). If it landed above the anchor, the test fails. Either
way, move it directly below the anchor.

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
