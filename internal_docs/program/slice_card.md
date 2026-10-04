<!-- Slice card template. To use it, paste this as the body of a GitHub issue.
     Labels: program, slice, chain:<X>, and mechanical or semantic.
     Keep the filled card at 1.2k tokens or fewer.
     Cite code as path::symbol, never as line numbers (lines drift). -->

**Link:** <X#>
**Chain issue:** #<n>
**Class:** mechanical | semantic
**Backend:** stock | fork | both | n/a
**Agent:** prog-<type>, with model and effort pinned by its definition
**Review:** none | prog-reviewer-<other family>
**Hub lock:** none | lock:<file> | freeze:<file> (until <date>)

### Goal
<!-- One paragraph: what changes and why. Link the report section that motivates it. -->

### Owned files
<!-- The only files the worker may edit. Derive this list, never write it: paste the output of the verification grep run on origin/main untruncated, and check each cited path with `git ls-tree origin/main <path>` or `nav.py where <symbol>`. -->
- `path::symbol`

### Append-only
<!-- For example, the CHANGELOG section or fragment. -->

### Read-only context
<!-- Name each item by path::symbol. -->
- `path::symbol`

### Do not read
Never read files over 2,000 lines whole. Use `python scripts/nav.py` once it exists; until then, grep an outline and read ranges.

### Verification
Run exactly these commands, and paste a summary of the results in the PR body:
- `<command>`

If this is a fix, the regression test must fail with the fix reverted. Say so in the PR body.

A ratchet or exception list pins a baseline (a count or a frozen set) and asserts current <= baseline; rejecting stale entries alone is not a ratchet. Raising the baseline is a maintainer gate (PROGRAM.md §4).

### Stop condition
<!-- When to stop and report instead of widening scope. -->

### Done when
<!-- Conditions someone can observe on main. -->

### Report
Send the orchestrator 300 words or fewer covering:
- the PR URL;
- what changed;
- the verification result;
- risks;
- follow-ups.
