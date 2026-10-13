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

### Seams
<!-- Semantic cards that touch a hub or write files only; delete otherwise. List each seam the change crosses: reload, shared namespace, a raise after registration, empty or degenerate names, partial writes. -->
- reload:
- shared namespace:
- a raise after registration:
- empty or degenerate names:
- partial writes:

### Owned files
<!-- The only files the worker may edit. Derive this list, never write it: paste the output of the verification grep run on origin/main untruncated, and check each cited path with `git ls-tree origin/main <path>` or `nav.py where <symbol>`.
     `land_pr.py` check 9 enforces it (lesson #1788): a PR labelled `program` must say `Slice: #<n>`, and every file it changes, both names of a rename included, must be named in backticks here, under Append-only or under Allowed. The rules:
     - `path::symbol` counts as `path`.
     - A trailing `/` means a directory. A `*` or `**` glob needs a literal top directory (`tests/x/**`, not `**/*.py`).
     - A bare name listed right after a path is its sibling: "`a/b/x.py`, `y.py`" owns `a/b/y.py`.
     - Placeholders such as `<slug>` and non-paths such as `lock-tests` are ignored.
     - Only `changelog.d/*.md` is implicit. Goldens, snapshots, `_api_index.json` and the skill mirror must be listed.
     - A section ends at the next `#` heading or bold-label line (`**Hub lock:**`). An `Owned:` label counts only where it opens a line or a sentence; "not owned:" is never a label.
     - An Owned line with a path and a negation is refused as ambiguous: list only what the PR may change, and put exclusions under another heading. A negation is `never`, `not` (also `do _not_`), `cannot`, any n't contraction (`don't`, `won't`, straight or curly apostrophe), `except`, `exclude`/`excluding`, `avoid`, `leave … alone` or `no changes to`/`no edits to`, outside the backticked paths (`land_pr.py::_NEGATION`).
     - A card with no Owned section, or one that names no path, is refused. -->
- `path::symbol`

### Append-only
<!-- For example, the CHANGELOG section or fragment. -->

### Allowed
<!-- Optional: files the PR may change but does not own, such as regenerated goldens or the skill mirror. -->

<!-- Amendments: to add files after the card is filed, post an issue comment whose first words are `**Card amendment` and name each path in backticks; check 9 adds them. An amendment line with a negation (the list above) adds no path; check 9 says it skipped it. A negated lead-in ("Do not touch:" then a list; any negated line that is not a bullet, or that ends in `:`) also drops every line after it up to a blank line, a heading or a bold label; a negated bullet drops itself and its sub-items. So put each addition in its own positive line, after a blank line if a negation precedes it. Any other comment, a maintainer ruling included, adds nothing. Limitation: the amendment's author is not checked, because every account here is the same user today.
     Override: the maintainer or coordinator applies the `scope-ok` label to the PR, and check 9 passes with an OVERRIDDEN notice, the way `human-gate` admits a baseline edit at check 8. -->


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
