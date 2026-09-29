# changelog.d — one fragment per PR

Add `changelog.d/<slug>.md` instead of editing `CHANGELOG.md`. The slug is
lowercase kebab-case, preferably date-prefixed (`2026-09-29-my-change.md`).
The file is exactly one section: one `### ADDED/FIXED/CHANGED — ...` header,
a blank line, the body, and a final newline. No `## ` header.

    python scripts/changelog.py --check      # validate fragments + CHANGELOG.md
    python scripts/changelog.py --assemble   # release time: fold in, delete

`--assemble` runs at release or in a maintainer housekeeping PR, never in
CI. This README is not a fragment. See `internal_docs/changelog_workflow.md`.
