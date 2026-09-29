### ADDED — changelog fragments: one `changelog.d/<slug>.md` per PR (program slice A2.1, #1219)

A PR now adds its own file under `changelog.d/` instead of inserting a section
at the `CHANGELOG.md` anchor, so two PRs never touch the same lines and the
union merge can no longer drop the blank line between sections.
`scripts/changelog.py --check` validates fragment names and content and the
`CHANGELOG.md` structure (including a `### ` header with no blank line above
it); `--assemble` folds fragments in below the anchor at release time and
deletes them. Direct sections at the anchor remain valid for old branches.
The adr-docs skill now says the ADR index is generated (#1220).
