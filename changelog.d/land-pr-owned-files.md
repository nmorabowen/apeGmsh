### ADDED — `land_pr.py` check 9 refuses a program PR whose files fall outside its slice card's Owned list (lesson #1788, #1855)

A PR labelled `program` must name its card as `Slice: #<n>`. Every file the PR changes,
both names of a rename included, must fall under the card's `### Owned files`,
`### Append-only` or `### Allowed` sections, or under a `**Card amendment` comment.
`changelog.d/*.md` is always allowed. A refusal names every out-of-scope file. The
`scope-ok` label overrides the check, and the script says so. `--owned-only` runs this
check alone on any PR, including merged ones. #1812 was a stale-tree commit that changed
52 files against 6 owned, and this check would have refused it.
