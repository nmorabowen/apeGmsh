### CHANGED — CI fails a PR that edits `CHANGELOG.md`; the union merge driver is gone

PRs add a `changelog.d/<slug>.md` fragment and leave `CHANGELOG.md`
alone. `python scripts/changelog.py --check --base REF` fails when the
diff `REF..HEAD` touches `CHANGELOG.md`, except for an assemble (it
deletes fragments) or a change to the tool itself, and the `lock-tests`
job runs it on every PR with `--base HEAD^1`. GitHub never honoured the
`merge=union` driver: #1267 and #1279 merged cleanly locally but showed
CONFLICTING, with `CHANGELOG.md` the only conflicting file, so concurrent
sessions kept rebasing each other. The anchor comment in `CHANGELOG.md`
now says "do not edit" instead of "insert here", and `.gitattributes`
drops the union driver so local merges report the conflict GitHub sees.
