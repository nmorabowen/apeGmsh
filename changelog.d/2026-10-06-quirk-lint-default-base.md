### CHANGED — `check_quirks.py` diffs against `origin/main` by default (program slice R0-L, #1543, lesson #1538)

A bare `python scripts/check_quirks.py` used to skip the diff-scoped rules (`comment-provenance`),
so a local run reported clean while CI's `--base origin/<base_ref>` failed (#1533, #1531). Without
`--base` it now diffs against `origin/main`, else `main`; when neither resolves it prints one
`note:` line naming the skipped rules and keeps the exit code. `--no-base` skips them on purpose.
`--base REF` and `scan(root, base)` are unchanged.
