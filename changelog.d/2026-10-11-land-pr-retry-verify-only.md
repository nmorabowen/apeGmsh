### FIXED — `land_pr.py` survives a transient post-merge `gh` failure (lesson #1682)

Every read after the merge call (mergeCommit lookup, `compare/main...<sha>`, `ls-remote`) is retried
with bounded backoff (5 tries, 1/2/4/8 s); the non-idempotent merge call never is. If a read keeps
failing, the script exits non-zero with `MERGED BUT UNVERIFIED: #N` and the command to run. New
`--verify-only <N>` skips the checks and the merge and runs only the post-merge proof for an
already-merged PR (refuses an unmerged one). Two landings (#1672, #1669) had skipped the proof this way.
