### FIXED — `land_pr.py` lists changed files through the files API, so a >20,000-line PR can land (#1775)

`scripts/land_pr.py` check 5 read the changed-file list with `gh pr diff --name-only`, which GitHub refuses (HTTP 406, "diff exceeded the maximum number of lines (20000)") for big PRs such as #1772, so the script crashed before merging. It now reads `gh api repos/{owner}/{repo}/pulls/<N>/files --paginate --jq '.[].filename'`, which has no line cap and no 100-file cap. The checks and their order are unchanged; `tests/test_land_pr.py` gains a case where `gh pr diff` answers 406.
