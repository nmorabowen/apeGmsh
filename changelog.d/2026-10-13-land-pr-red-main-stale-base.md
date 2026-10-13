### ADDED — `land_pr.py` checks 10 and 11: no landing on a red `main`, none from a stale base (#1877)

On 2026-10-13, `3af31061` merged #1698, whose required checks were green on a merge base
two days old. In between, X4-a had added a retirement-ledger row for the xfail #1698
deletes, so the merge turned `lock-tests` red. #1858 and #1866 then landed while `main`'s
run on `3af31061` was still in progress, on bases without it.

Check 10 reads `main`'s recent commits and, from each `tests.yml` push run, the five
required jobs. Other jobs never block. A commit counts as completed once its five have
finished. Check 10 refuses in these cases:

- The latest completed commit, in commit order, has a required job that failed or timed
  out.
- A required job was cancelled or skipped; the message asks for a re-run of `main`'s Tests.
- `main`'s head H has no run.
- H's five are still running, unless the PR's checks ran on a merge-ref base that contains H.
- A read fails.

The `fixes-red-main` label lets the fix or the revert through.

Check 11 refuses a PR that touches `tests/` or a baseline file when its required checks
ran on a base more than 10 commits behind `main`, or when `main`'s tip landed more than
24 h after those checks started. The base comes from the job log's `Merge <head> into
<base>` checkout line, and both checks read it once. Replays of `3af31061`, #1858, #1866
and `471432f3` are in `tests/test_land_pr.py`.
