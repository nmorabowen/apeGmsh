### ADDED — `land_pr.py` checks 10 and 11: no landing on a red `main`, none from a stale base (#1877)

On 2026-10-13, `3af31061` merged #1698, whose required checks were green on a merge base
two days old. In between, X4-a had added a retirement-ledger row for the xfail #1698
deletes, so the merge turned `lock-tests` red. #1858 and #1866 then landed while `main`'s
run on `3af31061` was still in progress. Check 10 refuses while `main`'s own required
checks are red, or cannot be read as completed and green. It makes one bounded `gh` read
and fails closed. The `fixes-red-main` label lets the fix or the revert through. Check 11
refuses a PR that touches `tests/` or a baseline file when its required checks ran on a
base more than 10 commits behind `main`, or when `main`'s tip landed more than 24 h after
those checks started. The base comes from the job log's `Merge <head> into <base>`
checkout line. The refusal asks for a fresh merge ref, which a Re-run does not give.
Replays of `3af31061` and `471432f3` are in `tests/test_land_pr.py`.
