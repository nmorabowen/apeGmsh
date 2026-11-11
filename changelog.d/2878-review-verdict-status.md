### ADDED — a `review-verdict` commit status on every PR head, and only a collaborator's comment counts as a verdict (#2878, lesson #2637)

A web merge never runs `land_pr.py`, so a program PR could merge with no
`Review-verdict` approve at its head and only the nightly orphan detector
would notice. The new workflow `.github/workflows/review-verdict.yml`
posts a `review-verdict` status on each PR head, on `pull_request_target`
and `issue_comment` events. Once the status is required on the `main`
ruleset, the merge button refuses a head that no verdict approved.

`scripts/check_review_verdict.py` decides the status by calling the
detector's own two functions, `find_orphans.verdict_required` and
`find_orphans.has_head_approval`, so the gate and the detector cannot
disagree. A status belongs to the commit, so the check decides every
open PR whose head is that commit: `success` only when each one needs no
verdict or its latest verdict at the head approves, `pending` otherwise
(naming the PR that waits), and `error` on the event's head when a read
or the decision fails. Runs that can post on one commit queue in one
concurrency group. The job checks out `main` only, never PR code, and its
one write permission is `statuses`. A file listing cut at the 2,000-file
page cap now counts as needing a verdict, in the detector too.

The repo is public, so `program_probe.verdicts` now counts a
`Review-verdict` line only from a comment whose author association is
`OWNER`, `MEMBER` or `COLLABORATOR`. That one parser feeds the status, the
orphan detector and the program probe. `find_orphans.has_head_approval`
now passes the REST `author_association` through to it.
