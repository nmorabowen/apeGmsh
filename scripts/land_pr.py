"""Land one PR on main: the "How work lands" checklist (AGENTS.md) as code.

    python scripts/land_pr.py <number>            # check, squash-merge, prove it landed
    python scripts/land_pr.py <number> --dry-run  # run the checks and stop before the merge

The checks run in order and the first failure refuses, naming the check and
the lesson. Then ``gh pr merge --squash --match-head-commit <checked head>``
(never ``--auto``, never ``--delete-branch``), so a push between the checks
and the merge is refused by GitHub; then ``compare/main...<squash sha>`` must
read ``behind`` or ``identical``; then the branch tip must still be the
merged head, so a push that raced the merge is flagged rather than left on
an orphaned branch. A push after the landing is the nightly orphan
detector's job (.github/workflows/orphans.yml, #1230).

GitHub is reached only through ``run_gh`` and the checkout only through
``run_git``, so tests/test_land_pr.py replays recorded responses through
those seams with neither ``gh`` nor a network. Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

Runner = Callable[[list[str]], str]

#: The checks `main` requires (AGENTS.md "How work lands"). One that has not
#: reported is a refusal: zero checks visible means a conflict or a stalled
#: Actions, not green (#630).
REQUIRED_CHECKS = ("lock-tests", "emit-cost-gate", "static-gates", "suite", "live-stock")
PR_FIELDS = "state,isDraft,baseRefName,headRefName,headRefOid,mergeable,statusCheckRollup"
COMPARE_TRIES = 5
COMPARE_WAIT_S = 3.0


class LandError(Exception):
    """A check refused, or the merge could not be proven to have reached main."""


def run_gh(args: list[str]) -> str:
    return subprocess.run(["gh", *args], check=True, stdout=subprocess.PIPE, text=True).stdout


def run_git(args: list[str]) -> str:
    return subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE, text=True).stdout


def _green(check: dict[str, Any]) -> bool:
    """A CheckRun carries status/conclusion; a commit StatusContext carries state."""
    if check.get("__typename") == "StatusContext":
        return check.get("state") == "SUCCESS"
    return check.get("status") == "COMPLETED" and check.get("conclusion") == "SUCCESS"


def check(number: int, gh: Runner, git: Runner) -> dict[str, Any]:
    """Checks 1-6, in order. Returns the PR record, or raises LandError."""
    pr = json.loads(gh(["pr", "view", str(number), "--json", PR_FIELDS]))
    head = pr["headRefOid"]
    if pr["baseRefName"] != "main":
        raise LandError(
            f"check 1: #{number} targets {pr['baseRefName']!r}, not main. A PR merged into "
            f"a stacked base never reaches main (#858 was missing for two months; #1057). "
            f"Retarget it (gh pr edit {number} --base main), then push a commit."
        )
    if pr["state"] == "MERGED":
        raise LandError(
            f"check 2: #{number} is already merged. A push to its branch now lands on an "
            f"orphaned branch (#335 -> #336); open a new PR for the change."
        )
    if pr["state"] != "OPEN" or pr["isDraft"]:
        state = "a draft" if pr["isDraft"] else pr["state"]
        raise LandError(f"check 2: #{number} is {state}, not an open PR.")
    local = git(["rev-parse", "HEAD"]).strip()
    if local != head:
        raise LandError(
            f"check 3: local HEAD {local[:12]} is not the PR head {head[:12]}. Push, then "
            f"wait until headRefOid equals your tip (#555 -> #556)."
        )
    if git(["status", "--porcelain"]).strip():
        raise LandError("check 4: the working tree is not clean (git status --porcelain).")
    labels = json.loads(gh(["label", "list", "--limit", "200", "--json", "name"]))
    frozen = [lab["name"][len("freeze:") :] for lab in labels if lab["name"].startswith("freeze:")]
    # `pr view --json files` stops at 100 files; the diff listing does not.
    for path in gh(["pr", "diff", str(number), "--name-only"]).splitlines():
        for pattern in frozen:
            if fnmatch.fnmatchcase(path, pattern):
                raise LandError(
                    f"check 5: {path} is inside the open freeze:{pattern} window "
                    f"(the board, #1203, names its expiry)."
                )
    if pr["mergeable"] != "MERGEABLE":
        why = "merge main locally and push" if pr["mergeable"] == "CONFLICTING" else "retry once GitHub has computed it"
        raise LandError(f"check 6: #{number} is {pr['mergeable']}, not MERGEABLE; {why}.")
    runs: dict[str, list[dict[str, Any]]] = {}
    for run in pr["statusCheckRollup"]:  # every run per name: a rerun does not hide a failure
        runs.setdefault(run.get("name") or run.get("context"), []).append(run)
    for name in REQUIRED_CHECKS:
        if name not in runs:
            raise LandError(
                f"check 6: required check {name!r} has not reported on {head[:12]}. Zero "
                f"checks visible means a conflict or a stalled Actions, not green (#630)."
            )
        for run in runs[name]:
            if not _green(run):
                seen = run.get("conclusion") or run.get("status") or run.get("state")
                raise LandError(
                    f"check 6: required check {name!r} is {seen} on {head[:12]}, not SUCCESS "
                    f"(every run of a required check must be). Pending is a refusal, not a wait."
                )
    return pr


def land(
    number: int,
    gh: Runner = run_gh,
    git: Runner = run_git,
    *,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    pr = check(number, gh, git)
    head = pr["headRefOid"]
    if dry_run:
        print(f"dry-run: #{number} passes checks 1-6 at {head[:12]}; not merging.")
        return 0
    gh(["pr", "merge", str(number), "--squash", "--match-head-commit", head])
    merge = json.loads(gh(["pr", "view", str(number), "--json", "mergeCommit"]))["mergeCommit"]
    if not merge:
        raise LandError(f"merged #{number} but gh reports no mergeCommit; confirm by hand.")
    sha = merge["oid"]
    status = ""
    for _ in range(COMPARE_TRIES):
        status = gh(["api", "repos/{owner}/{repo}/compare/main..." + sha, "--jq", ".status"]).strip()
        if status in ("behind", "identical"):
            break
        if status == "diverged":
            raise LandError(
                f"merged #{number} as {sha[:12]}, but compare/main...{sha[:12]} reads "
                f"'diverged': the squash commit is not on main (#858). Recover it by hand."
            )
        sleep(COMPARE_WAIT_S)
    else:
        raise LandError(
            f"compare/main...{sha[:12]} still reads {status!r} after {COMPARE_TRIES} tries; "
            f"confirm by hand before trusting the merge."
        )
    tip = git(["ls-remote", "origin", f"refs/heads/{pr['headRefName']}"]).split()
    if tip and tip[0] != head:
        raise LandError(
            f"merged #{number} as {sha[:12]}, but {pr['headRefName']} now reads {tip[0][:12]}, "
            f"not the merged head {head[:12]}: a push raced the merge and that commit is on "
            f"an orphaned branch, not on main. Open a new PR for it."
        )
    if not tip:
        print(f"{pr['headRefName']} is gone (auto-deleted); nothing raced the merge.")
    print(f"landed #{number} as {sha[:12]} on main (compare: {status}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("number", type=int, help="the PR number")
    parser.add_argument("--dry-run", action="store_true", help="run the checks, then stop before the merge")
    args = parser.parse_args(argv)
    try:
        return land(args.number, run_gh, run_git, dry_run=args.dry_run)
    except LandError as exc:
        print(f"land_pr: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
