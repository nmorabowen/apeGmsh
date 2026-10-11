"""Land one PR on main: the "How work lands" checklist (AGENTS.md) as code.

    python scripts/land_pr.py <number>            # check, squash-merge, prove it landed
    python scripts/land_pr.py <number> --dry-run  # run the checks and stop before the merge
    python scripts/land_pr.py <number> --verify-only  # only the post-merge proof, for a merged PR

The checks run in order and the first failure refuses, naming the check and
the lesson. Then ``gh pr merge --squash --match-head-commit <checked head>``
(never ``--auto``, never ``--delete-branch``), so a push between the checks
and the merge is refused by GitHub; then ``compare/main...<squash sha>`` must
read ``behind`` or ``identical``; then the branch tip must still be the
merged head, so a push that raced the merge is flagged rather than left on
an orphaned branch. A push after the landing is the nightly orphan
detector's job (.github/workflows/orphans.yml, #1230).

Every read after the merge is retried with bounded backoff (the merge call
never is); if one keeps failing the script exits non-zero with ``MERGED BUT
UNVERIFIED: #N`` and the ``--verify-only`` command to run (#1682).

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


class UnverifiedError(LandError):
    """The merge call went through but the post-merge proof could not be completed."""


#: Transient failures of a read: a non-zero gh/git exit, a missing binary, a cut-off body.
class _NotYet(Exception):
    """A read that came back empty right after the merge (eventual consistency)."""


_TRANSIENT = (subprocess.CalledProcessError, OSError, json.JSONDecodeError, _NotYet)
READ_TRIES = 5
READ_WAIT_S = 1.0  # doubles each retry: 1, 2, 4, 8 s


def _retry(fn: Callable[[], Any], what: str, sleep: Callable[[float], None]) -> Any:
    """Run a post-merge READ with bounded backoff (never the merge call itself:
    it is not idempotent). Exhaustion raises UnverifiedError (#1682)."""
    wait = READ_WAIT_S
    for attempt in range(1, READ_TRIES + 1):
        try:
            return fn()
        except _TRANSIENT as exc:
            if attempt == READ_TRIES:
                raise UnverifiedError(
                    f"{what} failed {READ_TRIES} times ({exc}); the merge itself went through."
                ) from exc
            sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")  # pragma: no cover


def _unverified(number: int, cause: UnverifiedError) -> UnverifiedError:
    return UnverifiedError(
        f"MERGED BUT UNVERIFIED: #{number} is merged, but the proof it reached main did "
        f"not complete: {cause} Run: python scripts/land_pr.py --verify-only {number}"
    )


def prove(
    number: int,
    gh: Runner,
    git: Runner,
    sleep: Callable[[float], None],
    head: str | None = None,
    branch: str | None = None,
) -> int:
    """The post-merge proof: compare/main...<squash> reads behind or identical, and
    the branch tip is still the merged head. Every read is retried; raises
    UnverifiedError (message starting MERGED BUT UNVERIFIED) if a read keeps failing."""
    try:
        if head is None or branch is None:
            view = _retry(
                lambda: json.loads(gh(["pr", "view", str(number), "--json", "state,headRefName,headRefOid"])),
                f"reading #{number}", sleep,
            )
            if view["state"] != "MERGED":
                raise LandError(f"#{number} is {view['state']}, not merged; --verify-only proves a merged PR.")
            head, branch = view["headRefOid"], view["headRefName"]

        def merge_oid() -> str:
            merge = json.loads(gh(["pr", "view", str(number), "--json", "mergeCommit"])).get("mergeCommit")
            if not merge or not merge.get("oid"):  # null right after the merge: retry
                raise _NotYet("gh reports no mergeCommit yet")
            return str(merge["oid"])

        sha = _retry(merge_oid, f"the mergeCommit lookup for #{number}", sleep)
        status = ""
        for _ in range(COMPARE_TRIES):
            status = _retry(
                lambda: gh(["api", "repos/{owner}/{repo}/compare/main..." + sha, "--jq", ".status"]).strip(),
                f"compare/main...{sha[:12]}", sleep,
            )
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
        tip = _retry(
            lambda: git(["ls-remote", "origin", f"refs/heads/{branch}"]).split(),
            f"git ls-remote for {branch}", sleep,
        )
    except UnverifiedError as exc:
        if str(exc).startswith("MERGED BUT UNVERIFIED"):
            raise
        raise _unverified(number, exc) from exc
    if tip and tip[0] != head:
        raise LandError(
            f"merged #{number} as {sha[:12]}, but {branch} now reads {tip[0][:12]}, "
            f"not the merged head {head[:12]}: a push raced the merge and that commit is on "
            f"an orphaned branch, not on main. Open a new PR for it."
        )
    if not tip:
        print(f"{branch} is gone (auto-deleted); nothing raced the merge.")
    print(f"landed #{number} as {sha[:12]} on main (compare: {status}).")
    return 0


def land(
    number: int,
    gh: Runner = run_gh,
    git: Runner = run_git,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    if verify_only:
        return prove(number, gh, git, sleep)
    pr = check(number, gh, git)
    head = pr["headRefOid"]
    if dry_run:
        print(f"dry-run: #{number} passes checks 1-6 at {head[:12]}; not merging.")
        return 0
    gh(["pr", "merge", str(number), "--squash", "--match-head-commit", head])  # never retried
    return prove(number, gh, git, sleep, head, pr["headRefName"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("number", type=int, help="the PR number")
    parser.add_argument("--dry-run", action="store_true", help="run the checks, then stop before the merge")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="skip the checks and the merge; run only the post-merge proof for an already-merged PR",
    )
    args = parser.parse_args(argv)
    if args.verify_only and args.dry_run:
        parser.error("--verify-only and --dry-run are mutually exclusive")  # exit 2
    try:
        return land(args.number, run_gh, run_git, dry_run=args.dry_run, verify_only=args.verify_only, sleep=time.sleep)
    except UnverifiedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except LandError as exc:
        print(f"land_pr: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
