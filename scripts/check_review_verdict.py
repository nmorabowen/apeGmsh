"""Post the ``review-verdict`` commit status on a PR's head (lesson #2637).

A web merge never runs ``land_pr``, and ``find_orphans.py`` reports a merge
without an approving verdict only afterwards. This script puts the same policy
on the PR head as a commit status, so a required ``review-verdict`` check makes
the merge button itself refuse:

- ``success`` when the T review policy needs no verdict for the PR;
- ``success`` when a ``Review-verdict: <model> approve <sha>`` comment names the
  current head and no later verdict at that head says ``changes``;
- ``pending`` otherwise, naming the head it waits for. A push moves the head,
  so an approval of an earlier head never carries over (#2628).

One policy, one implementation: the decision is ``find_orphans.verdict_required``
and ``find_orphans.has_head_approval`` (which uses ``program_probe``'s verdict
parser), loaded read-only, so the detector and this check cannot disagree.

    python scripts/check_review_verdict.py --pr 2628 --dry-run
    python scripts/check_review_verdict.py --pr 2628 --dry-run --at-merge

``--dry-run`` prints the status it would post and posts nothing. Exit 0 when
the status was decided (and posted), 2 when reading the PR, deciding or posting
failed. A failed decision posts ``error`` on the head it read, so an earlier
``success`` on that head cannot stand.
Stdlib only; all network access goes through ``fetch`` and ``post``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# The policy and the verdict matcher live in find_orphans.py (which loads
# program_probe.py); one copy, never re-derived here.
_spec = importlib.util.spec_from_file_location(
    "find_orphans", Path(__file__).resolve().parent / "find_orphans.py"
)
_orphans = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orphans)
verdict_required = _orphans.verdict_required
has_head_approval = _orphans.has_head_approval
gh_fetch = _orphans.gh_fetch
DEFAULT_REPO: str = _orphans.DEFAULT_REPO

Fetch = Callable[[str], Any]
Post = Callable[[str, dict], Any]

CONTEXT = "review-verdict"
#: GitHub truncates nothing; it rejects a description over 140 characters.
MAX_DESCRIPTION = 140


def read_pr(fetch: Fetch, repo: str, number: int, head: str | None = None) -> dict:
    """PR ``number`` as the REST API returns it, its head replaced by ``head``."""
    pr = fetch(f"repos/{repo}/pulls/{number}")
    if pr is None:
        raise RuntimeError(f"PR #{number} not found in {repo}")
    if head is not None:
        pr = {**pr, "head": {**pr["head"], "sha": head}}
    return pr


def decide(fetch: Fetch, repo: str, pr: dict) -> dict:
    """The ``review-verdict`` status for ``pr`` at ``pr["head"]["sha"]``."""
    sha = pr["head"]["sha"]
    if not verdict_required(fetch, repo, pr):
        state, description = "success", "Review verdict not required for this PR."
    elif has_head_approval(fetch, repo, pr):
        state, description = "success", f"Approved by a Review-verdict at head {sha[:12]}."
    else:
        state = "pending"
        description = (f"Waiting for 'Review-verdict: <model> approve {sha[:12]}' "
                       "at the current head.")
    return {"sha": sha, "state": state, "context": CONTEXT,
            "description": description[:MAX_DESCRIPTION]}


def as_of(fetch: Fetch, pr: dict) -> Fetch:
    """``fetch`` with the PR's comments cut at its merge instant (a replay).

    A comment posted after the merge did not exist when the merge button was
    pressed (#2628's approval of e2c3099a came four minutes after its merge).
    An edit made after the merge cannot be undone from the API; it is reported.
    """
    merged_at = pr.get("merged_at")
    if not merged_at:
        raise RuntimeError(f"PR #{pr['number']} is not merged: nothing to replay at merge")
    cut = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    comments = f"/issues/{pr['number']}/comments?"

    def replay(path: str) -> Any:
        rows = fetch(path)
        if comments not in path or not rows:
            return rows
        kept = []
        for c in rows:
            if datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")) > cut:
                continue
            updated = c.get("updated_at")
            if updated and datetime.fromisoformat(updated.replace("Z", "+00:00")) > cut:
                print(f"WARNING: comment {c.get('id')} was edited after the merge; "
                      "its current body is replayed", file=sys.stderr)
            kept.append(c)
        return kept
    return replay


def gh_post(path: str, fields: dict) -> Any:
    """POST ``fields`` to ``path`` through ``gh api``."""
    args = ["gh", "api", "-X", "POST", path]
    for key, value in fields.items():
        args += ["-f", f"{key}={value}"]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api POST {path} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout or "null")


def post_status(post: Post, repo: str, status: dict, target_url: str | None) -> None:
    fields = {k: status[k] for k in ("state", "context", "description")}
    if target_url:
        fields["target_url"] = target_url
    post(f"repos/{repo}/statuses/{status['sha']}", fields)


def _run_url() -> str | None:
    server, repo, run = (os.environ.get(k) for k in
                         ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None


def main(argv: list[str] | None = None, fetch: Fetch | None = None,
         post: Post | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pr", type=int, required=True, help="pull request number")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the status it would post; post nothing")
    ap.add_argument("--head", help="decide at this head sha instead of the PR's "
                    "current one (replay; needs --dry-run)")
    ap.add_argument("--at-merge", action="store_true",
                    help="replay a merged PR as its merge button saw it: only "
                    "comments posted by the merge count (needs --dry-run)")
    args = ap.parse_args(argv)
    if (args.head or args.at_merge) and not args.dry_run:
        ap.error("--head and --at-merge replay the past and post nothing: pass --dry-run")
    fetch = fetch or gh_fetch
    post = post or gh_post
    try:
        pr = read_pr(fetch, args.repo, args.pr, args.head)
        if args.at_merge:
            fetch = as_of(fetch, pr)
    except Exception as exc:  # noqa: BLE001 - no head known, so nothing to post
        print(f"ERROR: reading #{args.pr} failed: {exc}", file=sys.stderr)
        return 2
    try:
        status = decide(fetch, args.repo, pr)
    except Exception as exc:  # noqa: BLE001 - fail closed on the known head
        print(f"ERROR: deciding #{args.pr} failed: {exc}", file=sys.stderr)
        # An earlier success on this head (say, before a `semantic` label) must
        # not stand when the re-evaluation could not finish.
        status = {"sha": pr["head"]["sha"], "state": "error", "context": CONTEXT,
                  "description": "The review-verdict check failed; see the run log."}
        code = 2
    else:
        code = 0
    print(json.dumps({"pr": args.pr, **status}, indent=2))
    if args.dry_run:
        return code
    try:
        post_status(post, args.repo, status, _run_url())
    except Exception as exc:  # noqa: BLE001 - reported; the job fails
        print(f"ERROR: posting the status failed: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
