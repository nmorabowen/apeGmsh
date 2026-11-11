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

A status belongs to the commit, not to the PR: every open PR whose head is that
commit is decided, and ``success`` is posted only when all of them pass. A
second PR on the same head that needs no verdict cannot overwrite the first
one's ``pending`` (#2879 review, finding 1).

    python scripts/check_review_verdict.py --pr 2879 --sha <40-hex head>
    python scripts/check_review_verdict.py --pr 2628 --dry-run
    python scripts/check_review_verdict.py --pr 2628 --dry-run --at-merge

``--sha`` pins the commit (the workflow takes it from the event), so a failed
read still posts ``error`` there and an earlier ``success`` cannot stand.
``--dry-run`` prints the status it would post and posts nothing; a dry run of a
closed PR, ``--head`` and ``--at-merge`` replay that PR alone. Exit 0 when the
status was decided and posted (or nothing was to be posted), 2 when a read, the
decision or the post failed.
Stdlib only; all network access goes through ``fetch`` and ``post``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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
#: One page of ``commits/{sha}/pulls``; a full page means PRs may be unread.
PULLS_PAGE = 100
_SHA = re.compile(r"[0-9a-f]{40}")


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


def open_prs_at(fetch: Fetch, repo: str, sha: str, pr: dict) -> list[dict]:
    """Every open PR whose head is ``sha``, ``pr`` included when it is one.

    ``pr`` (the full PR the event named) is added even if the commit's PR list
    lags behind it. Raises when the list cannot be read whole, so the caller
    posts ``error`` rather than decide over a partial set.
    """
    rows = fetch(f"repos/{repo}/commits/{sha}/pulls?per_page={PULLS_PAGE}")
    if rows is None:
        raise RuntimeError(f"commit {sha} not found in {repo}")
    if len(rows) >= PULLS_PAGE:
        raise RuntimeError(f"{len(rows)} PRs list commit {sha}: the page may be cut")
    found = {p["number"]: p for p in rows
             if p["state"] == "open" and p["head"]["sha"] == sha}
    if pr["state"] == "open" and pr["head"]["sha"] == sha:
        found[pr["number"]] = pr
    return [found[n] for n in sorted(found)]


def decide_head(fetch: Fetch, repo: str, sha: str, prs: list[dict]) -> dict:
    """One status for commit ``sha``: ``success`` only if every PR in ``prs`` passes.

    A commit status belongs to the commit and the latest one per context wins,
    so a PR that needs no verdict must not post ``success`` over another PR at
    the same head that is still waiting.
    """
    results = [(p["number"], decide(fetch, repo, p)) for p in prs]
    waiting = [(n, s) for n, s in results if s["state"] != "success"]
    if waiting:
        number, status = waiting[0]
        more = f" (+{len(waiting) - 1} more)" if len(waiting) > 1 else ""
        description = f"#{number}{more}: {status['description']}"
    elif len(results) == 1:
        status, description = results[0][1], results[0][1]["description"]
    else:
        status = results[0][1]
        description = ("Every open PR at this head passes: "
                       + ", ".join(f"#{n}" for n, _ in results) + ".")
    return {**status, "sha": sha, "description": description[:MAX_DESCRIPTION]}


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
    ap.add_argument("--sha", help="the head commit to decide and post on, pinned "
                    "from the event: a failed read then posts error there")
    args = ap.parse_args(argv)
    if (args.head or args.at_merge) and not args.dry_run:
        ap.error("--head and --at-merge replay the past and post nothing: pass --dry-run")
    if args.sha and (args.head or args.at_merge):
        ap.error("--sha pins the live head; --head and --at-merge replay a past one")
    if args.sha and not _SHA.fullmatch(args.sha):
        ap.error(f"--sha must be a 40-digit lowercase hex commit, not {args.sha!r}")
    fetch = fetch or gh_fetch
    post = post or gh_post
    sha: str | None = args.sha
    prs: list[dict] = []
    try:
        pr = read_pr(fetch, args.repo, args.pr, args.head)
        sha = sha or pr["head"]["sha"]
        if args.at_merge:
            fetch = as_of(fetch, pr)
        if args.head or args.at_merge or (args.dry_run and pr["state"] != "open"):
            prs = [pr]  # a replay: the PRs open at that head then are not recorded
            print(f"NOTE: #{args.pr} replayed alone at {sha[:12]}", file=sys.stderr)
        else:
            prs = open_prs_at(fetch, args.repo, sha, pr)
        status = decide_head(fetch, args.repo, sha, prs) if prs else None
    except Exception as exc:  # noqa: BLE001 - fail closed on the head, if known
        print(f"ERROR: deciding #{args.pr} failed: {exc}", file=sys.stderr)
        if sha is None:
            return 2  # no commit known, so there is nothing to post on
        # An earlier success on this head (say, before a `semantic` label) must
        # not stand when the re-evaluation could not finish.
        status = {"sha": sha, "state": "error", "context": CONTEXT,
                  "description": "The review-verdict check failed; see the run log."}
        code = 2
    else:
        code = 0
    if status is None:
        print(f"No open PR has head {sha}; nothing to post.")
        return 0
    print(json.dumps({"pr": args.pr, "prs": [p["number"] for p in prs], **status},
                     indent=2))
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
