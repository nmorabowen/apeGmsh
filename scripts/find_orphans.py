"""Flag merged PRs whose work never reached ``main``.

Two detections over PRs merged in the last ``--since`` days:

1. ``not-on-main``: the merge commit is not an ancestor of ``main``
   (``compare/main...<oid>`` reads ``diverged`` or ``ahead``).
2. ``pushed-after-merge``: the head branch still exists and its tip is not
   the PR's head sha. A branch deleted after merge is not flagged.

    python scripts/find_orphans.py --format md    # exit 1 when anything is found

All network access goes through ``fetch(path)``, one wrapper over ``gh api``
that returns parsed JSON, or ``None`` on HTTP 404. Tests replay recorded
responses through it. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

Fetch = Callable[[str], Any]

DEFAULT_REPO = "nmorabowen/apeGmsh"
DEFAULT_SINCE_DAYS = 90
PER_PAGE = 100
MAX_PAGES = 20


def gh_fetch(path: str) -> Any:
    """GET ``path`` through ``gh api``; ``None`` on a 404."""
    proc = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        if "404" in proc.stderr or "Not Found" in proc.stderr:
            return None
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def merged_prs(fetch: Fetch, repo: str, since: datetime) -> list[dict]:
    """PRs merged at or after ``since``, read newest-updated first, paged."""
    found: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        rows = fetch(
            f"repos/{repo}/pulls?state=closed&sort=updated&direction=desc"
            f"&per_page={PER_PAGE}&page={page}"
        )
        if not rows:
            break
        for pr in rows:
            merged_at = pr.get("merged_at")
            if merged_at and _parse_ts(merged_at) >= since:
                found.append(pr)
        # `updated` order: once a whole page is older than the window, stop.
        if all(_parse_ts(pr["updated_at"]) < since for pr in rows):
            break
        if len(rows) < PER_PAGE:
            break
    return found


def find_orphans(fetch: Fetch, repo: str, since: datetime) -> list[dict]:
    """Return one finding per detection hit."""
    findings: list[dict] = []
    for pr in merged_prs(fetch, repo, since):
        number = pr["number"]
        oid = pr.get("merge_commit_sha")
        base = pr["base"]["ref"]
        if oid:
            cmp = fetch(f"repos/{repo}/compare/main...{oid}")
            status = cmp["status"] if cmp else "missing"
            if status not in ("behind", "identical"):
                findings.append({
                    "kind": "not-on-main", "pr": number, "title": pr["title"],
                    "base": base, "sha": oid, "detail": f"compare reads {status}",
                })
        head = pr.get("head") or {}
        head_repo = (head.get("repo") or {}).get("full_name")
        if head_repo == repo and head.get("ref"):
            branch = fetch(f"repos/{repo}/branches/{head['ref']}")
            if branch is not None:
                tip = branch["commit"]["sha"]
                if tip != head["sha"]:
                    findings.append({
                        "kind": "pushed-after-merge", "pr": number,
                        "title": pr["title"], "base": base, "sha": tip,
                        "detail": (
                            f"branch {head['ref']} tip {tip[:8]} "
                            f"!= PR head {head['sha'][:8]}"
                        ),
                    })
    return findings


def render_md(findings: list[dict]) -> str:
    if not findings:
        return "No orphaned work found.\n"
    lines = ["| Kind | PR | Base | Detail |", "|---|---|---|---|"]
    for f in findings:
        lines.append(
            f"| {f['kind']} | #{f['pr']} {f['title']} | `{f['base']}` | {f['detail']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, fetch: Fetch | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=int, default=DEFAULT_SINCE_DAYS,
                    help="look back this many days (default %(default)s)")
    ap.add_argument("--format", choices=("json", "md"), default="md")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    args = ap.parse_args(argv)
    since = datetime.now(timezone.utc) - timedelta(days=args.since)
    findings = find_orphans(fetch or gh_fetch, args.repo, since)
    if args.format == "json":
        print(json.dumps(findings, indent=2))
    else:
        print(render_md(findings), end="")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
