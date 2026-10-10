"""Report stalled program PRs and lock-label conflicts (lessons #1567, #1544).

Four findings over open ``program`` PRs and open issues:

1. ``approved-not-landed``: approved at the head sha (a
   ``Review-verdict: <model> approve <sha>`` comment whose sha is the head),
   every required check green, and not landed for more than ``--approved-hours``
   since the later of the approval and the last required check to finish.
2. ``no-verdict``: a non-draft PR open more than ``--verdict-hours`` with no
   ``Review-verdict`` comment at all.
3. ``stale-lock``: an issue carrying a ``lock:<file>`` label whose linked
   open program PR (``Slice: #N`` in the body, or a closing reference) has had
   no activity for more than ``--lock-hours``; with no linked PR, the issue's
   own last activity is used.
4. ``double-lock``: one ``lock:<file>`` label on more than one open issue.

    python scripts/program_probe.py            # report, exit 0
    python scripts/program_probe.py --issue    # open/update/close the tracking issue

All GitHub reads go through ``gh`` (JSON). ``--from-json FILE`` reads
``{"prs": [...], "issues": [...]}`` in the same shape instead, and ``--now``
pins the clock, so the probe runs offline. Stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

Runner = Callable[[list[str]], str]

DEFAULT_REPO = "nmorabowen/apeGmsh"
ISSUE_TITLE = "[Program probe] stalled program PRs and lock conflicts"
PR_FIELDS = (
    "number,title,createdAt,updatedAt,isDraft,headRefOid,body,comments,"
    "statusCheckRollup,closingIssuesReferences"
)
ISSUE_FIELDS = "number,title,labels,updatedAt"
LIMIT = "200"

_VERDICT = re.compile(
    r"^Review-verdict:\s*(?P<model>.+?)\s+(?P<verdict>approve|changes)\s+"
    r"(?P<sha>[0-9a-f]{7,40})\b",
    re.MULTILINE,
)
_SLICE = re.compile(r"Slice:\s*#(\d+)")

# The required-check list and the green test live in land_pr.py; one copy.
_spec = importlib.util.spec_from_file_location(
    "land_pr", Path(__file__).resolve().parent / "land_pr.py"
)
_land_pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_land_pr)
REQUIRED_CHECKS: tuple[str, ...] = _land_pr.REQUIRED_CHECKS
_green = _land_pr._green


def run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hours(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 3600.0


def fetch(gh: Runner, repo: str) -> dict[str, list[dict]]:
    prs = json.loads(gh(["pr", "list", "--repo", repo, "--state", "open",
                         "--label", "program", "--limit", LIMIT, "--json", PR_FIELDS]))
    issues = json.loads(gh(["issue", "list", "--repo", repo, "--state", "open",
                            "--limit", LIMIT, "--json", ISSUE_FIELDS]))
    return {"prs": prs, "issues": issues}


def verdicts(pr: dict) -> list[dict]:
    """Every ``Review-verdict`` line on the PR, oldest comment first."""
    found = []
    for c in sorted(pr.get("comments") or [], key=lambda c: c["createdAt"]):
        for m in _VERDICT.finditer(c.get("body") or ""):
            found.append({"model": m["model"], "verdict": m["verdict"],
                          "sha": m["sha"], "at": _ts(c["createdAt"])})
    return found


def approval_at_head(pr: dict) -> datetime | None:
    """Time of the approval at the head sha, or ``None``.

    Only verdicts naming the head sha count; the latest of those decides, so a
    ``changes`` after an ``approve`` at the same sha withdraws it.
    """
    head = pr["headRefOid"]
    at_head = [v for v in verdicts(pr) if head.startswith(v["sha"])]
    if not at_head or at_head[-1]["verdict"] != "approve":
        return None
    return at_head[-1]["at"]


def last_green(pr: dict) -> datetime | None:
    """When the last required check finished, if every run of each is green."""
    runs: dict[str, list[dict]] = {}
    for run in pr.get("statusCheckRollup") or []:
        runs.setdefault(run.get("name") or run.get("context"), []).append(run)
    latest: datetime | None = None
    for name in REQUIRED_CHECKS:
        if name not in runs or not all(_green(r) for r in runs[name]):
            return None
        for r in runs[name]:
            stamp = r.get("completedAt") or r.get("startedAt")
            if stamp and not stamp.startswith("0001"):
                t = _ts(stamp)
                latest = t if latest is None or t > latest else latest
    return latest


def _linked_issues(pr: dict) -> set[int]:
    nums = {int(n) for n in _SLICE.findall(pr.get("body") or "")}
    nums |= {ref["number"] for ref in pr.get("closingIssuesReferences") or []}
    return nums


def probe(data: dict, now: datetime, *, approved_hours: float = 4.0,
          verdict_hours: float = 6.0, lock_hours: float = 6.0) -> list[dict]:
    findings: list[dict] = []
    prs = data["prs"]
    for pr in prs:
        approved = approval_at_head(pr)
        green = last_green(pr)
        if approved is not None and green is not None:
            since = max(approved, green)
            age = _hours(now, since)
            if age > approved_hours:
                findings.append({
                    "kind": "approved-not-landed", "ref": f"#{pr['number']}",
                    "title": pr["title"],
                    "detail": f"approved at head {pr['headRefOid'][:8]} and green; "
                              f"unlanded {age:.1f} h",
                })
        if not pr.get("isDraft") and not verdicts(pr):
            age = _hours(now, _ts(pr["createdAt"]))
            if age > verdict_hours:
                findings.append({
                    "kind": "no-verdict", "ref": f"#{pr['number']}", "title": pr["title"],
                    "detail": f"open {age:.1f} h with no Review-verdict",
                })

    holders: dict[str, list[dict]] = {}
    for issue in data["issues"]:
        for label in issue.get("labels") or []:
            if label["name"].startswith("lock:"):
                holders.setdefault(label["name"], []).append(issue)
    for label in sorted(holders):
        issues = holders[label]
        for issue in issues:
            linked = [p for p in prs if issue["number"] in _linked_issues(p)]
            if linked:
                pr = max(linked, key=lambda p: p["updatedAt"])
                last, where = _ts(pr["updatedAt"]), f"PR #{pr['number']}"
            else:
                last, where = _ts(issue["updatedAt"]), "the issue (no linked PR)"
            age = _hours(now, last)
            if age > lock_hours:
                findings.append({
                    "kind": "stale-lock", "ref": f"#{issue['number']}",
                    "title": issue["title"],
                    "detail": f"`{label}`: no activity on {where} for {age:.1f} h",
                })
        if len(issues) > 1:
            refs = ", ".join(f"#{i['number']}" for i in sorted(issues, key=lambda i: i["number"]))
            findings.append({
                "kind": "double-lock", "ref": refs, "title": label,
                "detail": f"`{label}` is on {len(issues)} open issues",
            })
    return findings


def render_md(findings: list[dict]) -> str:
    if not findings:
        return "No stalled program PRs or lock conflicts found.\n"
    lines = ["| Kind | Ref | Title | Detail |", "|---|---|---|---|"]
    for f in findings:
        lines.append(f"| {f['kind']} | {f['ref']} | {f['title']} | {f['detail']} |")
    return "\n".join(lines) + "\n"


def sync_issue(gh: Runner, repo: str, findings: list[dict], now: datetime) -> str:
    """Open or update the one tracking issue; close it when clean."""
    rows = json.loads(gh(["issue", "list", "--repo", repo, "--state", "open",
                          "--label", "program", "--search", f'"{ISSUE_TITLE}" in:title',
                          "--json", "number,title"]))
    existing = next((r["number"] for r in rows if r["title"] == ISSUE_TITLE), None)
    body = (f"Program probe run {now:%Y-%m-%d %H:%M} UTC "
            f"(`scripts/program_probe.py`, lessons #1567 and #1544).\n\n"
            + render_md(findings))
    if findings:
        if existing is None:
            gh(["issue", "create", "--repo", repo, "--title", ISSUE_TITLE,
                "--label", "program", "--body", body])
            return "created"
        gh(["issue", "edit", str(existing), "--repo", repo, "--body", body])
        return f"updated #{existing}"
    if existing is not None:
        gh(["issue", "close", str(existing), "--repo", repo,
            "--comment", f"Clean on {now:%Y-%m-%d %H:%M} UTC: no findings."])
        return f"closed #{existing}"
    return "nothing to do"


def main(argv: list[str] | None = None, gh: Runner = run_gh) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    ap.add_argument("--approved-hours", type=float, default=4.0)
    ap.add_argument("--verdict-hours", type=float, default=6.0)
    ap.add_argument("--lock-hours", type=float, default=6.0)
    ap.add_argument("--now", type=_ts, default=None,
                    help="ISO timestamp to use as the current time (tests)")
    ap.add_argument("--from-json", type=Path, default=None,
                    help='read {"prs": [...], "issues": [...]} instead of GitHub')
    ap.add_argument("--format", choices=("json", "md"), default="md")
    ap.add_argument("--issue", action="store_true",
                    help="open/update/close the tracking issue")
    args = ap.parse_args(argv)
    now = args.now or datetime.now(timezone.utc)
    if args.from_json is not None:
        data = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        data = fetch(gh, args.repo)
    findings = probe(data, now, approved_hours=args.approved_hours,
                     verdict_hours=args.verdict_hours, lock_hours=args.lock_hours)
    if args.format == "json":
        print(json.dumps(findings, indent=2))
    else:
        print(render_md(findings), end="")
    if args.issue:
        print(f"tracking issue: {sync_issue(gh, args.repo, findings, now)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
