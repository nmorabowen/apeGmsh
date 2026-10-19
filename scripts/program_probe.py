"""Report stalled program PRs and lock-label conflicts (lessons #1567, #1544).

Seven findings over open ``program`` PRs, open issues and scheduled runs:

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
5. ``red-lane``: a watched scheduled workflow (``WATCHED_WORKFLOWS``) whose last
   two completed ``schedule`` runs on ``main`` both concluded ``failure``,
   ``timed_out`` or ``startup_failure``. ``cancelled`` and ``skipped`` do not
   count: a cancel is a concurrency or manual act, not the lane being broken.
   Fewer than two completed runs is not a finding.
6. ``unlabelled-slice``: an open PR whose body has ``Slice: #<n>`` but no
   ``program`` label, so it escapes the program-wide WIP count (#2012).
7. ``wip-over-cap``: more than ``WIP_CAP`` (8) counted open program PRs now, or a
   reconstructed peak above it in the last 24 h (lesson #2200). ``wip_count``
   is shared with ``scripts/open_program_pr.py``: a PR labelled ``human-gate``
   that is green and approved waits only on the maintainer's click and counts
   nothing (ruling on #2200, 2026-10-19). The 24 h peak cannot replay that
   exclusion, so it drops every PR that carries ``human-gate`` now, and the
   finding says it is approximate.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

Runner = Callable[[list[str]], str]

DEFAULT_REPO = "nmorabowen/apeGmsh"
ISSUE_TITLE = "[Program probe] stalled program PRs and lock conflicts"
PR_FIELDS = (
    "number,title,createdAt,updatedAt,isDraft,headRefOid,body,comments,"
    "statusCheckRollup,closingIssuesReferences,labels"
)
HISTORY_FIELDS = "number,title,createdAt,closedAt,mergedAt,labels"
WIP_CAP = 8
HUMAN_GATE = "human-gate"
ISSUE_FIELDS = "number,title,labels,updatedAt"
ALL_PR_FIELDS = "number,title,body,labels"
LIMIT = "200"

# Scheduled lanes nobody watches (Benchmarks sat red 10-15..10-18, #2144).
WATCHED_WORKFLOWS: tuple[str, ...] = (
    "live-fork.yml", "benchmarks.yml", "deps-canary.yml", "orphans.yml",
)
RED_CONCLUSIONS = frozenset({"failure", "timed_out", "startup_failure"})
RUN_FIELDS = "conclusion,status,createdAt,url,headSha"

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
    """Parse an ISO timestamp; a value with no offset is taken as UTC."""
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _hours(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 3600.0


def fetch(gh: Runner, repo: str) -> dict[str, list[dict]]:
    prs = json.loads(gh(["pr", "list", "--repo", repo, "--state", "open",
                         "--label", "program", "--limit", LIMIT, "--json", PR_FIELDS]))
    issues = json.loads(gh(["issue", "list", "--repo", repo, "--state", "open",
                            "--limit", LIMIT, "--json", ISSUE_FIELDS]))
    open_prs = json.loads(gh(["pr", "list", "--repo", repo, "--state", "open",
                              "--limit", LIMIT, "--json", ALL_PR_FIELDS]))
    runs = {
        wf: json.loads(gh(["run", "list", "--repo", repo, "--workflow", wf,
                           "--branch", "main", "--event", "schedule",
                           "--limit", "2", "--json", RUN_FIELDS]))
        for wf in WATCHED_WORKFLOWS
    }
    history = json.loads(gh(["pr", "list", "--repo", repo, "--state", "all",
                             "--label", "program", "--limit", LIMIT,
                             "--json", HISTORY_FIELDS]))
    return {"prs": prs, "issues": issues, "open_prs": open_prs, "runs": runs,
            "history": history}


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


def _labels(pr: dict) -> set[str]:
    return {label["name"] for label in pr.get("labels") or []}


def wip_count(prs: list[dict]) -> tuple[list[tuple[dict, str]], list[tuple[dict, str]]]:
    """Split open ``program`` PRs into (counted, excluded), each with a reason.

    Excluded: labelled ``human-gate`` and green and approved (``approval_at_head``,
    a Review-verdict approve on the head sha; GitHub's reviewDecision is not
    tied to the head and main does not dismiss stale reviews). Every other PR counts, drafts too.
    """
    counted: list[tuple[dict, str]] = []
    excluded: list[tuple[dict, str]] = []
    for pr in prs:
        if HUMAN_GATE not in _labels(pr):
            counted.append((pr, "counts"))
        elif last_green(pr) is None:
            counted.append((pr, "human-gate but not green: counts"))
        elif approval_at_head(pr) is None:
            counted.append((pr, "human-gate, green but not approved: counts"))
        else:
            excluded.append((pr, "human-gate, approved and green: waits on the maintainer"))
    return counted, excluded


def wip_peak(history: list[dict], now: datetime, *, hours: float = 24.0) -> tuple[int, datetime]:
    """Approximate peak of open program PRs over the last ``hours`` hours.

    Replays createdAt / closedAt / mergedAt; PRs carrying ``human-gate`` now are
    dropped, because the green-and-approved test cannot be replayed.
    """
    start = now - timedelta(hours=hours)
    spans = []
    for pr in history:
        if HUMAN_GATE in _labels(pr):
            continue
        end = pr.get("closedAt") or pr.get("mergedAt")
        spans.append((_ts(pr["createdAt"]), _ts(end) if end else None))
    moments = [start] + [c for c, _ in spans if start < c <= now]
    best = (0, start)
    for t in moments:
        n = sum(1 for c, e in spans if c <= t and (e is None or e > t))
        if n > best[0]:
            best = (n, t)
    return best


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

    for wf in sorted(data.get("runs") or {}):
        done = [r for r in data["runs"][wf] if r.get("status") == "completed"]
        done.sort(key=lambda r: r["createdAt"], reverse=True)
        last2 = done[:2]
        if len(last2) == 2 and all(r.get("conclusion") in RED_CONCLUSIONS for r in last2):
            detail = "; ".join(
                f"{r['conclusion']} {_ts(r['createdAt']):%Y-%m-%d} ({r['url']})"
                for r in last2)
            findings.append({
                "kind": "red-lane", "ref": wf, "title": f"{wf} red two scheduled runs",
                "detail": detail,
            })

    for pr in data.get("open_prs") or []:
        names = {label["name"] for label in pr.get("labels") or []}
        if "program" not in names and _SLICE.search(pr.get("body") or ""):
            findings.append({
                "kind": "unlabelled-slice", "ref": f"#{pr['number']}",
                "title": pr["title"],
                "detail": "body has `Slice: #<n>` but no `program` label",
            })

    counted, excluded = wip_count(prs)
    if len(counted) > WIP_CAP:
        refs = ", ".join("#" + str(p["number"]) for p, _ in counted)
        findings.append({
            "kind": "wip-over-cap", "ref": "program",
            "title": f"{len(counted)} counted open program PRs (cap {WIP_CAP})",
            "detail": f"counted {refs}; excluded {len(excluded)} human-gate approved-and-green",
        })
    if data.get("history"):
        peak, when = wip_peak(data["history"], now)
        if peak > WIP_CAP:
            findings.append({
                "kind": "wip-over-cap", "ref": "program-24h",
                "title": f"peak {peak} open program PRs in the last 24 h (cap {WIP_CAP})",
                "detail": f"peak at {when:%Y-%m-%d %H:%M} UTC; approximate: PRs "
                          f"labelled human-gate now are dropped, so the exclusion "
                          f"is not replayed",
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
    """Open or update the one tracking issue; close it when clean.

    More than one open exact-title match (two overlapping runs that each
    created one) is collapsed: the lowest number is kept, the rest are closed.
    """
    rows = json.loads(gh(["issue", "list", "--repo", repo, "--state", "open",
                          "--label", "program", "--search", f'"{ISSUE_TITLE}" in:title',
                          "--json", "number,title"]))
    matches = sorted(r["number"] for r in rows if r["title"] == ISSUE_TITLE)
    body = (f"Program probe run {now:%Y-%m-%d %H:%M} UTC "
            f"(`scripts/program_probe.py`, lessons #1567 and #1544).\n\n"
            + render_md(findings))
    if findings:
        keep, extras = (matches[0], matches[1:]) if matches else (None, [])
        for n in extras:
            gh(["issue", "close", str(n), "--repo", repo,
                "--comment", f"Duplicate of #{keep}; the probe tracks findings there."])
        dup = f"; closed duplicates {', '.join(f'#{n}' for n in extras)}" if extras else ""
        if keep is None:
            gh(["issue", "create", "--repo", repo, "--title", ISSUE_TITLE,
                "--label", "program", "--body", body])
            return "created"
        gh(["issue", "edit", str(keep), "--repo", repo, "--body", body])
        return f"updated #{keep}{dup}"
    for n in matches:
        gh(["issue", "close", str(n), "--repo", repo,
            "--comment", f"Clean on {now:%Y-%m-%d %H:%M} UTC: no findings."])
    if matches:
        return "closed " + ", ".join(f"#{n}" for n in matches)
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
    if args.issue and (args.from_json is not None or args.now is not None):
        # The offline mode must never write to the live tracking issue.
        ap.error("--issue cannot be combined with --from-json or --now")
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
