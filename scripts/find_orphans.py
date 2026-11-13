"""Flag merged PRs whose work never reached ``main``.

Three detections over PRs merged in the last ``--since`` days:

1. ``not-on-main``: the merge commit is not an ancestor of ``main``
   (``compare/main...<oid>`` reads ``diverged`` or ``ahead``).
2. ``pushed-after-merge``: the head branch still exists and its tip is not
   the PR's head sha. A branch deleted after merge is not flagged.
3. ``merged-without-verdict`` (lesson #2637, reported as its own section): a
   ``program`` PR that the T review policy required a verdict for (``semantic``
   label, or a change under ``tests/`` / ``.github/workflows/``, or to
   ``land_pr.py`` / ``check_quirks.py``, a BASELINE or a golden file) was
   merged with no ``Review-verdict: <model> approve <sha>`` comment bound to
   the merged head. A non-``semantic`` PR whose body carries ``verify_move``
   evidence is exempt. Only the last ``VERDICT_WINDOW_DAYS`` days are scanned.
   A web merge skips ``land_pr``, and this detector cannot block one.

    python scripts/find_orphans.py --format md    # exit 1 when anything is found,
                                                  # exit 2 when a detection failed

All network access goes through ``fetch(path)``, one wrapper over ``gh api``
that returns parsed JSON, or ``None`` on HTTP 404. Tests replay recorded
responses through it. Stdlib only.
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
from typing import Any, Callable

# The verdict format and the "approval at the head sha" rule live in
# program_probe.py (one parser); the BASELINE list lives in land_pr.py.
_spec = importlib.util.spec_from_file_location(
    "program_probe", Path(__file__).resolve().parent / "program_probe.py"
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
BASELINE: tuple[str, ...] = _probe._land_pr.BASELINE

Fetch = Callable[[str], Any]

DEFAULT_REPO = "nmorabowen/apeGmsh"
DEFAULT_SINCE_DAYS = 90
PER_PAGE = 100
MAX_PAGES = 20

#: The T review policy was posted at this instant (lesson #2637); a PR merged
#: earlier is not held to it. The #2627 landing is exempt by that record.
#: The verdict scan covers a rolling window, so its call count stops growing.
VERDICT_WINDOW_DAYS = 14
#: A move PR's body quotes `verify_move`'s success line ("OK, 59 defs, identical
#: multiset"); a non-semantic PR carrying that line is exempt (T ruling on #2679),
#: unless it also changes a file that always needs a verdict (`_always_verdict`).
_MOVE_EVIDENCE = re.compile(
    r"^[^\n]*\bOK\b\W{1,6}\d+ defs?, (?:identical multiset|\d+ moved)", re.MULTILINE)
#: Prose paths never count as golden files.
_PROSE_PREFIXES = ("docs/", "internal_docs/")
POLICY_START = datetime(2026, 10, 30, 19, 37, tzinfo=timezone.utc)
VERDICT_KIND = "merged-without-verdict"
VERDICT_NOTE = (
    "This section is a detector: it reports after the merge and cannot block one."
)
#: Paths whose change makes the T review policy require a verdict (besides a
#: `semantic` label). `tests/` and the workflows are prefixes.
POLICY_PREFIXES = ("tests/", ".github/workflows/")
POLICY_FILES = frozenset(
    {"scripts/land_pr.py", "scripts/check_quirks.py", *BASELINE}
)


# Rows verified redone on ``main`` (issue #1237, maintainer ruling 2026-10-11).
# Key: (orphan PR number, finding kind) -> (redoing PR(s), one-line reason).
# A finding with a matching key is dropped from the report; any other
# orphan, or the same PR under a different kind, still reports.
ACKNOWLEDGED: dict[tuple[int, str], tuple[str, str]] = {
    (858, "not-on-main"): (
        "#1169",
        "merged into a stacked base after its parent squashed; LogStrain2D "
        "and geom= recovered by #1169",
    ),
    (1057, "not-on-main"): (
        "#1058",
        "merged into a stacked base; von_mises_shell thickness-per-element "
        "carried to main by #1058 (merge 125cdd36)",
    ),
    (1097, "pushed-after-merge"): (
        "#1099",
        "post-merge commit a7237433 (-maxSubsteps, convergence-test gate, "
        "ADR 0103) landed as #1099",
    ),
    (2628, "pushed-after-merge"): (
        "#2636",
        "orphaned commit 4e847b8a re-applied in #2636 (be0dc12f4cba); "
        "lesson #2637",
    ),
    (2682, "merged-without-verdict"): (
        "#2872",
        "ADR amendment the maintainer merged from the web under the "
        "standing ADR rule; owner ruling recorded on #2872 (2026-11-13)",
    ),
    (2656, "merged-without-verdict"): (
        "#2872",
        "VA0p, landed 2026-11-01; owner ruling recorded on #2872 "
        "(2026-11-13)",
    ),
}


def gh_fetch(path: str) -> Any:
    """GET ``path`` through ``gh api``; ``None`` on a 404."""
    proc = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        if "(HTTP 404)" in proc.stderr:
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


def _paged(fetch: Fetch, path: str) -> list[dict]:
    """All pages of a list endpoint ``path`` (which has no query string)."""
    rows: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        chunk = fetch(f"{path}?per_page={PER_PAGE}&page={page}")
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < PER_PAGE:
            break
    return rows


def _always_verdict(name: str) -> bool:
    """Paths no move evidence can exempt: workflows, gate scripts, golden files."""
    return (name.startswith(".github/workflows/") or name in POLICY_FILES
            or _is_golden(name))


def _is_golden(name: str) -> bool:
    return "golden" in name.lower() and not name.startswith(_PROSE_PREFIXES)


def _needs_verdict_path(name: str) -> bool:
    return (name.startswith(POLICY_PREFIXES) or name in POLICY_FILES
            or _is_golden(name))


def verdict_required(fetch: Fetch, repo: str, pr: dict) -> bool:
    """Whether the T review policy required a verdict for this merged PR."""
    labels = {label["name"] for label in pr.get("labels") or []}
    if "program" not in labels:
        return False
    if "semantic" in labels:
        return True
    exempt = bool(_MOVE_EVIDENCE.search(pr.get("body") or ""))
    rows = _paged(fetch, f"repos/{repo}/pulls/{pr['number']}/files")
    if len(rows) >= PER_PAGE * MAX_PAGES:
        return True  # the listing hit the page cap: an unread path may need one
    for row in rows:
        names = [n for n in (row.get("filename"), row.get("previous_filename")) if n]
        if any(_always_verdict(n) for n in names):
            return True
        if not exempt and any(_needs_verdict_path(n) for n in names):
            return True
    return False


def has_head_approval(fetch: Fetch, repo: str, pr: dict) -> bool:
    """An approve bound to the merged head; the latest verdict at it decides."""
    comments = [
        {"body": c.get("body"), "createdAt": c["created_at"],
         "authorAssociation": c.get("author_association")}
        for c in _paged(fetch, f"repos/{repo}/issues/{pr['number']}/comments")
    ]
    probe_pr = {"headRefOid": pr["head"]["sha"], "comments": comments}
    return _probe.approval_at_head(probe_pr) is not None


def find_unreviewed(fetch: Fetch, repo: str, since: datetime,
                    now: datetime | None = None) -> list[dict]:
    """``merged-without-verdict`` findings (detection 3)."""
    now = now or datetime.now(timezone.utc)
    since = max(since, POLICY_START, now - timedelta(days=VERDICT_WINDOW_DAYS))
    findings: list[dict] = []
    for pr in merged_prs(fetch, repo, since):
        if not verdict_required(fetch, repo, pr):
            continue
        if has_head_approval(fetch, repo, pr):
            continue
        sha = pr["head"]["sha"]
        findings.append({
            "kind": VERDICT_KIND, "pr": pr["number"], "title": pr["title"],
            "base": pr["base"]["ref"], "sha": sha,
            "detail": f"no Review-verdict approve at merged head {sha[:8]}",
        })
    return findings


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
    return [f for f in findings if (f["pr"], f["kind"]) not in ACKNOWLEDGED]


def _table(findings: list[dict]) -> list[str]:
    lines = ["| Kind | PR | Base | Detail |", "|---|---|---|---|"]
    for f in findings:
        lines.append(
            f"| {f['kind']} | #{f['pr']} {f['title']} | `{f['base']}` | {f['detail']} |"
        )
    return lines


def render_md(findings: list[dict]) -> str:
    if not findings:
        return "No orphaned work found.\n"
    orphans = [f for f in findings if f["kind"] != VERDICT_KIND]
    unreviewed = [f for f in findings if f["kind"] == VERDICT_KIND]
    out: list[str] = []
    if orphans:
        out += _table(orphans) + [""]
    if unreviewed:
        out += ["## Merged without an approving Review-verdict (lesson #2637)", "",
                VERDICT_NOTE, ""] + _table(unreviewed) + [""]
    return "\n".join(out).rstrip("\n") + "\n"


def main(argv: list[str] | None = None, fetch: Fetch | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=int, default=DEFAULT_SINCE_DAYS,
                    help="look back this many days (default %(default)s)")
    ap.add_argument("--format", choices=("json", "md"), default="md")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    args = ap.parse_args(argv)
    since = datetime.now(timezone.utc) - timedelta(days=args.since)
    fetch = fetch or gh_fetch
    findings: list[dict] = []
    errors: list[str] = []
    # Each detection runs alone: one failing never loses the other's results.
    for name, detect in (("orphans", find_orphans), ("verdicts", find_unreviewed)):
        try:
            findings += detect(fetch, args.repo, since)
        except Exception as exc:  # noqa: BLE001 - reported, then exit 2
            errors.append(f"{name} detection failed: {exc}")
    if args.format == "json":
        print(json.dumps(findings, indent=2))
    else:
        print(render_md(findings) if findings or not errors else "", end="")
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
        if args.format == "md":
            print(f"\n**ERROR: {err}**")
    if errors:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
