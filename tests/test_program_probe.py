"""Offline tests for ``scripts/program_probe.py`` (lessons #1567, #1544).

Each finding fires on a constructed case and stays silent just under its
threshold. Fixtures use the ``gh pr list`` / ``gh issue list`` JSON shapes.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "program_probe",
    Path(__file__).resolve().parent.parent / "scripts" / "program_probe.py",
)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
HEAD = "a" * 40
OLD = "b" * 40


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _checks(hours_ago: float, conclusion: str = "SUCCESS") -> list[dict]:
    return [
        {"__typename": "CheckRun", "name": n, "status": "COMPLETED",
         "conclusion": conclusion, "startedAt": _iso(hours_ago + 0.1),
         "completedAt": _iso(hours_ago)}
        for n in pp.REQUIRED_CHECKS
    ]


def _verdict(hours_ago: float, verdict: str = "approve", sha: str = HEAD) -> dict:
    return {"body": f"Review-verdict: Opus 5.5 {verdict} {sha}\n\nnotes",
            "createdAt": _iso(hours_ago)}


def _pr(number=1, *, created=1.0, updated=0.5, comments=(), checks=(),
        draft=False, body="Slice: #900", head=HEAD) -> dict:
    return {"number": number, "title": f"pr {number}", "createdAt": _iso(created),
            "updatedAt": _iso(updated), "isDraft": draft, "headRefOid": head,
            "body": body, "comments": list(comments),
            "statusCheckRollup": list(checks), "closingIssuesReferences": []}


def _issue(number, labels, updated=0.5) -> dict:
    return {"number": number, "title": f"issue {number}", "updatedAt": _iso(updated),
            "labels": [{"name": n} for n in labels]}


def _kinds(prs=(), issues=()) -> list[str]:
    return [f["kind"] for f in pp.probe({"prs": list(prs), "issues": list(issues)}, NOW)]


# 1. approved at head, green, unlanded > 4 h -----------------------------------

def test_approved_green_unlanded_fires_past_4h():
    pr = _pr(created=10, comments=[_verdict(5)], checks=_checks(6))
    assert _kinds([pr]) == ["approved-not-landed"]


def test_approved_green_silent_just_under_4h():
    # approval 3.9 h ago is the later of approval and green
    pr = _pr(created=10, comments=[_verdict(3.9)], checks=_checks(6))
    assert _kinds([pr]) == []


def test_clock_runs_from_last_green_when_later():
    pr = _pr(created=10, comments=[_verdict(8)], checks=_checks(3.9))
    assert _kinds([pr]) == []


def test_verdict_at_older_sha_is_not_approval_at_head():
    pr = _pr(created=10, comments=[_verdict(5, sha=OLD)], checks=_checks(6))
    assert _kinds([pr]) == []


def test_short_sha_matches_head():
    pr = _pr(created=10, comments=[_verdict(5, sha=HEAD[:12])], checks=_checks(6))
    assert _kinds([pr]) == ["approved-not-landed"]


def test_changes_after_approve_at_head_withdraws_approval():
    pr = _pr(created=10, comments=[_verdict(6), _verdict(5, "changes")],
             checks=_checks(6))
    assert _kinds([pr]) == []


def test_red_or_missing_required_check_is_not_green():
    red = _pr(created=10, comments=[_verdict(5)], checks=_checks(6, "FAILURE"))
    missing = _pr(created=10, comments=[_verdict(5)], checks=_checks(6)[1:])
    assert _kinds([red, missing]) == []


# 2. open > 6 h with no verdict -----------------------------------------------

def test_no_verdict_fires_past_6h():
    assert _kinds([_pr(created=6.1)]) == ["no-verdict"]


def test_no_verdict_silent_just_under_6h():
    assert _kinds([_pr(created=5.9)]) == []


def test_no_verdict_skips_drafts_and_counts_any_verdict():
    assert _kinds([_pr(created=10, draft=True)]) == []
    assert _kinds([_pr(created=10, comments=[_verdict(9, "changes", OLD)])]) == []


# 3. lock holder with no activity > 6 h -----------------------------------------

def test_stale_lock_uses_linked_pr_activity():
    issue = _issue(900, ["lock:a.py"], updated=0.1)
    stale = _pr(created=1, updated=6.1, comments=[_verdict(1)])
    fresh = _pr(created=1, updated=5.9, comments=[_verdict(1)])
    assert _kinds([stale], [issue]) == ["stale-lock"]
    assert _kinds([fresh], [issue]) == []


def test_stale_lock_falls_back_to_issue_without_linked_pr():
    assert _kinds([], [_issue(900, ["lock:a.py"], updated=6.1)]) == ["stale-lock"]
    assert _kinds([], [_issue(900, ["lock:a.py"], updated=5.9)]) == []


def test_stale_lock_uses_most_recent_linked_pr():
    # card behaviour 3: the most recently updated linked PR decides
    issue = _issue(900, ["lock:a.py"], updated=10)
    old = _pr(1, created=12, updated=10, comments=[_verdict(11)])
    new = _pr(2, created=12, updated=1, comments=[_verdict(11)])
    assert _kinds([old, new], [issue]) == []
    assert _kinds([new, old], [issue]) == []


def test_only_lock_colon_labels_are_locks():
    assert _kinds([], [_issue(1, ["locked"], updated=10),
                       _issue(2, ["locked"], updated=10)]) == []


def test_closing_reference_links_pr_to_lock_issue():
    pr = _pr(updated=0.1, body="", comments=[_verdict(1)])
    pr["closingIssuesReferences"] = [{"number": 900}]
    assert _kinds([pr], [_issue(900, ["lock:a.py"], updated=10)]) == []


# 4. one lock label on more than one open issue -------------------------------

def test_double_lock_fires_on_two_holders():
    issues = [_issue(1, ["lock:a.py"]), _issue(2, ["lock:a.py", "program"])]
    found = pp.probe({"prs": [], "issues": issues}, NOW)
    assert [f["kind"] for f in found] == ["double-lock"]
    assert found[0]["ref"] == "#1, #2"


def test_single_holder_per_lock_is_silent():
    assert _kinds([], [_issue(1, ["lock:a.py"]), _issue(2, ["lock:b.py"])]) == []


# CLI and the tracking issue ---------------------------------------------------

def test_cli_from_json_reports_and_exits_zero(tmp_path, capsys):
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"prs": [_pr(created=7)], "issues": []}))
    rc = pp.main(["--from-json", str(fixture), "--now", NOW.isoformat(),
                  "--format", "json"], gh=_no_gh)
    assert rc == 0
    assert [f["kind"] for f in json.loads(capsys.readouterr().out)] == ["no-verdict"]
    # thresholds are flags
    rc = pp.main(["--from-json", str(fixture), "--now", NOW.isoformat(),
                  "--verdict-hours", "8"], gh=_no_gh)
    assert "No stalled" in capsys.readouterr().out


def _no_gh(args):
    raise AssertionError(f"unexpected gh call {args}")


def test_cli_refuses_issue_in_offline_mode(tmp_path, capsys):
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"prs": [_pr(created=7)], "issues": []}))
    import pytest
    for argv in (["--from-json", str(fixture), "--now", NOW.isoformat(), "--issue"],
                 ["--from-json", str(fixture), "--issue"],
                 ["--now", NOW.isoformat(), "--issue"]):
        with pytest.raises(SystemExit) as exc:
            pp.main(argv, gh=_no_gh)
        assert exc.value.code == 2
    assert "--issue cannot be combined" in capsys.readouterr().err


def test_cli_naive_now_is_utc(tmp_path, capsys):
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"prs": [_pr(created=7)], "issues": []}))
    naive = NOW.replace(tzinfo=None).isoformat()
    assert pp.main(["--from-json", str(fixture), "--now", naive,
                    "--format", "json"], gh=_no_gh) == 0
    found = json.loads(capsys.readouterr().out)
    assert [f["kind"] for f in found] == ["no-verdict"]
    assert "open 7.0 h" in found[0]["detail"]


class FakeGh:
    def __init__(self, open_issues):
        self.open_issues = open_issues
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append(args)
        if args[:2] == ["issue", "list"]:
            return json.dumps(self.open_issues)
        return ""


def test_sync_issue_creates_updates_and_closes():
    finding = [{"kind": "no-verdict", "ref": "#1", "title": "t", "detail": "d"}]
    gh = FakeGh([])
    assert pp.sync_issue(gh, "o/r", finding, NOW) == "created"
    assert gh.calls[-1][:2] == ["issue", "create"] and "program" in gh.calls[-1]

    gh = FakeGh([{"number": 77, "title": pp.ISSUE_TITLE},
                 {"number": 78, "title": "other"}])
    assert pp.sync_issue(gh, "o/r", finding, NOW) == "updated #77"
    assert gh.calls[-1][:3] == ["issue", "edit", "77"]
    assert pp.sync_issue(gh, "o/r", [], NOW) == "closed #77"
    assert gh.calls[-1][:3] == ["issue", "close", "77"]

    gh = FakeGh([])
    assert pp.sync_issue(gh, "o/r", [], NOW) == "nothing to do"
    assert len(gh.calls) == 1


def test_sync_issue_collapses_duplicate_tracking_issues():
    finding = [{"kind": "no-verdict", "ref": "#1", "title": "t", "detail": "d"}]
    dupes = [{"number": 6, "title": pp.ISSUE_TITLE},
             {"number": 5, "title": pp.ISSUE_TITLE},
             {"number": 7, "title": "other"}]

    gh = FakeGh(dupes)
    assert pp.sync_issue(gh, "o/r", [], NOW) == "closed #5, #6"
    closed = [c[2] for c in gh.calls if c[:2] == ["issue", "close"]]
    assert closed == ["5", "6"]

    gh = FakeGh(dupes)
    assert pp.sync_issue(gh, "o/r", finding, NOW) == "updated #5; closed duplicates #6"
    assert [c[:3] for c in gh.calls[1:]] == [["issue", "close", "6"],
                                             ["issue", "edit", "5"]]
