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


# 5-6. red-lane and unlabelled-slice ---------------------------------------------

def _run(days_ago, conclusion="failure", status="completed"):
    return {"conclusion": conclusion, "status": status,
            "createdAt": _iso(24 * days_ago), "headSha": HEAD,
            "url": f"https://example/runs/{days_ago}"}


def _lane(*runs, wf="benchmarks.yml"):
    return {"prs": [], "issues": [], "runs": {wf: list(runs)}}


def _lane_findings(data):
    return [f for f in pp.probe(data, NOW) if f["kind"] == "red-lane"]


def test_two_red_scheduled_runs_give_one_red_lane():
    found = _lane_findings(_lane(_run(1), _run(2, "timed_out")))
    assert len(found) == 1 and found[0]["ref"] == "benchmarks.yml"
    assert "runs/1" in found[0]["detail"] and "runs/2" in found[0]["detail"]


def test_red_then_green_is_silent():
    assert not _lane_findings(_lane(_run(1), _run(2, "success")))
    assert not _lane_findings(_lane(_run(1, "success"), _run(2)))


def test_single_run_or_in_progress_is_silent():
    assert not _lane_findings(_lane(_run(1)))
    assert not _lane_findings(_lane(_run(1, None, "in_progress"), _run(2)))


def test_cancelled_counts_as_non_green():
    # lesson #2556: a job at its own timeout-minutes concludes cancelled, and
    # Deps canary sat cancelled 7/7 nights; until #2556 this asserted silence
    found = _lane_findings(_lane(_run(1, "cancelled"), _run(2)))
    assert len(found) == 1 and "cancelled" in found[0]["detail"]


def test_unlabelled_slice_pr_is_flagged():
    def open_pr(n, body, labels):
        return {"number": n, "title": f"pr {n}", "body": body,
                "labels": [{"name": x} for x in labels]}
    data = {"prs": [], "issues": [], "open_prs": [
        open_pr(1, "Slice: #5\n", []),
        open_pr(2, "Slice: #5\n", ["program"]),
        open_pr(3, "no slice here", []),
    ]}
    found = [f for f in pp.probe(data, NOW) if f["kind"] == "unlabelled-slice"]
    assert [f["ref"] for f in found] == ["#1"]


def test_fetch_reads_runs_for_every_watched_workflow():
    calls = []

    def gh(args):
        calls.append(args)
        if args[0] == "run":
            return json.dumps([_run(1), _run(2)])
        return "[]"
    data = pp.fetch(gh, "o/r")
    assert set(data["runs"]) == set(pp.WATCHED_WORKFLOWS)
    run_calls = [a for a in calls if a[0] == "run"]
    assert all("--event" in a and "schedule" in a and "main" in a for a in run_calls)


# 7. wip-over-cap (lesson #2200) ------------------------------------------------

def _wip_pr(n, gate=False):
    pr = _pr(n)
    pr["labels"] = [{"name": "program"}] + ([{"name": "human-gate"}] if gate else [])
    return pr


def test_wip_over_cap_fires_at_nine_counted_and_not_at_eight():
    assert "wip-over-cap" in _kinds([_wip_pr(n) for n in range(9)])
    assert "wip-over-cap" not in _kinds([_wip_pr(n) for n in range(8)])


def test_wip_gated_approved_green_does_not_count():
    gated = _pr(99, comments=[_verdict(1)], checks=_checks(1))
    gated["labels"] = [{"name": "human-gate"}]
    assert "wip-over-cap" not in _kinds([_wip_pr(n) for n in range(8)] + [gated])


def test_wip_draft_does_not_count_and_ready_does():
    base = [_wip_pr(n) for n in range(8)]
    draft = _wip_pr(98)
    draft["isDraft"] = True
    ready = _wip_pr(99)
    counted, excluded = pp.wip_count(base + [draft])
    assert len(counted) == 8 and [p["number"] for p, _ in excluded] == [98]
    assert "wip-over-cap" not in _kinds(base + [draft])
    assert "wip-over-cap" in _kinds(base + [ready])


def _hist(n, created, closed=None, gate=False):
    return {"number": n, "title": f"h{n}", "createdAt": _iso(created),
            "closedAt": _iso(closed) if closed is not None else None, "mergedAt": None,
            "labels": [{"name": "program"}] + ([{"name": "human-gate"}] if gate else [])}


def _peak_kinds(history):
    return [f for f in pp.probe({"prs": [], "issues": [], "history": history}, NOW)
            if f["kind"] == "wip-over-cap"]


def test_wip_peak_reconstructed_from_24h_history():
    # nine overlap for two hours, then all close: the open set now is empty
    hist = [_hist(n, 10, closed=8) for n in range(9)]
    found = _peak_kinds(hist)
    assert len(found) == 1 and "peak 9" in found[0]["title"]
    assert "approximate" in found[0]["detail"]


def test_wip_peak_silent_at_eight_and_when_not_overlapping():
    assert _peak_kinds([_hist(n, 10, closed=8) for n in range(8)]) == []
    assert _peak_kinds([_hist(n, 20 - 2 * n, closed=19 - 2 * n) for n in range(9)]) == []


def test_wip_peak_drops_human_gate_now_and_old_prs_count_from_window_start():
    assert _peak_kinds([_hist(n, 10, closed=8, gate=(n == 0)) for n in range(9)]) == []
    # opened 3 days ago, still open: present at the window start
    assert len(_peak_kinds([_hist(n, 72) for n in range(9)])) == 1


# 8. one issue per red lane (lesson #2556) ---------------------------------------

def _night(days_ago, conclusion, wf):
    run = _run(days_ago, conclusion)
    run["url"] = f"https://example/{wf}/{days_ago}"
    return run


def _runs_1026():
    """The 10-26 triage, newest first: live-fork F,F,F; Benchmarks red 5
    nights; Deps canary cancelled 7/7; orphans green."""
    return {
        "live-fork.yml": [_night(d, "failure", "lf") for d in (1, 2, 3)]
                         + [_night(4, "success", "lf")],
        "benchmarks.yml": [_night(d, "failure", "bm") for d in range(1, 6)]
                          + [_night(6, "success", "bm"), _night(7, "success", "bm")],
        "deps-canary.yml": [_night(d, "cancelled", "dc") for d in range(1, 8)],
        "orphans.yml": [_night(d, "success", "or") for d in range(1, 8)],
    }


class LaneGh:
    """Fake ``gh``: ``issue list`` returns the given open issues."""

    def __init__(self, open_issues=()):
        self.open_issues = list(open_issues)
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append(args)
        if args[:2] == ["issue", "list"]:
            return json.dumps(self.open_issues)
        return ""

    def verb(self, name):
        return [c for c in self.calls if c[:2] == ["issue", name]]


def _flag(call, flag):
    return [call[i + 1] for i, a in enumerate(call) if a == flag]


def test_lane_issues_replay_1026_opens_exactly_three():
    gh = LaneGh()
    actions = pp.sync_lane_issues(gh, "o/r", _runs_1026(), NOW)
    created = gh.verb("create")
    assert len(created) == 3 and not gh.verb("comment") and not gh.verb("close")
    by_title = {_flag(c, "--title")[0]: _flag(c, "--label") for c in created}
    assert by_title == {
        pp.lane_title("live-fork.yml"): ["program", "chain:F"],
        pp.lane_title("benchmarks.yml"): ["program", "chain:S"],
        pp.lane_title("deps-canary.yml"): ["program", "chain:C"],
    }
    body = {_flag(c, "--title")[0]: _flag(c, "--body")[0] for c in created}
    assert "7 consecutive" in body[pp.lane_title("deps-canary.yml")]
    assert "5 consecutive" in body[pp.lane_title("benchmarks.yml")]
    assert "bm/6" not in body[pp.lane_title("benchmarks.yml")]  # green ends the streak
    assert sorted(actions) == ["benchmarks.yml: created", "deps-canary.yml: created",
                               "live-fork.yml: created"]
    # the tracking-issue finding agrees: three red lanes
    reds = _lane_findings({"prs": [], "issues": [], "runs": _runs_1026()})
    assert sorted(f["ref"] for f in reds) == ["benchmarks.yml", "deps-canary.yml",
                                              "live-fork.yml"]


def test_lane_issue_next_night_bumps_instead_of_duplicating():
    runs = {"deps-canary.yml": _runs_1026()["deps-canary.yml"]}
    title = pp.lane_title("deps-canary.yml")
    # the decoy sorts first: only the exact title may be bumped
    gh = LaneGh([{"number": 40, "title": title + " (old)"},
                 {"number": 41, "title": title}])
    assert pp.sync_lane_issues(gh, "o/r", runs, NOW) == ["deps-canary.yml: bumped #41"]
    assert not gh.verb("create") and not gh.verb("close")
    assert [c[2] for c in gh.verb("comment")] == ["41"]
    listed = gh.verb("list")[0]
    assert title in _flag(listed, "--search")[0] and _flag(listed, "--label") == ["program"]


def test_lane_issue_duplicates_collapse_to_the_lowest_number():
    runs = {"deps-canary.yml": _runs_1026()["deps-canary.yml"]}
    title = pp.lane_title("deps-canary.yml")
    gh = LaneGh([{"number": 44, "title": title}, {"number": 41, "title": title},
                 {"number": 40, "title": "other"}])
    assert pp.sync_lane_issues(gh, "o/r", runs, NOW) == [
        "deps-canary.yml: bumped #41; closed duplicates #44"]
    assert [c[2] for c in gh.verb("close")] == ["44"]
    assert "Duplicate of #41" in _flag(gh.verb("close")[0], "--comment")[0]
    assert [c[2] for c in gh.verb("comment")] == ["41"]


def test_lane_missing_chain_label_retries_and_keeps_later_lanes():
    class NoChainGh(LaneGh):
        def __call__(self, args):
            if args[:2] == ["issue", "create"] and "chain:S" in args:
                self.calls.append(args)
                raise RuntimeError("could not add label: 'chain:S' not found")
            return super().__call__(args)

    gh = NoChainGh()
    actions = pp.sync_lane_issues(gh, "o/r", _runs_1026(), NOW)
    labels = {_flag(c, "--title")[0]: _flag(c, "--label") for c in gh.verb("create")}
    # benchmarks sorts first: the failed attempt, then a retry with program
    # only; the later lanes still get their issues
    assert [_flag(c, "--label") for c in gh.verb("create")
            if _flag(c, "--title")[0] == pp.lane_title("benchmarks.yml")] == [
        ["program", "chain:S"], ["program"]]
    assert labels[pp.lane_title("deps-canary.yml")] == ["program", "chain:C"]
    assert labels[pp.lane_title("live-fork.yml")] == ["program", "chain:F"]
    assert any(a.startswith("benchmarks.yml: created without `chain:S`") for a in actions)
    assert len(actions) == 3


def test_lane_one_red_night_then_green_opens_nothing():
    for runs in ([_night(1, "success", "x"), _night(2, "failure", "x")],   # red, then green
                 [_night(1, "failure", "x"), _night(2, "success", "x")],   # one red so far
                 [_night(1, "cancelled", "x")]):                           # one night only
        gh = LaneGh()
        assert pp.sync_lane_issues(gh, "o/r", {"benchmarks.yml": runs}, NOW) == []
        assert not gh.verb("create") and not gh.verb("comment")


def test_lane_green_closes_the_open_issue_and_never_reopens():
    title = pp.lane_title("live-fork.yml")
    runs = {"live-fork.yml": [_night(1, "success", "lf"), _night(2, "failure", "lf"),
                              _night(3, "failure", "lf")]}
    gh = LaneGh([{"number": 50, "title": title}])
    assert pp.sync_lane_issues(gh, "o/r", runs, NOW) == ["live-fork.yml: closed #50"]
    assert [c[2] for c in gh.verb("close")] == ["50"]
    assert not gh.verb("reopen")
    # a green lane with no open issue only looks
    gh = LaneGh()
    assert pp.sync_lane_issues(gh, "o/r", runs, NOW) == []
    assert [c[:2] for c in gh.calls] == [["issue", "list"]]


def test_lane_without_completed_runs_makes_no_call():
    gh = LaneGh()
    runs = {"live-fork.yml": [_run(1, None, "in_progress")], "orphans.yml": []}
    assert pp.sync_lane_issues(gh, "o/r", runs, NOW) == []
    assert gh.calls == []


def test_every_watched_lane_has_a_chain_label():
    assert set(pp.WATCHED_WORKFLOWS) == set(pp.LANE_CHAINS)
    assert all(v.startswith("chain:") for v in pp.LANE_CHAINS.values())


def test_cli_issue_runs_lane_sync(monkeypatch, capsys):
    gh = LaneGh()
    monkeypatch.setattr(pp, "fetch", lambda g, repo: {
        "prs": [], "issues": [], "open_prs": [], "history": [], "runs": _runs_1026()})
    assert pp.main(["--issue", "--repo", "o/r"], gh=gh) == 0
    assert capsys.readouterr().out.count("lane issue:") == 3
    lane_creates = [c for c in gh.verb("create")
                    if _flag(c, "--title")[0].startswith(pp.LANE_TITLE_PREFIX)]
    assert len(lane_creates) == 3
