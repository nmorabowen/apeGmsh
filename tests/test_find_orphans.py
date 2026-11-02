"""Replay recorded GitHub responses through the ``fetch`` seam of
``scripts/find_orphans.py``. No network.

The PR fields (number, title, head/base refs, timestamps) are recorded from
#858 and #1097; the merge shas and compare statuses are the shapes those
cases produce (#858's merge commit is on no ref, so compare reads
``diverged``; #1097's branch was pushed to after the merge).
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "nmorabowen/apeGmsh"
SINCE = datetime(2026, 7, 1, tzinfo=timezone.utc)

_spec = importlib.util.spec_from_file_location(
    "find_orphans",
    Path(__file__).resolve().parent.parent / "scripts" / "find_orphans.py",
)
fo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fo)


def _pr(number, title, head_ref, head_sha, base, merge, merged_at):
    return {
        "number": number, "title": title, "merged_at": merged_at,
        "updated_at": merged_at, "merge_commit_sha": merge,
        "head": {"ref": head_ref, "sha": head_sha, "repo": {"full_name": REPO}},
        "base": {"ref": base},
    }


PR858 = _pr(858, "feat(opensees): 2-D finite strain", "claude/apegmsh-2d-finite-strain",
            "7c05d2a8b927815c825b70448abb2de2b97e24cb",
            "claude/apegmsh-facet-extractor-bug-e37f5c",
            "a858" + "0" * 36, "2026-07-25T04:33:36Z")
PR1097 = _pr(1097, "feat(opensees): act on the SANISAND integrator guide",
             "claude/ladruno-sanisand-integrator-guide-adb316",
             "4b8fdfdb7121405e0d6523669196d508a32026d3", "main",
             "b109" + "0" * 36, "2026-09-06T05:01:49Z")
PR_CLEAN = _pr(1200, "clean change", "feature/clean", "c" * 40, "main",
               "d200" + "0" * 36, "2026-09-10T00:00:00Z")
PR_DELETED = _pr(1201, "branch auto-deleted", "feature/gone", "e" * 40, "main",
                 "e201" + "0" * 36, "2026-09-11T00:00:00Z")
PR_OLD = _pr(5, "outside window", "old", "f" * 40, "main", "f005" + "0" * 36,
             "2026-01-01T00:00:00Z")

COMPARE = {
    PR858["merge_commit_sha"]: "diverged",
    PR1097["merge_commit_sha"]: "behind",
    PR_CLEAN["merge_commit_sha"]: "identical",
    PR_DELETED["merge_commit_sha"]: "behind",
}
BRANCHES = {
    "claude/apegmsh-2d-finite-strain": PR858["head"]["sha"],
    "claude/ladruno-sanisand-integrator-guide-adb316": "9" * 40,  # moved past head
    "feature/clean": "c" * 40,
    "feature/gone": None,  # auto-deleted: the API answers 404
}


def make_fetch(prs):
    def fetch(path):
        if path.startswith(f"repos/{REPO}/pulls?"):
            return prs if "page=1" in path else []
        if "/compare/main..." in path:
            return {"status": COMPARE[path.rsplit("...", 1)[1]]}
        if "/branches/" in path:
            tip = BRANCHES.get(path.split("/branches/", 1)[1])
            return None if tip is None else {"commit": {"sha": tip}}
        raise AssertionError(path)
    return fetch


def _run():
    fetch = make_fetch([PR858, PR1097, PR_CLEAN, PR_DELETED, PR_OLD])
    return fo.find_orphans(fetch, REPO, SINCE)


def _no_acks(monkeypatch):
    monkeypatch.setattr(fo, "ACKNOWLEDGED", {})


def test_858_diverged_is_flagged(monkeypatch):
    _no_acks(monkeypatch)
    hits = [f for f in _run() if f["pr"] == 858]
    assert [h["kind"] for h in hits] == ["not-on-main"]
    assert "diverged" in hits[0]["detail"]


def test_1097_branch_moved_past_head_is_flagged(monkeypatch):
    _no_acks(monkeypatch)
    hits = [f for f in _run() if f["pr"] == 1097]
    assert [h["kind"] for h in hits] == ["pushed-after-merge"]


def test_clean_pr_not_flagged():
    assert not [f for f in _run() if f["pr"] == 1200]


def test_deleted_branch_not_flagged_by_detection_2():
    assert not [f for f in _run() if f["pr"] == 1201]


def test_old_pr_outside_window_ignored():
    assert not [f for f in _run() if f["pr"] == 5]


def test_main_exit_codes_and_formats(capsys, monkeypatch):
    _no_acks(monkeypatch)
    dirty = make_fetch([PR858, PR1097])
    assert fo.main(["--since", "36500", "--format", "json"], fetch=dirty) == 1
    assert {f["pr"] for f in json.loads(capsys.readouterr().out)} == {858, 1097}
    assert fo.main(["--since", "36500", "--format", "md"], fetch=dirty) == 1
    assert "#858" in capsys.readouterr().out
    clean = make_fetch([PR_CLEAN, PR_DELETED])
    assert fo.main(["--since", "36500"], fetch=clean) == 0


def test_acknowledged_rows_are_suppressed():
    assert (858, "not-on-main") in fo.ACKNOWLEDGED
    assert (1097, "pushed-after-merge") in fo.ACKNOWLEDGED
    assert (1057, "not-on-main") in fo.ACKNOWLEDGED
    assert not [f for f in _run() if f["pr"] in (858, 1097)]


def test_unacknowledged_orphan_still_reports():
    # Same PR as an acknowledged one but a different kind, plus a new PR.
    new = _pr(1300, "new orphan", "feature/new", "a" * 40, "main",
              "a300" + "0" * 36, "2026-09-12T00:00:00Z")
    COMPARE[new["merge_commit_sha"]] = "diverged"
    BRANCHES["feature/new"] = "a" * 40
    try:
        hits = fo.find_orphans(make_fetch([PR858, new]), REPO, SINCE)
    finally:
        COMPARE.pop(new["merge_commit_sha"])
        BRANCHES.pop("feature/new")
    assert [(h["pr"], h["kind"]) for h in hits] == [(1300, "not-on-main")]
    assert fo.main(["--since", "36500"],
                   fetch=make_fetch([PR858])) == 0


# --- detection 3: merged without an approving Review-verdict (lesson #2637) ---

POST = datetime(2026, 10, 31, tzinfo=timezone.utc)  # a window after the policy
HEAD = "a1b2c3d4" + "0" * 32
OTHER = "99999999" + "0" * 32


def _vpr(number, labels, head=HEAD, merged_at=None, body=""):
    merged_at = merged_at or (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pr = _pr(number, f"pr {number}", f"b{number}", head, "main",
             f"m{number}" + "0" * 30, merged_at)
    pr["labels"] = [{"name": n} for n in labels]
    pr["body"] = body
    return pr


def _comment(body, at="2026-11-01T00:30:00Z"):
    return {"body": body, "created_at": at}


def make_vfetch(prs, files=None, comments=None):
    files = files or {}
    comments = comments or {}

    def fetch(path):
        if path.startswith(f"repos/{REPO}/pulls?"):
            return prs if "page=1" in path else []
        if "/compare/main..." in path:
            return {"status": "identical"}
        if "/branches/" in path:
            return None
        if "/pulls/" in path and "/files" in path:
            n = int(path.split("/pulls/")[1].split("/")[0])
            return [f if isinstance(f, dict) else {"filename": f} for f in files.get(n, [])] if "page=1" in path else []
        if "/issues/" in path and "/comments" in path:
            n = int(path.split("/issues/")[1].split("/")[0])
            return comments.get(n, []) if "page=1" in path else []
        raise AssertionError(path)
    return fetch


def _unreviewed(prs, **kw):
    return fo.find_unreviewed(make_vfetch(prs, **kw), REPO, POST)


def test_semantic_pr_without_verdict_is_flagged():
    hits = _unreviewed([_vpr(1, ["program", "semantic"])])
    assert [(h["kind"], h["pr"]) for h in hits] == [("merged-without-verdict", 1)]


def test_approve_at_merged_head_clears_it():
    c = {1: [_comment(f"Review-verdict: Opus 5.5 approve {HEAD[:12]}")]}
    assert not _unreviewed([_vpr(1, ["program", "semantic"])], comments=c)


def test_approve_at_older_sha_does_not_count():
    # the #2637 shape: approval bound to the pre-merge head, merged at another
    c = {1: [_comment(f"Review-verdict: Opus 5.5 approve {OTHER[:12]}")]}
    assert len(_unreviewed([_vpr(1, ["program", "semantic"])], comments=c)) == 1


def test_changes_verdict_does_not_count_and_withdraws_approve():
    c = {1: [_comment(f"Review-verdict: Opus changes {HEAD[:12]}")],
         2: [_comment(f"Review-verdict: Opus approve {HEAD[:12]}", "2026-11-01T00:10:00Z"),
             _comment(f"Review-verdict: Opus changes {HEAD[:12]}", "2026-11-01T00:20:00Z")]}
    prs = [_vpr(1, ["program", "semantic"]), _vpr(2, ["program", "semantic"])]
    assert {h["pr"] for h in _unreviewed(prs, comments=c)} == {1, 2}


def test_scope_by_changed_files():
    prs = [_vpr(n, ["program"]) for n in (10, 11, 12, 13, 14, 15)]
    files = {
        10: ["tests/test_x.py"], 11: [".github/workflows/orphans.yml"],
        12: ["scripts/land_pr.py"], 13: ["tests/lazy_import_snapshot.txt"],
        14: ["docs/a.md", "changelog.d/x.md"],  # docs/changelog only: out of scope
        15: ["scripts/verify_move.py"],
    }
    assert {h["pr"] for h in _unreviewed(prs, files=files)} == {10, 11, 12, 13}


def test_non_program_pr_is_out_of_scope():
    assert not _unreviewed([_vpr(20, ["semantic"])])


def test_pr_merged_before_policy_is_exempt():
    old = _vpr(30, ["program", "semantic"], merged_at="2026-10-30T19:36:00Z")
    since = datetime(2026, 7, 1, tzinfo=timezone.utc)  # a wide window is clamped
    assert not fo.find_unreviewed(make_vfetch([old]), REPO, since)


def test_report_has_separate_section_and_detector_note(capsys):
    fetch = make_vfetch([_vpr(1, ["program", "semantic"])])
    assert fo.main(["--since", "36500"], fetch=fetch) == 1
    out = capsys.readouterr().out
    assert "## Merged without an approving Review-verdict" in out
    assert "cannot block" in out
    assert "#1 pr 1" in out


# --- review of #2679: exemption, window, errors, scope mutations ---

MOVE_BODY = """Slice: #1965
Author-model: Sonnet
Class: mechanical

Moves `LadrunoUP` from `element/solid.py` to `element/solid_up.py`, verbatim.

## Verification
- `verify_move --base 27a95f769 --head bf30ccbba` (pushed sha): `OK, 59 defs, identical multiset`
"""


def test_verify_move_evidence_exempts_a_non_semantic_pr():
    pr = _vpr(40, ["program"], body=MOVE_BODY)
    assert not _unreviewed([pr], files={40: ["tests/families.py", "src/a.py"]})


def test_semantic_pr_is_never_exempt_by_move_evidence():
    pr = _vpr(41, ["program", "semantic"], body=MOVE_BODY)
    assert [h["pr"] for h in _unreviewed([pr])] == [41]


def test_body_without_move_evidence_is_still_flagged():
    pr = _vpr(42, ["program"], body="Slice: #1\nverify_move was not run here.")
    assert [h["pr"] for h in _unreviewed([pr], files={42: ["tests/a.py"]})] == [42]


def test_golden_clause_outside_tests():
    # not under tests/, so only the "golden" clause puts it in scope
    gold = "apeGmshViewer/fixtures/line_oracle_basic_only.golden.json"
    assert [h["pr"] for h in _unreviewed([_vpr(50, ["program"])], files={50: [gold]})] == [50]


def test_golden_in_a_docs_path_is_not_a_golden_file():
    prs = [_vpr(51, ["program"]), _vpr(52, ["program"])]
    files = {51: ["docs/golden-stills.md"], 52: ["internal_docs/golden/plan.md"]}
    assert not _unreviewed(prs, files=files)


def test_rename_out_of_tests_is_in_scope_via_previous_filename():
    row = {"filename": "src/apeGmsh/moved_test_helper.py",
           "previous_filename": "tests/helper.py"}
    assert [h["pr"] for h in _unreviewed([_vpr(53, ["program"])], files={53: [row]})] == [53]


def test_window_limits_scan_and_fetches_nothing_extra():
    calls = []
    inner = make_vfetch([_vpr(60, ["program", "semantic"], merged_at="2026-11-01T01:00:00Z")])

    def fetch(path):
        calls.append(path)
        return inner(path)

    now = datetime(2026, 11, 30, tzinfo=timezone.utc)  # 29 days later: outside 14 d
    assert fo.find_unreviewed(fetch, REPO, POST, now=now) == []
    assert not [c for c in calls if "/comments" in c or "/files" in c]


def test_comments_fetched_only_for_in_scope_prs():
    calls = []
    inner = make_vfetch([_vpr(70, ["program"]), _vpr(71, ["semantic"]),
                         _vpr(72, ["program", "semantic"])],
                        files={70: ["docs/a.md"]})

    def fetch(path):
        calls.append(path)
        return inner(path)

    fo.find_unreviewed(fetch, REPO, POST)
    commented = [c for c in calls if "/comments" in c]
    assert len(commented) == 1 and "/issues/72/" in commented[0]


def _failing(inner, needle):
    def fetch(path):
        if needle in path:
            raise RuntimeError(f"gh api {path} failed: 403")
        return inner(path)
    return fetch


def _diverged_and_unreviewed():
    orphan = _pr(858, "orphan", "claude/x", "7" * 40, "main", "a858" + "0" * 36,
                 (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    orphan["labels"] = []
    return [orphan, _vpr(80, ["program", "semantic"])]


def test_verdict_failure_keeps_orphan_results_and_exits_2(capsys, monkeypatch):
    _no_acks(monkeypatch)
    inner = make_vfetch(_diverged_and_unreviewed())

    def fetch(path):
        if "/compare/main..." in path:
            return {"status": "diverged"}
        return inner(path)
    assert fo.main(["--since", "3"], fetch=_failing(fetch, "/issues/80/comments")) == 2
    out = capsys.readouterr().out
    assert "not-on-main" in out and "verdicts detection failed" in out


def test_orphan_failure_keeps_verdict_results_and_exits_2(capsys, monkeypatch):
    _no_acks(monkeypatch)
    inner = make_vfetch(_diverged_and_unreviewed())
    assert fo.main(["--since", "3"], fetch=_failing(inner, "/compare/main...")) == 2
    out = capsys.readouterr().out
    assert "merged-without-verdict" in out and "orphans detection failed" in out
