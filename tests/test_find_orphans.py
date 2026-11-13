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


def test_resolved_2872_rows_are_acknowledged():
    for key in ((2628, "pushed-after-merge"),
                (2682, "merged-without-verdict"),
                (2656, "merged-without-verdict")):
        assert key in fo.ACKNOWLEDGED
    # PR 2628 yields two findings: the acknowledged kind is dropped, the
    # other kind for the same PR is kept.
    pr = _pr(2628, "pushed after merge", "feature/p2628", "c" * 40, "main",
             "a2628" + "0" * 35, "2026-09-12T00:00:00Z")
    COMPARE[pr["merge_commit_sha"]] = "diverged"
    BRANCHES["feature/p2628"] = "d" * 40
    try:
        hits = fo.find_orphans(make_fetch([pr]), REPO, SINCE)
    finally:
        COMPARE.pop(pr["merge_commit_sha"])
        BRANCHES.pop("feature/p2628")
    assert [(h["pr"], h["kind"]) for h in hits] == [(2628, "not-on-main")]


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


def _comment(body, at="2026-11-01T00:30:00Z", assoc="OWNER"):
    return {"body": body, "created_at": at, "author_association": assoc}


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


def test_approve_by_a_non_collaborator_does_not_clear_it():
    # #2878: the REST author_association reaches the parser; a stranger's approve
    # at the merged head leaves the merge flagged.
    for assoc in ("NONE", "CONTRIBUTOR", None):
        c = {1: [_comment(f"Review-verdict: Opus 5.5 approve {HEAD[:12]}", assoc=assoc)]}
        assert len(_unreviewed([_vpr(1, ["program", "semantic"])], comments=c)) == 1, assoc


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
    pinned = datetime(2026, 11, 2, tzinfo=timezone.utc)  # pinned: no time rot
    assert not fo.find_unreviewed(make_vfetch([old]), REPO, since, now=pinned)


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


# --- delta review of #2679 ---

import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("body", [
    "verify_move failed: not an identical multiset",
    "verify_move: FAILED - 3 defs, not an identical multiset",
    "I did not run verify_move; the multiset is identical, trust me.",
])
def test_move_evidence_must_be_verify_moves_ok_line(body):
    pr = _vpr(90, ["program"], body=body)
    assert [h["pr"] for h in _unreviewed([pr], files={90: ["tests/a.py"]})] == [90]


@pytest.mark.parametrize("line", [
    "verify_move: OK — 59 defs, identical multiset",
    "verify_move: OK — 12 defs, 3 moved",
])
def test_real_verify_move_output_lines_exempt(line):
    pr = _vpr(91, ["program"], body=f"Slice: #1\n\n```\n{line}\n```\n")
    assert not _unreviewed([pr], files={91: ["tests/a.py"]})


@pytest.mark.parametrize("path", [
    ".github/workflows/tests.yml", "scripts/land_pr.py", "scripts/check_quirks.py",
    "tests/freeze_census.py", "internal_docs/program/retirement_ledger.toml",
])
def test_move_evidence_never_exempts_a_gate_file(path):
    pr = _vpr(92, ["program"], body=MOVE_BODY)
    assert [h["pr"] for h in _unreviewed([pr], files={92: ["tests/a.py", path]})] == [92]


def test_move_evidence_does_not_exempt_a_rename_out_of_a_gate_file():
    row = {"filename": "scripts/old_land.py", "previous_filename": "scripts/land_pr.py"}
    pr = _vpr(93, ["program"], body=MOVE_BODY)
    assert [h["pr"] for h in _unreviewed([pr], files={93: [row]})] == [93]


def _fake_run(stderr):
    def run(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr=stderr)
    return run


def test_gh_fetch_404_is_not_found(monkeypatch):
    monkeypatch.setattr(fo.subprocess, "run", _fake_run("gh: Not Found (HTTP 404)"))
    assert fo.gh_fetch("repos/x/branches/gone") is None


@pytest.mark.parametrize("stderr", [
    "gh: Resource not accessible by integration (HTTP 403) see /branches/404-fix",
    "gh: API rate limit exceeded (HTTP 403); request 404ab was refused",
    "gh: Not Found",
])
def test_gh_fetch_other_errors_raise(monkeypatch, stderr):
    monkeypatch.setattr(fo.subprocess, "run", _fake_run(stderr))
    with pytest.raises(RuntimeError):
        fo.gh_fetch("repos/x/branches/b")


# The workflow's find step, run for real against stub or crashing scripts.

def _find_step_script():
    text = (ROOT / ".github" / "workflows" / "orphans.yml").read_text(encoding="utf-8")
    m = re.search(r"- name: Find orphaned work\n.*?run: \|\n(.*?)\n\n      - name:",
                  text, re.DOTALL)
    assert m, "find step not found"
    body = "\n".join(line[10:] for line in m.group(1).splitlines())
    return body.replace("python scripts/", f'"{Path(sys.executable).as_posix()}" scripts/')


def _run_step(tmp_path, source):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    (tmp_path / "scripts").mkdir(parents=True)
    for name, text in source.items():
        (tmp_path / "scripts" / name).write_text(text, encoding="utf-8")
    out = tmp_path / "gh_output"
    out.write_text("")
    env = dict(os.environ, GITHUB_OUTPUT=out.as_posix())
    subprocess.run([bash, "-c", _find_step_script()], cwd=tmp_path, env=env,
                   capture_output=True, text=True, check=False)
    return dict(line.split("=", 1) for line in out.read_text().split())


def _stub(code, text):
    return {"find_orphans.py": f"import sys\nprint({text!r}, end='')\nsys.exit({code})\n"}


def test_workflow_import_crash_fails_the_job(tmp_path):
    # Rename BASELINE in land_pr.py: find_orphans dies at import with exit 1 and
    # an empty report, which must not read as "findings".
    scripts = ROOT / "scripts"
    land = (scripts / "land_pr.py").read_text(encoding="utf-8")
    assert "\nBASELINE = (" in land
    src = {n: (scripts / n).read_text(encoding="utf-8")
           for n in ("find_orphans.py", "program_probe.py")}
    src["land_pr.py"] = land.replace("\nBASELINE = (", "\nBASELINE_RENAMED = (")
    got = _run_step(tmp_path, src)
    assert got.get("failed") == "1" and "found" not in got


def test_workflow_codes_need_their_positive_marker(tmp_path):
    assert _run_step(tmp_path / "a", _stub(1, "")) == {"failed": "1"}
    assert _run_step(tmp_path / "b", _stub(0, "")) == {"failed": "1"}
    assert _run_step(tmp_path / "c", _stub(0, "No orphaned work found.\n")) == {"found": "0"}
    assert _run_step(tmp_path / "d", _stub(1, "| k | #1 t | `main` | d |\n")) == {"found": "1"}
    got = _run_step(tmp_path / "e", _stub(2, "| k | #1 t | `main` | d |\n"))
    assert got == {"failed": "1", "found": "1"}
    assert _run_step(tmp_path / "f", _stub(2, "**ERROR: x**\n")) == {"failed": "1"}


def test_move_evidence_never_exempts_a_golden_file():
    pr = _vpr(95, ["program"], body=MOVE_BODY)
    gold = "tests/opensees/golden/cells/arch_with_orientation_fan_out/flat/py.h5dump"
    assert [h["pr"] for h in _unreviewed([pr], files={95: [gold]})] == [95]
    docs = _vpr(96, ["program"], body=MOVE_BODY)
    assert not _unreviewed([docs], files={96: ["docs/golden-stills.md"]})
