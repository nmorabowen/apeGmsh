"""The ``review-verdict`` commit status (lesson #2637, slice #2878).

Recorded-shape GitHub responses replay through the ``fetch`` / ``post`` seams
of ``scripts/check_review_verdict.py``. No network.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO = "nmorabowen/apeGmsh"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crv = _load("check_review_verdict")
fo = _load("find_orphans")

OLD = "e2c3099a" + "1" * 32   # the head a review approved
NEW = "5a5b5c5d" + "2" * 32   # the head after a later push (#2628)


def _pr(number=7, labels=("program", "semantic"), head=NEW, body="", draft=False,
        fork=False):
    return {
        "number": number, "title": f"pr {number}", "body": body, "draft": draft,
        "labels": [{"name": n} for n in labels],
        "head": {"ref": f"b{number}", "sha": head,
                 "repo": {"full_name": "someone/apeGmsh" if fork else REPO}},
        "base": {"ref": "main"},
        "merged_at": "2026-11-09T12:00:00Z", "updated_at": "2026-11-09T12:00:00Z",
        "merge_commit_sha": "m" * 40,
    }


def _c(body, at="2026-11-09T10:00:00Z"):
    return {"body": body, "created_at": at}


def make_fetch(prs, comments=None, files=None):
    prs = {pr["number"]: pr for pr in prs}
    comments = comments or {}
    files = files or {}

    def fetch(path):
        m = re.fullmatch(rf"repos/{REPO}/pulls/(\d+)", path)
        if m:
            return prs.get(int(m[1]))
        if path.startswith(f"repos/{REPO}/pulls?"):  # find_orphans' merged list
            return list(prs.values()) if "page=1" in path else []
        m = re.fullmatch(rf"repos/{REPO}/pulls/(\d+)/files\?per_page=100&page=(\d+)", path)
        if m:
            return [{"filename": f} for f in files.get(int(m[1]), [])] if m[2] == "1" else []
        m = re.fullmatch(rf"repos/{REPO}/issues/(\d+)/comments\?per_page=100&page=(\d+)", path)
        if m:
            return comments.get(int(m[1]), []) if m[2] == "1" else []
        raise AssertionError(path)
    return fetch


def _status(pr, comments=(), files=(), head=None):
    fetch = make_fetch([pr], {pr["number"]: list(comments)}, {pr["number"]: list(files)})
    return crv.decide(fetch, REPO, crv.read_pr(fetch, REPO, pr["number"], head))


# --- one policy, one implementation ---------------------------------------

def test_the_check_and_the_detector_share_the_policy_functions():
    assert crv.verdict_required is crv._orphans.verdict_required
    assert crv.has_head_approval is crv._orphans.has_head_approval
    assert Path(crv._orphans.__file__).resolve() == ROOT / "scripts" / "find_orphans.py"
    # Same code object as an independent load of the detector: not a copy.
    assert crv.verdict_required.__code__.co_filename == fo.verdict_required.__code__.co_filename
    assert crv.verdict_required.__code__.co_code == fo.verdict_required.__code__.co_code
    assert crv.has_head_approval.__code__.co_code == fo.has_head_approval.__code__.co_code
    src = (ROOT / "scripts" / "check_review_verdict.py").read_text(encoding="utf-8")
    assert "_VERDICT" not in src and "startswith" not in src, "a second verdict rule"


CASES = [
    # (labels, files, comments at head NEW) -> detector flags iff check not success
    (("program", "semantic"), (), ()),
    (("program", "semantic"), (), (_c(f"Review-verdict: Opus 5.5 approve {NEW[:12]}"),)),
    (("program", "semantic"), (), (_c(f"Review-verdict: Opus 5.5 approve {OLD[:12]}"),)),
    (("program", "semantic"), (), (_c(f"Review-verdict: Opus approve {NEW[:12]}"),
                                   _c(f"Review-verdict: Opus changes {NEW[:12]}",
                                      "2026-11-09T11:00:00Z"))),
    (("program",), ("docs/x.md",), ()),
    (("program",), ("tests/test_x.py",), ()),
    (("program",), (".github/workflows/x.yml",), ()),
    ((), (".github/workflows/x.yml",), ()),
]


@pytest.mark.parametrize("labels,files,comments", CASES)
def test_the_check_and_the_detector_cannot_disagree(labels, files, comments):
    pr = _pr(labels=labels)
    fetch = make_fetch([pr], {7: list(comments)}, {7: list(files)})
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    flagged = bool(fo.find_unreviewed(fetch, REPO, fo.POLICY_START, now=now))
    status = crv.decide(fetch, REPO, crv.read_pr(fetch, REPO, 7))
    assert flagged == (status["state"] != "success")


# --- the states -------------------------------------------------------------

def test_non_program_pr_is_not_required():
    got = _status(_pr(labels=()))
    assert got["state"] == "success" and "not required" in got["description"]


def test_mechanical_pr_outside_policy_paths_is_not_required():
    got = _status(_pr(labels=("program", "mechanical")), files=["src/apeGmsh/x.py"])
    assert got["state"] == "success" and "not required" in got["description"]


def test_mechanical_pr_with_move_evidence_is_exempt():
    pr = _pr(labels=("program", "mechanical"),
             body="verify_move: OK, 59 defs, identical multiset")
    assert _status(pr, files=["tests/test_x.py"])["state"] == "success"


def test_semantic_pr_with_zero_comments_waits_for_its_head():
    got = _status(_pr())
    assert got == {"sha": NEW, "state": "pending", "context": "review-verdict",
                   "description": got["description"]}
    assert NEW[:12] in got["description"]


def test_approve_at_current_head_is_success():
    got = _status(_pr(), [_c(f"Review-verdict: Opus 5.5 approve {NEW[:12]}")])
    assert got["state"] == "success" and NEW[:12] in got["description"]


def test_a_push_after_the_approval_does_not_inherit_it():
    # The #2628 case: the approval names the earlier head; the PR moved on.
    # This is the regression test: it fails with the head comparison removed.
    comments = [_c(f"Review-verdict: Opus 5.5 approve {OLD[:12]}")]
    assert _status(_pr(head=OLD), comments)["state"] == "success"
    got = _status(_pr(head=NEW), comments)
    assert got["state"] == "pending" and got["sha"] == NEW
    assert NEW[:12] in got["description"]


def test_a_later_changes_at_the_same_head_wins():
    comments = [_c(f"Review-verdict: Opus approve {NEW[:12]}", "2026-11-09T10:00:00Z"),
                _c(f"Review-verdict: Opus changes {NEW[:12]}", "2026-11-09T11:00:00Z")]
    assert _status(_pr(), comments)["state"] == "pending"
    # Ordered by when the comment was posted, not by its place in the reply.
    assert _status(_pr(), comments[::-1])["state"] == "pending"


def test_a_short_prefix_verdict_counts_down_to_seven_hex_digits():
    assert _status(_pr(), [_c(f"Review-verdict: Opus approve {NEW[:7]}")])["state"] == "success"
    assert _status(_pr(), [_c(f"Review-verdict: Opus approve {NEW[:6]}")])["state"] == "pending"


def test_an_edited_verdict_comment_is_read_as_it_now_stands():
    # The API returns the current body; the issue_comment `edited` event re-runs.
    edited = [_c(f"Review-verdict: Opus changes {NEW[:12]}")]
    assert _status(_pr(), edited)["state"] == "pending"


def test_a_draft_pr_is_decided_like_any_other():
    assert _status(_pr(draft=True))["state"] == "pending"
    assert _status(_pr(draft=True), [_c(f"Review-verdict: O approve {NEW[:12]}")])["state"] == "success"


def test_a_fork_pr_is_decided_like_any_other():
    assert _status(_pr(fork=True))["state"] == "pending"


def test_replay_at_a_past_head_decides_there():
    comments = [_c(f"Review-verdict: Opus approve {OLD[:12]}")]
    assert _status(_pr(head=NEW), comments, head=OLD)["state"] == "success"


@pytest.mark.parametrize("labels", [(), ("program", "semantic")])
def test_descriptions_fit_githubs_limit(labels):
    assert len(_status(_pr(labels=labels))["description"]) <= 140


# --- main: what reaches GitHub ----------------------------------------------

def _main(argv, pr, comments=()):
    posted = []
    fetch = make_fetch([pr] if pr else [], {7: list(comments)})
    code = crv.main(argv + ["--repo", REPO], fetch=fetch,
                    post=lambda path, fields: posted.append((path, fields)))
    return code, posted


def test_main_posts_the_status_on_the_head_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    code, posted = _main(["--pr", "7"], _pr())
    assert code == 0
    (path, fields), = posted
    assert path == f"repos/{REPO}/statuses/{NEW}"
    assert fields["state"] == "pending" and fields["context"] == "review-verdict"
    assert fields["target_url"] == f"https://github.com/{REPO}/actions/runs/42"


def test_dry_run_posts_nothing(capsys):
    code, posted = _main(["--pr", "7", "--dry-run"], _pr())
    assert code == 0 and posted == []
    assert '"state": "pending"' in capsys.readouterr().out


def test_head_override_needs_dry_run():
    with pytest.raises(SystemExit):
        _main(["--pr", "7", "--head", OLD], _pr())


def test_unknown_pr_exits_2_and_posts_nothing():
    code, posted = _main(["--pr", "7"], None)
    assert code == 2 and posted == []


def test_a_failed_decision_posts_error_on_the_head(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("files endpoint down")
    monkeypatch.setattr(crv, "verdict_required", boom)
    code, posted = _main(["--pr", "7"], _pr())
    assert code == 2
    (path, fields), = posted
    assert path.endswith(NEW) and fields["state"] == "error"


def test_a_failed_post_exits_2():
    def post(_path, _fields):
        raise RuntimeError("HTTP 403")
    code = crv.main(["--pr", "7", "--repo", REPO], fetch=make_fetch([_pr()]), post=post)
    assert code == 2


# --- the workflow's trust boundary ------------------------------------------

WF = (ROOT / ".github" / "workflows" / "review-verdict.yml").read_text(encoding="utf-8")


def _on_block():
    m = re.search(r"^on:\n(.*?)^\S", WF, re.MULTILINE | re.DOTALL)
    assert m
    return m.group(1)


def test_workflow_triggers_cover_every_state_change():
    on = _on_block()
    assert re.search(r"^  pull_request_target:\n    types: \[([^\]]*)\]", on, re.MULTILINE)
    types = re.search(r"pull_request_target:\n    types: \[([^\]]*)\]", on)[1]
    for t in ("opened", "synchronize", "labeled", "unlabeled", "reopened"):
        assert t in types
    comment_types = re.search(r"issue_comment:\n    types: \[([^\]]*)\]", on)[1]
    for t in ("created", "edited", "deleted"):
        assert t in comment_types
    # A plain pull_request trigger would run without a write token on forks and
    # invite checking out PR code: the two write-token triggers only.
    assert set(re.findall(r"^  (\w+):", on, re.MULTILINE)) == {
        "pull_request_target", "issue_comment"}


def test_workflow_never_runs_pr_code_and_writes_only_statuses():
    assert "ref: ${{ github.event.repository.default_branch }}" in WF
    assert "persist-credentials: false" in WF
    # Nothing from the PR head is checked out or interpolated into a command.
    assert not re.search(r"github\.event\.pull_request\.head|head_ref|head\.sha", WF)
    assert "permissions: {}" in WF
    writes = re.findall(r"^\s+(\w[\w-]*): write", WF, re.MULTILINE)
    assert writes == ["statuses"]
    assert "python scripts/check_review_verdict.py --pr \"$PR_NUMBER\"" in WF
