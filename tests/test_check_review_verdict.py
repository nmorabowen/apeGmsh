"""The ``review-verdict`` commit status (lesson #2637, slice #2878).

Recorded-shape GitHub responses replay through the ``fetch`` / ``post`` seams
of ``scripts/check_review_verdict.py``. No network.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
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


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    # A retried failure must not cost the suite wall time.
    monkeypatch.setattr(crv, "_sleep", lambda _seconds: None)

OLD = "e2c3099a" + "1" * 32   # the head a review approved
NEW = "5a5b5c5d" + "2" * 32   # the head after a later push (#2628)


def _pr(number=7, labels=("program", "semantic"), head=NEW, body="", draft=False,
        fork=False, state="open"):
    return {
        "number": number, "title": f"pr {number}", "body": body, "draft": draft,
        "state": state,
        "labels": [{"name": n} for n in labels],
        "head": {"ref": f"b{number}", "sha": head,
                 "repo": {"full_name": "someone/apeGmsh" if fork else REPO}},
        "base": {"ref": "main"},
        "merged_at": "2026-11-09T12:00:00Z", "updated_at": "2026-11-09T12:00:00Z",
        "merge_commit_sha": "m" * 40,
    }


def _c(body, at="2026-11-09T10:00:00Z", assoc="OWNER"):
    return {"body": body, "created_at": at, "author_association": assoc}


def make_fetch(prs, comments=None, files=None):
    """GitHub's REST answers for ``prs``, as the live API gives them.

    Checked read-only on cli/cli (2026-11-11): ``commits/{sha}/pulls`` returned
    ``[]`` for a fork PR's head and the PR itself for a same-repo head, while
    ``pulls?state=open`` listed the fork PR with that ``head.sha``.
    """
    prs = {pr["number"]: pr for pr in prs}
    comments = comments or {}
    files = files or {}

    def fetch(path):
        m = re.fullmatch(rf"repos/{REPO}/pulls/(\d+)", path)
        if m:
            return prs.get(int(m[1]))
        if path.startswith(f"repos/{REPO}/pulls?state=closed&"):  # find_orphans' merged list
            return list(prs.values()) if path.endswith("&page=1") else []
        m = re.fullmatch(rf"repos/{REPO}/pulls\?state=open&per_page=100&page=(\d+)", path)
        if m:  # every open PR, forks included, newest first
            rows = sorted((p for p in prs.values() if p["state"] == "open"),
                          key=lambda p: -p["number"])
            page = int(m[1])
            return rows[100 * (page - 1):100 * page]
        m = re.fullmatch(rf"repos/{REPO}/commits/([0-9a-f]{{40}})/pulls\?per_page=100", path)
        if m:  # PRs (any state) at that head, but never one from a fork
            return [p for p in prs.values() if p["head"]["sha"] == m[1]
                    and p["head"]["repo"]["full_name"] == REPO]
        m = re.fullmatch(rf"repos/{REPO}/pulls/(\d+)/files\?per_page=100&page=(\d+)", path)
        if m:
            names = files.get(int(m[1]), [])
            page = int(m[2])
            return [{"filename": f} for f in names[100 * (page - 1):100 * page]]
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
    (("program", "semantic"), (), (_c(f"Review-verdict: Opus 5.5 approve {NEW[:12]}",
                                      assoc="NONE"),)),
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


@pytest.mark.parametrize("assoc", ["NONE", "CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", None])
def test_an_approve_from_outside_the_repo_leaves_it_pending(assoc):
    # #2878 card probe: the repo is public; only OWNER/MEMBER/COLLABORATOR count.
    got = _status(_pr(), [_c(f"Review-verdict: Opus 5.5 approve {NEW[:12]}", assoc=assoc)])
    assert got["state"] == "pending"


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


def test_at_merge_replay_drops_an_approval_posted_after_the_merge(capsys):
    # #2628's shape: merged 19:36:41, its head approved at 19:40:33.
    pr = {**_pr(head=OLD), "merged_at": "2026-10-30T19:36:41Z"}
    late = [_c(f"Review-verdict: opus approve {OLD}", "2026-10-30T19:40:33Z")]
    assert _main(["--pr", "7", "--dry-run"], pr, late)[0] == 0
    assert '"state": "success"' in capsys.readouterr().out
    code, posted = _main(["--pr", "7", "--dry-run", "--at-merge"], pr, late)
    assert code == 0 and posted == []
    assert '"state": "pending"' in capsys.readouterr().out
    on_time = [{**late[0], "created_at": "2026-10-30T19:30:00Z"}]
    _main(["--pr", "7", "--dry-run", "--at-merge"], pr, on_time)
    assert '"state": "success"' in capsys.readouterr().out


def test_at_merge_refuses_an_unmerged_pr_and_a_live_post():
    pr = {**_pr(), "merged_at": None}
    assert _main(["--pr", "7", "--dry-run", "--at-merge"], pr)[0] == 2
    with pytest.raises(SystemExit):
        _main(["--pr", "7", "--at-merge"], _pr())


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
    types = re.search(r"^  pull_request_target:\n(?:    #.*\n)*    types: \[([^\]]*)\]",
                      on, re.MULTILINE)[1]
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
    # Nothing from the PR head is checked out or interpolated into a command:
    # the head sha (40 hex, set by GitHub) enters through one env value only,
    # and the step refuses anything that is not 40 hex.
    head_refs = re.findall(r"^.*(?:github\.event\.pull_request\.head|head_ref).*$",
                           WF, re.MULTILINE)
    assert head_refs == ["          EVENT_SHA: ${{ github.event.pull_request.head.sha }}"]
    assert '[[ "$sha" =~ ^[0-9a-f]{40}$ ]] ||' in WF
    assert "permissions: {}" in WF
    writes = re.findall(r"^\s+(\w[\w-]*): write", WF, re.MULTILINE)
    assert writes == ["statuses"]
    assert ('python scripts/check_review_verdict.py --pr "$PR_NUMBER" --sha "$HEAD_SHA"'
            in WF)


def test_workflow_serialises_status_runs_per_head_commit():
    # #2879 review, finding 3. A status belongs to the commit, so every run that
    # can post on commit S must queue in one group: per head sha, not per PR.
    # Per PR, the runs of two PRs at S could interleave and a stale `success`
    # land last. `cancel-in-progress: false` lets the newest queued run (which
    # decides every open PR at S) finish after the earlier ones.
    assert not re.search(r"^concurrency:", WF, re.MULTILINE), "a workflow-level group"
    status_job = re.search(r"^  status:\n(.*)", WF, re.MULTILINE | re.DOTALL)[1]
    assert re.search(r"^    needs: head$", status_job, re.MULTILINE)
    block = re.search(r"^    concurrency:\n((?:      .*\n)+)", status_job, re.MULTILINE)[1]
    assert block == ("      group: review-verdict-${{ needs.head.outputs.sha }}\n"
                     "      cancel-in-progress: false\n")
    assert "HEAD_SHA: ${{ needs.head.outputs.sha }}" in status_job
    head_job = re.search(r"^  head:\n(.*?)^  status:", WF, re.MULTILINE | re.DOTALL)[1]
    assert "sha: ${{ steps.head.outputs.sha }}" in head_job
    assert re.search(r"^      pull-requests: read", head_job, re.MULTILINE)
    assert not re.search(r": write", head_job)


def _head_step_script():
    m = re.search(r"^      - id: head\n(?:        [^\n]*\n)*?        run: \|\n"
                  r"((?:          [^\n]*\n)+)", WF, re.MULTILINE)
    assert m, "head step not found"
    script = "\n".join(line[10:] for line in m.group(1).splitlines())
    assert script.endswith('echo "sha=$sha" >> "$GITHUB_OUTPUT"')
    return script


def _run_head_step(tmp_path, failures, event_sha=""):
    """The head step under ``bash -eo pipefail`` (GitHub's shell); ``gh`` fails
    ``failures`` times and then prints NEW, ``sleep`` only records."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    tmp_path.mkdir(parents=True, exist_ok=True)
    out, log = tmp_path / "gh_output", tmp_path / "log"
    out.write_text("")
    log.write_text("")
    # `gh` runs inside $(...), a subshell: it counts its calls in the log file.
    prelude = (
        f'gh() {{ echo "gh $*" >> log; '
        f'if [[ $(grep -c "^gh " log) -le {failures} ]]; then echo "HTTP 502" >&2; return 1; fi; '
        f'echo {NEW}; }}\n'
        f'sleep() {{ echo "sleep $1" >> log; }}\n'
    )
    env = dict(os.environ, GITHUB_OUTPUT="gh_output", REPO=REPO, PR_NUMBER="7",
               EVENT_SHA=event_sha, GH_TOKEN="x")
    proc = subprocess.run([bash, "-eo", "pipefail", "-c", prelude + _head_step_script()],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          check=False)
    return proc.returncode, out.read_text().strip(), log.read_text().split("\n")[:-1]


def test_workflow_head_read_is_retried_with_backoff(tmp_path):
    # #2879 second review, finding 2: a comment event's head read fails twice
    # and then succeeds; the job still pins the head.
    code, output, log = _run_head_step(tmp_path / "a", failures=2)
    assert code == 0 and output == f"sha={NEW}"
    assert [line for line in log if line.startswith("sleep")] == [
        "sleep 0", "sleep 5", "sleep 15"]
    assert sum(line.startswith("gh api repos/") for line in log) == 3
    # Always failing: three tries, then the job fails red as before.
    code, output, log = _run_head_step(tmp_path / "b", failures=99)
    assert code == 1 and output == ""
    assert sum(line.startswith("gh api") for line in log) == 3
    # A PR event names its head: no read at all.
    code, output, log = _run_head_step(tmp_path / "c", failures=99, event_sha=OLD)
    assert code == 0 and output == f"sha={OLD}" and log == []


def test_workflow_reruns_when_a_pr_leaves_a_shared_head():
    types = re.search(r"pull_request_target:\n(?:    #.*\n)*    types: \[([^\]]*)\]",
                      _on_block())[1]
    assert "closed" in [t.strip() for t in types.split(",")]


# --- a status belongs to the commit (#2879 review, finding 1) ----------------

APPROVE_NEW = f"Review-verdict: Opus 5.5 approve {NEW[:12]}"


def _post_runs(order, prs, comments, close=()):
    """Run ``main`` once per PR number in ``order``; return the posts on NEW."""
    posted = []
    prs = {p["number"]: p for p in prs}
    for n in close:
        prs[n] = {**prs[n], "state": "closed"}
    fetch = make_fetch(list(prs.values()), comments)
    for number in order:
        code = crv.main(["--pr", str(number), "--repo", REPO], fetch=fetch,
                        post=lambda path, fields: posted.append((path, fields)))
        assert code == 0
    assert all(path == f"repos/{REPO}/statuses/{NEW}" for path, _ in posted)
    return [fields for _, fields in posted]


@pytest.mark.parametrize("order", [(7, 8), (8, 7)])
def test_an_unlabelled_pr_on_the_same_head_cannot_post_success_over_a_waiting_one(order):
    # The refuting probe: PR 7 (semantic, no approval) and PR 8 (no `program`
    # label) share head NEW. Whichever runs last, the commit reads pending, and
    # the description names the PR that waits. Fails with finding 1 reverted.
    prs = [_pr(7), _pr(8, labels=())]
    posted = _post_runs(order, prs, {})
    assert [f["state"] for f in posted] == ["pending", "pending"]
    assert posted[-1]["description"].startswith("#7: ")
    assert NEW[:12] in posted[-1]["description"]


@pytest.mark.parametrize("order", [(7, 8), (8, 7)])
def test_both_prs_on_the_head_passing_is_success(order):
    prs = [_pr(7), _pr(8, labels=())]
    posted = _post_runs(order, prs, {7: [_c(APPROVE_NEW)]})
    assert [f["state"] for f in posted] == ["success", "success"]
    assert "#7" in posted[-1]["description"] and "#8" in posted[-1]["description"]


def test_an_approval_on_the_other_pr_does_not_count_for_this_one():
    # Verdicts are read per PR: an approve on #8's thread does not clear #7.
    prs = [_pr(7), _pr(8, labels=("program", "semantic"))]
    posted = _post_runs((7, 8), prs, {8: [_c(APPROVE_NEW)]})
    assert [f["state"] for f in posted] == ["pending", "pending"]
    assert posted[-1]["description"].startswith("#7: ")


def test_closing_the_unlabelled_pr_keeps_the_waiting_one_pending():
    # The `closed` event of #8 re-decides #7, still open on the same head.
    posted = _post_runs((8,), [_pr(7), _pr(8, labels=())], {}, close=(8,))
    assert [f["state"] for f in posted] == ["pending"]
    assert posted[0]["description"].startswith("#7: ")


def test_closing_the_waiting_pr_frees_the_head_for_the_other():
    posted = _post_runs((7,), [_pr(7), _pr(8, labels=())], {}, close=(7,))
    assert [f["state"] for f in posted] == ["success"]
    assert "not required" in posted[0]["description"]


def test_a_closed_pr_alone_on_its_head_posts_nothing(capsys):
    assert _post_runs((7,), [_pr(7)], {}, close=(7,)) == []
    assert "nothing to post" in capsys.readouterr().out


def test_a_pr_whose_head_moved_past_the_pinned_sha_is_not_decided_there():
    # The event pinned NEW; the PR has since moved to OLD (its synchronize run
    # decides OLD). No open PR is left at NEW, so nothing is posted on it.
    posted = []
    code = crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW],
                    fetch=make_fetch([_pr(7, head=OLD)]),
                    post=lambda path, fields: posted.append((path, fields)))
    assert code == 0 and posted == []


def test_the_pinned_sha_is_where_the_status_goes():
    posted = []
    code = crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW],
                    fetch=make_fetch([_pr(7), _pr(8, labels=())]),
                    post=lambda path, fields: posted.append((path, fields)))
    assert code == 0
    (path, fields), = posted
    assert path == f"repos/{REPO}/statuses/{NEW}" and fields["state"] == "pending"


@pytest.mark.parametrize("bad", ["e2c3099a", NEW.upper(), NEW + "0", "x" * 40])
def test_sha_must_be_a_full_commit(bad):
    with pytest.raises(SystemExit):
        crv.main(["--pr", "7", "--repo", REPO, "--sha", bad], fetch=make_fetch([_pr()]),
                 post=lambda *_: None)


def test_sha_is_live_only():
    with pytest.raises(SystemExit):
        crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW, "--dry-run", "--head", OLD],
                 fetch=make_fetch([_pr()]), post=lambda *_: None)


def test_a_dry_run_of_a_closed_pr_replays_it_alone(capsys):
    fetch = make_fetch([_pr(7, state="closed")], {7: [_c(APPROVE_NEW)]})
    assert crv.main(["--pr", "7", "--repo", REPO, "--dry-run"], fetch=fetch,
                    post=lambda *_: None) == 0
    out = capsys.readouterr()
    assert '"state": "success"' in out.out and "replayed alone" in out.err


# --- fail closed on a read failure (#2879 review, finding 2) -----------------

def _failing(fetch, fragment):
    def wrapped(path):
        if fragment in path:
            raise RuntimeError(f"HTTP 502 on {path}")
        return fetch(path)
    return wrapped


@pytest.mark.parametrize("fragment", [
    "pulls/7",                         # the PR itself
    "pulls?state=open",                # the PRs at the head
    "issues/7/comments",               # the verdicts
    "pulls/8/files",                   # a sibling's changed files
])
def test_a_failed_read_posts_error_on_the_pinned_head(fragment):
    # Always failing: every retry fails too, and the check fails closed.
    prs = [_pr(7), _pr(8, labels=("program",))]
    fetch = _failing(make_fetch(prs, {7: [_c(APPROVE_NEW)]}), fragment)
    posted, slept = [], []
    code = crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW], fetch=fetch,
                    post=lambda path, fields: posted.append((path, fields)),
                    sleep=slept.append)
    assert code == 2
    (path, fields), = posted
    assert path == f"repos/{REPO}/statuses/{NEW}" and fields["state"] == "error"
    assert slept == list(crv.RETRY_DELAYS)  # three tries, then the error


def test_a_failed_read_without_a_pinned_head_posts_nothing_and_fails():
    posted = []
    code = crv.main(["--pr", "7", "--repo", REPO],
                    fetch=_failing(make_fetch([_pr()]), "pulls/7"),
                    post=lambda path, fields: posted.append((path, fields)))
    assert code == 2 and posted == []


def test_a_missing_or_cut_open_pr_listing_fails_closed():
    def no_listing(path):
        return None if "pulls?state=open" in path else make_fetch([_pr()])(path)
    posted = []
    assert crv.main(["--pr", "7", "--repo", REPO], fetch=no_listing,
                    post=lambda p, f: posted.append(f)) == 2
    assert [f["state"] for f in posted] == ["error"]
    # Every page up to the cap is full: PRs past it are unread.
    cap = crv.PULLS_PAGE * crv.PULLS_MAX_PAGES
    many = [_pr(n, labels=(), head=OLD) for n in range(100, 100 + cap - 1)] + [_pr(7)]
    posted.clear()
    assert crv.main(["--pr", "7", "--repo", REPO], fetch=make_fetch(many),
                    post=lambda p, f: posted.append(f)) == 2
    assert [f["state"] for f in posted] == ["error"]
    # One PR fewer is read whole: #7 alone is at NEW, and it waits.
    posted.clear()
    assert crv.main(["--pr", "7", "--repo", REPO], fetch=make_fetch(many[1:]),
                    post=lambda p, f: posted.append(f)) == 0
    assert [f["state"] for f in posted] == ["pending"]


def test_a_failed_post_with_a_pinned_head_exits_2():
    calls = []

    def post(_path, _fields):
        calls.append(1)
        raise RuntimeError("HTTP 403")
    assert crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW],
                    fetch=make_fetch([_pr()]), post=post) == 2
    assert len(calls) == 1 + len(crv.RETRY_DELAYS)


# --- a transient failure is retried (#2879 second review, finding 2) ----------

def _flaky(call, fragment, failures=2):
    """``call``, failing its first ``failures`` calls whose path holds ``fragment``."""
    left = [failures]

    def wrapped(path, *rest):
        if fragment in path and left[0]:
            left[0] -= 1
            raise RuntimeError(f"HTTP 502 on {path}")
        return call(path, *rest)
    return wrapped


@pytest.mark.parametrize("fragment", [
    "pulls/7", "pulls?state=open", "issues/7/comments", "pulls/8/files"])
def test_a_read_that_fails_twice_then_succeeds_decides_normally(fragment):
    # #7 is approved at NEW and #8 is a program PR off the policy paths, so the
    # head passes; a read that recovers on its third try must not post error.
    prs = [_pr(7), _pr(8, labels=("program",))]
    fetch = _flaky(make_fetch(prs, {7: [_c(APPROVE_NEW)]}, {8: ["docs/x.md"]}), fragment)
    posted, slept = [], []
    code = crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW], fetch=fetch,
                    post=lambda path, fields: posted.append((path, fields)),
                    sleep=slept.append)
    assert code == 0
    (path, fields), = posted
    assert path == f"repos/{REPO}/statuses/{NEW}" and fields["state"] == "success"
    assert slept == list(crv.RETRY_DELAYS)


def test_a_post_that_fails_twice_then_succeeds_posts_once():
    posted, slept = [], []
    post = _flaky(lambda path, fields: posted.append((path, fields)), "statuses/")
    code = crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW],
                    fetch=make_fetch([_pr()]), post=post, sleep=slept.append)
    assert code == 0 and slept == list(crv.RETRY_DELAYS)
    (path, fields), = posted
    assert path.endswith(NEW) and fields["state"] == "pending"


def test_a_404_is_an_answer_not_retried():
    slept = []
    assert crv.main(["--pr", "7", "--repo", REPO], fetch=make_fetch([]),
                    post=lambda *_: None, sleep=slept.append) == 2
    assert slept == []


# --- fork PRs share heads too (#2879 second review, findings 1 and 4) ---------

@pytest.mark.parametrize("pinned", [True, False])
def test_a_waiting_fork_pr_on_the_head_keeps_it_pending(pinned):
    # The refuting probe: fork PR A (#7, semantic, no approval) and same-repo
    # PR B (#8, no `program` label) share head NEW. B's event decides NEW: the
    # stub, like GitHub, leaves A out of commits/NEW/pulls, so a check that
    # reads that endpoint posts success over A. Fails with the fix reverted.
    prs = [_pr(7, fork=True), _pr(8, labels=())]
    posted = []
    code = crv.main(["--pr", "8", "--repo", REPO] + (["--sha", NEW] if pinned else []),
                    fetch=make_fetch(prs),
                    post=lambda path, fields: posted.append((path, fields)))
    assert code == 0
    (path, fields), = posted
    assert path == f"repos/{REPO}/statuses/{NEW}"
    assert fields["state"] == "pending" and fields["description"].startswith("#7: ")


def test_an_open_pr_on_another_head_is_not_decided_here(capsys):
    # Kills the mutant "drop the head.sha == sha filter": #9 waits at OLD, and
    # its verdict has nothing to do with commit NEW, where #7 needs none.
    prs = [_pr(7, labels=()), _pr(9, head=OLD)]
    posted = []
    assert crv.main(["--pr", "7", "--repo", REPO, "--sha", NEW], fetch=make_fetch(prs),
                    post=lambda path, fields: posted.append((path, fields))) == 0
    (path, fields), = posted
    assert path.endswith(NEW) and fields["state"] == "success"
    assert '"prs": [\n    7\n  ]' in capsys.readouterr().out


# --- a cut file listing fails closed (#2879 review, finding 5) ---------------

def test_a_file_listing_cut_at_the_page_cap_requires_a_verdict():
    # 2,000 prose files fill every page; the path that needs a verdict is not
    # read. The policy must not read the PR as "not required".
    cap = fo.PER_PAGE * fo.MAX_PAGES
    names = [f"docs/p{i}.md" for i in range(cap)] + ["tests/test_late.py"]
    pr = _pr(labels=("program", "mechanical"))
    assert _status(pr, files=names)["state"] == "pending"
    # One page short of the cap is read whole, and prose alone is not required.
    assert _status(pr, files=names[:cap - 1])["state"] == "success"
    assert _status(pr, files=names[:cap - 1] + ["tests/t.py"])["state"] == "pending"
