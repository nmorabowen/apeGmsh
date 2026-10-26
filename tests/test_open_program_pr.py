"""Offline tests for ``scripts/open_program_pr.py`` (lesson #2200, slice #2211).

``gh`` is a fake: a function over the argument list. The open set is built from
the ``gh pr list`` JSON shape that ``program_probe`` reads.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


op = _load("open_program_pr")
pp = op._probe

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
HEAD = "c" * 40


def _iso(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pr(n, *, gate=False, green=False, approved=False, decision="", created=3.0):
    checks = [{"__typename": "CheckRun", "name": c, "status": "COMPLETED",
               "conclusion": "SUCCESS" if green else "FAILURE",
               "startedAt": _iso(2), "completedAt": _iso(1.9)}
              for c in pp.REQUIRED_CHECKS]
    comments = ([{"body": f"Review-verdict: Opus approve {HEAD}", "createdAt": _iso(1)}]
                if approved else [])
    return {"number": n, "title": f"pr {n}", "createdAt": _iso(created),
            "updatedAt": _iso(0.5), "isDraft": False, "headRefOid": HEAD,
            "body": "Slice: #1", "comments": comments, "statusCheckRollup": checks,
            "closingIssuesReferences": [], "reviewDecision": decision,
            "labels": [{"name": "program"}] + ([{"name": "human-gate"}] if gate else [])}


class FakeGh:
    def __init__(self, prs=(), fail_list=False, fail_create=False, remote=("slice-branch",)):
        self.prs, self.fail_list, self.fail_create = list(prs), fail_list, fail_create
        self.remote = set(remote)  # branches that exist on the remote
        self.created = []

    def __call__(self, args):
        if args[:2] == ["pr", "list"]:
            if self.fail_list:
                raise RuntimeError("gh pr list failed: network")
            return json.dumps(self.prs)
        if args[:2] == ["pr", "create"]:
            if self.fail_create:
                raise RuntimeError("gh pr create failed")
            self.created.append(args)
            return "https://github.com/x/y/pull/9\n"
        if args[0] == "api":
            branch = args[1].split("/git/ref/heads/", 1)[1]
            if branch not in self.remote:
                raise RuntimeError("gh api failed: HTTP 404 Not Found")
            return "{}"
        raise AssertionError(args)


def fake_git(branch="slice-branch"):
    def git(args):
        assert args == ["rev-parse", "--abbrev-ref", "HEAD"], args
        return branch + "\n"
    return git


@pytest.fixture
def body(tmp_path):
    f = tmp_path / "body.md"
    f.write_text("Slice: #2211\nAuthor-model: Sonnet\n", encoding="utf-8")
    return f


def _open(gh, body, *extra, git=None):
    return op.main(["--title", "t", "--body-file", str(body), *extra], gh=gh,
                   git=git or fake_git())


def test_seven_counted_opens(body):
    gh = FakeGh([_pr(n) for n in range(7)])
    assert _open(gh, body) == 0
    cmd = gh.created[0]
    assert cmd[cmd.index("--base") + 1] == "main"
    assert cmd.count("--label") == 1 and "program" in cmd


def test_eight_counted_refuses(body, capsys):
    gh = FakeGh([_pr(n) for n in range(8)])
    assert _open(gh, body) == 1
    assert gh.created == []
    assert "WIP is full" in capsys.readouterr().err


def test_eight_open_one_gated_approved_green_opens(body):
    gh = FakeGh([_pr(n) for n in range(7)] + [_pr(99, gate=True, green=True, approved=True)])
    assert _open(gh, body) == 0


def test_stale_approval_does_not_exclude():
    pr = _pr(1, gate=True, green=True, approved=True)
    pr["headRefOid"] = "d" * 40  # new push after the approval
    pr["reviewDecision"] = "APPROVED"
    counted, excluded = pp.wip_count([pr])
    assert len(counted) == 1 and not excluded


@pytest.mark.parametrize("kw", [
    dict(gate=True, green=False, approved=True),   # red
    dict(gate=True, green=True, approved=False),   # unapproved
    dict(gate=False, green=True, approved=True),   # approved and green, but no human-gate
])
def test_other_prs_still_count(kw, body):
    gh = FakeGh([_pr(n) for n in range(7)] + [_pr(99, **kw)])
    assert _open(gh, body) == 1


def test_body_without_slice_refuses(tmp_path):
    f = tmp_path / "b.md"
    f.write_text("no slice here\n", encoding="utf-8")
    gh = FakeGh([])
    assert _open(gh, f) == 1 and gh.created == []


def test_failed_gh_read_refuses_and_count_too(body):
    gh = FakeGh(fail_list=True)
    assert _open(gh, body) == 1 and gh.created == []
    assert op.main(["--count"], gh=gh) == 1


def test_failed_create_is_nonzero(body):
    assert _open(FakeGh(fail_create=True), body) == 1


def test_count_prints_each_pr_with_reason_and_exits_zero(capsys):
    gh = FakeGh([_pr(1), _pr(2, gate=True, green=True, approved=True)])
    assert op.main(["--count"], gh=gh) == 0
    out = capsys.readouterr().out
    assert "WIP count: 1 of 8" in out
    assert "counted  #1" in out and "excluded #2" in out


def test_replay_of_2200_moment(body):
    """Real open set at 2026-10-19T03:16:02Z: 9 open; #2197 and #2186 human-gate, green, approved."""
    gated = [_pr(2197, gate=True, green=True, approved=True),
             _pr(2186, gate=True, green=True, approved=True)]
    others = [_pr(n) for n in (2199, 2198, 2190, 2184, 2180, 2179, 2174)]
    prs = gated + others
    assert len(prs) == 9  # the old rule: every open program PR
    counted, excluded = pp.wip_count(prs)
    assert (len(counted), len(excluded)) == (7, 2)
    assert _open(FakeGh(prs), body) == 0


def test_bom_crlf_body_is_accepted(tmp_path):
    f = tmp_path / "b.md"
    f.write_bytes("\ufeffSlice: #2211\r\nAuthor-model: x\r\n".encode("utf-8"))
    gh = FakeGh([])
    assert _open(gh, f) == 0 and len(gh.created) == 1


def test_cp1252_console_with_arrow_title(monkeypatch):
    import io
    import sys
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", out)
    pr = _pr(1)
    pr["title"] = "A \u2192 B"
    assert op.main(["--count"], gh=FakeGh([pr])) == 0
    out.flush()


def _head_of(gh):
    cmd = gh.created[0]
    return cmd[cmd.index("--head") + 1]


def test_head_flag_is_passed_even_when_checkout_is_another_branch(body):
    """Lesson #2325, occurrence 2: the worktree holds a different branch than the slice's."""
    gh = FakeGh([], remote=("k-2195-twin",))
    assert _open(gh, body, "--head", "k-2195-twin", git=fake_git("worktree-agent-x")) == 0
    assert _head_of(gh) == "k-2195-twin"


def test_head_derived_from_current_branch(body):
    gh = FakeGh([])
    assert _open(gh, body) == 0 and _head_of(gh) == "slice-branch"


def test_detached_head_refuses_by_name(body, capsys):
    gh = FakeGh([])
    assert _open(gh, body, git=fake_git("HEAD")) == 1
    assert gh.created == [] and "detached" in capsys.readouterr().err


@pytest.mark.parametrize("extra,cur", [(("--head", "nope"), "slice-branch"), ((), "unpushed")])
def test_unpushed_branch_refuses_naming_it(body, capsys, extra, cur):
    """Lesson #2325, occurrence 1: the head is not on the remote."""
    gh = FakeGh([])
    assert _open(gh, body, *extra, git=fake_git(cur)) == 1
    assert gh.created == []
    err = capsys.readouterr().err
    assert (extra[1] if extra else cur) in err and "not on the remote" in err
