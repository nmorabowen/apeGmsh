"""scripts/land_pr.py replays: the lost-work PRs are refused or flagged, a clean PR lands.

The recordings are what `gh pr view --json ...`, `gh api .../pulls/N/files`,
`gh api .../compare` and `git ls-remote` returned for the real PRs, trimmed
to the fields the script reads. Nothing here needs `gh` or a network: both
seams are replaced by `Replay`, which fails on any call that was not
recorded.

- #858 and #1057 merged into a stacked base (#858 reached main 62 days later,
  in #1169): refused at check 1.
- #1097's SHAs (merged head `4b8fdfdb`, squash `740107cc`, the commit that
  stayed on its branch, `a7237433`) exercise the post-merge race path: a
  branch tip that is not the merged head is flagged. (As it happened, #1097
  was pushed to after its landing; that is the nightly orphan detector's
  job, #1230, not this script's.)
- #1218 (A1.2): every check passes, the merge runs, compare reads `behind`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "land_pr.py"


def _load():
    spec = importlib.util.spec_from_file_location("land_pr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


land_pr = _load()


@pytest.fixture(autouse=True)
def _as_mains_copy(monkeypatch):
    """Run `main` in-process as if it were origin/main's copy (no re-exec); the
    ruling-2 tests at the bottom drop this to exercise the hop itself."""
    monkeypatch.setenv(land_pr.MAIN_COPY_ENV, os.path.abspath(land_pr.__file__))


class Replay:
    """Recorded responses keyed by the joined argv; refuses an unrecorded call."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, args: list[str]) -> str:
        key = " ".join(args)
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unrecorded call: {key}")
        return self.responses[key]


#: The Actions run every `_rollup` check belongs to (check 11 reads its job log), and when it started.
RUN_ID = "38000000001"
STARTED = "2026-10-13T10:00:00Z"


JOB_IDS = {name: 9000 + i for i, name in enumerate((*land_pr.REQUIRED_CHECKS, "qt-window-tests", "build"))}


def _job(name, run=RUN_ID):
    return f"https://github.com/nmorabowen/apeGmsh/actions/runs/{run}/job/{JOB_IDS[name]}"


def _rollup(names=land_pr.REQUIRED_CHECKS, **overrides):
    runs = [{"__typename": "CheckRun", "name": n, "status": "COMPLETED", "conclusion": "SUCCESS",
             "startedAt": STARTED, "detailsUrl": _job(n)} for n in names]
    return [dict(run, **overrides.get(run["name"], {})) for run in runs]


def _context(name, state):
    return {"__typename": "StatusContext", "context": name, "state": state}


# gh pr view <n> --json ..., as each PR stood before its merge.
PR_858 = {
    "state": "OPEN", "isDraft": False,
    "baseRefName": "claude/apegmsh-facet-extractor-bug-e37f5c",
    "headRefName": "claude/apegmsh-2d-finite-strain",
    "headRefOid": "7c05d2a8b927815c825b70448abb2de2b97e24cb",
    "mergeable": "MERGEABLE", "statusCheckRollup": [],
}
PR_1057 = {
    "state": "OPEN", "isDraft": False,
    "baseRefName": "claude/adr-0098-derived-contour",
    "headRefName": "claude/adr-0098-shell-thickness",
    "headRefOid": "04de26dd42da83756d1f015d0e6b83e6aaeef42d",
    "mergeable": "MERGEABLE", "statusCheckRollup": [],
}
PR_1097 = {
    "state": "OPEN", "isDraft": False, "baseRefName": "main",
    "headRefName": "claude/ladruno-sanisand-integrator-guide-adb316",
    "headRefOid": "4b8fdfdb7121405e0d6523669196d508a32026d3",
    "mergeable": "MERGEABLE", "statusCheckRollup": _rollup(),
}
FILES_1097 = ["src/apeGmsh/opensees/_internal/ns/nd.py", "src/apeGmsh/studio/_api_index.json"]
SQUASH_1097 = "740107cc2cb361f93b581b64bc405cd1f7d3945f"
ORPHAN_1097 = "a7237433ab270c4280735f29242e4a8d5b00554b"
PR_1218 = {
    "state": "OPEN", "isDraft": False, "baseRefName": "main",
    "headRefName": "guppi/vibrant-meitner-w7x8bs-a1-apiindex",
    "headRefOid": "704e2bed841c1866e54ddc193d31f1c1320a5295",
    "mergeable": "MERGEABLE",
    "statusCheckRollup": _rollup(names=(*land_pr.REQUIRED_CHECKS, "qt-window-tests", "build")),
}
FILES_1218 = [
    "CHANGELOG.md", "src/apeGmsh/studio/_api_index.json",
    "src/apeGmsh/studio/_index_build.py", "tests/studio/test_lookup.py",
]
SQUASH_1218 = "21de232c633892deb7d5e38da3fbd3af207c7617"
COMPARE = "api repos/{owner}/{repo}/compare/main..."  # the post-merge proof, not check 11
MAIN_COMMITS_CALL = f"api {land_pr.MAIN_COMMITS} --jq {land_pr.MAIN_COMMITS_JQ}"
MAIN_RUNS_CALL = f"api {land_pr.MAIN_RUNS} --jq {land_pr.MAIN_RUNS_JQ}"
MAIN_TIP = "6f66fd2383255cc7d19daed275ce7665e0b2853a"
FRESH_BASE = "f" * 40
LABELS = [{"name": "program"}, {"name": "slice"}, {"name": "semantic"}]
FREEZE_STUDIO = [*LABELS, {"name": "freeze:src/apeGmsh/studio/*"}]


def _files_call(number):
    """The one files listing checks 5, 7 and 8 read (#1775, #1686)."""
    return f"api repos/{{owner}}/{{repo}}/pulls/{number}/files --paginate --jq {land_pr.FILES_JQ}"


def _row(path, status="modified", previous=None, patch="@@ -1 +1 @@\n-a\n+b"):
    """One `gh api .../pulls/N/files --jq FILES_JQ` line: [filename, status, previous, changes, patch]."""
    changes = 0 if patch is None else sum(ln[:1] in "+-" for ln in patch.splitlines() if not ln.startswith("@@"))
    return json.dumps([path, status, previous, changes, patch])


def _run_row(run_id, sha, status="completed", conclusion="success"):
    """One row of `gh api MAIN_RUNS --jq MAIN_RUNS_JQ`: main's Tests push runs, newest first."""
    return {"id": run_id, "head_sha": sha, "status": status, "conclusion": conclusion}


#: By default main's head is MAIN_TIP and its Tests run there is completed and green.
MAIN_GREEN = [_run_row(38373760364, MAIN_TIP)]


def _ancestry_call(tip, base):
    """Check 10's question: does the checks' base contain main's head (ahead or identical)?"""
    return f"api repos/{{owner}}/{{repo}}/compare/{tip}...{base} --jq .status"


def _checkout(head, base=FRESH_BASE):
    """The job-log line actions/checkout prints for a pull_request run (check 11 reads it)."""
    return f"2026-10-13T10:00:05.0000000Z HEAD is now at 54b621b Merge {head} into {base}\n"


def _log_call(job_url):
    return f"api --allow-escape-sequences repos/{{owner}}/{{repo}}/actions/jobs/{job_url.rsplit('/', 1)[1]}/logs"


def _behind_call(base=FRESH_BASE):
    return f"api repos/{{owner}}/{{repo}}/compare/{base}...main --jq {land_pr.BEHIND_JQ}"


def _recording(number, pr, squash, files=FILES_1218, *, compare="behind", tip="", labels=LABELS, local=None,
               porcelain="", rows=None, issues=None, shows=None, cards=None, main=MAIN_GREEN, main_order=(MAIN_TIP,),
               logs=None, behind=(0, None)):
    pr = {"labels": [], "body": "", "closingIssuesReferences": [], **pr}
    head = pr["headRefOid"]
    if rows is None:
        rows = [_row(f) for f in files]
    gh = Replay({
        f"pr view {number} --json {land_pr.PR_FIELDS}": json.dumps(pr),
        "label list --limit 200 --json name": json.dumps(labels),
        _files_call(number): "".join(r + "\n" for r in rows),
        **{
            f"issue view {n} --json labels": json.dumps({"labels": [{"name": lab} for lab in names]})
            for n, names in (issues or {}).items()
        },
        **{f"issue view {n} --json body,comments": json.dumps(card) for n, card in (cards or {}).items()},
        f"pr merge {number} --squash --match-head-commit {head}": "",
        f"pr view {number} --json mergeCommit": json.dumps({"mergeCommit": {"oid": squash}}),
        f"api repos/{{owner}}/{{repo}}/compare/main...{squash} --jq .status": compare + "\n",
        MAIN_COMMITS_CALL: json.dumps(list(main_order)),
        MAIN_RUNS_CALL: json.dumps(main),
        **{
            _log_call(run["detailsUrl"]): _checkout(head)
            for run in pr["statusCheckRollup"] if "/actions/runs/" in (run.get("detailsUrl") or "")
        },
        **(logs or {}),
        _behind_call(): json.dumps(list(behind)),
    })
    git = Replay({
        "rev-parse HEAD": (local or head) + "\n",
        "status --porcelain": porcelain,
        f"ls-remote origin refs/heads/{pr['headRefName']}": tip,
        **{f"show {head}:{path}": text for path, text in (shows or {}).items()},
    })
    return gh, git


def _run(number, gh, git, **kw):
    return land_pr.land(number, gh, git, sleep=lambda _s: None, **kw)


def _merged(gh: Replay) -> bool:
    return any(call.startswith("pr merge") for call in gh.calls)


@pytest.mark.parametrize("number, pr", [(858, PR_858), (1057, PR_1057)])
def test_stacked_base_is_refused_at_check_1(number, pr):
    gh, git = _recording(number, pr, "unused")
    with pytest.raises(land_pr.LandError, match=r"check 1: .*not main.*#858"):
        _run(number, gh, git)
    assert not _merged(gh)
    assert git.calls == [], "check 1 refuses before anything local is read"


def test_push_racing_the_merge_is_flagged_1097_shas():
    tip = f"{ORPHAN_1097}\trefs/heads/{PR_1097['headRefName']}\n"
    gh, git = _recording(1097, PR_1097, SQUASH_1097, FILES_1097, tip=tip)
    with pytest.raises(land_pr.LandError, match=rf"now reads {ORPHAN_1097[:12]}.*raced the merge"):
        _run(1097, gh, git)
    assert _merged(gh), "the race is caught after a real merge"


def test_clean_landing_1218(capsys):
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    assert _run(1218, gh, git) == 0
    merge = f"pr merge 1218 --squash --match-head-commit {PR_1218['headRefOid']}"
    assert merge in gh.calls, "the merge pins the head the checks saw"
    assert not any("--auto" in c or "--delete-branch" in c for c in gh.calls)
    assert gh.calls.index(merge) > gh.calls.index(_files_call(1218))
    out = capsys.readouterr().out
    assert f"landed #1218 as {SQUASH_1218[:12]}" in out
    assert f"{PR_1218['headRefName']} is gone" in out


def test_dry_run_checks_and_never_merges(capsys):
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    assert _run(1218, gh, git, dry_run=True) == 0
    assert not _merged(gh)
    assert "dry-run" in capsys.readouterr().out


def test_check_5_reads_the_diff_not_the_100_file_view():
    """`gh pr view --json files` stops at 100 entries; the frozen file is the 101st."""
    view = {**PR_1218, "files": [{"path": f"src/apeGmsh/opensees/f{i}.py"} for i in range(100)]}
    files = [f["path"] for f in view["files"]] + ["src/apeGmsh/studio/_api_index.json"]
    gh, git = _recording(1218, view, SQUASH_1218, files, labels=FREEZE_STUDIO)
    with pytest.raises(land_pr.LandError, match=r"check 5: .*_api_index\.json.*freeze:"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_a_failed_run_before_a_green_rerun_refuses():
    """The rollup keeps every run of a name; a later SUCCESS does not hide a FAILURE."""
    history = [*_rollup(names=("suite",), suite={"conclusion": "FAILURE"}), *PR_1218["statusCheckRollup"]]
    gh, git = _recording(1218, {**PR_1218, "statusCheckRollup": history}, SQUASH_1218)
    with pytest.raises(land_pr.LandError, match=r"check 6: .*'suite' is FAILURE"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_status_context_success_is_green():
    """Check 6 reads a StatusContext's state. The PR touches no tests/, so check 11, which
    needs an Actions job for every required check, does not apply."""
    rollup = [_context("lock-tests", "SUCCESS"), *_rollup(names=land_pr.REQUIRED_CHECKS[1:])]
    gh, git = _recording(1218, {**PR_1218, "statusCheckRollup": rollup}, SQUASH_1218, FILES_1097)
    assert _run(1218, gh, git) == 0
    assert _merged(gh)


# Each refusal: (overrides to the #1218 recording, the message it must carry).
REFUSALS = {
    "2-draft": (dict(pr={"isDraft": True}), r"check 2: .*draft"),
    "2-merged": (dict(pr={"state": "MERGED"}), r"check 2: .*already merged.*#335"),
    "3-local-head": (dict(local="0" * 40), r"check 3: .*#555"),
    "4-dirty-tree": (dict(porcelain=" M src/x.py\n"), r"check 4"),
    "5-freeze-label": (dict(labels=FREEZE_STUDIO), r"check 5: .*_api_index\.json.*freeze:"),
    "6-conflicting": (dict(pr={"mergeable": "CONFLICTING"}), r"check 6: .*CONFLICTING, not MERGEABLE"),
    "6-unknown": (dict(pr={"mergeable": "UNKNOWN"}), r"check 6: .*UNKNOWN, not MERGEABLE"),
    "6-pending": (dict(pr={"statusCheckRollup": _rollup(suite={"status": "IN_PROGRESS", "conclusion": None})}), r"check 6: .*'suite' is IN_PROGRESS"),
    "6-failed": (dict(pr={"statusCheckRollup": _rollup(**{"live-stock": {"conclusion": "FAILURE"}})}), r"check 6: .*'live-stock' is FAILURE"),
    "6-missing": (dict(pr={"statusCheckRollup": _rollup(names=land_pr.REQUIRED_CHECKS[:-1])}), r"check 6: .*'live-stock' has not reported"),
    "6-status-context-pending": (
        dict(pr={"statusCheckRollup": [_context("lock-tests", "PENDING"), *_rollup(names=land_pr.REQUIRED_CHECKS[1:])]}),
        r"check 6: .*'lock-tests' is PENDING",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_check_refuses(case):
    overrides, message = REFUSALS[case]
    overrides = dict(overrides)
    pr = {**PR_1218, **overrides.pop("pr", {})}
    gh, git = _recording(1218, pr, SQUASH_1218, **overrides)
    with pytest.raises(land_pr.LandError, match=message):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_diverged_compare_fails_after_merge():
    gh, git = _recording(1218, PR_1218, SQUASH_1218, compare="diverged")
    with pytest.raises(land_pr.LandError, match=r"diverged.*#858"):
        _run(1218, gh, git)


def test_compare_retry_is_bounded():
    gh, git = _recording(1218, PR_1218, SQUASH_1218, compare="ahead")
    with pytest.raises(land_pr.LandError, match=rf"after {land_pr.COMPARE_TRIES} tries"):
        _run(1218, gh, git)
    assert sum(c.startswith(COMPARE) for c in gh.calls) == land_pr.COMPARE_TRIES


def test_main_reports_refusal_on_stderr(monkeypatch, capsys):
    gh, git = _recording(858, PR_858, "unused")
    monkeypatch.setattr(land_pr, "run_gh", gh)
    monkeypatch.setattr(land_pr, "run_git", git)
    assert land_pr.main(["858"]) == 1
    assert "land_pr: check 1" in capsys.readouterr().err


# --- #1682: post-merge reads are retried; --verify-only; MERGED BUT UNVERIFIED -------------------


class Flaky(Replay):
    """Fails the first `fail` calls whose key starts with `prefix`, then replays."""

    def __init__(self, responses, prefix, fail):
        super().__init__(responses)
        self.prefix, self.fail, self.failed = prefix, fail, 0

    def __call__(self, args):
        key = " ".join(args)
        if key.startswith(self.prefix) and self.failed < self.fail:
            self.failed += 1
            self.calls.append(key)
            raise land_pr.subprocess.CalledProcessError(1, ["gh", *args])
        return super().__call__(args)


def _flaky(prefix, fail):
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    return Flaky(gh.responses, prefix, fail), git


def test_first_post_merge_read_fails_then_proof_completes(capsys):
    gh, git = _flaky("pr view 1218 --json mergeCommit", 1)
    sleeps: list[float] = []
    assert land_pr.land(1218, gh, git, sleep=sleeps.append) == 0
    assert gh.calls.count("pr view 1218 --json mergeCommit") == 2
    assert sleeps == [land_pr.READ_WAIT_S]
    assert f"landed #1218 as {SQUASH_1218[:12]}" in capsys.readouterr().out


def test_compare_read_is_retried():
    gh, git = _flaky(COMPARE, 2)
    assert _run(1218, gh, git) == 0
    assert gh.failed == 2


def test_ls_remote_read_is_retried():
    gh, git0 = _recording(1218, PR_1218, SQUASH_1218)
    git = Flaky(git0.responses, "ls-remote", 1)
    assert _run(1218, gh, git) == 0
    assert git.failed == 1


def test_all_retries_fail_is_merged_but_unverified(monkeypatch, capsys):
    gh, git = _flaky("pr view 1218 --json mergeCommit", 99)
    monkeypatch.setattr(land_pr, "run_gh", gh)
    monkeypatch.setattr(land_pr, "run_git", git)
    monkeypatch.setattr(land_pr.time, "sleep", lambda _s: None)
    assert land_pr.main(["1218"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("MERGED BUT UNVERIFIED: #1218")
    assert "land_pr.py --verify-only 1218" in err
    assert gh.calls.count("pr view 1218 --json mergeCommit") == land_pr.READ_TRIES


def test_the_merge_call_is_never_retried():
    gh, git = _flaky("pr view 1218 --json mergeCommit", 99)
    with pytest.raises(land_pr.UnverifiedError, match="MERGED BUT UNVERIFIED"):
        _run(1218, gh, git)
    assert sum(c.startswith("pr merge") for c in gh.calls) == 1

    gh, git = _flaky("pr merge", 99)  # the merge call itself fails: one attempt, no retry
    with pytest.raises(land_pr.subprocess.CalledProcessError):
        _run(1218, gh, git)
    assert gh.failed == 1


def _verify_recording(state, squash=SQUASH_1218, tip=""):
    view = {"state": state, "headRefName": PR_1218["headRefName"], "headRefOid": PR_1218["headRefOid"]}
    gh = Replay({
        "pr view 1218 --json state,headRefName,headRefOid": json.dumps(view),
        "pr view 1218 --json mergeCommit": json.dumps({"mergeCommit": {"oid": squash}}),
        f"api repos/{{owner}}/{{repo}}/compare/main...{squash} --jq .status": "identical\n",
    })
    git = Replay({f"ls-remote origin refs/heads/{PR_1218['headRefName']}": tip})
    return gh, git


def test_verify_only_on_a_merged_pr(capsys):
    gh, git = _verify_recording("MERGED")
    assert _run(1218, gh, git, verify_only=True) == 0
    assert not _merged(gh)
    assert git.calls == [f"ls-remote origin refs/heads/{PR_1218['headRefName']}"], "no pre-checks"
    assert f"landed #1218 as {SQUASH_1218[:12]} on main (compare: identical)" in capsys.readouterr().out


def test_verify_only_refuses_an_open_pr():
    gh, git = _verify_recording("OPEN")
    with pytest.raises(land_pr.LandError, match="not merged"):
        _run(1218, gh, git, verify_only=True)
    assert git.calls == []


def test_main_verify_only_exit_code(monkeypatch, capsys):
    gh, git = _verify_recording("OPEN")
    monkeypatch.setattr(land_pr, "run_gh", gh)
    monkeypatch.setattr(land_pr, "run_git", git)
    assert land_pr.main(["1218", "--verify-only"]) == 1
    assert "not merged" in capsys.readouterr().err


def test_null_merge_commit_is_retried_then_succeeds():
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    answers = iter([{"mergeCommit": None}, {"mergeCommit": {"oid": SQUASH_1218}}])
    responses = dict(gh.responses)
    key = "pr view 1218 --json mergeCommit"

    class Seq(Replay):
        def __call__(self, args):
            if " ".join(args) == key:
                self.calls.append(key)
                return json.dumps(next(answers))
            return super().__call__(args)

    seq = Seq(responses)
    assert _run(1218, seq, git) == 0
    assert seq.calls.count(key) == 2


def test_verify_only_with_dry_run_is_refused(capsys):
    with pytest.raises(SystemExit) as exc:
        land_pr.main(["1218", "--verify-only", "--dry-run"])
    assert exc.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


# --- #1686: ADR 0121 D4 (invariant 8) and the check-5 freeze holder ---------------------------------
#
# The two ADR patches are what `gh api repos/{owner}/{repo}/pulls/N/files` returned for the real
# PRs, trimmed to the header: #1512 flipped ADR 0116 from Proposed to Accepted; #1522 added ADR
# 0117 born Accepted. Both landed before this check existed.

PATCH_1512 = (
    "@@ -1,7 +1,7 @@\n"
    " # ADR 0116 — Render technologies: three.js in apeGmshViewer and matplotlib; VTK in sunset\n"
    " \n"
    "-**Status:** Proposed (2026-10-05). The maintainer accepts it by merging the\n"
    "-X1-c slice (#1483), which is a human gate (PROGRAM.md §4).\n"
    "+**Status:** Accepted (2026-10-06). The maintainer accepted it by merging the\n"
    "+X1-c slice (#1483) as #1493 (`2e6626d6`), and confirmed it in the T session.\n"
    " \n"
    " **Owner:** nmora\n"
)
ROWS_1512 = [
    _row("architecture/decisions/0116-render-technologies.md", patch=PATCH_1512),
    _row("architecture/decisions/README.md"),
]
PATCH_1522 = (
    "@@ -0,0 +1,383 @@\n"
    "+# ADR 0117 — Assembly (compose v2): instances of model files, rehydrated into one forward bridge\n"
    "+\n"
    "+**Status:** Accepted (2026-10-06). Design brief by the Fable architect on\n"
    "+issue #1517 (grounded on `main` `ad515ffe`), ratified by the maintainer on\n"
)
ROWS_1522 = [
    _row("architecture/decisions/0117-assembly-compose-v2.md", "added", patch=PATCH_1522),
    _row("architecture/decisions/README.md"),
]


def _d4(rows, **kw):
    return _recording(1218, kw.pop("pr", PR_1218), SQUASH_1218, [json.loads(r)[0] for r in rows], rows=rows, **kw)


def test_proposed_to_accepted_flip_is_refused_1512():
    gh, git = _d4(ROWS_1512)
    with pytest.raises(land_pr.LandError, match=r"check 7: .*0116-render-technologies\.md sets Status Accepted"):
        _run(1218, gh, git)
    assert not _merged(gh)


#: The head's copy of the born file, as `git show <head>:<path>` returns it.
HEAD_0117 = "".join(line[1:] + "\n" for line in PATCH_1522.splitlines()[1:])
ADR_0117 = "architecture/decisions/0117-assembly-compose-v2.md"
PROPOSED = "# ADR 0131 — new\n\n**Status:** Proposed (2026-10-11).\n"


def test_new_adr_born_accepted_is_refused_1522():
    gh, git = _d4(ROWS_1522, shows={ADR_0117: HEAD_0117})
    with pytest.raises(land_pr.LandError, match=r"check 7: .*0117-assembly-compose-v2\.md is a new ADR born with Status Accepted"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_human_gate_does_not_lift_the_accepted_refusal():
    """D4 (a) has no label exemption: accepting an ADR is the maintainer's web merge."""
    gh, git = _d4(ROWS_1512, pr={**PR_1218, "labels": [{"name": land_pr.HUMAN_GATE}]})
    with pytest.raises(land_pr.LandError, match=r"check 7"):
        _run(1218, gh, git)


@pytest.mark.parametrize("patch", [
    # an Accepted ADR's Status line reworded: it was already Accepted
    "@@ -1,3 +1,3 @@\n # ADR 0121 — x\n \n-**Status:** Accepted (2026-10-11).\n+**Status:** Accepted (2026-10-11; errata below).\n",
    # a body edit that never touches the Status line (#1689's errata)
    "@@ -449,3 +449,5 @@ them\n   its scope\n+\n+## Errata (2026-10-11)\n",
    # a Status line quoted below the 30-line header parse_adr reads
    "@@ -40,2 +40,3 @@\n text\n+**Status:** Accepted is what the maintainer sets.\n text\n",
    # a new ADR born Proposed
    "@@ -0,0 +1,3 @@\n+# ADR 0199 — x\n+\n+**Status:** Proposed (2026-10-11).\n",
])
def test_status_edits_that_do_not_accept_pass(patch):
    gh, git = _d4([_row("architecture/decisions/0121-charter-v2-b-with-amendments.md", patch=patch)])
    assert _run(1218, gh, git) == 0


def test_adr_with_no_patch_fails_closed():
    rows = [json.dumps([ADR_0117, "modified", None, 383, None])]
    gh, git = _d4(rows)
    with pytest.raises(land_pr.LandError, match=r"check 7: GitHub returned no patch"):
        _run(1218, gh, git)


# The review's rename hole (round 2): a rename's patch is against the OLD file, so an
# Accepted header inherited through the rename never shows as an added line.
ADR_0131 = "architecture/decisions/0131-new.md"
REPRO_B = (  # the reviewer's reproducer B: 0050 renamed to 0131, header rewritten, Accepted kept
    "@@ -1,3 +1,3 @@\n-# ADR 0050 — old\n+# ADR 0131 — new\n \n"
    "-**Status:** Accepted (2025).\n+**Status:** Accepted (2026-10-11).\n"
)
ACCEPTED_0131 = "# ADR 0131 — new\n\n**Status:** Accepted (2026-10-11).\n"


@pytest.mark.parametrize("previous, patch", [
    ("internal_docs/plan_new.md", None),  # a non-ADR file renamed in whole (pure rename: no patch)
    ("internal_docs/plan_new.md", "@@ -1 +1 @@\n-# Plan\n+# ADR 0131 — new\n"),  # Status line untouched
    ("architecture/decisions/0050-old.md", REPRO_B),  # one ADR number to another (reproducer B)
    ("architecture/decisions/0050-old.md", None),  # 0050 -> 0131 with no content change
])
def test_rename_into_a_new_adr_born_accepted_is_refused(previous, patch):
    row = json.dumps([ADR_0131, "renamed", previous, 0 if patch is None else 4, patch])
    gh, git = _d4([row], shows={ADR_0131: ACCEPTED_0131})
    with pytest.raises(land_pr.LandError, match=rf"check 7: .*0131-new\.md \(renamed from {previous}\) is a new ADR born with Status Accepted"):
        _run(1218, gh, git)
    assert not _merged(gh)


@pytest.mark.parametrize("previous", ["architecture/decisions/0050-old.md", "internal_docs/plan_new.md"])
def test_copy_into_a_new_adr_born_accepted_is_refused(previous):
    """A copied row has no patch against an empty file either: judge the head's header."""
    row = json.dumps([ADR_0131, "copied", previous, 0, None])
    gh, git = _d4([row], shows={ADR_0131: ACCEPTED_0131})
    with pytest.raises(land_pr.LandError, match=r"check 7: .*0131-new\.md is a new ADR born with Status Accepted"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_reproducer_b_patch_alone_reads_as_no_flip():
    """Why the rename needs the head's file: judged as a patch, B is not a flip."""
    assert land_pr._sets_accepted(REPRO_B) is False


@pytest.mark.parametrize("status, previous", [("added", None), ("renamed", "internal_docs/plan_new.md")])
def test_new_adr_born_proposed_passes(status, previous):
    gh, git = _d4([json.dumps([ADR_0131, status, previous, 3, None])], shows={ADR_0131: PROPOSED})
    assert _run(1218, gh, git) == 0


def test_rename_within_one_adr_number_is_judged_as_a_patch():
    """0121's slug renamed with its Accepted Status untouched: not a birth, not a flip."""
    row = json.dumps(["architecture/decisions/0121-charter-v2.md", "renamed",
                      "architecture/decisions/0121-charter-v2-b-with-amendments.md", 0, None])
    gh, git = _d4([row])
    assert _run(1218, gh, git) == 0


def test_status_reading_agrees_with_adr_index_on_every_adr():
    """`_accepted` mirrors scripts/adr_index.py::parse_adr; if that parser changes, this fails."""
    root = SCRIPT.parents[1]
    spec = importlib.util.spec_from_file_location("adr_index", root / "scripts" / "adr_index.py")
    assert spec is not None and spec.loader is not None
    adr_index = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adr_index)
    adrs = sorted((root / "architecture" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert len(adrs) > 100
    for path in adrs:
        header = path.read_text(encoding="utf-8").splitlines()[: land_pr.ADR_HEADER_LINES]
        status = next(ln for ln in header if ln.startswith("**Status:**"))
        assert land_pr._accepted(status) == (adr_index.parse_adr(path)[1] == "Accepted"), path.name


def test_baseline_names_existing_files_including_the_retirement_ledger():
    """A misspelt BASELINE path would never match a PR file, so check 8 would never fire."""
    root = SCRIPT.parents[1]
    assert "internal_docs/program/retirement_ledger.toml" in land_pr.BASELINE  # X4-a, #1806
    assert [p for p in land_pr.BASELINE if not (root / p).is_file()] == []


@pytest.mark.parametrize("path", land_pr.BASELINE)
def test_baseline_edit_without_human_gate_is_refused(path):
    gh, git = _d4([_row(path)])
    with pytest.raises(land_pr.LandError, match=rf"check 8: .*{path}.*'human-gate'"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_baseline_renamed_away_is_refused():
    gh, git = _d4([_row("tests/freeze_census_v2.py", "renamed", previous="tests/freeze_census.py")])
    with pytest.raises(land_pr.LandError, match=r"check 8: .*tests/freeze_census\.py"):
        _run(1218, gh, git)


def test_baseline_edit_with_human_gate_lands():
    gh, git = _d4([_row("tests/freeze_census.py")], pr={**PR_1218, "labels": [{"name": land_pr.HUMAN_GATE}]})
    assert _run(1218, gh, git) == 0
    assert _merged(gh)


# Check 5's holder: S2-a #1702 (body "Slice: #1696 ...") split build.py under the
# freeze:src/apeGmsh/opensees/_internal/build.py label that its slice issue #1696 carried.
# The label had to be lifted by hand before #1702 could land.
BUILD = "src/apeGmsh/opensees/_internal/build.py"
FREEZE_BUILD = [*LABELS, {"name": f"freeze:{BUILD}"}]
BODY_1702 = "Slice: #1696 (link S2-a, chain #1200)\nAuthor-model: Sonnet (prog-mechanic)\n"


def test_freeze_holder_via_slice_issue_lands_1702(capsys):
    gh, git = _recording(1218, {**PR_1218, "body": BODY_1702}, SQUASH_1218, [BUILD], labels=FREEZE_BUILD,
                         issues={1696: ["program", "slice", f"freeze:{BUILD}"]})
    assert _run(1218, gh, git) == 0
    assert "issue view 1696 --json labels" in gh.calls
    assert "held by #1696; allowed" in capsys.readouterr().out


def test_freeze_holder_via_closing_reference_lands():
    pr = {**PR_1218, "closingIssuesReferences": [{"number": 1696}]}
    gh, git = _recording(1218, pr, SQUASH_1218, [BUILD], labels=FREEZE_BUILD, issues={1696: [f"freeze:{BUILD}"]})
    assert _run(1218, gh, git) == 0


def test_freeze_holder_via_pr_label_lands():
    pr = {**PR_1218, "labels": [{"name": f"freeze:{BUILD}"}]}
    gh, git = _recording(1218, pr, SQUASH_1218, [BUILD], labels=FREEZE_BUILD)
    assert _run(1218, gh, git) == 0


@pytest.mark.parametrize("body, issues", [
    ("", {}),  # names no issue
    ("Slice: #1661\n", {1661: ["program", "slice"]}),  # names an issue that does not hold the freeze
    ("Overlap with #1696 (whichever lands second rebases).\n", {}),  # a mention is not a Slice: line
])
def test_freeze_non_holder_is_refused(body, issues):
    gh, git = _recording(1218, {**PR_1218, "body": body}, SQUASH_1218, [BUILD], labels=FREEZE_BUILD, issues=issues)
    with pytest.raises(land_pr.LandError, match=r"check 5: .*build\.py.*holds the freeze"):
        _run(1218, gh, git)
    assert not _merged(gh)


# --- Ruling 2 on #1686: the rules that run are origin/main's, never the checkout's ----------------

SHOW_MAIN = f"show origin/main:{land_pr.SELF}"


def _load_as(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_runs_origin_mains_copy_in_a_subprocess(monkeypatch, capfd):
    """The checkout's copy fetches main, writes main's file out and runs THAT; the
    copy recognises itself by MAIN_COPY_ENV naming its own path."""
    monkeypatch.delenv(land_pr.MAIN_COPY_ENV)
    fake_main = (
        "import os, sys\n"
        f"print('main copy', os.environ[{land_pr.MAIN_COPY_ENV!r}] == os.path.abspath(__file__), sys.argv[1:])\n"
        "sys.exit(7)\n"
    )
    git = Replay({"fetch origin main": "", SHOW_MAIN: fake_main})
    monkeypatch.setattr(land_pr, "run_git", git)
    monkeypatch.setattr(land_pr, "run_gh", Replay({}))  # hermetic if the hop is reverted
    assert land_pr.main(["1218", "--dry-run"]) == 7
    assert git.calls == ["fetch origin main", SHOW_MAIN]
    assert "main copy True ['1218', '--dry-run']" in capfd.readouterr().out


@pytest.mark.parametrize("env", [None, "C:/elsewhere/land_pr.py"])
def test_a_missing_or_stale_marker_still_hops(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(land_pr.MAIN_COPY_ENV)
    else:
        monkeypatch.setenv(land_pr.MAIN_COPY_ENV, env)
    seen: list[list[str]] = []
    monkeypatch.setattr(land_pr, "run_gh", Replay({}))  # hermetic if the hop is reverted
    monkeypatch.setattr(land_pr, "run_git", Replay({"fetch origin main": "", SHOW_MAIN: "pass\n"}))
    monkeypatch.setattr(land_pr, "run_python", lambda argv, e: seen.append(argv) or 0)
    assert land_pr.main(["1218", "--verify-only"]) == 0
    assert seen and seen[0][1:] == ["1218", "--verify-only"]


def test_mains_copy_does_not_hop_again(monkeypatch):
    """The loop guard: with the marker naming this file, main never re-executes."""
    def no_hop(*_a):
        raise AssertionError("main's copy re-executed itself")
    monkeypatch.setattr(land_pr, "run_python", no_hop)
    monkeypatch.setattr(land_pr, "run_git", Replay({}))
    gh, _ = _verify_recording("OPEN")
    monkeypatch.setattr(land_pr, "run_gh", gh)
    assert land_pr.main(["1218", "--verify-only"]) == 1


def test_a_pr_that_strips_check_7_is_still_judged_by_it(monkeypatch, tmp_path, capsys):
    """The revert-proof: the PR's checkout of land_pr.py drops the D4 checks; run from
    that checkout, main's copy (this file's real text) still refuses #1512's flip."""
    call = "    _check_adr_and_baseline(number, pr, rows, git)\n"
    real = SCRIPT.read_text(encoding="utf-8")
    assert real.count(call) == 1
    stripped_path = tmp_path / "checkout" / "land_pr.py"
    stripped_path.parent.mkdir()
    stripped_path.write_text(real.replace(call, ""), encoding="utf-8")
    pr_copy = _load_as(stripped_path, "land_pr_checkout")

    # The mutation is real: run as if it were main's copy, the stripped file lands the flip.
    gh, git = _d4(ROWS_1512)
    assert pr_copy.land(1218, gh, git, sleep=lambda _s: None) == 0

    def run_main_copy(argv, env):  # stands in for the subprocess, with the recorded seams
        main_copy = _load_as(Path(argv[0]), "land_pr_main_copy")
        main_gh, main_git = _d4(ROWS_1512)
        main_copy.run_gh, main_copy.run_git = main_gh, main_git
        monkeypatch.setenv(land_pr.MAIN_COPY_ENV, env[land_pr.MAIN_COPY_ENV])
        rc = main_copy.main(argv[1:])
        assert not _merged(main_gh)
        return rc

    monkeypatch.delenv(land_pr.MAIN_COPY_ENV)
    # Hermetic even with the hop reverted: the checkout's own seams are recorded too, so
    # an in-process run lands the flip offline (and the assertion below fails fast).
    local_gh, local_git = _d4(ROWS_1512)
    pr_copy.run_gh = local_gh
    pr_copy.run_git = Replay({**local_git.responses, "fetch origin main": "", SHOW_MAIN: real})
    pr_copy.run_python = run_main_copy
    assert pr_copy.main(["1218"]) == 1
    err = capsys.readouterr().err
    assert "running origin/main's scripts/land_pr.py" in err
    assert "check 7: architecture/decisions/0116-render-technologies.md sets Status Accepted" in err


def test_big_pr_file_list_never_calls_pr_diff(capsys):
    """#1775: `gh pr diff` is refused with HTTP 406 past 20000 lines (#1772). The
    file list must come from the paginated files endpoint, never from `pr diff`."""
    import subprocess

    gh, git = _recording(1218, PR_1218, SQUASH_1218, [f"src/f{i}.py" for i in range(250)])
    inner = gh.__call__

    def guarded(args):
        if args[:2] == ["pr", "diff"]:
            gh.calls.append(" ".join(args))
            raise subprocess.CalledProcessError(1, ["gh", *args], stderr="HTTP 406: diff exceeded the maximum number of lines (20000)")
        return inner(args)

    assert _run(1218, guarded, git) == 0
    assert not any(c.startswith("pr diff") for c in gh.calls)
    assert gh.calls.count(_files_call(1218)) == 1, "checks 5, 7 and 8 share one listing"
    assert "landed #1218" in capsys.readouterr().out


def test_rename_out_of_a_frozen_file_is_refused():
    """A rename lists the new name in `filename` and the old in `previous_filename`;
    check 5 must see both, or moving a file out of a freeze window slips past."""
    rows = [
        _row("src/apeGmsh/opensees/moved.py", "renamed", previous="src/apeGmsh/studio/old.py"),
        _row("src/apeGmsh/opensees/other.py"),
    ]
    gh, git = _recording(1218, PR_1218, SQUASH_1218, [], labels=FREEZE_STUDIO, rows=rows)
    with pytest.raises(land_pr.LandError, match=r"check 5: src/apeGmsh/studio/old\.py.*freeze:"):
        _run(1218, gh, git)
    assert not _merged(gh)


# One `gh api repos/{owner}/{repo}/pulls/N/files` page as GitHub returns it: the rename carries
# `previous_filename`, the binary file has no `patch`, and unread fields ride along.
FILES_PAGE = [
    {"sha": "a1", "filename": "src/apeGmsh/opensees/moved.py", "status": "renamed", "additions": 1,
     "deletions": 1, "changes": 2, "previous_filename": "src/apeGmsh/studio/old.py",
     "patch": "@@ -1 +1 @@\n-a\n+b"},
    {"sha": "b2", "filename": "docs/img/shot.png", "status": "added", "additions": 0, "deletions": 0,
     "changes": 0},
    {"sha": "c3", "filename": "README.md", "status": "modified", "additions": 1, "deletions": 0,
     "changes": 1, "patch": "@@ -1 +1,2 @@\n a\n+b"},
]
FILES_ROWS = [
    ["src/apeGmsh/opensees/moved.py", "renamed", "src/apeGmsh/studio/old.py", 2, "@@ -1 +1 @@\n-a\n+b"],
    ["docs/img/shot.png", "added", None, 0, None],
    ["README.md", "modified", None, 1, "@@ -1 +1,2 @@\n a\n+b"],
]


def _jq_filter(expression, page):
    """Apply a real jq expression to a JSON page: the `jq` binary, else the `jq` module."""
    import shutil
    import subprocess

    binary = shutil.which("jq")
    if binary:
        return subprocess.run([binary, "-r", expression], input=json.dumps(page), capture_output=True,
                              text=True, check=True).stdout
    try:
        import jq
    except ImportError:
        pytest.skip("neither the jq binary nor the jq module is available")
    return "".join(f"{row}\n" for row in jq.compile(expression).input_value(page).all())


def test_files_jq_run_for_real_yields_the_fields_the_checks_read():
    """`gh api --jq` is jq. The replays key on `land_pr.FILES_JQ` itself, so a broken
    expression (say `.previous_filename` turned into `null`) passed them (#1715)."""
    seen = []

    def gh(args):
        assert args[:3] == ["api", "repos/{owner}/{repo}/pulls/9/files", "--paginate"]
        seen.append(args[args.index("--jq") + 1])
        return _jq_filter(seen[-1], FILES_PAGE)

    assert land_pr._changed_files(9, gh) == FILES_ROWS
    assert seen == [land_pr.FILES_JQ]


# --- Check 9 (lesson #1788, #1855): a program PR stays inside its card's Owned list ---------
# #1812 (B, card #1801) was a whole-tree commit from a stale worktree: 52 files against 6
# owned. Squash-merged, it would have reverted #1761, #1796 and #1803.
OWNED_1812 = [
    "src/apeGmsh/opensees/_internal/compose.py",
    "src/apeGmsh/opensees/opensees_model.py",
    "tests/opensees/golden/roundtrip_oracle_ledger.txt",
    "tests/opensees/golden/test_roundtrip_oracle.py",
    "tests/opensees/h5/test_h5_replay_eq_and_fix_ndf.py",
    "changelog.d/k2a-roundtrip-oracle.md",
]
FOREIGN_1812 = [
    "src/apeGmsh/viewers/session/_select_all.py",  # #1796
    "src/apeGmsh/viewers/session/_realize.py",  # #1796
    "tests/viewers/test_select_all_stage_activation.py",  # #1796
    "internal_docs/program/prototypes/legacy.py",  # #1803 deleted it; the stale tree re-adds it
    "internal_docs/program/prototypes/clones.py",
    "internal_docs/program/prototypes/contract_cov.py",
    *(f"apeGmshViewer/src/contour/part{i:02d}.ts" for i in range(40)),  # #1761
]
CARD_1812 = "### Goal\nK2a.\n\n### Owned files\n" + "".join(f"- `{p}`\n" for p in OWNED_1812) + "\n### Verification\n- `pytest`\n"
PROGRAM_PR = {**PR_1218, "labels": [{"name": "program"}, {"name": "semantic"}], "body": "Slice: #1801\nClass: semantic\n"}


def _scope(files=None, *, card=CARD_1812, comments=(), pr=PROGRAM_PR, rows=None):
    files = OWNED_1812 if files is None else files
    cards = {1801: {"body": card, "comments": [{"body": c} for c in comments]}}
    return _recording(1218, pr, SQUASH_1218, files, rows=rows, cards=cards)


def test_1812_stale_tree_is_refused_and_every_foreign_file_named():
    """Revert-proof: with `check_owned` removed from `check`, this PR merges."""
    assert len(OWNED_1812) == 6 and len(OWNED_1812 + FOREIGN_1812) == 52
    gh, git = _scope(OWNED_1812 + FOREIGN_1812)
    with pytest.raises(land_pr.LandError, match=r"check 9: #1218 changes 46 file\(s\) outside slice card #1801") as exc:
        _run(1218, gh, git)
    assert all(f"\n  {path}\n" in str(exc.value) for path in FOREIGN_1812)
    assert not any(f"\n  {path}\n" in str(exc.value) for path in OWNED_1812)
    assert not _merged(gh)


@pytest.mark.parametrize("files, card, comments", [
    pytest.param(OWNED_1812[:3], CARD_1812, (), id="within-owned"),
    pytest.param(["src/apeGmsh/opensees/emitter/verbs.py"],
                 "### Owned files\n- `src/apeGmsh/opensees/emitter/verbs.py::VERBS` (the embedded_rebar row)\n", (),
                 id="path-symbol"),
    pytest.param(["tests/viewers/golden_stills/a.png", "tests/fixtures/schema_corpus/x/opensees_2.29.h5"],
                 "### Owned files\n- `tests/viewers/golden_stills/`\n- `tests/fixtures/schema_corpus/**`\n", (),
                 id="directory-and-glob"),
    pytest.param(["internal_docs/program/board_notes.md"], CARD_1812,
                 ("**Card amendment (T):** also re-date a row in `internal_docs/program/board_notes.md`.",),
                 id="bold-amendment"),
    pytest.param(["internal_docs/program/board_notes.md"], CARD_1812,
                 ("Card amendment (orchestrator): Owned files add `internal_docs/program/board_notes.md`.",),
                 id="plain-amendment-1807"),
    pytest.param(["changelog.d/any-slug.md"], CARD_1812, (), id="changelog-fragment"),
    pytest.param(["internal_docs/program/prototypes/legacy.py", "internal_docs/program/prototypes/README.md"],
                 "### Owned files\n- `internal_docs/program/prototypes/contract_cov.py`, `legacy.py`, `clones.py`: delete.\n"
                 "- `internal_docs/program/prototypes/README.md`\n", (), id="sibling-list-1802"),
    pytest.param(["src/x.py", "docs/y.md"], "### Owned files\n- `src/x.py`\n\n### Allowed\n- `docs/y.md`\n", (),
                 id="allowed-section"),
    pytest.param(["tests/test_fork_token_ratchet.py"],
                 "**Maintainer ruling:** YES. Owned: `tests/test_fork_token_ratchet.py` (that one value).", (),
                 id="owned-label-1851"),
])
def test_files_inside_the_card_pass(files, card, comments, capsys):
    gh, git = _scope(files, card=card, comments=comments)
    assert _run(1218, gh, git, dry_run=True) == 0
    assert "check 9: #1218's" in capsys.readouterr().out


def test_rename_with_both_names_owned_passes():
    gh, git = _scope([], rows=[_row(OWNED_1812[1], "renamed", previous=OWNED_1812[0])])
    assert _run(1218, gh, git, dry_run=True) == 0


@pytest.mark.parametrize("new, old", [
    ("src/apeGmsh/opensees/unowned.py", OWNED_1812[0]),  # renamed INTO an unowned path
    (OWNED_1812[0], "src/apeGmsh/opensees/unowned.py"),  # renamed out of an unowned path
])
def test_rename_touching_an_unowned_path_is_refused(new, old):
    gh, git = _scope([], rows=[_row(new, "renamed", previous=old)])
    with pytest.raises(land_pr.LandError, match=r"check 9: .*1 file\(s\) outside(.|\n)*\n  src/apeGmsh/opensees/unowned\.py"):
        _run(1218, gh, git)


@pytest.mark.parametrize("files, card, pr, message", [
    pytest.param(OWNED_1812, CARD_1812, {**PROGRAM_PR, "body": "Class: semantic\n"},
                 r"check 9: .*no 'Slice: #<n>' line", id="no-slice-line"),
    pytest.param(OWNED_1812, "### Goal\nFix `src/x.py`.\n", PROGRAM_PR,
                 r"check 9: slice card #1801 has no '### Owned files'", id="no-owned-section"),
    pytest.param(OWNED_1812, "### Owned files\nDerived after Phase 1 and recorded on this issue.\n", PROGRAM_PR,
                 r"check 9: slice card #1801's Owned section names no backticked path", id="empty-owned-1801"),
    pytest.param(["src/apeGmsh/rebar/_geometry.py"], "### Owned files\n- `src/apeGmsh/rebar/__init__.py`\n\n"
                 "### Verification\n- `src/apeGmsh/rebar/_geometry.py` stays green\n", PROGRAM_PR,
                 r"check 9: (.|\n)*_geometry\.py", id="path-outside-the-section"),
    pytest.param(["src/apeGmsh/core/x.py"], "### Owned files\n- `*`\n- `**/*.py`\n- `src/apeGmsh/rebar/`\n", PROGRAM_PR,
                 r"check 9: (.|\n)*src/apeGmsh/core/x\.py", id="bare-globs-are-dropped"),
    pytest.param(["internal_docs/x.md"], CARD_1812, PROGRAM_PR, r"check 9: (.|\n)*internal_docs/x\.md",
                 id="not-a-changelog-fragment"),
])
def test_scope_refusals(files, card, pr, message):
    gh, git = _scope(files, card=card, pr=pr)
    with pytest.raises(land_pr.LandError, match=message):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_comment_that_is_not_an_amendment_adds_nothing():
    gh, git = _scope(["src/extra.py"], comments=("Locks granted: `src/extra.py`.",))
    with pytest.raises(land_pr.LandError, match=r"check 9: (.|\n)*src/extra\.py"):
        _run(1218, gh, git)


def test_scope_ok_overrides_with_a_notice(capsys):
    pr = {**PROGRAM_PR, "labels": [*PROGRAM_PR["labels"], {"name": "scope-ok"}]}
    gh, git = _scope(OWNED_1812 + FOREIGN_1812, pr=pr)
    assert _run(1218, gh, git) == 0
    assert "check 9: OVERRIDDEN by the 'scope-ok' label" in capsys.readouterr().out


def test_non_program_pr_is_not_checked_and_says_so(capsys):
    gh, git = _scope(OWNED_1812 + FOREIGN_1812, pr=PR_1218)
    assert _run(1218, gh, git, dry_run=True) == 0
    assert "check 9: #1218 is not labelled 'program'" in capsys.readouterr().out
    assert not any(call.startswith("issue view") for call in gh.calls)


def test_owned_only_runs_check_9_alone_on_a_merged_pr(monkeypatch, capsys):
    gh = Replay({
        "pr view 1812 --json labels,body": json.dumps({"labels": PROGRAM_PR["labels"], "body": PROGRAM_PR["body"]}),
        _files_call(1812): "".join(_row(f) + "\n" for f in OWNED_1812 + FOREIGN_1812),
        "issue view 1801 --json body,comments": json.dumps({"body": CARD_1812, "comments": []}),
    })
    monkeypatch.setattr(land_pr, "run_gh", gh)
    monkeypatch.setattr(land_pr, "run_git", Replay({}))
    assert land_pr.main(["1812", "--owned-only"]) == 1
    assert "check 9: #1812 changes 46 file(s)" in capsys.readouterr().err
    assert not _merged(gh)


def test_owned_only_with_dry_run_is_refused(capsys):
    with pytest.raises(SystemExit) as exc:
        land_pr.main(["1218", "--owned-only", "--dry-run"])
    assert exc.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("line, paths", [
    ("- `.github/workflows/tests.yml`: `lock-tests`.", [".github/workflows/tests.yml"]),  # 1807
    ("- `changelog.d/<slug>.md`", []),  # a placeholder
    ("- `skills/apegmsh/references/rebar.md` (then `sync_skill.py`)",  # prose, not a list: no sibling
     ["skills/apegmsh/references/rebar.md", "sync_skill.py"]),
    ("- `tests/opensees/golden/cells/**` + `MANIFEST.json`",
     ["tests/opensees/golden/cells/**", "tests/opensees/golden/cells/MANIFEST.json"]),
    ("- `./AGENTS.md` and `scripts/land_pr.py::BASELINE`", ["AGENTS.md", "scripts/land_pr.py"]),
])
def test_line_paths(line, paths):
    assert land_pr._line_paths(line) == paths


@pytest.mark.parametrize("card", [
    # review of #1858, finding 1: a negated "not owned:" in a Stop line is not a label
    "### Owned files\n- `scripts/a.py`\n\n### Stop conditions\n- A file not owned: `src/apeGmsh/opensees/apesees.py`\n",
    "### Goal\nThe unowned: `src/apeGmsh/opensees/apesees.py` stays.\n\n### Owned files\n- `scripts/a.py`\n",
    # finding 2: a bold pseudo-heading ends the Owned section
    "### Owned files\n- `scripts/a.py`\n\n**Do not read:** `src/apeGmsh/opensees/apesees.py`\n",
    "### Owned files\n- `scripts/a.py`\n**Hub lock:**\n- `src/apeGmsh/opensees/apesees.py`\n",
    # the template's HTML guidance is not card text
    "### Owned files\n<!-- e.g. `src/apeGmsh/opensees/apesees.py` -->\n- `scripts/a.py`\n",
])
def test_paths_outside_a_real_owned_list_are_not_owned(card):
    assert land_pr._card_paths(card) == ["scripts/a.py"]


@pytest.mark.parametrize("line", [
    "- `scripts/a.py` (never touch `src/x.py`)",
    "- `src/apeGmsh/rebar/`, except `src/apeGmsh/rebar/detailing.py`",
    "- Do not edit `src/x.py`",
    "- `scripts/a.py`; `src/x.py` is not owned",
])
def test_negated_owned_line_fails_closed(line):
    gh, git = _scope(["scripts/a.py"], card=f"### Owned files\n{line}\n")
    with pytest.raises(land_pr.LandError, match=r"check 9: an Owned line negates.*ambiguous") as exc:
        _run(1218, gh, git)
    assert line.strip() in str(exc.value)
    assert not _merged(gh)


def test_negated_amendment_line_adds_no_path(capsys):
    amendment = "**Card amendment (T):** also edit `src/a.py`.\nDo NOT touch `src/b.py`.\n"
    gh, git = _scope(["src/a.py"], comments=(amendment,))
    assert _run(1218, gh, git, dry_run=True) == 0
    assert "skipped a negated card-amendment line" in capsys.readouterr().out
    gh, git = _scope(["src/a.py", "src/b.py"], comments=(amendment,))
    with pytest.raises(land_pr.LandError, match=r"check 9: #1218 changes 1 file\(s\)(.|\n)*\n  src/b\.py"):
        _run(1218, gh, git)


def test_owned_label_mid_line_after_a_sentence_still_counts_1851():
    card = "**Maintainer ruling (2026-10-13): YES.** Lower it. Owned: `tests/x.py` (that one value)."
    assert land_pr._card_paths(card) == ["tests/x.py"]


# --- #1872 (follow-up to #1858): the amendment negation guard failed open ------------------
def _refused_b(amendment):
    """`src/b.py` is changed but must not be whitelisted; `src/a.py` stays owned."""
    gh, git = _scope(["src/a.py", "src/b.py"], comments=(amendment,))
    with pytest.raises(land_pr.LandError, match=r"check 9: #1218 changes 1 file\(s\)(.|\n)*\n  src/b\.py"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_negation_opening_a_multi_line_list_excludes_the_list_1872(capsys):
    """Revert-proof: with the lead-in's `excluding` state removed, `src/b.py` is owned."""
    amendment = "**Card amendment:** add `src/a.py`.\nDo not touch:\n- `src/b.py`\n"
    _refused_b(amendment)
    assert "skipped a negated card-amendment line on #1801: '- `src/b.py`'" in capsys.readouterr().out
    gh, git = _scope(["src/a.py"], comments=(amendment,))  # the positive line still adds its path
    assert _run(1218, gh, git, dry_run=True) == 0


@pytest.mark.parametrize("amendment", [
    pytest.param("**Card amendment:** add `src/a.py`.\n- Do not touch `src/b0.py` or these\n  - `src/b.py`\n",
                 id="sub-items"),
    pytest.param("**Card amendment:** add `src/a.py`.\n- Do not touch:\n- `src/b.py`\n", id="bullet-ending-in-colon"),
    # review of #1883, F2 (a): an unmarked continuation line, at any indent, stays in the item
    pytest.param("**Card amendment:** add `src/a.py`.\n- do not touch `src/b0.py`,\n`src/b.py`, `src/b3.py`\n",
                 id="unindented-continuation"),
    # F2 (b): a negated bullet naming no path is a lead-in for the sibling bullets after it
    pytest.param("**Card amendment:**\n- add `src/a.py`\n- Do not touch the following\n- `src/b.py`\n",
                 id="pathless-bullet-lead-in"),
])
def test_a_negated_bullet_excludes_the_list_it_opens_1872(amendment):
    """Revert-proof, one rule per case: without the bullet's sub-item rule, the trailing-colon
    lead-in, the unmarked-continuation rule, or the pathless-bullet lead-in, `src/b.py` is owned."""
    _refused_b(amendment)


@pytest.mark.parametrize("gap", ["\n", "\n\n"])
def test_a_colon_lead_in_spans_blank_lines_to_its_list_1872(gap):
    """Review of #1883, F1: a blank line between "Do not touch:" and its list (23 of 355
    program issues use that layout) must not end the exclusion. Revert-proof: with the
    lead-in ending at a blank line, `src/b.py` is owned."""
    _refused_b(f"**Card amendment:** add `src/a.py`.\n\nDo not touch:{gap}\n- `src/b.py`\n")


def test_a_negated_bullet_does_not_swallow_its_siblings_1872():
    """Amendments are often prose bullets naming a path (#1663, #1722): one negated bullet
    must not drop the paths its siblings add. Fails if a negated bullet excluded to the
    paragraph's end."""
    amendment = "**Card amendment:**\n- add `src/a.py`\n- do not reformat `src/z.py`\n- add `src/c.py`\n"
    gh, git = _scope(["src/a.py", "src/c.py"], comments=(amendment,))
    assert _run(1218, gh, git, dry_run=True) == 0


@pytest.mark.parametrize("tail", [
    pytest.param("\nAlso add `src/c.py`.", id="blank-line"),
    pytest.param("### Also\n- `src/c.py`", id="heading"),
    pytest.param("**Also add:** `src/c.py`", id="bold-label"),
])
def test_a_negated_list_ends_where_an_owned_section_does_1872(tail):
    """The list a negation opens ends at a blank line, a heading or a bold label, as in
    `_card_paths`; a positive line after it adds its path again (the card's stop condition)."""
    amendment = "**Card amendment:** add `src/a.py`.\nDo not touch:\n- `src/b.py`\n" + tail
    gh, git = _scope(["src/a.py", "src/c.py"], comments=(amendment,))
    assert _run(1218, gh, git, dry_run=True) == 0


@pytest.mark.parametrize("line", [
    pytest.param("Don\u2019t touch `src/b.py`.", id="curly-apostrophe"),
    pytest.param("Do _not_ touch `src/b.py`.", id="underscore-emphasis"),
    pytest.param("It won't touch `src/b.py`.", id="won't"),
    pytest.param("Leave `src/b.py` alone.", id="leave-alone"),
    pytest.param("Avoid `src/b.py`.", id="avoid"),
    pytest.param("No changes to `src/b.py`.", id="no-changes-to"),
    pytest.param("It cannot touch `src/b.py`.", id="cannot"),
    pytest.param("Excluding `src/b.py`.", id="excluding"),
    # review of #1883, F4
    pytest.param("Keep `src/b.py` untouched.", id="untouched"),
    pytest.param("Editing `src/b.py` is forbidden.", id="forbidden"),
    pytest.param("`src/b.py` is out of scope.", id="out-of-scope"),
    pytest.param("`src/b.py` is removed from Owned.", id="removed-from-1856"),
    pytest.param("No `src/b.py`.", id="bare-no"),
    pytest.param("Neither `src/b.py` nor `src/b2.py`.", id="neither"),
])
def test_negation_vocabulary_adds_no_path_1872(line):
    """Revert-proof, one form per case: drop the form from `_NEGATION` and `src/b.py` is owned."""
    _refused_b(f"**Card amendment (T):** also edit `src/a.py`.\n{line}\n")


@pytest.mark.parametrize("line", [
    "- `scripts/a.py` (don\u2019t touch `src/x.py`)",
    "- `scripts/a.py`; leave `src/x.py` alone",
])
def test_new_negation_vocabulary_refuses_an_owned_line_1872(line):
    with pytest.raises(land_pr.LandError, match=r"check 9: an Owned line negates"):
        land_pr._card_paths(f"### Owned files\n{line}\n")


def test_a_negation_word_inside_a_backticked_path_is_not_a_negation_1872():
    """Revert-proof: a letter boundary matches `not` in `not_found.py` (a word boundary did
    not), so without masking path spans this Owned line refuses and the amendment adds nothing."""
    card = "### Owned files\n- `src/x/not_found.py`, `test_never_twice.py`\n"
    assert land_pr._card_paths(card) == ["src/x/not_found.py", "src/x/test_never_twice.py"]
    gh, git = _scope(["tests/avoid-cannot.py"], comments=("**Card amendment:** add `tests/avoid-cannot.py`.",))
    assert _run(1218, gh, git, dry_run=True) == 0


def test_a_negation_word_inside_a_quoted_code_string_is_not_a_negation_1857():
    """#1857's Owned line (open PR #1887) quotes `cannot join:` templates. Revert-proof:
    masking only path spans keeps `cannot` and refuses the card."""
    card = ("### Owned files\n- `apeGmshViewer/src/panels/windows/analysis.ts` (the note strings `JOIN_GAP`, "
            "`AMBIGUOUS`, the two `cannot join:` templates, and the file's header comment only)\n")
    assert land_pr._card_paths(card) == ["apeGmshViewer/src/panels/windows/analysis.ts"]


@pytest.mark.parametrize("line", [
    pytest.param("- `do not` touch `src/b.py`", id="ticked-do-not"),
    pytest.param("`never` `src/b.py`", id="ticked-never"),
])
def test_a_backticked_negation_phrase_still_negates_g1(line):
    """Review of #1883, G1. Revert-proof: masking every span hides the negation and
    `src/b.py` is owned."""
    _refused_b(f"**Card amendment (T):** also edit `src/a.py`.\n{line}\n")


def test_a_backticked_negation_in_an_owned_line_refuses_g1():
    """Review of #1883, G1. Revert-proof: masking every span lets this Owned line own `b.py`."""
    with pytest.raises(land_pr.LandError, match=r"check 9: an Owned line negates"):
        land_pr._card_paths("### Owned files\n- `src/a.py` (`never` `src/b.py`)\n")


def test_mispaired_backticks_do_not_hide_a_negation_1250():
    """A ``double`` run pairs the following ticks wrongly, so masking would swallow the
    prose "Never" (#1250's Owned line). Revert-proof: masking regardless of the pairing
    lets this line through."""
    line = ("- `scripts/a.py`: RST ``double`` docstrings get `(pinned)`. Never put a URL in single "
            "backticks: the `doc-path` rule reads `x/y.md`.")
    with pytest.raises(land_pr.LandError, match=r"check 9: an Owned line negates"):
        land_pr._card_paths(f"### Owned files\n{line}\n")


# --- Lesson #1877: checks 10 (red main) and 11 (stale base) ---------------------------------------
# 3af31061 squash-merged #1698, whose required checks ran on 10-11 21:25Z on the merge ref
# `Merge b2540a96 into e6bd0b6d`. X4-a (#1819, 10-12) added the retirement-ledger row for the
# Linux xfail #1698 deletes, so the merge turned lock-tests red. #1858 (471432f3) and #1866
# then landed while main's run on 3af31061 was still in progress, and main stayed red.
# Recorded with `gh api` on 2026-10-13, trimmed to the fields the script reads.

HEAD_1698 = "b2540a96713010baf320df83f13a60232c67cb7e"
BASE_1698 = "e6bd0b6dc5eb378cf228dab65bef272fd9deecbd"
RUN_1698 = "38176278368"
JOBS_1698 = {  # statusCheckRollup detailsUrl job ids, run 38176278368
    "lock-tests": 114579297270, "emit-cost-gate": 114579297255, "static-gates": 114579297339,
    "suite": 114579297267, "live-stock": 114579297189,
}
PR_1698 = {
    "state": "OPEN", "isDraft": False, "baseRefName": "main",
    "headRefName": "prog/f3f-sanisand-flip-pin", "headRefOid": HEAD_1698, "mergeable": "MERGEABLE",
    "labels": [{"name": "program"}, {"name": "chain:F"}, {"name": "semantic"}],
    "body": "Slice: #1680\nAuthor-model: Opus 5.5\nClass: semantic\n",
    "statusCheckRollup": [
        {"__typename": "CheckRun", "name": name, "status": "COMPLETED", "conclusion": "SUCCESS",
         "startedAt": "2026-10-11T21:25:27Z",
         "detailsUrl": f"https://github.com/nmorabowen/apeGmsh/actions/runs/{RUN_1698}/job/{job}"}
        for name, job in JOBS_1698.items()
    ],
}
FILES_1698 = [
    "architecture/fork-divergence-roster.md",
    "changelog.d/f3f-sanisand-flip-pin.md",
    "tests/opensees/integration_ladruno/test_ladruno_sanisand_live.py",
]
CARD_1680 = "### Owned files\n" + "".join(f"- `{p}`\n" for p in FILES_1698) + "\n### The effect test\n"
#: Job 114579297270 (lock-tests), lines 105 and 124 of its log.
LOG_1698 = (
    "2026-10-11T21:25:32.2069551Z ##[group]Checking out the ref\n"
    f"2026-10-11T21:25:32.6027970Z HEAD is now at 54b621b Merge {HEAD_1698} into {BASE_1698}\n"
)
#: compare/e6bd0b6d...6f66fd23 (main's tip just before 3af31061): ahead_by, the tip's date.
BEHIND_1698 = (68, "2026-10-13T09:04:47Z")
SHA_3AF = "3af31061bd34d600f42f1a2d3eeaea20e14d6f9d"
SHA_471 = "471432f39b9096a0db0c8cf32b8885dfe8d2d812"
SHA_1ACF = "1acff395a9cf66b3bd3688f5c285a511aea44d34"
SHA_53F = "53f1c4b87f780be46fbc52550fd270f2f1bb28d5"
#: main's Tests push runs (newest first) at each landing, rebuilt from the recorded run list's
#: created_at/updated_at: 38428776331 on 3af31061 ran 16:13:51Z-16:30:16Z (failure),
#: 38429075356 on 471432f3 16:16:00Z-16:35:38Z (failure), 38429723003 on 1acff395
#: 16:20:42Z-16:42:00Z (failure); 38373760364 on 6f66fd23 had completed green at 09:24:56Z.
RUN_6F66 = _run_row(38373760364, MAIN_TIP)
RUN_53F1 = _run_row(38373316012, SHA_53F)
RUNS_AT_1858 = [_run_row(38428776331, SHA_3AF, "in_progress", None), RUN_6F66, RUN_53F1]  # 16:15:57Z
RUNS_AT_1866 = [  # 16:20:38Z
    _run_row(38429075356, SHA_471, "in_progress", None), _run_row(38428776331, SHA_3AF, "in_progress", None),
    RUN_6F66, RUN_53F1,
]
RUNS_RED_471 = [  # 16:36Z: 471432f3's run is the latest completed, and red
    _run_row(38429723003, SHA_1ACF, "in_progress", None), _run_row(38429075356, SHA_471, "completed", "failure"),
    _run_row(38428776331, SHA_3AF, "completed", "failure"), RUN_6F66,
]


def _landed_pr(head, run, jobs, started, files, slice_no):
    """A PR as it stood at its landing: green required checks from one Actions run."""
    rollup = [
        {"__typename": "CheckRun", "name": name, "status": "COMPLETED", "conclusion": "SUCCESS", "startedAt": started,
         "detailsUrl": f"https://github.com/nmorabowen/apeGmsh/actions/runs/{run}/job/{job}"}
        for name, job in jobs.items()
    ]
    pr = {"state": "OPEN", "isDraft": False, "baseRefName": "main", "headRefName": f"pr-{slice_no}",
          "headRefOid": head, "mergeable": "MERGEABLE", "labels": [{"name": "program"}, {"name": "semantic"}],
          "body": f"Slice: #{slice_no}\n", "statusCheckRollup": rollup}
    card = {"body": "### Owned files\n" + "".join(f"- `{p}`\n" for p in files), "comments": []}
    return pr, files, {slice_no: card}


#: #1858 (squash 471432f3): checks in run 38368430040, merge ref `Merge a756c51e into a46fe539`.
HEAD_1858, BASE_1858 = "a756c51e4058da90b26749158157792ea745826f", "a46fe53964eb7088d6a02a8e0775ed0d5f177406"
PR_1858, FILES_1858, CARD_1858 = _landed_pr(
    HEAD_1858, 38368430040,
    {"lock-tests": 115199113581, "emit-cost-gate": 115199113454, "static-gates": 115199113732,
     "suite": 115199112987, "live-stock": 115199113353},
    "2026-10-13T08:19:07Z",
    ["AGENTS.md", "changelog.d/land-pr-owned-files.md", "internal_docs/program/slice_card.md",
     "scripts/land_pr.py", "tests/test_land_pr.py"],
    1855,
)
#: #1866 (squash 1acff395): checks in run 38369618270, merge ref `Merge 2da1057d into 686840f6`.
HEAD_1866, BASE_1866 = "2da1057d07edeb9f59a9d7d89c73bac44a92d0c8", "686840f602ee08383460732eb15a870c146d25ee"
PR_1866, FILES_1866, CARD_1866 = _landed_pr(
    HEAD_1866, 38369618270,
    {"lock-tests": 115203195794, "emit-cost-gate": 115203196012, "static-gates": 115203196086,
     "suite": 115203196122, "live-stock": 115203196098},
    "2026-10-13T08:30:00Z",
    ["changelog.d/2026-10-12-chunk-owned-nodes.md", "src/apeGmsh/opensees/_internal/build/ndf.py",
     "tests/opensees/unit/test_emit_owned_nodes.py"],
    1758,
)


def _1698(**kw):
    gh, git = _recording(1698, PR_1698, "unused", FILES_1698, cards={1680: {"body": CARD_1680, "comments": []}},
                         logs={_log_call(f"x/{JOBS_1698['lock-tests']}"): LOG_1698}, **kw)
    gh.responses[_behind_call(BASE_1698)] = json.dumps(list(BEHIND_1698))
    return gh, git


def test_1698_stale_base_is_refused_3af31061():
    """Replay (b): #1698 as it stood before 3af31061. Revert-proof: without
    `check_fresh_base` in `check`, this PR merges."""
    gh, git = _1698()
    with pytest.raises(land_pr.LandError, match=(
        r"check 11: #1698 touches tests/opensees/integration_ladruno/test_ladruno_sanisand_live\.py but its "
        r"required checks in run 38176278368 \(job 114579297270\) ran on base e6bd0b6dc5eb, 68 commit\(s\) "
        r"and 36 h behind main.*fresh merge ref"
    )):
        _run(1698, gh, git)
    assert not _merged(gh)
    assert sum("/logs" in c for c in gh.calls) == 1, "one log per workflow run, not per job"


#: main's commit order (newest first) at each landing, as `commits?sha=main` lists it.
ORDER_AT_1858 = (SHA_3AF, MAIN_TIP, SHA_53F)
ORDER_AT_1866 = (SHA_471, SHA_3AF, MAIN_TIP, SHA_53F)
ORDER_RED = (SHA_1ACF, SHA_471, SHA_3AF, MAIN_TIP)


def _landing(number, pr, files, cards, base, *, main, order, ancestry, behind):
    """A PR whose required checks ran on `base`, landing on main as `main`/`order` read."""
    job = pr["statusCheckRollup"][0]["detailsUrl"]
    gh, git = _recording(number, pr, "unused", files, cards=cards, main=main, main_order=order,
                         logs={_log_call(job): _checkout(pr["headRefOid"], base)})
    gh.responses[_ancestry_call(order[0], base)] = ancestry + "\n"
    gh.responses[_behind_call(base)] = json.dumps(list(behind))
    return gh, git


def test_1858_landing_on_an_in_progress_main_is_refused():
    """Replay: #1858 landed (16:15:57Z) while 3af31061's Tests run was in progress, on its real
    merge-ref base a46fe539, which lacks 3af31061; the latest completed run (6f66fd23) was green.
    Revert-proof: without the ancestry branch (or `check_main_green`), it merges; check 11
    passes it (9 commits, 8 h behind)."""
    gh, git = _landing(1858, PR_1858, FILES_1858, CARD_1858, BASE_1858, main=RUNS_AT_1858, order=ORDER_AT_1858,
                       ancestry="behind", behind=(9, "2026-10-13T16:13:47Z"))
    with pytest.raises(land_pr.LandError, match=(
        r"check 10: main unverified at 3af31061bd34: its Tests run is in progress and #1858's required checks "
        r"in run 38368430040 \(job 115199113581\) ran on base a46fe53964eb, which does not contain it "
        r"\(compare: behind\)\. Merge main and re-run, or wait"
    )):
        _run(1858, gh, git)
    assert not _merged(gh)


def test_1866_landing_on_an_in_progress_main_is_refused():
    """Replay: #1866 landed (16:20:38Z) while 471432f3's and 3af31061's runs were in progress,
    on base 686840f6. Revert-proof: without the ancestry branch, it merges."""
    gh, git = _landing(1866, PR_1866, FILES_1866, CARD_1866, BASE_1866, main=RUNS_AT_1866, order=ORDER_AT_1866,
                       ancestry="behind", behind=(8, "2026-10-13T16:15:56Z"))
    with pytest.raises(land_pr.LandError, match=r"check 10: main unverified at 471432f39b90: .*686840f602ee"):
        _run(1866, gh, git)
    assert not _merged(gh)


@pytest.mark.parametrize("ancestry", ["identical", "ahead"])
def test_a_base_that_contains_the_in_progress_head_lands(ancestry, capsys):
    """The ruling's throughput clause: a PR that has just merged main lands while main's run
    is in progress. Revert-proof: without the ancestry branch (refusing every in-progress
    head), it is refused."""
    gh, git = _landing(1218, PR_1218, FILES_1218, None, SHA_3AF, main=RUNS_AT_1858, order=ORDER_AT_1858,
                       ancestry=ancestry, behind=(0, None))
    assert _run(1218, gh, git) == 0
    assert _ancestry_call(SHA_3AF, SHA_3AF) in gh.calls
    assert "check 10: main's Tests run on 3af31061bd34 is in progress; the latest completed one (6f66fd238325)" in (
        capsys.readouterr().out
    )


def test_a_src_only_pr_still_has_its_base_read_by_check_10():
    """Check 11 skips a PR outside tests/, but check 10 needs its base whenever H is in
    progress. Revert-proof: if check 10 skipped the base read, this merges."""
    gh, git = _recording(1097, PR_1097, SQUASH_1097, FILES_1097, main=RUNS_AT_1858, main_order=ORDER_AT_1858)
    gh.responses[_log_call(_job("lock-tests"))] = "no checkout line\n"
    with pytest.raises(land_pr.LandError, match=r"check 10: the base #1097's required checks ran on .*could not be read"):
        _run(1097, gh, git)


@pytest.mark.parametrize("number, pr, files", [(1218, PR_1218, FILES_1218), (1097, PR_1097, FILES_1097)])
def test_any_pr_is_refused_while_main_is_red_471432f3(number, pr, files):
    """Replay: at 16:36Z, 471432f3's run is main's latest completed one, and red; no PR lands,
    whatever its base. Revert-proof: without the red branch, both merge."""
    gh, git = _recording(number, pr, "unused", files, main=RUNS_RED_471, main_order=ORDER_RED)
    with pytest.raises(land_pr.LandError, match=(
        r"check 10: main is red: its latest completed Tests run \(38429075356, on 471432f39b90\) is failure.*"
        r"'fixes-red-main'"
    )):
        _run(number, gh, git)
    assert not _merged(gh)


def test_latest_completed_is_by_commit_order_not_by_listing():
    """Two runs created in the same second list in either order (01e7d4a7 and 9e44756b, 17:03:27Z).
    The latest completed run is the newest COMMIT's: here X, green, not Y, red. Revert-proof:
    taking the first completed run in the listing refuses this PR as red."""
    tip, x, y = "a" * 40, "b" * 40, "c" * 40
    runs = [_run_row(3, y, "completed", "failure"), _run_row(2, tip, "in_progress", None), _run_row(1, x)]
    gh, git = _landing(1218, PR_1218, FILES_1218, None, tip, main=runs, order=(tip, x, y),
                       ancestry="identical", behind=(0, None))
    assert _run(1218, gh, git) == 0


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", "timed_out", None])
def test_a_cancelled_or_skipped_run_is_red(conclusion):
    """Only success proves main. Revert-proof: judging only `failure` as red lands these."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218, main=[_run_row(1, MAIN_TIP, "completed", conclusion)])
    with pytest.raises(land_pr.LandError, match=rf"check 10: main is red: .* is {conclusion}\."):
        _run(1218, gh, git)


def test_no_run_for_main_head_refuses():
    """H was pushed but its Tests run has not been created (or is off the page). Revert-proof:
    without the no-run branch, the PR is judged on 6f66fd23's green run and merges."""
    tip = "d" * 40
    gh, git = _landing(1218, PR_1218, FILES_1218, None, tip, main=[RUN_6F66], order=(tip, MAIN_TIP),
                       ancestry="identical", behind=(0, None))
    with pytest.raises(land_pr.LandError, match=r"check 10: main unverified at dddddddddddd: no Tests run on main's head"):
        _run(1218, gh, git)


def test_no_completed_run_refuses():
    """Revert-proof: without the no-completed branch, this raises a TypeError, not a refusal."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218, main=[_run_row(1, MAIN_TIP, "in_progress", None)])
    with pytest.raises(land_pr.LandError, match=r"check 10: main unverified at 6f66fd238325: none of main's last 1"):
        _run(1218, gh, git)


@pytest.mark.parametrize("call, response, message", [
    pytest.param(MAIN_COMMITS_CALL, None, r"CalledProcessError", id="commits-call-fails"),
    pytest.param(MAIN_RUNS_CALL, None, r"CalledProcessError", id="runs-call-fails"),
    pytest.param(MAIN_COMMITS_CALL, '["main"]', r"main's commits read", id="commits-garbled"),
    pytest.param(MAIN_COMMITS_CALL, "[]", r"main's commits read", id="commits-empty"),
    pytest.param(MAIN_RUNS_CALL, '[{"id": 1}]', r"KeyError", id="runs-garbled"),
])
def test_a_failed_or_garbled_main_read_refuses(call, response, message):
    """Fail closed, one bounded read each, no retry. Revert-proof: without the try/except
    (or the sha validation), these raise a non-LandError or merge."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    if response is None:
        gh = Flaky(gh.responses, call, 99)
    else:
        gh.responses[call] = response
    with pytest.raises(land_pr.LandError, match=rf"check 10: main unverified: .*{message}"):
        _run(1218, gh, git)
    assert not _merged(gh)
    if response is None:
        assert gh.failed == 1


def test_a_failed_ancestry_read_refuses():
    gh, git = _landing(1218, PR_1218, FILES_1218, None, SHA_3AF, main=RUNS_AT_1858, order=ORDER_AT_1858,
                       ancestry="identical", behind=(0, None))
    gh = Flaky(gh.responses, _ancestry_call(SHA_3AF, SHA_3AF), 99)
    with pytest.raises(land_pr.LandError, match=r"check 10: main unverified at 3af31061bd34: compare with .*failed"):
        _run(1218, gh, git)


def test_only_tests_yml_push_runs_are_read():
    """Review of #1898, point 1: deps-canary's scheduled `suite` (check suite 104084451041 on
    6f66fd23) must not count; only tests.yml's push runs do."""
    assert "/actions/workflows/tests.yml/runs?branch=main&event=push&" in land_pr.MAIN_RUNS


def test_fixes_red_main_label_admits_the_fix(capsys):
    """The named override: the fix (or the revert) lands on a red main. Revert-proof:
    without the override branch, it is refused as red."""
    pr = {**PR_1218, "labels": [{"name": land_pr.FIXES_RED_MAIN}]}
    gh, git = _recording(1218, pr, SQUASH_1218, main=RUNS_RED_471, main_order=ORDER_RED)
    assert _run(1218, gh, git) == 0
    assert MAIN_COMMITS_CALL not in gh.calls and MAIN_RUNS_CALL not in gh.calls
    assert "check 10: OVERRIDDEN by the 'fixes-red-main' label" in capsys.readouterr().out


def test_fixes_red_main_does_not_lift_check_11():
    """The override is check 10's alone: a stale base still refuses. Revert-proof: without
    `check_fresh_base`, it merges."""
    pr = {**PR_1218, "labels": [{"name": land_pr.FIXES_RED_MAIN}]}
    gh, git = _recording(1218, pr, SQUASH_1218, behind=(land_pr.STALE_COMMITS + 1, STARTED))
    with pytest.raises(land_pr.LandError, match=r"check 11: "):
        _run(1218, gh, git)


def test_green_main_and_fresh_base_land(capsys):
    """Revert-proof: without `check_main_green` or `check_fresh_base`, its line is not printed."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    assert _run(1218, gh, git) == 0
    out = capsys.readouterr().out
    assert f"check 10: main's Tests run on its head {MAIN_TIP[:12]} is green." in out
    assert f"check 11: run {RUN_ID} (job {JOB_IDS['lock-tests']}) ran on base ffffffffffff, 0 commit(s)" in out


@pytest.mark.parametrize("behind, refused", [
    pytest.param((land_pr.STALE_COMMITS, "2026-10-14T09:59:00Z"), False, id="N-commits-23h59"),
    pytest.param((land_pr.STALE_COMMITS + 1, "2026-10-13T10:30:00Z"), True, id="N+1-commits"),
    pytest.param((1, "2026-10-14T10:01:00Z"), True, id="1-commit-24h01"),
    pytest.param((0, None), False, id="current"),
])
def test_stale_base_limits(behind, refused, capsys):
    """N commits and 24 h, from the checks' start to main's tip. Revert-proof: without
    `check_fresh_base`, the refused cases merge and the others print no check-11 line."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218, behind=behind)
    if refused:
        with pytest.raises(land_pr.LandError, match=r"check 11: .*limits: 10 commits, 24 h"):
            _run(1218, gh, git)
    else:
        assert _run(1218, gh, git) == 0
        assert f"ran on base ffffffffffff, {behind[0]} commit(s)" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["internal_docs/program/retirement_ledger.toml", "tests/test_x.py"])
def test_a_ledger_or_tests_edit_on_a_stale_base_refuses(path):
    """The ledger is outside tests/ but is a BASELINE file. Revert-proof: without
    `check_fresh_base`, both merge."""
    pr = {**PR_1218, "labels": [{"name": land_pr.HUMAN_GATE}]}
    gh, git = _d4([_row(path)], pr=pr, behind=BEHIND_1698)
    with pytest.raises(land_pr.LandError, match=rf"check 11: #1218 touches {path} .*68 commit"):
        _run(1218, gh, git)


def test_a_pr_outside_tests_and_baseline_skips_check_11(capsys):
    """No log is read for a src-only PR on a stale base. Revert-proof: without
    `check_fresh_base`, the 'neither' line is not printed."""
    gh, git = _recording(1097, PR_1097, SQUASH_1097, FILES_1097, behind=BEHIND_1698)
    assert _run(1097, gh, git) == 0
    assert not any("/logs" in c for c in gh.calls)
    assert "check 11: #1097 touches neither tests/ nor a baseline file" in capsys.readouterr().out


@pytest.mark.parametrize("log", [
    pytest.param("2026-10-13T10:00:05Z HEAD is now at 54b621b Some other commit\n", id="no-checkout-line"),
    pytest.param(_checkout("0" * 40), id="another-head"),
    pytest.param("", id="empty-log"),
])
def test_an_unreadable_base_refuses(log):
    """Fail closed: a log that does not name this head's merge ref is not a fresh base.
    Revert-proof: without `check_fresh_base`, each merges."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    gh.responses[_log_call(_job("lock-tests"))] = log
    with pytest.raises(land_pr.LandError, match=r"check 11: the base #1218's required checks ran on .*could not be read"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_a_failed_log_read_refuses():
    """Revert-proof: without `check_fresh_base`, it merges."""
    gh, git = _recording(1218, PR_1218, SQUASH_1218)
    gh = Flaky(gh.responses, "api --allow-escape-sequences", 99)
    with pytest.raises(land_pr.LandError, match=r"check 11: .*could not be read \(CalledProcessError"):
        _run(1218, gh, git)
    assert not _merged(gh)


def test_a_required_status_context_cannot_prove_its_base():
    """A required check reported as a commit status names no Actions job. Revert-proof:
    without `check_fresh_base`, it merges."""
    rollup = [_context("lock-tests", "SUCCESS"), *_rollup(names=land_pr.REQUIRED_CHECKS[1:])]
    gh, git = _recording(1218, {**PR_1218, "statusCheckRollup": rollup}, SQUASH_1218)
    with pytest.raises(land_pr.LandError, match=r"check 11: required check 'lock-tests' names no Actions job"):
        _run(1218, gh, git)


def test_every_workflow_run_is_judged_the_stale_one_refuses():
    """A second run of the required checks (after a reopen, say) does not hide a stale one,
    as check 6 judges every run. Revert-proof: without `check_fresh_base`, it merges."""
    stale_run = "37000000001"
    rollup = [*_rollup(), *[dict(run, detailsUrl=_job(run["name"], stale_run)) for run in _rollup()]]
    gh, git = _recording(1218, {**PR_1218, "statusCheckRollup": rollup}, SQUASH_1218)
    gh.responses[_log_call(_job("lock-tests", stale_run))] = _checkout(PR_1218["headRefOid"], BASE_1698)
    gh.responses[_behind_call(BASE_1698)] = json.dumps(list(BEHIND_1698))
    with pytest.raises(land_pr.LandError, match=rf"check 11: .*run {stale_run} .*68 commit"):
        _run(1218, gh, git)
