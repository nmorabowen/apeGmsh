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


def _rollup(names=land_pr.REQUIRED_CHECKS, **overrides):
    runs = [{"__typename": "CheckRun", "name": n, "status": "COMPLETED", "conclusion": "SUCCESS"} for n in names]
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
COMPARE = "api repos/{owner}/{repo}/compare/"
LABELS = [{"name": "program"}, {"name": "slice"}, {"name": "semantic"}]
FREEZE_STUDIO = [*LABELS, {"name": "freeze:src/apeGmsh/studio/*"}]


def _files_call(number):
    """The one files listing checks 5, 7 and 8 read (#1775, #1686)."""
    return f"api repos/{{owner}}/{{repo}}/pulls/{number}/files --paginate --jq {land_pr.FILES_JQ}"


def _row(path, status="modified", previous=None, patch="@@ -1 +1 @@\n-a\n+b"):
    """One `gh api .../pulls/N/files --jq FILES_JQ` line: [filename, status, previous, changes, patch]."""
    changes = 0 if patch is None else sum(ln[:1] in "+-" for ln in patch.splitlines() if not ln.startswith("@@"))
    return json.dumps([path, status, previous, changes, patch])


def _recording(number, pr, squash, files=FILES_1218, *, compare="behind", tip="", labels=LABELS, local=None,
               porcelain="", rows=None, issues=None, shows=None):
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
        f"pr merge {number} --squash --match-head-commit {head}": "",
        f"pr view {number} --json mergeCommit": json.dumps({"mergeCommit": {"oid": squash}}),
        f"api repos/{{owner}}/{{repo}}/compare/main...{squash} --jq .status": compare + "\n",
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
    rollup = [_context("lock-tests", "SUCCESS"), *_rollup(names=land_pr.REQUIRED_CHECKS[1:])]
    gh, git = _recording(1218, {**PR_1218, "statusCheckRollup": rollup}, SQUASH_1218)
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
