"""scripts/land_pr.py replays: the lost-work PRs are refused or flagged, a clean PR lands.

The recordings are what `gh pr view --json ...`, `gh pr diff --name-only`,
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
LABELS = [{"name": "program"}, {"name": "slice"}, {"name": "semantic"}]
FREEZE_STUDIO = [*LABELS, {"name": "freeze:src/apeGmsh/studio/*"}]


def _recording(number, pr, squash, files=FILES_1218, *, compare="behind", tip="", labels=LABELS, local=None, porcelain=""):
    head = pr["headRefOid"]
    gh = Replay({
        f"pr view {number} --json {land_pr.PR_FIELDS}": json.dumps(pr),
        "label list --limit 200 --json name": json.dumps(labels),
        f"pr diff {number} --name-only": "".join(f + "\n" for f in files),
        f"pr merge {number} --squash --match-head-commit {head}": "",
        f"pr view {number} --json mergeCommit": json.dumps({"mergeCommit": {"oid": squash}}),
        f"api repos/{{owner}}/{{repo}}/compare/main...{squash} --jq .status": compare + "\n",
    })
    git = Replay({
        "rev-parse HEAD": (local or head) + "\n",
        "status --porcelain": porcelain,
        f"ls-remote origin refs/heads/{pr['headRefName']}": tip,
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
    assert gh.calls.index(merge) > gh.calls.index("pr diff 1218 --name-only")
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
    assert sum(c.startswith("api ") for c in gh.calls) == land_pr.COMPARE_TRIES


def test_main_reports_refusal_on_stderr(monkeypatch, capsys):
    gh, git = _recording(858, PR_858, "unused")
    monkeypatch.setattr(land_pr, "run_gh", gh)
    monkeypatch.setattr(land_pr, "run_git", git)
    assert land_pr.main(["858"]) == 1
    assert "land_pr: check 1" in capsys.readouterr().err
