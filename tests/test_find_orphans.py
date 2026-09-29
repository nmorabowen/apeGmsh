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
from datetime import datetime, timezone
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


def test_858_diverged_is_flagged():
    hits = [f for f in _run() if f["pr"] == 858]
    assert [h["kind"] for h in hits] == ["not-on-main"]
    assert "diverged" in hits[0]["detail"]


def test_1097_branch_moved_past_head_is_flagged():
    hits = [f for f in _run() if f["pr"] == 1097]
    assert [h["kind"] for h in hits] == ["pushed-after-merge"]


def test_clean_pr_not_flagged():
    assert not [f for f in _run() if f["pr"] == 1200]


def test_deleted_branch_not_flagged_by_detection_2():
    assert not [f for f in _run() if f["pr"] == 1201]


def test_old_pr_outside_window_ignored():
    assert not [f for f in _run() if f["pr"] == 5]


def test_main_exit_codes_and_formats(capsys):
    dirty = make_fetch([PR858, PR1097])
    assert fo.main(["--since", "36500", "--format", "json"], fetch=dirty) == 1
    assert {f["pr"] for f in json.loads(capsys.readouterr().out)} == {858, 1097}
    assert fo.main(["--since", "36500", "--format", "md"], fetch=dirty) == 1
    assert "#858" in capsys.readouterr().out
    clean = make_fetch([PR_CLEAN, PR_DELETED])
    assert fo.main(["--since", "36500"], fetch=clean) == 0
