"""Structural guards for CHANGELOG.md under the union merge driver.

CHANGELOG.md merges with git's built-in ``merge=union`` driver
(``.gitattributes``), which NEVER reports a conflict — it keeps both
sides of any overlapping edit. That is safe for the repo's
insert-only convention (one contiguous ``###`` section per PR at the
anchor comment), but it silently mangles the file when a PR edits an
existing line. The known mangling mode: a PR that appends to the
frozen single-line ``## Unreleased — …`` ledger merges into TWO
near-identical header lines instead of conflicting.

These tests run in the curated suite, so a mangled merge turns main
red loudly instead of shipping a corrupted changelog. See
``internal_docs/changelog_workflow.md``.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _changelog_lines() -> list[str]:
    return (_REPO_ROOT / "CHANGELOG.md").read_text(
        encoding="utf-8",
    ).splitlines()


def test_exactly_one_unreleased_header() -> None:
    """The frozen ledger line exists exactly once.

    Two ``## Unreleased`` lines is the union-merge mangling signature:
    a PR edited the frozen line (old convention) and the driver kept
    both versions. Fix: delete the stale duplicate, fold the PR's item
    into a ``###`` section at the anchor instead.
    """
    headers = [
        ln for ln in _changelog_lines() if ln.startswith("## Unreleased")
    ]
    assert len(headers) == 1, (
        f"CHANGELOG.md has {len(headers)} '## Unreleased' header lines "
        f"(expected exactly 1). This is the union-merge mangling mode: "
        f"some PR appended to the FROZEN ledger line instead of adding "
        f"a '###' section at the anchor comment. Keep the longer line, "
        f"delete the other, and move the missing item into a section."
    )


def test_entry_anchor_comment_present() -> None:
    """The insertion anchor survives — without it contributors lose
    the documented single insertion point and drift back to editing
    the header."""
    assert any(
        "NEW ENTRIES GO DIRECTLY BELOW THIS COMMENT" in ln
        for ln in _changelog_lines()
    ), (
        "CHANGELOG.md lost its entry-anchor comment (the '<!-- ⚓ NEW "
        "ENTRIES ... -->' block under the Unreleased header). Restore "
        "it from internal_docs/changelog_workflow.md."
    )


def test_entry_anchor_is_top_of_unreleased() -> None:
    """The anchor is the first thing under the ``## Unreleased`` header,
    and there is exactly one.

    "Directly below the anchor" and "the top of Unreleased" are the same
    place only while nothing sits between the header and the anchor. One
    section inserted above it (#974, 2026-08-15) split them; 32 more
    followed, the anchor sank to line 868, and PRs landed on both sides.
    Two anchors is the union-merge signature of a stale branch that
    inserted at an old anchor position.
    """
    lines = _changelog_lines()
    anchors = [
        i for i, ln in enumerate(lines)
        if "NEW ENTRIES GO DIRECTLY BELOW THIS COMMENT" in ln
    ]
    assert len(anchors) == 1, (
        f"CHANGELOG.md has {len(anchors)} entry-anchor comments at lines "
        f"{[i + 1 for i in anchors]} (expected exactly 1, directly under "
        f"the '## Unreleased' header). Keep that one, delete the others."
    )
    header = next(
        i for i, ln in enumerate(lines) if ln.startswith("## Unreleased")
    )
    first = next(i for i in range(header + 1, len(lines)) if lines[i].strip())
    assert first == anchors[0], (
        f"CHANGELOG.md line {first + 1} sits between the '## Unreleased' "
        f"header and the entry anchor (line {anchors[0] + 1}). New "
        f"sections go directly BELOW the anchor; move this one there."
    )


def test_no_merge_conflict_markers() -> None:
    """Belt-and-braces: no conflict markers committed.

    Union merges don't produce markers, but a hand-resolved conflict
    elsewhere in the file can leave them behind.
    """
    bad = [
        i + 1 for i, ln in enumerate(_changelog_lines())
        if ln.startswith(("<<<<<<<", ">>>>>>>", "======="))
        and not ln.startswith("=========")  # markdown setext rules are longer
    ]
    assert not bad, f"CHANGELOG.md has conflict markers at lines {bad}."


def test_union_driver_registered() -> None:
    """.gitattributes keeps the union driver on CHANGELOG.md — the
    insert-only convention is only conflict-free while it is active."""
    attrs = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert any(
        ln.split("#", 1)[0].split() == ["CHANGELOG.md", "merge=union"]
        for ln in attrs.splitlines()
    ), (
        ".gitattributes no longer declares 'CHANGELOG.md merge=union'. "
        "Without it every concurrent CHANGELOG edit conflicts again "
        "(the pre-2026-06-12 treadmill)."
    )
