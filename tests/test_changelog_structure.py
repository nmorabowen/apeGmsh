"""Structural guards for CHANGELOG.md, and the rule that PRs leave it alone.

PRs add ``changelog.d/`` fragments; ``scripts/changelog.py --base`` fails
a PR that edits CHANGELOG.md, because GitHub ignored the ``merge=union``
driver the file used to carry and flagged every pair of such PRs
CONFLICTING (#1267, #1279). The structure checks below stay: they catch
the mangling modes union left behind while it was active
(2026-06-12 to 2026-10-03):

* a PR that appends to the frozen single-line ``## Unreleased — …``
  ledger merges into TWO near-identical header lines instead of
  conflicting;
* two PRs that each insert a section at the anchor merge with the blank
  line between them dropped, so one section's last line runs straight
  into the next ``###`` header. It happened three times on 2026-09-25
  alone (#1172 twice, #1169's rebase), and 56 older sections carried it.

These tests run in the curated suite, so a mangled merge turns main
red loudly instead of shipping a corrupted changelog. See
``internal_docs/changelog_workflow.md``.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- changelog fragments (scripts/changelog.py) -----------------------------

_spec = importlib.util.spec_from_file_location(
    "changelog_tool", _REPO_ROOT / "scripts" / "changelog.py",
)
changelog_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog_tool)  # type: ignore[union-attr]


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


def _headings_without_blank_line(lines: list[str]) -> list[int]:
    """1-based line numbers of ``###`` headings whose previous line is not
    blank. Headings inside fenced code blocks (a ``# comment`` in a code
    sample) are not headings and are skipped."""
    bad: list[int] = []
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or i == 0 or not ln.startswith("### "):
            continue
        if lines[i - 1].strip():
            bad.append(i + 1)
    return bad


def test_every_section_heading_has_a_blank_line_before_it() -> None:
    """The second union-merge mangling mode: a dropped blank line.

    When two PRs insert sections at the anchor, the union driver keeps
    both but can lose the blank line between them, and the next ``###``
    header becomes a continuation of the previous section's last
    paragraph. Fix: insert the blank line; it is an insertion, so the
    insert-only rule allows it.
    """
    bad = _headings_without_blank_line(_changelog_lines())
    assert not bad, (
        f"CHANGELOG.md has '###' headings with no blank line before them "
        f"at lines {bad}. A union merge dropped the separator between two "
        f"sections; insert a blank line above each."
    )


def test_the_blank_line_check_catches_the_merge_shape() -> None:
    """The check sees the exact shape a union merge leaves behind, and
    ignores a ``#`` line inside a code sample."""
    merged = [
        "### FIXED — one section",
        "",
        "its last line.",
        "### ADDED — the next section",
        "",
        "```python",
        "x = 1",
        "### not a heading, a comment in a sample",
        "```",
    ]
    assert _headings_without_blank_line(merged) == [4]


def test_entry_anchor_comment_present() -> None:
    """The insertion anchor survives — without it contributors lose
    the documented single insertion point and drift back to editing
    the header."""
    assert any(
        changelog_tool.ANCHOR_MARK in ln
        for ln in _changelog_lines()
    ), (
        "CHANGELOG.md lost its entry-anchor comment (the '<!-- ⚓ "
        "FRAGMENTS ARE ASSEMBLED ... -->' block under the Unreleased "
        "header). Restore it from git history."
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
        if changelog_tool.ANCHOR_MARK in ln
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


def test_union_driver_not_registered() -> None:
    """.gitattributes keeps no union driver on CHANGELOG.md. GitHub never
    honoured it (#1267, #1279), so it only made local merges disagree with
    GitHub and hid real conflicts."""
    attrs = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert not any(
        ln.split("#", 1)[0].split()[:1] == ["CHANGELOG.md"]
        for ln in attrs.splitlines()
    ), ".gitattributes sets an attribute on CHANGELOG.md again; see its comment."


_ANCHOR_REGION = """\
# Changelog

## Unreleased — frozen ledger line

<!-- ⚓ FRAGMENTS ARE ASSEMBLED BELOW THIS COMMENT (newest first).
     Do not edit this file in a PR. -->

### CHANGED — an older section

Its body.
"""


def _frag(title: str) -> str:
    return f"### ADDED — {title}\n\nBody of {title}.\n"


def _make_root(tmp_path: Path, changelog: str = _ANCHOR_REGION) -> Path:
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tmp_path


def test_repo_changelog_check_passes() -> None:
    """The real repo: every fragment and CHANGELOG.md are well formed."""
    assert changelog_tool.check(_REPO_ROOT) == []


def test_check_flags_bad_fragment_names_and_content(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    d = root / "changelog.d"
    (d / "README.md").write_text("not a fragment\n", encoding="utf-8")
    (d / "Bad_Name.md").write_text(_frag("x"), encoding="utf-8")
    (d / "no-header.md").write_text("just text\n", encoding="utf-8")
    (d / "two-headers.md").write_text(
        _frag("a") + "\n### ADDED — b\n\nmore\n", encoding="utf-8")
    (d / "h2.md").write_text(
        "### ADDED — a\n\nx\n\n## Unreleased\n", encoding="utf-8")
    (d / "no-newline.md").write_text("### ADDED — a\n\nx", encoding="utf-8")
    problems = "\n".join(changelog_tool.check(root))
    for name in ("Bad_Name", "no-header", "two-headers", "h2", "no-newline"):
        assert name in problems, problems
    assert "README" not in problems
    assert changelog_tool.main(["--check", "--root", str(root)]) == 1


def test_check_flags_glued_header_in_changelog(tmp_path: Path) -> None:
    """#1219: a '### ' header whose previous line is not blank."""
    glued = _ANCHOR_REGION.replace(
        "Its body.\n", "Its body.\n### ADDED — glued on\n\nx\n")
    root = _make_root(tmp_path, glued)
    problems = changelog_tool.check(root)
    assert any("no blank" in p for p in problems), problems
    assert changelog_tool.main(["--check", "--root", str(root)]) == 1


def test_assemble_is_ordered_and_idempotent(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    (root / "changelog.d" / "b-second.md").write_text(_frag("B"), encoding="utf-8")
    (root / "changelog.d" / "a-first.md").write_text(_frag("A"), encoding="utf-8")
    assert changelog_tool.assemble(root) == ["a-first.md", "b-second.md"]
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (
        "-->\n\n### ADDED — A\n\nBody of A.\n\n### ADDED — B\n\nBody of B."
        "\n\n### CHANGED — an older"
    ) in text
    assert changelog_tool.check(root) == []
    assert changelog_tool.assemble(root) == []
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == text


def test_three_concurrent_fragment_branches_merge_clean(tmp_path: Path) -> None:
    """The link's done-when: three PRs, three fragments, no conflict,
    blank-separated sections after assemble, second assemble a no-op."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    root = _make_root(tmp_path)
    (root / "changelog.d" / "README.md").write_text("readme\n", encoding="utf-8")
    shutil.copy(_REPO_ROOT / ".gitattributes", root / ".gitattributes")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t",
             "-c", "commit.gpgsign=false", *args],
            cwd=root, check=True, capture_output=True,
        )

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    for n in ("one", "two", "three"):
        git("checkout", "-q", "-b", f"pr-{n}", "main")
        (root / "changelog.d" / f"2026-09-29-{n}.md").write_text(
            _frag(n), encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", n)
    git("checkout", "-q", "main")
    for n in ("one", "two", "three"):
        git("merge", "-q", "--no-edit", f"pr-{n}")  # check=True: no conflict

    assert changelog_tool.check(root) == []
    assert len(changelog_tool.fragment_paths(root)) == 3
    changelog_tool.assemble(root)
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.count("\n### ") == 4  # 3 new + the older section
    assert "\n\n\n" not in text
    for a, b in (("one", "three"), ("three", "two")):  # sorted filenames
        assert text.index(f"— {a}\n") < text.index(f"— {b}\n")
    assert changelog_tool.check(root) == []
    assert changelog_tool.assemble(root) == []
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == text


def _git_repo(root: Path):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t",
             "-c", "commit.gpgsign=false", *args],
            cwd=root, check=True, capture_output=True,
        )

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    return git


def test_pr_diff_rejects_a_direct_changelog_section(tmp_path: Path) -> None:
    """The #1267 / #1279 shape: a feature PR inserts its section into
    CHANGELOG.md. GitHub flags every pair of these CONFLICTING."""
    root = _make_root(tmp_path)
    git = _git_repo(root)
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        text.replace("-->\n", "-->\n\n" + _frag("direct")), encoding="utf-8")
    (root / "feature.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "feature")
    problems = changelog_tool.check_pr_diff("HEAD^1", root)
    assert len(problems) == 1 and "must not edit" in problems[0]
    assert changelog_tool.main(["--check", "--root", str(root),
                                "--base", "HEAD^1"]) == 1


def test_pr_diff_allows_fragments_assemble_and_tool_changes(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    git = _git_repo(root)
    (root / "changelog.d" / "2026-10-03-frag.md").write_text(
        _frag("frag"), encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "fragment PR")
    assert changelog_tool.check_pr_diff("HEAD^1", root) == []

    changelog_tool.assemble(root)  # edits CHANGELOG.md, deletes the fragment
    git("add", "-A")
    git("commit", "-q", "-m", "assemble PR")
    assert changelog_tool.check_pr_diff("HEAD^1", root) == []

    (root / "scripts").mkdir()
    (root / "scripts" / "changelog.py").write_text("# tool\n", encoding="utf-8")
    cl = root / "CHANGELOG.md"
    cl.write_text(cl.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "tooling PR")
    assert changelog_tool.check_pr_diff("HEAD^1", root) == []
