"""Every ``qt``-marked test file lies under a directory the qt lane runs.

#1241: ``tests/sections/test_builder_gui_b6.py`` was marked ``qt`` (so the
``suite`` lane deselects it) while the ``qt-window-tests`` step only searched
``tests/viewers``: the file ran nowhere. This reads the workflow text, so a
marked file outside the lane's search roots fails here instead of silently
vanishing.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
TESTS = ROOT / "tests"


def _lane_roots() -> list[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r'grep -rl "pytest\\\.mark\\\.qt"([^\n|]*?)--include', text)
    assert m, "qt-window-tests discovery loop not found in tests.yml"
    roots = [t.rstrip("/") for t in m.group(1).split() if t.startswith("tests/")]
    assert roots, "qt lane names no directories"
    return roots


def _qt_marked_files() -> list[Path]:
    out = []
    for p in sorted(TESTS.rglob("*.py")):
        if p.resolve() == Path(__file__).resolve():
            continue
        if "pytest.mark.qt" in p.read_text(encoding="utf-8", errors="replace"):
            out.append(p)
    return out


def test_every_qt_marked_file_is_in_a_lane_root():
    roots = _lane_roots()
    orphans = [
        p.relative_to(ROOT).as_posix()
        for p in _qt_marked_files()
        if not any(p.relative_to(ROOT).as_posix().startswith(r + "/") for r in roots)
    ]
    assert not orphans, (
        f"qt-marked files outside the qt lane roots {roots} run nowhere "
        f"(#1241): {orphans}. Add their directory to the grep in the "
        "qt-window-tests step of .github/workflows/tests.yml."
    )


def test_lane_roots_exist():
    for r in _lane_roots():
        assert (ROOT / r).is_dir(), r
