"""X4-b ratchet: the freeze-listed surfaces may not grow (shrink-only LOC).

Program link X4-b (#1807); rulings on #1199.  Each entry is a surface the
program is retiring or folding; none of them may get bigger while it waits.

LOC definition (stable, do not change): the number of physical lines of a
``.py`` file that contain a non-whitespace character.  Blank lines count 0;
comment and docstring lines count 1.  A surface is the sum over its files.
``Results.from_recorders`` is counted over its AST line span
(``lineno`` through ``end_lineno``, decorators excluded), non-blank lines only.

The assertion is ``current <= baseline``: growth fails; a shrink passes and
prints the lower number to adopt (lowering a baseline is optional and
human-gated, so a deletion PR need not touch this file).  Baselines are the
exact counts at the X4-b landing commit,
with no slack.  Raising one is a maintainer gate (PROGRAM section 4,
``human-gate`` label), never a fix for a red test.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "apeGmsh"

# surface -> pinned LOC (measured at the X4-b landing commit).
BASELINE: dict[str, int] = {
    "viewers": 57452,
    "studio": 9679,
    "hpc": 554,
    "ground_motions": 979,
    "from_recorders": 1237,
    "section_gui": 1784,
    "rebar": 1924,
}

_DIRS = {
    "viewers": ("viewers",),
    "studio": ("studio",),
    "hpc": ("hpc",),
    "ground_motions": ("ground_motions",),
    "rebar": ("rebar",),
}
_FILES = {
    "section_gui": (
        "sections/_builder_gui.py",
        "sections/_inspector.py",
        "sections/_properties.py",
    ),
    "rebar": ("core/RebarComposite.py",),
}
_TRANSCODERS = "results/transcoders"
_RESULTS = "results/Results.py"


def loc(text: str) -> int:
    """Physical lines with at least one non-whitespace character."""
    return sum(1 for line in text.splitlines() if line.strip())


def _tree_loc(rel: str, root: Path) -> int:
    return sum(loc(p.read_text(encoding="utf-8"))
               for p in sorted((root / rel).rglob("*.py")))


def from_recorders_loc(source: str) -> int:
    """Non-blank lines in the span of ``Results.from_recorders``."""
    lines = source.splitlines()
    for cls in ast.parse(source).body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Results":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "from_recorders":
                    return loc("\n".join(lines[fn.lineno - 1:fn.end_lineno]))
    raise AssertionError("Results.from_recorders not found")


def count(surface: str, root: Path = SRC) -> int:
    if surface == "from_recorders":
        return (_tree_loc(_TRANSCODERS, root)
                + from_recorders_loc((root / _RESULTS).read_text(encoding="utf-8")))
    total = sum(_tree_loc(d, root) for d in _DIRS.get(surface, ()))
    for f in _FILES.get(surface, ()):
        total += loc((root / f).read_text(encoding="utf-8"))
    return total


def _verdict(surface: str, current: int, baseline: int) -> tuple[bool, str | None]:
    """(ok, message): growth is not ok; a shrink is ok and names the number to adopt."""
    if current > baseline:
        return False, (f"{surface}: {current} LOC > baseline {baseline}. This surface is "
                       "frozen or being retired; it may only shrink. Do not raise the baseline.")
    if current < baseline:
        return True, (f"{surface}: {current} LOC < baseline {baseline}. Shrinking is "
                      f"welcome: lower BASELINE[{surface!r}] to {current}.")
    return True, None


@pytest.mark.parametrize("surface", sorted(BASELINE))
def test_loc_does_not_grow(surface: str) -> None:
    ok, msg = _verdict(surface, count(surface), BASELINE[surface])
    if ok and msg:
        print(msg)  # visible with -s / in the CI log summary
    assert ok, msg


def test_loc_definition() -> None:
    assert loc("a = 1\n\n   \n# c\n'''d\n\ne'''\n") == 4


def test_self_test_growth_fails_and_shrink_prints_number(tmp_path: Path) -> None:
    pkg = tmp_path / "hpc"
    pkg.mkdir()
    (pkg / "m.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    base = count("hpc", tmp_path)
    assert base == 2 and _verdict("hpc", base, base) == (True, None)
    (pkg / "m.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    grown = _verdict("hpc", count("hpc", tmp_path), base)
    assert grown[0] is False and "Do not raise the baseline" in grown[1]
    (pkg / "m.py").write_text("x = 1\n", encoding="utf-8")
    shrunk = _verdict("hpc", count("hpc", tmp_path), base)
    assert shrunk[0] is True and "lower BASELINE['hpc'] to 1" in shrunk[1]


def test_from_recorders_span() -> None:
    src = (
        "class Results:\n"
        "    def other(self):\n"
        "        return 1\n"
        "\n"
        "    @classmethod\n"
        "    def from_recorders(cls):\n"
        "        a = 1\n"
        "\n"
        "        return a\n"
        "\n"
        "    def after(self):\n"
        "        pass\n"
    )
    assert from_recorders_loc(src) == 3
