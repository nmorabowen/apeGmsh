"""Golden emit corpus: every cell's deck and h5 dump against its committed golden.

The oracle for pure moves (panel plan ``plan_expert_panel_2026-09.md``,
"The proof regime").  One test per cell of the 7 fixtures x 5 modes x 3
outputs grid; ``README.md`` beside this file says what each axis means.
This module never writes a golden.  On a mismatch it fails with a unified
diff of the differing cell and the regen command; a deliberate emit change
regenerates with ``python -m tests.opensees.golden.regen`` and lists the
changed cells in the PR body.
"""
from __future__ import annotations

import difflib
import inspect
import json
import re
from pathlib import Path

import pytest

from tests.opensees.fixtures import fem_stub
from tests.opensees.golden import builder

_DIFF_LINES = 80
_CELLS = builder.all_cells()


def _read_golden(path: Path) -> str:
    # EOL-normalised: a CRLF checkout is the OS's byte, not the emitter's.
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _assert_same(cell: str, kind: str, golden: Path, got: str) -> None:
    assert golden.exists(), (
        f"{cell}: missing golden {golden.relative_to(builder.GOLDEN_DIR)}; "
        f"run `{builder.REGEN_COMMAND}` and review the new cell"
    )
    want = _read_golden(golden)
    if kind == "deck":
        # Float literals within builder.FLOAT_REL_TOL (cross-platform libm
        # ulps); every other byte exact.
        bad = builder.first_deck_mismatch(want, got)
    else:
        # The dump already rounds floats before hashing: exact text.
        bad = None if got == want else next(
            (i for i, (a, b) in enumerate(
                zip(want.split("\n"), got.split("\n"))) if a != b),
            min(want.count("\n"), got.count("\n")),
        )
    if bad is None:
        return
    wl, gl = want.split("\n"), got.split("\n")
    first = (
        f"first difference beyond tolerance at line {bad + 1}:\n"
        f"  golden : {wl[bad] if bad < len(wl) else '<EOF>'!r}\n"
        f"  emitted: {gl[bad] if bad < len(gl) else '<EOF>'!r}\n"
    )
    diff = list(difflib.unified_diff(
        want.splitlines(), got.splitlines(),
        fromfile=f"golden/{cell}/{kind}", tofile=f"emitted/{cell}/{kind}",
        lineterm="", n=2,
    ))
    shown = diff[:_DIFF_LINES]
    if len(diff) > _DIFF_LINES:
        shown.append(f"... ({len(diff) - _DIFF_LINES} more diff lines)")
    pytest.fail(
        f"golden {kind} mismatch in cell {cell}\n" + first + "\n".join(shown)
        + f"\nIf the change is deliberate, run `{builder.REGEN_COMMAND}` "
        "and list the changed cells in the PR body.",
        pytrace=False,
    )


_CI_LINE = "geomTransf Linear 3 0.25881904510252085 0.0 0.9659258262890684"


@pytest.mark.parametrize(
    ("got", "matches"),
    [
        # #1258: the CI runner's libm, one ulp off in two components.
        ("geomTransf Linear 3 0.2588190451025208 0.0 0.9659258262890682", True),
        # A 1e-9 and a 1e-6 relative change are real changes.
        ("geomTransf Linear 3 0.25881904536133989 0.0 0.9659258262890684",
         False),
        ("geomTransf Linear 3 0.2588193039215659 0.0 0.9659258262890684",
         False),
        # Integers and text stay exact: tag, token, whitespace, sign.
        ("geomTransf Linear 4 0.25881904510252085 0.0 0.9659258262890684",
         False),
        ("geomTransf PDelta 3 0.25881904510252085 0.0 0.9659258262890684",
         False),
        (_CI_LINE + "  ", False),
        ("geomTransf Linear 3 -0.25881904510252085 0.0 0.9659258262890684",
         False),
        ("geomTransf Linear 3 0.25881904510252085 0.0", False),
    ],
)
def test_deck_comparison_tolerates_last_ulp_only(
    got: str, matches: bool,
) -> None:
    want = f"=== model.tcl ===\n{_CI_LINE}\n"
    bad = builder.first_deck_mismatch(want, f"=== model.tcl ===\n{got}\n")
    assert (bad is None) is matches
    if not matches:
        assert bad == 1


@pytest.mark.parametrize(
    ("fixture", "mode", "output"), _CELLS,
    ids=[builder.cell_id(*c) for c in _CELLS],
)
def test_cell_matches_golden(
    fixture: str, mode: str, output: str, tmp_path: Path,
) -> None:
    cell = builder.cell_id(fixture, mode, output)
    deck, dump = builder.golden_paths(fixture, mode, output)
    reason = builder.applicability(fixture, mode, output)
    if reason is not None:
        assert not deck.exists() and not dump.exists(), (
            f"{cell} is n/a ({reason}) but has a committed golden"
        )
        return
    _assert_same(
        cell, "deck", deck,
        builder.render_deck(fixture, mode, output, tmp_path / "deck"),
    )
    _assert_same(
        cell, "h5dump", dump,
        builder.render_h5_dump(fixture, mode, output, tmp_path / "h5"),
    )


def test_manifest_is_the_grid() -> None:
    """MANIFEST.json lists all 105 cells, each golden or n/a with a reason."""
    text = _read_golden(builder.MANIFEST_PATH)
    assert text == builder.manifest_text(), (
        f"MANIFEST.json is stale; run `{builder.REGEN_COMMAND}`"
    )
    cells = json.loads(text)["cells"]
    assert len(cells) == 7 * 5 * 3 == len(_CELLS)
    for cell, entry in cells.items():
        status = entry["status"]
        assert status == "golden" or re.fullmatch(r"n/a: \S.{9,}", status), (
            f"{cell}: status must be 'golden' or 'n/a: <reason>', "
            f"got {status!r}"
        )


def test_fixture_roster_is_every_make_factory() -> None:
    """A new ``make_*`` factory must join the grid, not fall outside it."""
    factories = {
        name for name, obj in vars(fem_stub).items()
        if name.startswith("make_") and inspect.isfunction(obj)
    }
    driven = {spec.factory.__name__ for spec in builder.FIXTURES.values()}
    assert driven == factories, (
        f"not in the corpus: {sorted(factories - driven)}; "
        f"unknown: {sorted(driven - factories)}"
    )


def test_no_orphan_goldens() -> None:
    """Every file under cells/ belongs to a golden cell (no silent leftovers)."""
    owned = {
        p for c in _CELLS if builder.applicability(*c) is None
        for p in builder.golden_paths(*c)
    }
    on_disk = {p for p in builder.CELLS_DIR.rglob("*") if p.is_file()}
    assert on_disk == owned, (
        f"orphans: {sorted(str(p) for p in on_disk - owned)}; "
        f"missing: {sorted(str(p) for p in owned - on_disk)}"
    )


def test_no_workflow_invokes_regen() -> None:
    """Regeneration is a maintainer act; CI must never rewrite the oracle."""
    workflows = builder.REPO_ROOT / ".github" / "workflows"
    files = sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml"))
    assert files, f"no workflows found under {workflows}"
    pattern = re.compile(r"golden[./\\]regen")
    offenders = [
        p.name for p in files
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"workflows invoke the golden regen: {offenders}"
