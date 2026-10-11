"""Lock on the warrant rows (ADR 0121 D3, invariants 9 and 10).

The registry is ``tests/opensees/warrant_rows.py``; the lane rule is applied
by ``tests/conftest.py`` when ``APEGMSH_WARRANT_LANE`` is set. This file
checks, without importing openseespy:

* each row names a member, a stock state (``refused``/``honest``) and two
  node ids that exist (AST of the file, module-level function);
* the stock test runs in ``live-stock`` (``live``, not ``ladruno_fork``) and
  the fork test runs in ``live-fork`` (``ladruno_fork``);
* invariant 10: a refused row's stock test holds a ``pytest.raises`` whose
  ``match`` names the fork; an honest row's stock test asserts;
* a row that cites a roster id is cited back by that roster row;
* invariant 9: every fork member (the live emitter's ``_FORK_ONLY_*`` sets
  and the materials whose docstring carries the fork command header or a
  33xxx tag) is grandfathered or has a refused row,
  and the grandfathered set never grows past its freeze-point count;
* the lane rule itself: a row test that skips in its lane fails it, and a
  ``ladruno_mkl`` skip counts for the fork lane but never for the member.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import apply_warrant_rule
from tests.opensees import warrant_rows as wr
from tests.opensees.unit.test_fork_divergence_roster import ROSTER, parse_roster

ROOT = Path(__file__).resolve().parents[3]
NODE_RE = re.compile(r"^(?P<path>tests/[\w./-]+\.py)::(?P<name>\w+)$")


@lru_cache(maxsize=None)
def _module(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _mark_names(node: ast.expr) -> set[str]:
    """``pytest.mark.X`` names in a decorator or a ``pytestmark`` value."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "mark"):
            names.add(sub.attr)
    return names


def _function(nodeid: str) -> tuple[ast.FunctionDef, set[str]]:
    """The test function a node id names, and the marks it carries."""
    m = NODE_RE.match(nodeid)
    assert m, f"{nodeid!r} is not a tests/...py::<module-level function> id"
    module = _module(m["path"])
    marks: set[str] = set()
    for stmt in module.body:
        if (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "pytestmark"
                        for t in stmt.targets)):
            marks |= _mark_names(stmt.value)
    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == m["name"]:
            for dec in stmt.decorator_list:
                marks |= _mark_names(dec)
            return stmt, marks
    raise AssertionError(f"{nodeid}: no module-level function {m['name']!r}")


def _raises_matches(fn: ast.FunctionDef) -> list[str | None]:
    """The ``match=`` of every ``pytest.raises(...)`` in ``fn``."""
    found: list[str | None] = []
    for sub in ast.walk(fn):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "raises"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "pytest"):
            match = next((k.value for k in sub.keywords if k.arg == "match"), None)
            found.append(match.value if isinstance(match, ast.Constant)
                         and isinstance(match.value, str) else None)
    return found


ROW_IDS = [row.member for row in wr.ROWS]


def test_there_is_one_refused_and_one_honest_exemplar() -> None:
    states = sorted(row.stock_state for row in wr.ROWS)
    assert "refused" in states and "honest" in states
    assert len(set(ROW_IDS)) == len(ROW_IDS), "one row per member"


@pytest.mark.parametrize("row", wr.ROWS, ids=ROW_IDS)
def test_row_shape(row: wr.WarrantRow) -> None:
    assert row.stock_state in wr.STOCK_STATES
    assert row.stock_test != row.fork_test


@pytest.mark.parametrize("row", wr.ROWS, ids=ROW_IDS)
def test_stock_test_runs_in_live_stock(row: wr.WarrantRow) -> None:
    _, marks = _function(row.stock_test)
    assert "live" in marks, f"{row.stock_test}: live-stock selects `live`"
    assert "ladruno_fork" not in marks, (
        f"{row.stock_test}: a ladruno_fork test skips on stock"
    )


@pytest.mark.parametrize("row", wr.ROWS, ids=ROW_IDS)
def test_fork_test_runs_in_live_fork(row: wr.WarrantRow) -> None:
    _, marks = _function(row.fork_test)
    assert "ladruno_fork" in marks, f"{row.fork_test}: live-fork selects `ladruno_fork`"
    assert not marks & {"qt", "subprocess", "bench"}, row.fork_test


@pytest.mark.parametrize("row", wr.ROWS, ids=ROW_IDS)
def test_stock_side_pins_its_state(row: wr.WarrantRow) -> None:
    """Invariant 10: a refused row's test holds ``pytest.raises`` (AST)."""
    fn, _ = _function(row.stock_test)
    matches = _raises_matches(fn)
    if row.stock_state == "refused":
        assert matches, f"{row.stock_test}: a refused row pins pytest.raises"
        assert any(m and "fork" in m.lower() for m in matches), (
            f"{row.stock_test}: the refusal's match= must name the fork"
        )
    else:
        assert not matches, f"{row.stock_test}: an honest row runs, it is not refused"
        assert any(isinstance(n, ast.Assert) for n in ast.walk(fn)), (
            f"{row.stock_test}: an honest row checks an oracle"
        )


@pytest.mark.parametrize("row", [r for r in wr.ROWS if r.roster],
                         ids=[r.member for r in wr.ROWS if r.roster])
def test_roster_row_cites_both_tests(row: wr.WarrantRow) -> None:
    rows, _ = parse_roster(ROSTER.read_text(encoding="utf-8"))
    cited = {r["id"]: r["pinning tests"] for r in rows}
    assert row.roster in cited, f"no roster row {row.roster}"
    for nodeid in (row.stock_test, row.fork_test):
        assert nodeid in cited[row.roster], f"roster {row.roster} must cite {nodeid}"


#: The two structural signals of a fork material in a class docstring: the
#: command header the fork classes carry, and a tag in the fork's private
#: 33xxx band. ``PlaneStressRebar`` names the fork only in passing, under a
#: "stock OpenSees" header, and stays out.
FORK_COMMAND_RE = re.compile(r"OpenSees command \(Ladruno fork[,)]")
FORK_TAG_RE = re.compile(r"``(?:ND|MAT)_TAG`` \*\*33\d{3}\*\*")


def is_fork_material(node: ast.ClassDef) -> bool:
    """A public, non-warning class whose docstring carries a fork signal."""
    doc = ast.get_docstring(node) or ""
    return (not node.name.startswith("_")
            and not any("Warning" in ast.unparse(b) for b in node.bases)
            and bool(FORK_COMMAND_RE.search(doc) or FORK_TAG_RE.search(doc)))


def fork_members() -> set[str]:
    """Fork elements and integrators (the live gate sets) and materials.

    A fork material is a class in ``src/apeGmsh/opensees/material/`` that
    :func:`is_fork_material` accepts; read on the AST, never by name.
    """
    from apeGmsh.opensees.emitter import live
    members = set(live._FORK_ONLY_ELEMENTS) | set(live._FORK_ONLY_INTEGRATORS)
    for path in sorted((ROOT / "src" / "apeGmsh" / "opensees" / "material").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and is_fork_material(node):
                members.add(node.name)
    return members


def test_the_classifier_reads_structure_not_names() -> None:
    def cls(src: str) -> ast.ClassDef:
        node = ast.parse(src).body[0]
        assert isinstance(node, ast.ClassDef)
        return node
    assert is_fork_material(cls(
        'class FooFork(NDMaterial):\n    """``nDMaterial Foo``.\n\n'
        '    ``ND_TAG`` **33099**\n    """'))
    assert is_fork_material(cls(
        'class Bar(NDMaterial):\n    """OpenSees command (Ladruno fork)::"""'))
    assert not is_fork_material(cls(
        'class LadrunoLookalike(NDMaterial):\n    """Tcl signature (stock OpenSees)."""'))
    assert not is_fork_material(cls(
        'class LadrunoW(UserWarning):\n    """OpenSees command (Ladruno fork)::"""'))


def test_fork_members_are_grandfathered_or_warranted() -> None:
    """Invariant 9: a fork member added after the freeze has a refused row."""
    members = fork_members()
    stale = wr.GRANDFATHERED_FORK_MEMBERS - members
    assert not stale, f"grandfathered names no longer fork-only; remove: {sorted(stale)}"
    refused = {r.member for r in wr.ROWS if r.stock_state == "refused"}
    new = members - wr.GRANDFATHERED_FORK_MEMBERS - refused
    assert not new, (
        f"fork members added after the freeze need a refused warrant row in "
        f"tests/opensees/warrant_rows.py (ADR 0121 invariant 9): {sorted(new)}"
    )
    assert len(wr.GRANDFATHERED_FORK_MEMBERS) <= wr.FREEZE_POINT_FORK_MEMBERS


# -- the lane rule -----------------------------------------------------------

TET10 = wr.ROWS[0]
assert TET10.member == "TenNodeTetrahedron"


def _skip_report(nodeid: str, reason: str, when: str = "setup") -> pytest.TestReport:
    return pytest.TestReport(
        nodeid=nodeid, location=(nodeid, 0, "t"), keywords={},
        outcome="skipped", longrepr=(nodeid, 1, f"Skipped: {reason}"), when=when,
    )


def _item(*marks: str) -> SimpleNamespace:
    return SimpleNamespace(iter_markers=lambda: [SimpleNamespace(name=m) for m in marks])


def test_lane_env_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert wr.lane_from_env({}) is None
    assert wr.lane_from_env({wr.LANE_ENV: "fork"}) == "fork"
    with pytest.raises(ValueError, match="not a warrant lane"):
        wr.lane_from_env({wr.LANE_ENV: "Stock"})


@pytest.mark.parametrize("lane", wr.LANES)
def test_a_row_test_that_skips_fails_its_lane(
    lane: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(wr.LANE_ENV, lane)
    nodeid = TET10.test_for(lane)
    report = _skip_report(nodeid, "stock-only")
    apply_warrant_rule(_item("live"), report)
    assert report.failed
    assert "TenNodeTetrahedron" in str(report.longrepr)
    assert "fails the lane" in str(report.longrepr)


def test_an_xfailed_row_test_fails_its_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(wr.LANE_ENV, "stock")
    report = _skip_report(TET10.stock_test, "", when="call")
    report.wasxfail = "known"
    apply_warrant_rule(_item("live"), report)
    assert report.failed and not hasattr(report, "wasxfail")


def test_other_lanes_and_tests_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fork side of a row skips in live-stock: not its lane.
    monkeypatch.setenv(wr.LANE_ENV, "stock")
    report = _skip_report(TET10.fork_test, "requires the Ladruno fork backend")
    apply_warrant_rule(_item("ladruno_fork"), report)
    assert report.skipped
    # A test that is no row.
    report = _skip_report("tests/opensees/live/test_live_minimal.py::test_x", "no")
    apply_warrant_rule(_item("live"), report)
    assert report.skipped
    # No lane set: the default suite is untouched.
    monkeypatch.delenv(wr.LANE_ENV)
    report = _skip_report(TET10.stock_test, "stock-only")
    apply_warrant_rule(_item("live"), report)
    assert report.skipped


def test_ladruno_mkl_skips_count_for_the_lane_never_for_the_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling 3. A row whose fork test needs Pardiso, skipped for it."""
    mkl_row = wr.WarrantRow("X", "refused", "tests/a.py::s", "tests/a.py::f")
    monkeypatch.setattr(wr, "ROWS", (mkl_row,))
    monkeypatch.setenv(wr.LANE_ENV, "fork")
    reason = "ladruno_mkl: Pardiso unavailable (system Pardiso returned -1)"

    report = _skip_report(mkl_row.fork_test, reason)
    apply_warrant_rule(_item("ladruno_fork", "ladruno_mkl"), report)
    assert report.skipped, "the lane keeps its ladruno_mkl skip (lane health)"
    assert not wr.side_green(report.outcome), "and the member is not fork green"

    # The exemption needs both the marker and the reason, in the fork lane.
    for marks, why, lane in (
        (("ladruno_fork",), reason, "fork"),
        (("ladruno_fork", "ladruno_mkl"), "fork build mismatch: pinned x", "fork"),
        (("live", "ladruno_mkl"), reason, "stock"),
    ):
        monkeypatch.setenv(wr.LANE_ENV, lane)
        report = _skip_report(mkl_row.test_for(lane), why)
        apply_warrant_rule(_item(*marks), report)
        assert report.failed, (marks, why, lane)


def test_only_a_pass_makes_a_side_green() -> None:
    assert wr.side_green("passed")
    for outcome in ("skipped", "failed", "error", "xfailed", "not run"):
        assert not wr.side_green(outcome)
