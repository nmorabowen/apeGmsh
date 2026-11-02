"""Both readers of ``/opensees/combinations`` give each row the same verdict.

``results/user`` (``resolve``, through ``read_model_combinations``) and
``opensees/_internal/_combinations_h5.py`` (``read_combinations``) read the
columns with their own code and judge the rows with one implementation,
``opensees/_internal/_combination_rules.py`` (VU2). The oracle for every row is
:data:`EXPECTED`, written by hand from the reasons of ADR 0130 A3; the two
readers must also agree with each other.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import h5py
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal import _combinations_h5 as archive
from apeGmsh.opensees._internal._combinations_h5 import CombinationTerm, CombinationWarning
from apeGmsh.opensees._internal.schema_version import OPENSEES_KEY
from apeGmsh.results.user import Combination, Term, UserFile, write_user_file
from apeGmsh.results.user._descriptor import resolve
from apeGmsh.results.user._reader import read_model_combinations
from tests.fixtures.schema import OPENSEES_CURRENT
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

D = CombinationTerm("stage", "D")
L = CombinationTerm("stage", "L")


def _link(source: str) -> CombinationTerm:
    return CombinationTerm("combination", source)


def _row(name: str, *terms: CombinationTerm, rule: str = "linear", **kw) -> archive.Combination:
    return archive.Combination(name, rule, terms, **kw)


def _mode(mode: int, source: str = "eig") -> CombinationTerm:
    return CombinationTerm("mode", source, 1.0, step=-1, mode=mode)


#: Each table is read as one file. Row names are unique unless a row tests that.
TABLES = {
    "good": [
        _row("uls", D, L),
        _row("env", _link("uls"), L, rule="envelope", envelope="absmax"),
        _row("cqc", _mode(1), _mode(2), rule="cqc", damping=0.05),
        _row("xyz", D, L, rule="directional", fraction=0.3),
        _row("srss", D, L, rule="srss"),
        _row("abs", D, rule="abs_sum"),
        _row("gravity", CombinationTerm("stage", "gravity")),
    ],
    "reaches a cycle with an absent name": [
        _row("a", _link("b")), _row("b", _link("a")),
        _row("c", _link("a"), _link("zz")), _row("s", _link("c")),
    ],
    "in a cycle with an absent name": [
        _row("a", _link("b"), _link("zz")), _row("b", _link("a")),
    ],
    "self": [_row("a", _link("a")), _row("keep", D)],
    "unresolved carries": [_row("r", _link("zz")), _row("s", _link("r")), _row("keep", D)],
    "names a refused row": [_row("bad", D, rule="median"), _row("r", _link("bad")), _row("keep", D)],
    "duplicates": [_row("same", D), _row("same", L), _row("keep", D)],
    "a bad row and its namesake": [_row("same", D, rule="median"), _row("same", L), _row("keep", D)],
    "names": [
        _row("", D), _row(" edge", D), _row("edge ", D), _row("ctl\x07", D),
        _row("é", D), _row("Delta_after (L/480)", D),
    ],
    "rules and params": [
        _row("rule", D, rule="median"),
        _row("no terms"),
        _row("env none", D, rule="envelope"),
        _row("env mean", D, rule="envelope", envelope="mean"),
        _row("env on linear", D, envelope="max"),
        _row("dir none", D, rule="directional"),
        _row("dir high", D, rule="directional", fraction=1.5),
        _row("frac on linear", D, fraction=0.3),
        _row("damp on srss", D, rule="srss", damping=0.05),
    ],
    "cqc": [
        _row("zero", _mode(1), rule="cqc", damping=0.0),
        _row("one", _mode(1), rule="cqc", damping=1.0),
        _row("nan", _mode(1), rule="cqc"),
        _row("stage", D, rule="cqc", damping=0.05),
        _row("two sets", _mode(1, "a"), _mode(2, "b"), rule="cqc", damping=0.05),
        _row("fine", _mode(1), rule="cqc", damping=0.05),
    ],
    "terms": [
        _row("kind", CombinationTerm("stack", "D")),
        _row("empty source", CombinationTerm("stage", "")),
        _row("spectrum", CombinationTerm("spectrum", "case")),
        _row("spectrum srss", CombinationTerm("spectrum", "case"), rule="srss"),
        _row("spectrum cqc", CombinationTerm("spectrum", "case"), rule="cqc"),
        _row("spectrum cqc damped", CombinationTerm("spectrum", "case"), rule="cqc", damping=0.05),
        _row("reference", CombinationTerm("reference", "case")),
        _row("stage step", CombinationTerm("stage", "D", step=-2)),
        _row("stage mode", CombinationTerm("stage", "D", mode=3)),
        _row("mode zero", CombinationTerm("mode", "eig", mode=0)),
        _row("combo step", CombinationTerm("combination", "keep", step=0)),
        _row("factor", CombinationTerm("stage", "D", float("inf"))),
        _row("keep", D),
    ],
}


#: The verdict of every refused row, by position, named by hand (A3).
EXPECTED: dict[str, dict[int, str]] = {
    "good": {},
    "reaches a cycle with an absent name": {0: "cycle", 1: "cycle", 2: "cycle", 3: "cycle"},
    "in a cycle with an absent name": {0: "cycle", 1: "cycle"},
    "self": {0: "cycle"},
    "unresolved carries": {0: "unresolved", 1: "unresolved"},
    "names a refused row": {0: "unknown-rule", 1: "unresolved"},
    "duplicates": {0: "duplicate", 1: "duplicate"},
    "a bad row and its namesake": {0: "unknown-rule", 1: "duplicate"},
    # Row 4 is ``e`` and a combining acute (NFD), so it is not NFC.
    "names": {i: "empty-name" for i in range(5)},
    "rules and params": {0: "unknown-rule", 1: "no-terms", **{i: "bad-param" for i in range(2, 9)}},
    "cqc": {0: "bad-param", 1: "bad-param", 2: "bad-param", 3: "bad-term", 4: "bad-term"},
    # Row 4, cqc over a case with no damping, is a bad term in model.h5: the
    # damping is not the fix, because no model.h5 row may name a case (#2699 review).
    "terms": {i: "bad-term" for i in range(12)},
}


def _write(path: Path, rows: list[archive.Combination]) -> Path:
    with h5py.File(path, "w") as f:
        archive.write_combinations_into(f, rows)
        f.create_group("meta").attrs[OPENSEES_KEY] = OPENSEES_CURRENT
    return path


def _archive_verdicts(path: Path) -> dict[int, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CombinationWarning)
        zone = archive.read_combinations(str(path))
    assert zone.error is None, zone.error
    return {r.index: r.reason for r in zone.refused}


def _user_verdicts(path: Path) -> dict[int, str]:
    table = resolve([], read_model_combinations(path))
    assert all(r.origin == "model" for r in table.refused)
    return {r.row: r.reason for r in table.refused}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_both_validators_give_each_row_the_same_verdict(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / "model.h5", TABLES[name])
    assert _user_verdicts(path) == _archive_verdicts(path)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_each_row_has_the_verdict_named_by_hand(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / "model.h5", TABLES[name])
    assert _user_verdicts(path) == EXPECTED[name]
    assert _archive_verdicts(path) == EXPECTED[name]


def test_the_comparison_can_fail(tmp_path: Path) -> None:
    """Self-test: the tables above are not all accepted, nor all refused."""
    reasons: set[str] = set()
    for i, rows in enumerate(TABLES.values()):
        reasons |= set(_user_verdicts(_write(tmp_path / f"{i}.h5", rows)).values())
    assert reasons == {
        "empty-name", "duplicate", "unknown-rule", "no-terms", "bad-param",
        "bad-term", "unresolved", "cycle",
    }


@pytest.mark.parametrize(
    "table, expected",
    [
        ("reaches a cycle with an absent name", {0: "cycle", 1: "cycle", 2: "cycle", 3: "cycle"}),
        ("in a cycle with an absent name", {0: "cycle", 1: "cycle"}),
    ],
)
def test_a_row_reaching_a_cycle_reads_cycle_in_both(tmp_path: Path, table: str, expected) -> None:
    path = _write(tmp_path / "model.h5", TABLES[table])
    assert _user_verdicts(path) == expected
    assert _archive_verdicts(path) == expected


# ---- a deep chain is linear (#2699 review: the old search was cubic) --------

DEPTH = 1000


def _chain(end: archive.CombinationTerm) -> list[archive.Combination]:
    """``c0 -> c1 -> ... -> c999 -> end``, written deepest first."""
    rows = [_row(f"c{i}", _link(f"c{i + 1}")) for i in range(DEPTH - 1)]
    rows.append(_row(f"c{DEPTH - 1}", end))
    return rows[::-1]


@pytest.mark.parametrize("end, reason", [
    (D, None),
    (_link("c0"), "cycle"),
    (_link("absent"), "unresolved"),
])
def test_a_deep_chain_is_judged_in_under_a_second(tmp_path: Path, end, reason) -> None:
    path = _write(tmp_path / "model.h5", _chain(end))
    want = {} if reason is None else {i: reason for i in range(DEPTH)}
    t0 = time.perf_counter()
    assert _user_verdicts(path) == want
    assert _archive_verdicts(path) == want
    assert time.perf_counter() - t0 < 1.0


# ---- a real 2.39 zone, as ops.combination writes it ------------------------


@pytest.fixture
def bridge_model(tmp_path: Path) -> tuple[Path, tuple[archive.Combination, ...]]:
    ops = apeSees(build_simple_frame_fem())
    ops.model(ndm=3, ndf=6)
    uls = ops.combination("Delta_after (L/480)", [("D", 1.2), ("L", 1.6)])
    declared = (
        uls,
        ops.combination("env", [(uls, 1.0), ("E", -1.0)], rule="envelope", envelope="absmax"),
        ops.combination("modal", [_mode(1), _mode(2)], rule="cqc", damping=0.05),
        ops.combination("xyz", [("Ex", 1.0), ("Ey", 1.0)], rule="directional", fraction=0.3),
    )
    path = tmp_path / "bridge.h5"
    ops.h5(str(path))
    return path, declared


def _as_user(c: archive.Combination) -> Combination:
    return Combination(
        c.name, c.rule,
        tuple(Term(t.kind, t.source, t.step, t.mode, t.factor) for t in c.terms),
        envelope=c.envelope, fraction=c.fraction, damping=c.damping,
    )


def test_read_model_combinations_reads_what_ops_combination_wrote(bridge_model) -> None:
    path, declared = bridge_model
    with h5py.File(path, "r") as f:
        assert str(f["meta"].attrs[OPENSEES_KEY]) == OPENSEES_CURRENT
    assert read_model_combinations(path) == [_as_user(c) for c in declared]


def test_a_user_row_names_a_row_ops_combination_wrote(bridge_model, tmp_path: Path) -> None:
    path, declared = bridge_model
    up = Combination("up", "linear", (Term("combination", "Delta_after (L/480)"),))
    user = write_user_file(tmp_path / "s.user.h5", path, combinations=[up])
    table = UserFile(user).combinations(model_h5=path)
    assert table.refused == () and table.rows == (up,)
    assert table.model_rows == tuple(_as_user(c) for c in declared)


def test_a_name_ops_combination_used_clashes_with_a_user_row(bridge_model, tmp_path: Path) -> None:
    path, _ = bridge_model
    user = write_user_file(tmp_path / "s.user.h5", path, combinations=[])
    with h5py.File(user, "a") as f:
        zone = f["combinations"]
        from apeGmsh.results.user._descriptor import write_generation
        write_generation(zone.create_group("g1"), [Combination("env", "linear", (Term("stage", "D"),))])
        zone.attrs["current"] = 1
    table = UserFile(user).combinations(model_h5=path)
    assert sorted((r.origin, r.reason) for r in table.refused) == [("model", "clash"), ("user", "clash")]
