"""``apeSees`` combinations archived in ``/opensees/combinations`` (ADR 0130 D4).

The oracle is the layout in ``architecture/h5-schema.md``, section
``<stem>.user.h5``, and the refusal reasons of Amendment 1 A3. Each test names
the right answer directly: the columns, the reason, or an unchanged hash.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import ModelData, OpenSeesModel, apeSees
from apeGmsh.opensees._internal._combinations_h5 import (
    Combination,
    CombinationError,
    CombinationTerm,
    CombinationWarning,
    name_clashes,
    read_combinations,
    write_combinations_into,
)
from apeGmsh.opensees._internal.lineage import (
    MODEL_HASH_EXCLUDED_CHILDREN,
    compute_model_hash,
)
from apeGmsh.opensees._internal.schema_version import SchemaVersionError
from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from tests.fixtures.schema import OPENSEES_CURRENT, OPENSEES_FLOOR
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _bridge() -> apeSees:
    ops = apeSees(build_simple_frame_fem())
    ops.model(ndm=3, ndf=6)
    return ops


def _declare_all_rules(ops: apeSees) -> tuple[Combination, ...]:
    """One combination per rule, nested, with a real name that needs no escape."""
    uls = ops.combination("Delta_after (L/480)", [("D", 1.2), ("L", 1.6)])
    return (
        uls,
        ops.combination("env", [(uls, 1.0), ("E", -1.0)],
                        rule="envelope", envelope="absmax"),
        ops.combination("modal", [
            CombinationTerm("mode", "eig", 1.0, step=-1, mode=1),
            CombinationTerm("mode", "eig", 1.0, step=-1, mode=2),
        ], rule="cqc", damping=0.05),
        ops.combination("cqc_one_set", [
            CombinationTerm("mode", "eig", 1.0, step=3, mode=1),
        ], rule="cqc", damping=0.02),
        ops.combination("xyz", [("Ex", 1.0), ("Ey", 1.0)],
                        rule="directional", fraction=0.3),
        ops.combination("srss", [("Ex", 1.0), ("Ey", 1.0)], rule="srss"),
        ops.combination("abs", [
            CombinationTerm("stage", "Ex", 2.0, step=7),
        ], rule="abs_sum"),
    )


def _rewrite_table(path: Path, **columns: object) -> None:
    """Hand-edit one ``g0`` column or ``terms/`` column of a written file."""
    with h5py.File(path, "a") as f:
        g = f["opensees/combinations/g0"]
        for key, values in columns.items():
            where = g["terms"] if key in ("kind", "source", "step", "mode", "factor") else g
            dtype = where[key].dtype
            del where[key]
            where.create_dataset(key, data=values, dtype=dtype)


def _write_rows(path: Path, rows: list[Combination]) -> None:
    with h5py.File(path, "w") as f:
        write_combinations_into(f, rows)


def _stage(name: str, *terms: CombinationTerm, rule: str = "linear") -> Combination:
    return Combination(name, rule, terms or (CombinationTerm("stage", "D"),))


def _combo_term(source: str) -> CombinationTerm:
    return CombinationTerm("combination", source)


# --- the layout -------------------------------------------------------------


def test_the_columns_are_the_schemas(tmp_path: Path) -> None:
    ops = _bridge()
    ops.combination("ULS", [("D", 1.2), ("L", 1.6)])
    ops.combination("xyz", [("Ex", 1.0)], rule="directional", fraction=0.3)
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with h5py.File(path, "r") as f:
        zone = f["opensees/combinations"]
        assert zone.attrs["current"] == 0 and zone.attrs["current"].dtype == np.int32
        g = zone["g0"]
        assert list(g["name"].asstr()[()]) == ["ULS", "xyz"]
        assert list(g["rule"].asstr()[()]) == ["linear", "directional"]
        assert list(g["envelope"].asstr()[()]) == ["", ""]
        assert g["fraction"].dtype == np.float64
        assert math.isnan(g["fraction"][0]) and g["fraction"][1] == 0.3
        assert all(math.isnan(v) for v in g["damping"][()])
        assert g["term_offsets"].dtype == np.int32
        assert g["term_offsets"][()].tolist() == [0, 2, 3]
        t = g["terms"]
        assert list(t["kind"].asstr()[()]) == ["stage"] * 3
        assert list(t["source"].asstr()[()]) == ["D", "L", "Ex"]
        assert t["step"].dtype == np.int32 and t["step"][()].tolist() == [-1] * 3
        assert t["mode"].dtype == np.int32 and t["mode"][()].tolist() == [0] * 3
        assert t["factor"][()].tolist() == [1.2, 1.6, 1.0]
        assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT


def test_the_bump_is_the_fixtures() -> None:
    assert SCHEMA_VERSION == OPENSEES_CURRENT


def test_no_combination_writes_no_group(tmp_path: Path) -> None:
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    with h5py.File(path, "r") as f:
        assert "combinations" not in f["opensees"]
    zone = read_combinations(str(path))
    assert zone.rows == () and zone.refused == () and zone.error is None


# --- reload -----------------------------------------------------------------


def test_every_rule_round_trips(tmp_path: Path) -> None:
    ops = _bridge()
    declared = _declare_all_rules(ops)
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    assert read_combinations(str(path)).rows == declared
    assert OpenSeesModel.from_h5(str(path)).combinations() == declared


def test_open_seesmodel_rewrite_echoes(tmp_path: Path) -> None:
    ops = _bridge()
    declared = _declare_all_rules(ops)
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    ops.h5(str(first))
    OpenSeesModel.from_h5(str(first)).to_h5(str(second))
    assert read_combinations(str(second)).rows == declared


def test_modeldata_rewrite_echoes(tmp_path: Path) -> None:
    ops = _bridge()
    declared = _declare_all_rules(ops)
    first, second, third = (tmp_path / n for n in ("a.h5", "b.h5", "c.h5"))
    ops.h5(str(first))
    md = ModelData.from_h5(str(first))
    md.write(str(second))
    md.write(str(third))  # twice: the echo is not one-shot
    assert read_combinations(str(second)).rows == declared
    assert read_combinations(str(third)).rows == declared


def test_a_rewrite_echoes_a_refused_row_verbatim(tmp_path: Path) -> None:
    """A rewrite keeps a row it refuses to use, rather than deleting it."""
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    ops = _bridge()
    ops.combination("good", [("D", 1.0)])
    ops.combination("bad", [("D", 1.0)])
    ops.h5(str(first))
    _rewrite_table(first, kind=np.array(["stage", "spectrum"], dtype=object))
    with pytest.warns(CombinationWarning, match="bad"):
        model = OpenSeesModel.from_h5(str(first))
    assert [c.name for c in model.combinations()] == ["good"]
    model.to_h5(str(second))
    with pytest.warns(CombinationWarning):
        zone = read_combinations(str(second))
    assert [t.kind for c in zone.rows for t in c.terms] == ["stage", "spectrum"]


def test_a_malformed_zone_opens_and_a_rewrite_refuses(tmp_path: Path) -> None:
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    ops = _bridge()
    ops.combination("good", [("D", 1.0)])
    ops.h5(str(first))
    with h5py.File(first, "a") as f:
        del f["opensees/combinations/g0/rule"]
    with pytest.warns(CombinationWarning, match="rule"):
        model = OpenSeesModel.from_h5(str(first))
    assert model.combinations() == ()
    with pytest.raises(ValueError, match="refused"):
        model.to_h5(str(second))
    with pytest.warns(CombinationWarning, match="rule"):
        md = ModelData.from_h5(str(first))
    with pytest.raises(ValueError, match="refused"):
        md.write(str(second))


# --- the hash ---------------------------------------------------------------


def _hash_of(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return str(f["meta/lineage"].attrs["model_hash"])


def test_a_combination_leaves_model_hash_unchanged(tmp_path: Path) -> None:
    bare, with_combo = tmp_path / "a.h5", tmp_path / "b.h5"
    _bridge().h5(str(bare))
    ops = _bridge()
    _declare_all_rules(ops)
    ops.h5(str(with_combo))
    assert _hash_of(with_combo) == _hash_of(bare)
    assert "combinations" in MODEL_HASH_EXCLUDED_CHILDREN
    with h5py.File(with_combo, "a") as f:
        recomputed = compute_model_hash("fem", f["opensees"])
        del f["opensees/combinations"]
        assert compute_model_hash("fem", f["opensees"]) == recomputed


def test_a_rewrite_keeps_model_hash(tmp_path: Path) -> None:
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    ops = _bridge()
    _declare_all_rules(ops)
    ops.h5(str(first))
    OpenSeesModel.from_h5(str(first)).to_h5(str(second))
    assert _hash_of(second) == _hash_of(first)


# --- refused when declared, with nothing registered -------------------------


@pytest.mark.parametrize("rule", ["linear", "cqc"])
@pytest.mark.parametrize("kind", ["spectrum", "reference"])
def test_a_case_term_is_refused_when_declared(kind: str, rule: str) -> None:
    """``cqc`` over a case with no damping reads ``bad-term`` as well: the term
    is what to fix, not the damping (#2699 review)."""
    ops = _bridge()
    with pytest.raises(CombinationError, match=rf"\[bad-term\]") as err:
        ops.combination("x", [CombinationTerm(kind, "case")], rule=rule)
    assert ".user.h5" in str(err.value)
    assert ops.build().combinations == ()


BAD_DECLARATIONS = [
    ("", {"terms": [("D", 1.0)]}, "empty-name"),
    (" edge", {"terms": [("D", 1.0)]}, "empty-name"),
    ("edge ", {"terms": [("D", 1.0)]}, "empty-name"),
    ("ctrl\x07", {"terms": [("D", 1.0)]}, "empty-name"),
    ("e\u0301", {"terms": [("D", 1.0)]}, "empty-name"),  # NFD, not NFC
    ("lone\ud800", {"terms": [("D", 1.0)]}, "empty-name"),  # no UTF-8 form
    ("x", {"terms": [("D", 1.0)], "rule": "geometric"}, "unknown-rule"),
    ("x", {"terms": []}, "no-terms"),
    ("x", {"terms": [("D", 1.0)], "rule": "envelope"}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "rule": "envelope", "envelope": "mean"}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "envelope": "max"}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "rule": "directional"}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "rule": "directional", "fraction": 1.5}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "fraction": 0.3}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "damping": 0.05}, "bad-param"),
    ("x", {"terms": [CombinationTerm("mode", "e", mode=1)], "rule": "cqc",
           "damping": 0.0}, "bad-param"),
    ("x", {"terms": [CombinationTerm("mode", "e", mode=1)], "rule": "cqc"}, "bad-param"),
    ("x", {"terms": [CombinationTerm("mode", "e", mode=1)], "rule": "cqc",
           "damping": math.nan}, "bad-param"),
    ("x", {"terms": [CombinationTerm("mode", "e", mode=1)], "rule": "cqc",
           "damping": 1.0}, "bad-param"),
    ("x", {"terms": [("D", 1.0)], "rule": "cqc", "damping": 0.05}, "bad-term"),
    ("x", {"terms": [CombinationTerm("mode", "e", mode=1),
                     CombinationTerm("mode", "f", mode=1)], "rule": "cqc",
           "damping": 0.05}, "bad-term"),
    ("x", {"terms": [("D", math.nan)]}, "bad-term"),
    ("x", {"terms": [("D", math.inf)]}, "bad-term"),
    ("x", {"terms": [("D", True)]}, "bad-term"),
    ("x", {"terms": [("", 1.0)]}, "bad-term"),
    ("x", {"terms": ["D"]}, "bad-term"),
    ("x", {"terms": [CombinationTerm("mode", "e")]}, "bad-term"),  # mode 0
    ("x", {"terms": [CombinationTerm("stage", "D", mode=1)]}, "bad-term"),
    ("x", {"terms": [CombinationTerm("stage", "D", step=2 ** 31)]}, "bad-term"),
    ("x", {"terms": [CombinationTerm("widget", "D")]}, "bad-term"),
    ("x", {"terms": [_combo_term("nobody")]}, "unresolved"),
    ("x", {"terms": [_combo_term("x")]}, "unresolved"),  # itself
]


@pytest.mark.parametrize("name, kwargs, reason", BAD_DECLARATIONS)
def test_a_refused_declaration_registers_nothing(
    tmp_path: Path, name: str, kwargs: dict[str, object], reason: str,
) -> None:
    ops = _bridge()
    terms = kwargs.pop("terms")
    with pytest.raises(CombinationError, match=rf"\[{reason}\]") as err:
        ops.combination(name, terms, **kwargs)  # type: ignore[arg-type]
    assert err.value.reason == reason
    assert ops.build().combinations == ()
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with h5py.File(path, "r") as f:
        assert "combinations" not in f["opensees"]
    # The refused name stays free.
    if reason not in ("empty-name",):
        ops.combination(name, [("D", 1.0)])


def test_a_handle_from_another_bridge_is_unresolved() -> None:
    """A handle is checked by value, so a same-named row of another model or
    an edited copy cannot pass as this model's."""
    other = _bridge().combination("base", [("D", 9.0)])
    ops = _bridge()
    with pytest.raises(CombinationError, match=r"\[unresolved\]"):
        ops.combination("top", [(other, 1.0)])
    mine = ops.combination("base", [("D", 1.0)])
    with pytest.raises(CombinationError, match=r"\[unresolved\]"):
        ops.combination("top", [(other, 1.0)])  # shares the name only
    top = ops.combination("top", [(mine, 1.0)])
    assert [c.name for c in ops.build().combinations] == ["base", "top"]
    assert top.terms == (CombinationTerm("combination", "base", 1.0),)


def test_a_duplicate_is_refused_and_the_first_kept() -> None:
    ops = _bridge()
    first = ops.combination("ULS", [("D", 1.2)])
    with pytest.raises(CombinationError, match=r"\[duplicate\]"):
        ops.combination("ULS", [("D", 9.9)])
    assert ops.build().combinations == (first,)


def test_a_combination_may_share_a_stage_name() -> None:
    """A term's ``kind`` separates the namespaces (A2)."""
    ops = _bridge()
    ops.combination("D", [("D", 1.2)])
    assert ops.build().combinations[0].name == "D"


def test_a_primitive_name_does_not_clash_with_a_combination() -> None:
    ops = _bridge()
    ops.combination("shared", [("D", 1.0)])
    ops.timeSeries.Linear(name="shared")
    assert [c.name for c in ops.build().combinations] == ["shared"]


def test_a_built_model_is_frozen_against_later_declarations() -> None:
    ops = _bridge()
    ops.combination("a", [("D", 1.0)])
    built = ops.build()
    ops.combination("b", [("D", 1.0)])
    assert [c.name for c in built.combinations] == ["a"]


# --- refused at read, row by row ---------------------------------------------


def _reasons(path: Path) -> dict[str, str]:
    with pytest.warns(CombinationWarning):
        zone = read_combinations(str(path))
    return {r.name: r.reason for r in zone.refused}


@pytest.mark.parametrize("row, reason", [
    (_stage(""), "empty-name"),
    (_stage(" edge"), "empty-name"),
    (_stage("e\u0301"), "empty-name"),
    (_stage("x", rule="geometric"), "unknown-rule"),
    (Combination("x", "linear", ()), "no-terms"),
    (Combination("x", "envelope", (CombinationTerm("stage", "D"),)), "bad-param"),
    (Combination("x", "linear", (CombinationTerm("stage", "D"),), fraction=0.3), "bad-param"),
    (Combination("x", "cqc", (CombinationTerm("mode", "e", mode=1),)), "bad-param"),
    (Combination("x", "cqc", (CombinationTerm("mode", "e", mode=1),), damping=1.5), "bad-param"),
    (_stage("x", CombinationTerm("spectrum", "case")), "bad-term"),
    (_stage("x", CombinationTerm("reference", "case")), "bad-term"),
    (_stage("x", CombinationTerm("stage", "D", math.nan)), "bad-term"),
    (_stage("x", _combo_term("in_user_h5")), "unresolved"),
])
def test_a_bad_row_refuses_alone(
    tmp_path: Path, row: Combination, reason: str,
) -> None:
    path = tmp_path / "t.h5"
    good = _stage("good")
    _write_rows(path, [good, row])
    with pytest.warns(CombinationWarning, match=reason):
        zone = read_combinations(str(path), opensees_root="/opensees")
    assert [(r.index, r.reason) for r in zone.refused] == [(1, reason)]
    assert zone.valid == (good,)
    assert zone.error is None


def test_a_duplicate_refuses_every_row_of_that_name(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("a"), _stage("a"), _stage("b")])
    with pytest.warns(CombinationWarning, match="duplicate"):
        zone = read_combinations(str(path))
    assert [(r.index, r.reason) for r in zone.refused] == [
        (0, "duplicate"), (1, "duplicate")]
    assert [c.name for c in zone.valid] == ["b"]


def test_a_two_row_cycle_refuses_both(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [
        _stage("a", _combo_term("b")),
        _stage("b", _combo_term("a")),
        _stage("free"),
    ])
    assert _reasons(path) == {"a": "cycle", "b": "cycle"}


def test_a_self_cycle_refuses(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("a", _combo_term("a")), _stage("free")])
    assert _reasons(path) == {"a": "cycle"}


def test_a_long_cycle_refuses_every_member(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [
        _stage("a", _combo_term("b")),
        _stage("b", _combo_term("c")),
        _stage("c", CombinationTerm("stage", "D"), _combo_term("a")),
    ])
    assert _reasons(path) == {"a": "cycle", "b": "cycle", "c": "cycle"}


def test_a_row_reaching_a_cycle_reads_cycle(tmp_path: Path) -> None:
    """At any depth, and even beside a term that is merely absent; a row on
    another refused row is ``unresolved``, and a chain carries that."""
    path = tmp_path / "t.h5"
    _write_rows(path, [
        _stage("top", _combo_term("mid"), _combo_term("absent")),
        _stage("mid", _combo_term("a")),
        _stage("a", _combo_term("b")),
        _stage("b", _combo_term("a")),
        _stage("rests_on_rests", _combo_term("rests_on_bad")),
        _stage("rests_on_bad", _combo_term("no_terms")),
        Combination("no_terms", "linear", ()),
        _stage("free"),
    ])
    assert _reasons(path) == {
        "top": "cycle", "mid": "cycle",
        "a": "cycle", "b": "cycle",
        "rests_on_rests": "unresolved", "rests_on_bad": "unresolved",
        "no_terms": "no-terms",
    }


def test_a_forward_reference_in_a_file_resolves(tmp_path: Path) -> None:
    """A file's row order is not a declaration order: no false refusal."""
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("top", _combo_term("base")), _stage("base")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", CombinationWarning)
        assert len(read_combinations(str(path)).valid) == 2


def test_a_good_file_reads_without_a_warning(tmp_path: Path) -> None:
    ops = _bridge()
    _declare_all_rules(ops)
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CombinationWarning)
        assert read_combinations(str(path)).refused == ()


# --- malformed columns refuse the zone ---------------------------------------


def _delete(path: Path, key: str) -> None:
    with h5py.File(path, "a") as f:
        del f[f"opensees/combinations/{key}"]


def _replace(path: Path, key: str, data: object) -> None:
    _delete(path, key)
    with h5py.File(path, "a") as f:
        f[f"opensees/combinations/{key}"] = data


def _attr(path: Path, value: object) -> None:
    with h5py.File(path, "a") as f:
        f["opensees/combinations"].attrs["current"] = value


def _second_generation(path: Path, current: int) -> None:
    with h5py.File(path, "a") as f:
        zone = f["opensees/combinations"]
        zone.copy("g0", "g1")
        zone.attrs["current"] = np.int32(current)


#: Damage a rewrite could not echo, or would change: each one once opened and
#: then truncated the source, or collapsed it, inside the rewrite (F1).
WIDENED = [
    pytest.param(
        lambda p: _replace(p, "g0/terms/step", np.array([2 ** 40, -1], dtype="i8")),
        id="i8 step, out of i4 range"),
    pytest.param(
        lambda p: _replace(p, "g0/terms/mode", np.array([0, 0], dtype="i8")),
        id="i8 mode"),
    pytest.param(
        lambda p: _replace(p, "g0/terms/mode", np.array([0, 0], dtype="u4")),
        id="u4 mode"),
    pytest.param(
        lambda p: _replace(p, "g0/term_offsets", np.array([0, 1, 2], dtype="i8")),
        id="i8 term_offsets"),
    pytest.param(
        lambda p: _replace(p, "g0/terms/factor", np.array([1.0, 1.0], dtype="f4")),
        id="f4 factor"),
    pytest.param(
        lambda p: _replace(p, "g0/fraction", np.array([0.1, 0.1], dtype="f4")),
        id="f4 fraction"),
    pytest.param(
        lambda p: _replace(p, "g0/damping", np.array([0.1, 0.1], dtype="f4")),
        id="f4 damping"),
    pytest.param(lambda p: _attr(p, np.int64(0)), id="i8 @current"),
    pytest.param(lambda p: _attr(p, np.int32(5)), id="@current names no group"),
    pytest.param(lambda p: _second_generation(p, 1), id="@current=1 with g0 and g1"),
    pytest.param(lambda p: _second_generation(p, 0), id="@current=0 with g0 and g1"),
]


@pytest.mark.parametrize("damage", [
    lambda p: _delete(p, "g0/name"),
    lambda p: _delete(p, "g0/fraction"),
    lambda p: _delete(p, "g0/term_offsets"),
    lambda p: _delete(p, "g0/terms"),
    lambda p: _delete(p, "g0/terms/factor"),
    lambda p: _delete(p, "g0"),
    lambda p: _replace(p, "g0/rule", np.array([1, 2], dtype="i4")),
    lambda p: _replace(p, "g0/fraction", np.array([0.1], dtype="f8")),
    lambda p: _replace(p, "g0/term_offsets", np.array([0, 2], dtype="i4")),
    lambda p: _replace(p, "g0/term_offsets", np.array([1, 2, 3], dtype="i4")),
    lambda p: _replace(p, "g0/term_offsets", np.array([0, 3, 2], dtype="i4")),
    lambda p: _replace(p, "g0/terms/factor", np.array(["x"], dtype=h5py.string_dtype())),
    lambda p: _attr(p, "zero"),
    *WIDENED,
])
def test_malformed_columns_refuse_the_zone(tmp_path: Path, damage: object) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("a"), _stage("b")])
    damage(path)  # type: ignore[operator]
    with pytest.warns(CombinationWarning, match="refused"):
        zone = read_combinations(str(path))
    assert zone.error is not None
    assert zone.rows == ()


def _bridge_file_with(path: Path, damage: object) -> bytes:
    """A real two-term model.h5 carrying ``damage``, and its bytes."""
    ops = _bridge()
    ops.combination("a", [("D", 1.0)])
    ops.combination("b", [("L", 1.0)])
    ops.h5(str(path))
    damage(path)  # type: ignore[operator]
    return path.read_bytes()


@pytest.mark.parametrize("damage", WIDENED)
def test_a_refused_zone_leaves_the_source_whole_on_rewrite(
    tmp_path: Path, damage: object,
) -> None:
    """Rewriting over the source raises before the file is opened for writing."""
    path = tmp_path / "m.h5"
    before = _bridge_file_with(path, damage)
    with pytest.warns(CombinationWarning, match="refused"):
        model = OpenSeesModel.from_h5(str(path))
    with pytest.raises(ValueError, match="refused"):
        model.to_h5(str(path))
    assert path.read_bytes() == before
    with pytest.warns(CombinationWarning, match="refused"):
        md = ModelData.from_h5(str(path))
    with pytest.raises(ValueError, match="refused"):
        md.write(str(path))
    assert path.read_bytes() == before


def test_a_non_utf8_cell_refuses_the_zone(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("a")])
    with h5py.File(path, "a") as f:
        f["opensees/combinations/g0/name"][0] = b"\xff"
    with pytest.warns(CombinationWarning, match="UTF-8"):
        assert read_combinations(str(path)).error is not None


# --- ModelData refuses a version it cannot carry (F2) -------------------------


def _stamp(path: Path, version: str) -> None:
    with h5py.File(path, "a") as f:
        f["meta"].attrs["opensees_schema_version"] = version


def _minor(version: str, minor: int) -> str:
    major, _, _ = version.split(".")
    return f"{major}.{minor}.0"


def test_modeldata_refuses_a_newer_opensees_zone(tmp_path: Path) -> None:
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    current_minor = int(OPENSEES_CURRENT.split(".")[1])
    _stamp(path, _minor(OPENSEES_CURRENT, current_minor + 1))
    with pytest.raises(SchemaVersionError, match="newer"):
        ModelData.from_h5(str(path))


def test_modeldata_refuses_below_the_floor(tmp_path: Path) -> None:
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    floor_minor = int(OPENSEES_FLOOR.split(".")[1])
    _stamp(path, _minor(OPENSEES_FLOOR, floor_minor - 1))
    with pytest.raises(SchemaVersionError, match="floor|too old|older"):
        ModelData.from_h5(str(path))


def test_modeldata_refuses_an_unstamped_opensees_zone(tmp_path: Path) -> None:
    """The refusal names the opensees zone. FEMData's own refusal of the empty
    envelope named the neutral one, and used to answer first (#2699 review)."""
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    with h5py.File(path, "a") as f:
        for key in ("opensees_schema_version", "schema_version"):
            if key in f["meta"].attrs:
                del f["meta"].attrs[key]
    with pytest.raises(MalformedH5Error, match="no opensees zone version"):
        ModelData.from_h5(str(path))


def test_modeldata_reads_the_envelope_as_the_opensees_version(tmp_path: Path) -> None:
    """A file stamped with the envelope alone is not unstamped (ADR 0023)."""
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    with h5py.File(path, "a") as f:
        del f["meta"].attrs["opensees_schema_version"]
        assert "schema_version" in f["meta"].attrs
    assert ModelData.from_h5(str(path)).ndm == 3


def test_modeldata_opens_a_file_with_no_opensees_zone(tmp_path: Path) -> None:
    """The guard reads the zone only when the zone is there."""
    path = tmp_path / "m.h5"
    _bridge().h5(str(path))
    current_minor = int(OPENSEES_CURRENT.split(".")[1])
    _stamp(path, _minor(OPENSEES_CURRENT, current_minor + 1))
    with h5py.File(path, "a") as f:
        del f["opensees"]
    assert ModelData.from_h5(str(path)).ndm == 3


def test_a_missing_current_refuses_the_zone(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    _write_rows(path, [_stage("a")])
    with h5py.File(path, "a") as f:
        del f["opensees/combinations"].attrs["current"]
    with pytest.warns(CombinationWarning, match="current"):
        assert read_combinations(str(path)).error is not None


# --- the helper VU2 joins on -------------------------------------------------


def test_name_clashes_is_byte_equal() -> None:
    model = [_stage("ULS"), _stage("e\u0301"), _stage("only_model")]
    user = [_stage("ULS"), _stage("\u00e9"), _stage("only_user")]
    assert name_clashes(model, user) == frozenset({"ULS"})


def test_name_clashes_empty() -> None:
    assert name_clashes([], [_stage("a")]) == frozenset()
