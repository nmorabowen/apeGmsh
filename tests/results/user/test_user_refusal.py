"""A bad row refuses alone and the file still opens; malformed columns refuse
the zone (ADR 0130 A3). Bad rows are written raw, since the writer refuses them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.schema_version import (
    OPENSEES,
    OPENSEES_KEY,
    SchemaVersionError,
    reader_version,
)
from apeGmsh.results.user import (
    Combination,
    Term,
    UserFile,
    UserFileRefused,
    UserWriteRefused,
    ZoneRefused,
    write_user_file,
)
from apeGmsh.results.user._descriptor import resolve, write_generation
from tests.fixtures.schema import OPENSEES_CURRENT

from .conftest import nodal

GOOD = Combination("good", "linear", (Term("stage", "D", -1, 0, 1.0),))
D = Term("stage", "D")


def _model_rows(model: Path, rows) -> None:
    with h5py.File(model, "a") as f:
        write_generation(f.create_group("opensees/combinations/g0"), rows)
        f["meta"].attrs[OPENSEES_KEY] = OPENSEES_CURRENT


def _raw(make_model, tmp_path: Path, rows, *, model_rows=None) -> tuple[Path, Path]:
    """A file holding ``rows`` as they are, and its model archive."""
    model = make_model()
    if model_rows is not None:
        _model_rows(model, model_rows)
    path = write_user_file(tmp_path / "s.user.h5", model, combinations=[])
    with h5py.File(path, "a") as f:
        zone = f["combinations"]
        write_generation(zone.create_group("g1"), rows)
        zone.attrs["current"] = np.int32(1)
    return path, model


def _reason(table, name: str) -> Optional[str]:
    return next((r.reason for r in table.refused if r.name == name), None)


@pytest.mark.parametrize(
    "row, reason",
    [
        (Combination("", "linear", (D,)), "empty-name"),
        (Combination(" edge", "linear", (D,)), "empty-name"),
        (Combination("ctl\x07", "linear", (D,)), "empty-name"),
        (Combination("é", "linear", (D,)), "empty-name"),
        (Combination("r", "median", (D,)), "unknown-rule"),
        (Combination("r", "linear", ()), "no-terms"),
        (Combination("r", "envelope", (D,)), "bad-param"),
        (Combination("r", "envelope", (D,), envelope="mean"), "bad-param"),
        (Combination("r", "linear", (D,), envelope="max"), "bad-param"),
        (Combination("r", "directional", (D,)), "bad-param"),
        (Combination("r", "directional", (D,), fraction=1.5), "bad-param"),
        (Combination("r", "linear", (D,), fraction=0.3), "bad-param"),
        (Combination("r", "srss", (D,), damping=0.05), "bad-param"),
        (Combination("r", "cqc", (Term("spectrum", "c"),), damping=1.0), "bad-param"),
        (Combination("r", "linear", (Term("stack", "D"),)), "bad-term"),
        (Combination("r", "linear", (Term("stage", ""),)), "bad-term"),
        (Combination("r", "linear", (Term("stage", "D", -2),)), "bad-term"),
        (Combination("r", "linear", (Term("stage", "D", -1, 3),)), "bad-term"),
        (Combination("r", "linear", (Term("mode", "eig", -1, 0),)), "bad-term"),
        (Combination("r", "linear", (Term("spectrum", "c", 2, 1),)), "bad-term"),
        (Combination("r", "linear", (Term("reference", "c", -1, 1),)), "bad-term"),
        (Combination("r", "linear", (Term("combination", "good", 0),)), "bad-term"),
        (Combination("r", "linear", (Term("stage", "D", factor=float("inf")),)), "bad-term"),
        (Combination("r", "linear", (Term("spectrum", "c"),)), "bad-term"),
        (Combination("r", "envelope", (Term("spectrum", "c"),), envelope="max"), "bad-term"),
        (Combination("r", "srss", (Term("spectrum", "c"), D)), "bad-term"),
        (Combination("r", "cqc", (D,)), "bad-term"),
        (Combination("r", "cqc", (Term("mode", "a", -1, 1), Term("mode", "b", -1, 2))), "bad-term"),
        (Combination("r", "linear", (Term("combination", "nobody"),)), "unresolved"),
    ],
)
def test_a_bad_row_refuses_alone_with_its_reason(make_model, tmp_path: Path, row, reason) -> None:
    path, _ = _raw(make_model, tmp_path, [GOOD, row])
    table = UserFile(path).combinations()  # the file opens and the zone reads
    assert [r.reason for r in table.refused] == [reason]
    assert table.rows == (GOOD,)
    assert table.refused[0].row == 1


@pytest.mark.parametrize(
    "row",
    [
        Combination("whole srss", "srss", (Term("spectrum", "c"),)),
        Combination("list srss", "srss", (D, Term("stage", "L"))),
        Combination("abs", "abs_sum", (Term("spectrum", "c"),)),
        Combination("cqc whole", "cqc", (Term("spectrum", "c"),)),
        Combination("cqc modes", "cqc", (Term("mode", "eig", -1, 1), Term("mode", "eig", -1, 2)), damping=0.05),
        Combination("cqc incr", "cqc", (Term("spectrum", "c", -1, 1), Term("spectrum", "c", -1, 2))),
        Combination("dir", "directional", (D, Term("stage", "L")), fraction=0.3),
        Combination("mode k", "linear", (Term("mode", "eig", 4, 2), Term("spectrum", "c", -1, 2))),
        Combination("ref", "linear", (Term("reference", "c"), D)),
        Combination("negative", "linear", (Term("stage", "D", 0, 0, -1.5),)),
    ],
)
def test_the_forms_the_rules_table_allows_stand(make_model, tmp_path: Path, row) -> None:
    path, _ = _raw(make_model, tmp_path, [row])
    table = UserFile(path).combinations()
    assert table.refused == () and table.rows == (row,)


def test_a_name_clash_refuses_both_rows(make_model, tmp_path: Path) -> None:
    a = Combination("same", "linear", (D,))
    b = Combination("same", "linear", (Term("stage", "L"),))
    path, _ = _raw(make_model, tmp_path, [GOOD, a, b])
    table = UserFile(path).combinations()
    assert table.rows == (GOOD,)
    assert [(r.name, r.reason, r.row) for r in table.refused] == [
        ("same", "duplicate", 1), ("same", "duplicate", 2),
    ]


def test_a_clash_across_model_and_user_refuses_both_rows(make_model, tmp_path: Path) -> None:
    mine = Combination("same", "linear", (D,))
    theirs = Combination("same", "linear", (Term("stage", "L"),))
    path, model = _raw(make_model, tmp_path, [mine, GOOD], model_rows=[theirs])
    table = UserFile(path).combinations(model_h5=model)
    assert table.rows == (GOOD,) and table.model_rows == ()
    assert sorted((r.origin, r.reason) for r in table.refused) == [("model", "clash"), ("user", "clash")]


def test_a_combination_may_share_a_stage_name(make_model, tmp_path: Path) -> None:
    row = Combination("gravity", "linear", (Term("stage", "gravity", -1, 0, 1.2),))
    path, _ = _raw(make_model, tmp_path, [row])
    table = UserFile(path).combinations()
    assert table.refused == () and table.rows == (row,)


def test_a_user_row_may_name_a_model_row_and_never_the_reverse(make_model, tmp_path: Path) -> None:
    base = Combination("base", "linear", (D,))
    upward = Combination("upward", "linear", (Term("combination", "user only"),))
    user = Combination("user only", "linear", (Term("combination", "base"),))
    path, model = _raw(make_model, tmp_path, [user], model_rows=[base, upward])
    table = UserFile(path).combinations(model_h5=model)
    assert table.rows == (user,) and table.model_rows == (base,)
    assert [(r.name, r.reason, r.origin) for r in table.refused] == [("upward", "unresolved", "model")]


def test_without_the_model_a_row_naming_a_model_row_is_unresolved(make_model, tmp_path: Path) -> None:
    row = Combination("up", "linear", (Term("combination", "base"),))
    path, _ = _raw(make_model, tmp_path, [row])
    assert _reason(UserFile(path).combinations(), "up") == "unresolved"


@pytest.mark.parametrize("kind", ["spectrum", "reference"])
def test_a_model_row_refuses_a_spectrum_or_reference_term(make_model, tmp_path: Path, kind) -> None:
    bad = Combination("m", "srss", (Term(kind, "case"),))
    path, model = _raw(make_model, tmp_path, [], model_rows=[bad])
    table = UserFile(path).combinations(model_h5=model)
    assert [(r.name, r.reason) for r in table.refused] == [("m", "bad-term")]


def test_a_row_naming_a_refused_row_is_unresolved_transitively() -> None:
    bad = Combination("bad", "median", (D,))
    one = Combination("one", "linear", (Term("combination", "bad"),))
    two = Combination("two", "linear", (Term("combination", "one"),))
    table = resolve([bad, one, two, GOOD])
    assert table.rows == (GOOD,)
    assert [(r.name, r.reason) for r in table.refused] == [
        ("bad", "unknown-rule"), ("one", "unresolved"), ("two", "unresolved"),
    ]


def test_a_cycle_refuses_every_row_in_it_or_reaching_it() -> None:
    a = Combination("a", "linear", (Term("combination", "b"),))
    b = Combination("b", "linear", (Term("combination", "a"),))
    reach = Combination("reach", "linear", (Term("combination", "a"),))
    selfie = Combination("self", "linear", (Term("combination", "self"),))
    below = Combination("below", "linear", (Term("combination", "good"),))
    table = resolve([a, b, reach, selfie, below, GOOD])
    assert {r.name: r.reason for r in table.refused} == dict.fromkeys(
        ["a", "b", "reach", "self"], "cycle")
    assert set(table.rows) == {below, GOOD}


def test_a_chain_in_any_order_stands() -> None:
    top = Combination("top", "linear", (Term("combination", "mid"),))
    mid = Combination("mid", "linear", (Term("combination", "good"),))
    assert resolve([top, mid, GOOD]).refused == ()


@pytest.mark.parametrize("column", ["name", "rule", "envelope", "terms/kind", "terms/source"])
def test_a_non_utf8_cell_refuses_the_zone(make_model, tmp_path: Path, column: str) -> None:
    path, _ = _raw(make_model, tmp_path, [GOOD, Combination("x", "linear", (D,))])
    write_user_file(path, make_model("m2.h5"), results=[nodal()])
    with h5py.File(path, "a") as f:
        f[f"combinations/g1/{column}"][1] = b"\xff\xfe"
    uf = UserFile(path)
    with pytest.raises(ZoneRefused, match="not UTF-8"):
        uf.combinations()
    assert uf.results().entries[0].name == nodal().name


def test_a_non_utf8_cell_in_states_refuses_the_zone(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()])
    with h5py.File(path, "a") as f:
        f["results/r0/STATES/source"][0] = b"\xff\xfe"
    with pytest.raises(ZoneRefused, match="not UTF-8"):
        UserFile(path).results()


@pytest.mark.parametrize(
    "row, reason",
    [
        (Combination("r", "cqc", (Term("spectrum", "c"),), damping=0.0), "bad-param"),
        (Combination("r", "cqc", (Term("mode", "eig", -1, 1), Term("mode", "eig", -1, 2))), "bad-param"),
    ],
)
def test_the_damping_rulings_refuse(row, reason) -> None:
    assert [r.reason for r in resolve([row]).refused] == [reason]


@pytest.mark.parametrize(
    "row",
    [
        Combination("whole", "cqc", (Term("spectrum", "c"),)),
        Combination("incr", "cqc", (Term("spectrum", "c", -1, 1), Term("spectrum", "c", -1, 2))),
        Combination("modes", "cqc", (Term("mode", "eig", -1, 1),), damping=0.05),
    ],
)
def test_nan_damping_stands_where_a_source_exists(row) -> None:
    assert resolve([row]).refused == ()


# ---- malformed columns refuse the zone --------------------------------------


def _mangle(path: Path, how) -> None:
    with h5py.File(path, "a") as f:
        how(f["combinations"], f["combinations/g1"])


def _del(name):
    return lambda z, g: g.__delitem__(name)


def _swap(name, data):
    def how(z, g):
        del g[name]
        g.create_dataset(name, data=data)
    return how


@pytest.mark.parametrize(
    "how, match",
    [
        (_del("rule"), "no rule column"),
        (_del("terms"), "no terms group"),
        (_del("term_offsets"), "no term_offsets"),
        (_swap("fraction", np.array([0.0, 0.0], dtype="f4")), "expected f8"),
        (_swap("term_offsets", np.array([0, 1, 2], dtype="i8")), "expected i4"),
        (_swap("fraction", np.array([0.0])), "1 rows, expected 2"),
        (_swap("term_offsets", np.array([0, 2, 1], dtype="i4")), "does not partition"),
        (_swap("term_offsets", np.array([1, 1, 2], dtype="i4")), "does not partition"),
        (_swap("term_offsets", np.array([0, 1, 9], dtype="i4")), "does not partition"),
        (_swap("rule", np.array([1, 2], dtype="i4")), "not a string column"),
        (lambda z, g: z.attrs.__delitem__("current"), "no @current"),
        (lambda z, g: z.attrs.__setitem__("current", np.int32(9)), "names g9"),
        (lambda z, g: z.attrs.__setitem__("current", "one"), "not an integer"),
    ],
)
def test_malformed_columns_refuse_the_zone_and_the_file_still_opens(
    make_model, tmp_path: Path, how, match,
) -> None:
    path, _ = _raw(make_model, tmp_path, [GOOD, Combination("two", "linear", (D,))])
    write_user_file(path, make_model(), results=[nodal()])
    _mangle(path, how)
    uf = UserFile(path)
    with pytest.raises(ZoneRefused, match=match):
        uf.combinations()
    assert uf.results().entries[0].name == nodal().name


def test_a_zone_without_its_key_or_a_key_without_its_zone_refuses(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[GOOD])
    with h5py.File(path, "a") as f:
        del f["meta"].attrs["combinations_schema_version"]
    with pytest.raises(ZoneRefused, match="no version key"):
        UserFile(path).combinations()
    other = write_user_file(tmp_path / "t.user.h5", make_model())
    with h5py.File(other, "a") as f:
        f["meta"].attrs["user_results_schema_version"] = "1.0.0"
    with pytest.raises(ZoneRefused, match="no /results"):
        UserFile(other).results()


# ---- /results entries refuse alone ------------------------------------------


@pytest.mark.parametrize(
    "how, reason",
    [
        (lambda r: r.attrs.__setitem__("name", " edge"), "empty-name"),
        (lambda r: r.attrs.__delitem__("name"), "empty-name"),
        (lambda r: r.attrs.__setitem__("target", "faces"), "bad-param"),
        (lambda r: r.attrs.__setitem__("basis", "vibes"), "bad-param"),
        (lambda r: r.attrs.__setitem__("per_state", np.int8(3)), "bad-param"),
        (lambda r: r.attrs.__delitem__("per_state"), "bad-param"),
        (lambda r: r.__delitem__("STATES"), "bad-param"),
        (lambda r: r.__delitem__("RESULTS"), "bad-param"),
        (lambda r: r["RESULTS/ON_NODES/values"].__delitem__("DATA"), "bad-param"),
        (lambda r: r["RESULTS/ON_NODES/values"].__delitem__("ID"), "bad-param"),
        (lambda r: r["RESULTS/ON_NODES/values"].attrs.__setitem__("COMPONENTS", "only one"), "bad-param"),
        (lambda r: r["STATES/kind"].__setitem__(0, "stack"), "bad-term"),
    ],
)
def test_a_bad_entry_refuses_alone(make_model, tmp_path: Path, how, reason) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("bad"), nodal("fine")])
    with h5py.File(path, "a") as f:
        how(f["results/r0"])
    uf = UserFile(path)
    index = uf.results()
    assert [e.name for e in index.entries] == ["fine"]
    assert [r.reason for r in index.refused] == [reason]
    assert np.isnan(uf.read_result("fine").data[0, 1, 0])


def test_a_broken_link_in_an_entry_refuses_that_entry(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("bad"), nodal("fine")])
    with h5py.File(path, "a") as f:
        values = f["results/r0/RESULTS/ON_NODES/values"]
        del values["DATA"]
        values["DATA"] = h5py.SoftLink("/nowhere")
    index = UserFile(path).results()
    assert [e.name for e in index.entries] == ["fine"]
    assert [r.name for r in index.refused] == ["bad"]


def test_a_dataset_where_results_belongs_refuses_that_entry(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("bad"), nodal("keep")])
    with h5py.File(path, "a") as f:
        del f["results/r0/RESULTS"]
        f["results/r0"].create_dataset("RESULTS", data=[1])
    index = UserFile(path).results()
    assert [e.name for e in index.entries] == ["keep"]
    assert [(r.name, r.reason) for r in index.refused] == [("bad", "bad-param")]


@pytest.mark.parametrize("extra", [1, -1])
def test_data_axis_0_must_match_the_states(make_model, tmp_path: Path, extra: int) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("bad"), nodal("keep")])
    with h5py.File(path, "a") as f:
        values = f["results/r0/RESULTS/ON_NODES/values"]
        s, n, c = values["DATA"].shape
        del values["DATA"], values["STEP"], values["TIME"]
        values.create_dataset("DATA", data=np.zeros((s + extra, n, c)))
        values.create_dataset("STEP", data=np.arange(s + extra, dtype="i4"))
        values.create_dataset("TIME", data=np.arange(s + extra, dtype="f8"))
    index = UserFile(path).results()
    assert [e.name for e in index.entries] == ["keep"]
    assert [r.reason for r in index.refused] == ["bad-param"]
    assert "rows on axis 0" in index.refused[0].detail


def test_a_state_free_result_has_one_row(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("bad")])
    with h5py.File(path, "a") as f:
        r = f["results/r0"]
        r.attrs["per_state"] = np.int8(0)
        del r["STATES"]
    index = UserFile(path).results()
    assert [r.reason for r in index.refused] == ["bad-param"]


@pytest.mark.parametrize("link", ["combinations", "combinations/g1/rule", "combinations/g1/terms/kind"])
def test_a_dangling_link_refuses_the_zone(make_model, tmp_path: Path, link: str) -> None:
    path, _ = _raw(make_model, tmp_path, [GOOD])
    write_user_file(path, make_model("m2.h5"), results=[nodal()])
    with h5py.File(path, "a") as f:
        del f[link]
        f[link] = h5py.SoftLink("/nowhere")
    uf = UserFile(path)
    with pytest.raises(ZoneRefused):
        uf.combinations()
    assert uf.results().entries[0].name == nodal().name


def test_a_dangling_results_zone_refuses_the_zone(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()], combinations=[GOOD])
    with h5py.File(path, "a") as f:
        del f["results"]
        f["results"] = h5py.SoftLink("/nowhere")
    uf = UserFile(path)
    with pytest.raises(ZoneRefused):
        uf.results()
    assert uf.combinations().rows == (GOOD,)


def test_model_combinations_are_read_under_the_opensees_version(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    model = make_model("newer.h5")
    _model_rows(model, [GOOD])
    reader = reader_version(OPENSEES)
    with h5py.File(model, "a") as f:
        f["meta"].attrs[OPENSEES_KEY] = f"{reader.major}.{reader.minor + 1}.0"
    with pytest.raises(SchemaVersionError, match="newer"):
        UserFile(path).combinations(model_h5=model)
    with h5py.File(model, "a") as f:
        del f["meta"].attrs[OPENSEES_KEY]
    with pytest.raises(ZoneRefused, match="opensees_schema_version"):
        UserFile(path).combinations(model_h5=model)


def test_a_model_without_the_zone_needs_no_opensees_version(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[GOOD])
    assert UserFile(path).combinations(model_h5=make_model()).rows == (GOOD,)


def test_a_model_that_is_not_hdf5_refuses_as_a_zone(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[GOOD])
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not hdf5")
    with pytest.raises(ZoneRefused):
        UserFile(path).combinations(model_h5=junk)


def test_a_damaged_meta_refuses_the_file(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    with h5py.File(path, "a") as f:
        attrs = dict(f["meta"].attrs)
        del f["meta"]
        f["meta"] = np.zeros(1)
        assert attrs
    with pytest.raises(UserFileRefused):
        UserFile(path)


def test_a_duplicate_result_name_refuses_both(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("a"), nodal("b")])
    with h5py.File(path, "a") as f:
        f["results/r1"].attrs["name"] = "a"
    index = UserFile(path).results()
    assert index.entries == () and [r.reason for r in index.refused] == ["duplicate", "duplicate"]
    with pytest.raises(KeyError, match="refused"):
        UserFile(path).read_result("a")


def test_an_absent_result_is_a_named_keyerror(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal("a")])
    with pytest.raises(KeyError, match="no user result named"):
        UserFile(path).read_result("nobody")


# ---- the writer refuses what the reader would ------------------------------


@pytest.mark.parametrize(
    "rows, reason",
    [
        ([Combination("a", "median", (D,))], "unknown-rule"),
        ([Combination("a", "linear", (D,)), Combination("a", "linear", (D,))], "duplicate"),
        ([Combination("a", "linear", (Term("combination", "a"),))], "cycle"),
        ([Combination("a", "linear", (Term("combination", "none"),))], "unresolved"),
    ],
)
def test_the_writer_refuses_a_row_the_reader_would(make_model, tmp_path: Path, rows, reason) -> None:
    target = tmp_path / "s.user.h5"
    with pytest.raises(UserWriteRefused, match=reason):
        write_user_file(target, make_model(), combinations=rows)
    assert not target.exists()


def test_the_writer_refuses_a_clash_with_the_model(make_model, tmp_path: Path) -> None:
    model = make_model()
    _model_rows(model, [GOOD])
    with pytest.raises(UserWriteRefused, match="clash"):
        write_user_file(tmp_path / "s.user.h5", model, combinations=[GOOD])


def test_a_row_may_name_a_model_row_through_the_writer(make_model, tmp_path: Path) -> None:
    model = make_model()
    _model_rows(model, [GOOD])
    up = Combination("up", "linear", (Term("combination", "good"),))
    path = write_user_file(tmp_path / "s.user.h5", model, combinations=[up])
    table = UserFile(path).combinations(model_h5=model)
    assert table.rows == (up,) and table.model_rows == (GOOD,)
