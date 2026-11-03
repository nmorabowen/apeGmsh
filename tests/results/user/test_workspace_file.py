"""Reading and writing ``<stem>.workspace.h5`` (ADR 0132 D2, D4-D6; h5-schema section).

The file-level, table-level and row-level refusals, the floor rule, the plot
grammar's closed keys, and the write: a whole rebuild, rows copied verbatim,
the revision check, a newer file never overwritten and an unreadable one
renamed aside.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.user import (
    PlotDef,
    SetDef,
    ViewDef,
    WorkspaceFile,
    WorkspaceRefused,
    WorkspaceRenamedAside,
    WorkspaceWriteRefused,
    ZoneRefused,
    _workspace,
    write_workspace,
)

from tests.results.user._workspace_helpers import frame

STR = h5py.string_dtype(encoding="utf-8")

HISTORY = {
    "kind": "history",
    "over": {"kind": "stage", "source": "pushover", "steps": [0, -1], "axis": "step"},
    "series": [{
        "label": "roof drift",
        "quantity": {"group": "results", "field": "DISPLACEMENT", "component": "Ux"},
        "target": {"kind": "set", "name": "roof"},
        "reduce": "mean",
        "minus": {"target": {"kind": "set", "name": "base"}, "reduce": "mean"},
        "scale": 1000.0,
    }],
}
XY = {
    "kind": "xy",
    "over": {"kind": "stage", "source": "pushover", "axis": "time"},
    "x": {"quantity": {"group": "results", "field": "DISPLACEMENT", "component": "Ux"},
          "target": {"kind": "label", "name": "roof"}, "reduce": "max"},
    "y": [{"quantity": {"group": "user_results", "field": "base shear", "component": "Vx"},
           "target": {"kind": "physical", "name": "Base"}, "reduce": "sum"}],
}
PROFILE = {
    "kind": "profile",
    "state": {"kind": "stage", "source": "gravity", "step": -1, "mode": 0},
    "axis": "z",
    "series": [{"quantity": {"group": "results", "field": "DISPLACEMENT", "component": "Ux"},
                "target": {"kind": "set", "name": "column line"}, "reduce": "each"}],
}
SHOT = {"name": "iso", "camera": {"preset": "iso"}}


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return frame(tmp_path / "frame.h5")


@pytest.fixture
def ws(model: Path) -> Path:
    return write_workspace(
        model.with_name("frame.workspace.h5"),
        sets=[SetDef("roof", "nodes", [31, 32, 33]), SetDef("springs", "elements", [501, 502])],
        plots=[PlotDef("drift", HISTORY), PlotDef("capacity", XY), PlotDef("sway", PROFILE)],
        views=[ViewDef("iso", SHOT)],
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hand_file(path: Path, *, version: str = "1.0.0", revision: int = 4,
               sets: tuple = (), plots: tuple = (), views: tuple = (),
               members: tuple = (), skip: tuple = ()) -> Path:
    """A workspace laid out by hand: ``sets`` rows are (name, kind, id_space, fem, model),
    ``members`` one (ids, xyz) per set row, plot and view rows (name, grammar, body)."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["workspace_schema_version"] = version
        meta.attrs["revision"] = np.int32(revision)
        meta.attrs["writer"] = "apeGmshViewer 0.9.0"
        meta.attrs["save_id"] = "viewer_save_0001"
        if "sets" not in skip:
            g = f.create_group("sets")
            for i, col in enumerate(("name", "kind", "id_space", "fem_hash", "model_hash")):
                g.create_dataset(col, data=[r[i] for r in sets], dtype=STR)
            sizes = [len(m[0]) for m in members]
            g.create_dataset("member_offsets", data=np.concatenate(([0], np.cumsum(sizes))).astype("i4"))
            g.create_dataset("ids", data=np.concatenate([np.asarray(m[0], "i4") for m in members]
                                                        or [np.zeros(0, "i4")]))
            g.create_dataset("xyz", data=np.concatenate([np.asarray(m[1], "f8").reshape(-1, 3)
                                                         for m in members] or [np.zeros((0, 3))]))
        for table, rows in (("plots", plots), ("views", views)):
            if table in skip:
                continue
            g = f.create_group(table)
            for i, col in enumerate(("name", "grammar", "body")):
                g.create_dataset(col, data=[r[i] for r in rows], dtype=STR)
    return path


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_a_write_reads_back_every_table(ws: Path) -> None:
    f = WorkspaceFile(ws)
    assert (f.version, f.revision) == ("1.0.0", 1)
    assert f.writer.startswith("apeGmsh ")
    assert re.fullmatch(r"[A-Za-z0-9_-]{8,64}", f.save_id)
    assert [s.name for s in f.sets] == ["roof", "springs"]
    assert [p.name for p in f.plots] == ["drift", "capacity", "sway"]
    assert f.plots[0].data == HISTORY and f.views[0].data == SHOT
    assert not f.refused


def test_empty_tables_are_written_with_offsets_0(model: Path) -> None:
    ws = write_workspace(model.with_name("frame.workspace.h5"), views=[ViewDef("iso", SHOT)])
    with h5py.File(ws, "r") as f:
        assert f["sets/member_offsets"][...].tolist() == [0]
        assert f["sets/xyz"].shape == (0, 3) and f["plots/name"].shape == (0,)
        assert f["sets/ids"].dtype == np.dtype("i4") and f["sets/xyz"].dtype == np.dtype("f8")
        assert h5py.check_string_dtype(f["views/body"].dtype).encoding == "utf-8"
    assert WorkspaceFile(ws).sets == ()


def test_a_second_save_adds_one_revision_and_replaces_by_name_in_place(ws: Path) -> None:
    before = WorkspaceFile(ws)
    write_workspace(ws, plots=[PlotDef("drift", XY)], sets=[SetDef("base", "nodes", [1, 2])])
    after = WorkspaceFile(ws)
    assert after.revision == before.revision + 1
    assert after.save_id != before.save_id
    assert [p.name for p in after.plots] == ["drift", "capacity", "sway"]
    assert after.plots[0].data == XY
    assert [s.name for s in after.sets] == ["roof", "springs", "base"]


def test_remove_drops_a_row_and_refuses_an_absent_one(ws: Path) -> None:
    write_workspace(ws, remove={"plots": ["sway"], "sets": ["springs"]})
    f = WorkspaceFile(ws)
    assert [p.name for p in f.plots] == ["drift", "capacity"]
    assert [s.name for s in f.sets] == ["roof"]
    with pytest.raises(WorkspaceWriteRefused, match="no plots named"):
        write_workspace(ws, remove={"plots": ["sway"]})
    with pytest.raises(WorkspaceWriteRefused, match="unknown table"):
        write_workspace(ws, remove={"curves": ["x"]})


def test_rows_the_writer_did_not_change_are_copied_verbatim_refused_rows_included(model: Path) -> None:
    target = _hand_file(
        model.with_name("frame.workspace.h5"),
        sets=[("weird", "faces", "opensees", "f" * 32, ""),
              ("roof", "nodes", "opensees", "f" * 32, "")],
        members=[([1], [[0, 0, 0]]), ([31, 32], [[0, 0, 9], [6, 0, 9]])],
        plots=[("old", "apegmsh-plot/9", "{not even json"), ("badé", "apegmsh-plot/1", "[]")],
        views=[("v", "apegmsh-view/1", '{"x": 1}')],
    )
    with h5py.File(target, "r") as f:
        before = {t: {c: f[t][c][...].tolist() for c in f[t]} for t in ("sets", "plots", "views")}
    assert {r.reason for r in WorkspaceFile(target).refused} == {
        "bad-param", "unknown-grammar", "bad-json"}

    write_workspace(target, views=[ViewDef("new", SHOT)])

    with h5py.File(target, "r") as f:
        after = {t: {c: f[t][c][...].tolist() for c in f[t]} for t in ("sets", "plots", "views")}
        assert f["meta"].attrs["revision"] == 5
    assert after["sets"] == before["sets"]
    assert after["plots"] == before["plots"]
    assert after["views"]["name"] == before["views"]["name"] + [b"new"]


def test_a_moved_set_takes_the_models_hashes_on_the_next_save(ws: Path, model: Path) -> None:
    frame(model, fem_hash="b" * 32)
    assert WorkspaceFile(ws).verdicts().by_name("roof").verdict == "moved"
    write_workspace(ws, views=[ViewDef("another", SHOT)])
    roof = WorkspaceFile(ws).set("roof")
    assert roof.fem_hash == "b" * 32
    assert WorkspaceFile(ws).verdicts().by_name("roof").verdict == "current"
    # The springs read lost (ambiguous), so their row is left as it was.
    assert WorkspaceFile(ws).set("springs").fem_hash == "f" * 32


# ---------------------------------------------------------------------------
# What the writer refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("defs, reason", [
    ([SetDef("roof", "nodes", [31, 999])], "bad-row"),
    ([SetDef("roof", "elements", [201, 999])], "bad-row"),
    ([SetDef("roof", "nodes", [])], "bad-members"),
    ([SetDef("roof", "nodes", [31, 31])], "bad-members"),
    ([SetDef("roof", "nodes", [True])], "bad-members"),
    ([SetDef("roof", "faces", [1])], "bad-param"),
    ([SetDef(" roof", "nodes", [1])], "empty-name"),
    ([SetDef("roof", "nodes", [1]), SetDef("roof", "nodes", [2])], "duplicate"),
])
def test_the_writer_never_writes_a_row_that_would_refuse(model: Path, defs, reason) -> None:
    target = model.with_name("frame.workspace.h5")
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(target, sets=defs)
    assert err.value.reason == reason
    assert not target.exists()
    assert not list(model.parent.glob("*.partial"))


@pytest.mark.parametrize("body, reason", [
    ({"kind": "history", "over": HISTORY["over"], "series": HISTORY["series"], "title": "x"},
     "bad-plot"),
    ("{\"kind\": \"history\", \"kind\": \"xy\"}", "bad-json"),
    ("[1, 2]", "bad-json"),
    ({"kind": "xy", **{k: v for k, v in XY.items() if k != "kind"}, "x": PROFILE["series"][0]},
     "bad-plot"),
])
def test_the_writer_refuses_a_bad_plot(model: Path, body, reason) -> None:
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(model.with_name("frame.workspace.h5"), plots=[PlotDef("p", body)])
    assert err.value.reason == reason


def test_an_element_set_needs_a_tag_map_and_a_model_hash(tmp_path: Path) -> None:
    frame(tmp_path / "a.h5", opensees=False)
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(tmp_path / "a.workspace.h5", sets=[SetDef("s", "elements", [201])])
    assert err.value.reason == "no-tag-map"
    frame(tmp_path / "b.h5", model_hash=None)
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(tmp_path / "b.workspace.h5", sets=[SetDef("s", "elements", [201])])
    assert err.value.reason == "bad-param"


def test_a_write_without_the_paired_model_refuses_unpaired(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(tmp_path / "lonely.workspace.h5", views=[ViewDef("v", SHOT)])
    assert err.value.reason == "unpaired"
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(tmp_path / ".workspace.h5", views=[ViewDef("v", SHOT)])
    assert err.value.reason == "bad-name"


@pytest.mark.parametrize("version, reason", [("1.1.0", "newer"), ("2.0.0", "major")])
def test_no_writer_writes_over_a_newer_file(model: Path, version: str, reason: str) -> None:
    target = _hand_file(model.with_name("frame.workspace.h5"), version=version)
    before = _digest(target)
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(target, views=[ViewDef("v", SHOT)])
    assert err.value.reason == reason
    assert _digest(target) == before
    assert not list(model.parent.glob("*.partial"))


@pytest.mark.parametrize("content", ["junk", "no-version", "two-part-version"])
def test_an_unreadable_file_is_renamed_aside_never_overwritten(model: Path, content: str) -> None:
    target = model.with_name("frame.workspace.h5")
    if content == "junk":
        target.write_bytes(b"this is not HDF5")
    else:
        _hand_file(target, version="1.0" if content == "two-part-version" else "1.0.0")
        if content == "no-version":
            with h5py.File(target, "a") as f:
                del f["meta"].attrs["workspace_schema_version"]
    before = target.read_bytes()
    with pytest.warns(WorkspaceRenamedAside, match="renamed aside"):
        write_workspace(target, views=[ViewDef("v", SHOT)])
    asides = sorted(model.parent.glob("frame.workspace.h5.unreadable-*"))
    assert len(asides) == 1
    assert re.fullmatch(r"frame\.workspace\.h5\.unreadable-\d{8}T\d{6}Z", asides[0].name)
    assert asides[0].read_bytes() == before
    assert WorkspaceFile(target).revision == 1


def test_a_second_aside_in_the_same_second_takes_a_counter(model: Path, monkeypatch) -> None:
    real = _workspace.time.gmtime
    monkeypatch.setattr(_workspace.time, "gmtime", lambda *a: real(0))
    target = model.with_name("frame.workspace.h5")
    for _ in range(2):
        target.write_bytes(b"junk")
        with pytest.warns(WorkspaceRenamedAside):
            write_workspace(target, views=[ViewDef("v", SHOT)])
    names = sorted(p.name for p in model.parent.glob("frame.workspace.h5.unreadable-*"))
    assert names == ["frame.workspace.h5.unreadable-19700101T000000Z",
                     "frame.workspace.h5.unreadable-19700101T000000Z-1"]


def test_the_second_of_two_writers_refuses_revision_moved(ws: Path, monkeypatch) -> None:
    real = _workspace._fsync

    def another_writer_lands(partial: Path) -> None:
        real(partial)
        with h5py.File(ws, "a") as f:  # what another writer's replace looks like
            f["meta"].attrs["revision"] = np.int32(9)

    monkeypatch.setattr(_workspace, "_fsync", another_writer_lands)
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(ws, views=[ViewDef("late", SHOT)])
    assert err.value.reason == "revision-moved"
    assert WorkspaceFile(ws).revision == 9
    assert [v.name for v in WorkspaceFile(ws).views] == ["iso"]
    assert not list(ws.parent.glob("*.partial"))


def test_a_file_created_during_a_first_write_is_not_overwritten(model: Path, monkeypatch) -> None:
    target = model.with_name("frame.workspace.h5")
    real = _workspace._fsync

    def appears(partial: Path) -> None:
        real(partial)
        _hand_file(target)

    monkeypatch.setattr(_workspace, "_fsync", appears)
    with pytest.raises(WorkspaceWriteRefused, match="revision-moved"):
        write_workspace(target, views=[ViewDef("v", SHOT)])
    assert WorkspaceFile(target).writer == "apeGmshViewer 0.9.0"


def test_a_partial_that_does_not_read_back_refuses(ws: Path, monkeypatch) -> None:
    real = _workspace._write_file

    def corrupt(out, meta, tables):
        real(out, meta, tables)
        out["sets/xyz"][0, 0] = 99.0

    monkeypatch.setattr(_workspace, "_write_file", corrupt)
    before = _digest(ws)
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(ws, views=[ViewDef("v2", SHOT)])
    assert err.value.reason == "partial-mismatch"
    assert _digest(ws) == before
    assert not list(ws.parent.glob("*.partial"))


# ---------------------------------------------------------------------------
# What the reader refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version, reason", [
    ("1.1.0", "newer"), ("2.0.0", "major"), ("0.9.0", "major"), ("1.0", "bad-version"),
    ("01.0.0", "bad-version"), ("1.0.0-rc1", "bad-version"),
])
def test_the_floor_rule(model: Path, version: str, reason: str) -> None:
    target = _hand_file(model.with_name("frame.workspace.h5"), version=version)
    with pytest.raises(WorkspaceRefused) as err:
        WorkspaceFile(target)
    assert err.value.reason == reason


def test_a_later_patch_reads(model: Path) -> None:
    assert WorkspaceFile(_hand_file(model.with_name("frame.workspace.h5"), version="1.0.7"))


@pytest.mark.parametrize("table", ["sets", "plots", "views"])
def test_a_missing_table_refuses_the_file(model: Path, table: str) -> None:
    target = _hand_file(model.with_name("frame.workspace.h5"), skip=(table,))
    with pytest.raises(WorkspaceRefused) as err:
        WorkspaceFile(target)
    assert err.value.reason == "missing-table"


def test_not_hdf5_and_no_version(model: Path) -> None:
    target = model.with_name("frame.workspace.h5")
    target.write_bytes(b"{}")
    with pytest.raises(WorkspaceRefused) as err:
        WorkspaceFile(target)
    assert err.value.reason == "not-hdf5"
    with pytest.raises(WorkspaceRefused) as err:
        WorkspaceFile(model)  # a model archive has no workspace version key
    assert err.value.reason == "no-version"


def test_malformed_columns_refuse_that_table_and_the_others_read(model: Path) -> None:
    target = _hand_file(model.with_name("frame.workspace.h5"),
                        sets=[("roof", "nodes", "opensees", "f" * 32, "")],
                        members=[([31, 32], [[0, 0, 9], [6, 0, 9]])],
                        views=[("v", "apegmsh-view/1", "{}")])
    with h5py.File(target, "a") as f:
        del f["sets/member_offsets"]
        f["sets"].create_dataset("member_offsets", data=np.array([0, 3], dtype="i4"))
    ws = WorkspaceFile(target)
    with pytest.raises(ZoneRefused, match="member_offsets"):
        ws.sets
    assert [v.name for v in ws.views] == ["v"]
    with pytest.raises(WorkspaceWriteRefused) as err:
        write_workspace(target, views=[ViewDef("w", SHOT)])
    assert err.value.reason == "bad-table"


def test_row_refusals(model: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "café")
    target = _hand_file(
        model.with_name("frame.workspace.h5"),
        sets=[("a", "nodes", "opensees", "f", ""), ("a", "nodes", "opensees", "f", ""),
              (decomposed, "nodes", "opensees", "f", ""), ("empty", "nodes", "opensees", "f", ""),
              ("twice", "nodes", "opensees", "f", ""), ("space", "nodes", "abaqus", "f", ""),
              ("no model hash", "elements", "opensees", "f", ""),
              ("ok", "elements", "opensees", "f", "m")],
        members=[([1], [[0, 0, 0]]), ([2], [[6, 0, 0]]), ([3], [[12, 0, 0]]), ([], []),
                 ([1, 1], [[0, 0, 0], [0, 0, 0]]), ([1], [[0, 0, 0]]), ([201], [[0, 0, 1.5]]),
                 ([201], [[0, 0, 1.5]])],
        plots=[("nan", "apegmsh-plot/1", '{"kind": "history", "x": NaN}'),
               ("closed", "apegmsh-plot/1", json.dumps({**HISTORY, "colour": "red"})),
               ("grammar", "apegmsh-plot/2", json.dumps(HISTORY)),
               ("fine", "apegmsh-plot/1", json.dumps(PROFILE))],
        views=[("list", "apegmsh-view/1", "[]"), ("shot", "apegmsh-view/1", json.dumps(SHOT))],
    )
    ws = WorkspaceFile(target)
    got = [(r.table, r.name, r.reason) for r in ws.refused]
    assert got == [
        ("sets", "a", "duplicate"), ("sets", "a", "duplicate"),
        ("sets", decomposed, "empty-name"), ("sets", "empty", "bad-members"),
        ("sets", "twice", "bad-members"), ("sets", "space", "bad-param"),
        ("sets", "no model hash", "bad-param"),
        ("plots", "nan", "bad-json"), ("plots", "closed", "bad-plot"),
        ("plots", "grammar", "unknown-grammar"), ("views", "list", "bad-json"),
    ]
    assert "'colour'" in next(r.detail for r in ws.refused if r.name == "closed")
    assert [s.name for s in ws.sets] == ["ok"]
    assert [p.name for p in ws.plots] == ["fine"] and [v.name for v in ws.views] == ["shot"]


# ---------------------------------------------------------------------------
# The plot grammar
# ---------------------------------------------------------------------------


def _with(plot: dict, path: tuple, value) -> dict:
    out = json.loads(json.dumps(plot))
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


@pytest.mark.parametrize("plot", [HISTORY, XY, PROFILE])
def test_the_three_plot_kinds_stand(plot: dict) -> None:
    assert _workspace.plot_problem(plot) is None


@pytest.mark.parametrize("plot, needle", [
    (_with(HISTORY, ("kind",), "bar"), "kind"),
    (_with(HISTORY, ("series",), []), "series"),
    (_with(HISTORY, ("over", "kind"), "mode"), "over.kind"),
    (_with(HISTORY, ("over", "steps"), [-2, 3]), "over.steps"),
    (_with(HISTORY, ("over", "steps"), [0, True]), "over.steps"),
    (_with(HISTORY, ("over", "axis"), "frequency"), "over.axis"),
    (_with(HISTORY, ("over", "source"), KeyError), "'source'"),
    (_with(HISTORY, ("series", 0, "reduce"), "median"), "reduce"),
    (_with(HISTORY, ("series", 0, "scale"), True), "scale"),
    (_with(HISTORY, ("series", 0, "scale"), 10 ** 400), "scale"),
    (_with(HISTORY, ("series", 0, "quantity", "group"), "mpco"), "quantity.group"),
    (_with(HISTORY, ("series", 0, "target", "kind"), "box"), "target.kind"),
    (_with(HISTORY, ("series", 0, "minus", "reduce"), "each"), "minus.reduce"),
    (_with(HISTORY, ("series", 0, "label"), " x"), "label"),
    (_with(XY, ("y", 0, "reduce"), "each"), "y[0].reduce"),
    (_with(PROFILE, ("series", 0, "reduce"), "sum"), "reduce"),
    (_with(PROFILE, ("axis",), "time"), "axis"),
    (_with(PROFILE, ("state", "step"), -2), "state"),
    (_with(PROFILE, ("state", "mode"), KeyError), "'mode'"),
])
def test_a_plot_that_breaks_the_grammar_names_the_key(plot: dict, needle: str) -> None:
    problem = _workspace.plot_problem(plot)
    assert problem is not None and needle in problem


def test_strict_json_refuses_nan_infinity_overflow_and_a_repeated_key() -> None:
    for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": 1e400}', '{"a": 1, "a": 2}'):
        with pytest.raises(ValueError):
            _workspace.strict_json(text)
    assert _workspace.strict_json('{"a": 1e400}', finite=False)["a"] == float("inf")
