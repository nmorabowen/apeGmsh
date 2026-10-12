"""ADR 0098 Amendment 7 (A7.3, A7.7) at the IR layer — no Qt, no GL.

* A7.3: the four colour-mapped slot records carry ``clim`` / ``cmap``.
  ``None`` is today's default (no default changes), ``cmap`` is any
  matplotlib name (``'jet'`` included) validated at assignment, and two
  occupied slots of one field that set them differently RAISE — a
  silent winner would be a wrong legend.
* A7.7: the writer writes snapshot v2, the reader takes v1 and v2, a v1
  ``path`` plot pane drops with a notice, and ``path`` left §6.

Every test here fails on the pre-A7 tree (the records had no style
fields, the snapshot was v1-only, and ``path`` was a plot kind).
"""
from __future__ import annotations

import json
import math

import pytest

from apeGmsh.results.session import (
    COLOUR_MAPPED,
    PLOT_KINDS,
    SLOT_CATALOG,
    SNAPSHOT_VERSION,
    Contour,
    Gauss,
    Line,
    Loads,
    PlotView,
    Reactions,
    ResultsSession,
    Sand,
    SnapshotError,
    Vector,
    restore_snapshot,
    slot_style,
    snapshot,
)

_STYLED = {"contour": Contour, "vector": Vector, "gauss": Gauss, "sand": Sand}


@pytest.fixture
def view():
    return ResultsSession().add_view()


# =====================================================================
# A7.3 — the stored style fields
# =====================================================================

def test_style_fields_live_on_exactly_the_colour_mapped_records():
    assert set(_STYLED) == set(COLOUR_MAPPED)
    for record_type in _STYLED.values():
        record = record_type("S")
        # No default changed: None means "the kind's existing default".
        assert (record.clim, record.cmap) == (None, None)
    for record_type, kwargs in (
        (Line, {"component": "M"}), (Loads, {}), (Reactions, {}),
    ):
        with pytest.raises(TypeError):
            record_type(**kwargs, cmap="viridis")


@pytest.mark.parametrize("category", sorted(_STYLED))
def test_jet_is_accepted_on_every_colour_mapped_record(category):
    record = _STYLED[category]("S", cmap="jet")
    assert record.cmap == "jet"
    assert slot_style(record) == (None, "jet")


@pytest.mark.parametrize("category", sorted(_STYLED))
def test_an_unknown_cmap_raises_at_assignment(category):
    with pytest.raises(ValueError, match="not a colormap"):
        _STYLED[category]("S", cmap="definitely_not_a_cmap")


@pytest.mark.parametrize("bad", [
    (1.0,), (0.0, 1.0, 2.0), (1.0, 1.0), (2.0, 1.0),
    (0.0, math.nan), (-math.inf, 1.0), ("0", "1"), "01", (True, 2.0), 3.0,
])
def test_a_malformed_clim_raises_at_assignment(bad):
    with pytest.raises(ValueError, match="clim"):
        Contour("S", clim=bad)


def test_clim_normalises_to_a_float_pair():
    # A JSON list and an int pair make the SAME record — the snapshot
    # reader depends on it to round-trip equal.
    assert Contour("S", clim=[0, 5]) == Contour("S", clim=(0.0, 5.0))
    assert Contour("S", clim=[0, 5]).clim == (0.0, 5.0)


def test_style_is_not_field_identity(view):
    # Display state: same quantity on two slots is still one scale.
    view.contour = Contour("S", clim=(0.0, 1.0), cmap="jet")
    view.gauss = Gauss("S")
    legends = view.legends()
    assert [lg.field for lg in legends] == ["S"]
    assert legends[0].categories == ("contour", "gauss")


# ---------------------------------------------------------------------
# The scale law: one field, one scale, no silent winner
# ---------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: (Contour("S", clim=(0.0, 1.0)), Gauss("S", clim=(0.0, 2.0))),
    lambda: (Contour("S", cmap="jet"), Gauss("S", cmap="viridis")),
    lambda: (Vector("S", cmap="jet"), Sand("S", cmap="coolwarm")),
], ids=["clim", "cmap", "vector-sand"])
def test_a_conflicting_explicit_style_raises_and_changes_nothing(
    view, make,
):
    first, second = make()
    categories = {Contour: "contour", Vector: "vector",
                  Gauss: "gauss", Sand: "sand"}
    setattr(view, categories[type(first)], first)
    before = dict(view.slots)

    with pytest.raises(ValueError, match="One field is one scale"):
        setattr(view, categories[type(second)], second)

    assert view.slots == before


def test_matching_or_unset_style_is_not_a_conflict(view):
    view.contour = Contour("S", clim=(0.0, 1.0))
    view.gauss = Gauss("S", clim=(0.0, 1.0), cmap="jet")  # same clim
    view.sand = Sand("S")                                 # defers
    (legend,) = view.legends()
    # The range and map come from the slot that sets them.
    assert (legend.clim, legend.cmap) == ((0.0, 1.0), "jet")


def test_different_fields_never_conflict(view):
    view.contour = Contour("S", clim=(0.0, 1.0), cmap="jet")
    view.gauss = Gauss("E", clim=(5.0, 9.0), cmap="viridis")
    styles = {lg.field: (lg.clim, lg.cmap) for lg in view.legends()}
    assert styles == {"S": ((0.0, 1.0), "jet"), "E": ((5.0, 9.0), "viridis")}


def test_replacing_the_occupant_is_not_a_conflict_with_itself(view):
    view.contour = Contour("S", clim=(0.0, 1.0))
    view.contour = Contour("S", clim=(0.0, 2.0))
    assert view.legends()[0].clim == (0.0, 2.0)


def test_an_unstyled_view_reports_default_legends(view):
    view.contour = Contour("S")
    (legend,) = view.legends()
    assert (legend.clim, legend.cmap) == (None, None)


# =====================================================================
# A7.7 — snapshot v2, the v1 reader, and `path` leaving §6
# =====================================================================

def test_path_is_no_longer_a_plot_kind():
    assert PLOT_KINDS == ("history", "xy")
    with pytest.raises(ValueError, match="PlotView.kind"):
        PlotView("plot-x", kind="path")


def test_the_writer_writes_v2():
    assert SNAPSHOT_VERSION == 2
    assert snapshot(ResultsSession())["version"] == 2


def test_v2_round_trips_clim_and_cmap_through_json():
    session = ResultsSession()
    v = session.add_view()
    v.contour = Contour("S", averaging="unaveraged",
                        clim=(-1.5, 3.0), cmap="jet")
    v.vector = Vector("U", cmap="viridis")
    v.gauss = Gauss("S", clim=(-1.5, 3.0))
    v.sand = Sand("P")

    payload = json.loads(json.dumps(snapshot(session)))
    assert payload["panes"][0]["slots"]["contour"]["clim"] == [-1.5, 3.0]
    restored = restore_snapshot(payload)

    assert restored.notices == ()
    assert restored.session.panes[0].slots == v.slots


#: A v1 file exactly as the pre-A7 writer wrote it: slot payloads with
#: no style keys, a `path` plot beside a history plot.
_V1 = {
    "kind": "apegmsh.results.session",
    "version": 1,
    "saved_at": "2026-10-01T00:00:00+00:00",
    "results_path": None,
    "time": None,
    "time_linked": True,
    "selection": {"kind": None},
    "panes": [
        {"pane": "mesh", "id": "mesh-1", "name": None,
         "slots": {"contour": {"quantity": "S", "averaging": "averaged"},
                   "gauss": {"quantity": "S"},
                   "line": {"component": "M"},
                   "reactions": {}},
         "legend_hidden": ["S"]},
        {"pane": "plot", "id": "plot-2", "kind": "history", "name": "h",
         "series": [{"source": {"kind": "node", "key": 7},
                     "quantity": "displacement_x"}],
         "cursor": None},
        {"pane": "plot", "id": "plot-3", "kind": "path", "name": "p",
         "series": [], "cursor": None},
    ],
}


def _v1(**changes):
    data = json.loads(json.dumps(_V1))
    data.update(changes)
    return data


def test_a_v1_snapshot_restores_identically_and_resaves_as_v2():
    restored = restore_snapshot(_v1(panes=_V1["panes"][:2]))
    assert restored.notices == ()
    mesh, plot = restored.session.panes

    # The style fields a v1 file cannot carry restore as the defaults.
    assert mesh.slots == {
        "contour": Contour("S"), "gauss": Gauss("S"),
        "line": Line("M"), "reactions": Reactions(),
    }
    assert [(lg.field, lg.hidden, lg.clim, lg.cmap)
            for lg in mesh.legends()] == [("S", True, None, None)]
    assert (plot.id, plot.kind, plot.name) == ("plot-2", "history", "h")

    again = snapshot(restored.session)
    assert again["version"] == 2
    assert again["panes"][0]["slots"]["contour"] == {
        "quantity": "S", "averaging": "averaged",
        "clim": None, "cmap": None,
    }


def test_a_v1_path_plot_is_dropped_with_a_notice():
    restored = restore_snapshot(_v1())

    assert [p.id for p in restored.session.panes] == ["mesh-1", "plot-2"]
    (notice,) = restored.notices
    assert "plot-3" in notice and "path" in notice and "A7.7" in notice


def test_a_v2_path_plot_is_a_schema_violation():
    with pytest.raises(ValueError, match="PlotView.kind"):
        restore_snapshot(_v1(version=2))


def test_a_v1_file_carrying_v2_style_fields_refuses():
    data = _v1(panes=[{"pane": "mesh", "id": "mesh-1", "slots": {
        "contour": {"quantity": "S", "cmap": "jet"}}}])
    with pytest.raises(SnapshotError, match="v2"):
        restore_snapshot(data)


@pytest.mark.parametrize("version", [1, 2])
def test_an_unknown_slot_category_still_refuses_in_both_versions(version):
    data = _v1(version=version, panes=[{
        "pane": "mesh", "id": "mesh-1",
        "slots": {"fiber_section": {"quantity": "S"}}}])
    with pytest.raises(SnapshotError, match="CLOSED"):
        restore_snapshot(data)


@pytest.mark.parametrize("version", [0, 3, "2", True, None])
def test_versions_outside_v1_v2_refuse(version):
    with pytest.raises(SnapshotError, match="this build reads v1 and v2"):
        restore_snapshot(_v1(version=version, panes=[]))


def test_a_bad_cmap_in_a_v2_file_refuses():
    data = _v1(version=2, panes=[{"pane": "mesh", "id": "mesh-1", "slots": {
        "contour": {"quantity": "S", "cmap": "nope"}}}])
    with pytest.raises(ValueError, match="not a colormap"):
        restore_snapshot(data)


def test_a_conflicting_v2_file_refuses():
    data = _v1(version=2, panes=[{"pane": "mesh", "id": "mesh-1", "slots": {
        "contour": {"quantity": "S", "clim": [0, 1]},
        "gauss": {"quantity": "S", "clim": [0, 2]}}}])
    with pytest.raises(ValueError, match="One field is one scale"):
        restore_snapshot(data)


def test_catalog_is_unchanged_by_a7_narrow():
    # KEEP ruling: A7 widens style, not the catalog.
    assert list(SLOT_CATALOG) == [
        "contour", "vector", "gauss", "line", "sand", "loads", "reactions",
    ]
