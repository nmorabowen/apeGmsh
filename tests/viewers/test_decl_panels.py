"""LoadsPanel / MassesPanel — callback-wiring (headless Qt).

Pure-UI panels: capture target + form params and fire one callback;
model_viewer owns the library call + dim validation. These tests
exercise that contract without VTK or a gmsh model.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# LoadsPanel
# =====================================================================


def test_loads_has_all_seven_types(qapp):
    from apeGmsh.viewers.ui._loads_panel import LoadsPanel, LOAD_TYPES

    p = LoadsPanel(
        get_target=lambda: None, get_patterns=lambda: [],
        on_apply=lambda *a: None, on_remove=lambda k: None,
        list_records=lambda: [],
    )
    assert len(LOAD_TYPES) == 7
    assert p._stack.count() == 7
    assert dict(LOAD_TYPES)["surface"] == 2
    assert dict(LOAD_TYPES)["gravity"] == 3


def test_loads_declare_requires_target(qapp):
    from apeGmsh.viewers.ui._loads_panel import LoadsPanel

    rec: list = []
    p = LoadsPanel(
        get_target=lambda: None, get_patterns=lambda: [],
        on_apply=lambda *a: rec.append(a),
        on_remove=lambda k: None, list_records=lambda: [],
    )
    p._combo_type.setCurrentText("gravity")
    p._declare()                       # no target set
    assert rec == []                   # callback not fired
    assert "target" in p._hint.text().lower()


def test_loads_gravity_payload(qapp):
    from apeGmsh.viewers.ui._loads_panel import LoadsPanel

    rec: list = []
    p = LoadsPanel(
        get_target=lambda: ("group", "Body"),
        get_patterns=lambda: ["dead"],
        on_apply=lambda *a: rec.append(a),
        on_remove=lambda k: None, list_records=lambda: [],
    )
    p._combo_type.setCurrentText("gravity")
    p._set_target()
    p._fields["gravity"]["density"].setValue(2500.0)
    p._fields["gravity"]["g"]["z"].setValue(-9.81)
    p._declare()
    op, pat, tgt, params = rec[-1]
    assert op == "gravity"
    assert tgt == ("group", "Body")
    assert params["density"] == pytest.approx(2500.0)
    assert params["g"] == [pytest.approx(0.0), pytest.approx(0.0),
                           pytest.approx(-9.81)]
    # blank Advanced text fields are omitted (library default holds)
    assert "reduction" not in params and "name" not in params


def test_loads_list_grouped_by_pattern_and_remove(qapp):
    from apeGmsh.viewers.ui._loads_panel import LoadsPanel

    removed: list = []
    recs = [
        {"key": 1, "pattern": "dead", "type": "gravity",
         "target": "Body", "name": None, "params": {}},
        {"key": 2, "pattern": "live", "type": "surface",
         "target": "Top", "name": "p", "params": {}},
    ]
    p = LoadsPanel(
        get_target=lambda: None, get_patterns=lambda: [],
        on_apply=lambda *a: None,
        on_remove=lambda k: removed.append(k),
        list_records=lambda: recs,
    )
    assert p._tree.topLevelItemCount() == 2          # dead + live
    dead = p._tree.topLevelItem(0)
    p._tree.setCurrentItem(dead.child(0))
    p._remove()
    assert removed == [1]


# =====================================================================
# MassesPanel
# =====================================================================


def test_masses_types_and_volume_payload(qapp):
    from apeGmsh.viewers.ui._masses_panel import MassesPanel, MASS_TYPES

    rec: list = []
    p = MassesPanel(
        get_target=lambda: ("label", "shaft"),
        on_apply=lambda *a: rec.append(a),
        on_remove=lambda k: None, list_records=lambda: [],
    )
    assert [t for t, _ in MASS_TYPES] == [
        "point", "line", "surface", "volume"
    ]
    assert dict(MASS_TYPES)["volume"] == 3
    p._combo_type.setCurrentText("volume")
    p._set_target()
    p._fields["volume"]["density"].setValue(7850.0)
    p._declare()
    op, tgt, params = rec[-1]
    assert op == "volume"
    assert tgt == ("label", "shaft")
    assert params["density"] == pytest.approx(7850.0)
    assert params["derive_rotational"] is False
