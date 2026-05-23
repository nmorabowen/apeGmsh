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


# =====================================================================
# _decl_record_view — LoadDef/MassDef → outline-row projection
# =====================================================================
#
# Regression for the three-broker (ADR 0020 / Phase 8) field rename.
# Pre-refactor model_viewer._rec_view read removed ``r.pg`` /
# ``r.label`` attributes; the outline always saw
# ``target_tuple = None`` and the edit-from-outline flow lost the
# group/label discrimination.  LoadDef and MassDef now expose
# ``target`` + ``target_source`` only (see
# ``apeGmsh/_kernel/defs/loads.py:21,26``).


def test_decl_record_view_pg_target_maps_to_group_tuple():
    from apeGmsh._kernel.defs.loads import PointLoadDef
    from apeGmsh.viewers.model_viewer import _decl_record_view

    r = PointLoadDef(
        target="Diaphragm", target_source="pg",
        pattern="dead", force_xyz=(0.0, 0.0, -100.0),
    )
    out = _decl_record_view(r, with_pattern=True)

    assert out["target_tuple"] == ("group", "Diaphragm")
    assert out["target"] == "Diaphragm"
    assert out["pattern"] == "dead"
    assert out["type"] == "point"
    assert out["params"]["force_xyz"] == (0.0, 0.0, -100.0)


def test_decl_record_view_label_target_maps_to_label_tuple():
    from apeGmsh._kernel.defs.masses import PointMassDef
    from apeGmsh.viewers.model_viewer import _decl_record_view

    r = PointMassDef(
        target="lumped_top", target_source="label",
        mass=50.0,
    )
    out = _decl_record_view(r, with_pattern=False)

    assert out["target_tuple"] == ("label", "lumped_top")
    assert out["target"] == "lumped_top"
    assert "pattern" not in out
    assert out["type"] == "point"


def test_decl_record_view_auto_target_yields_none_tuple():
    """target_source='auto' (e.g. user passed a Part / DimTag list)
    must not be coerced into a group/label tuple — the outline shows
    the bare target string and the edit flow leaves the type picker
    on its current value."""
    from apeGmsh._kernel.defs.loads import GravityLoadDef
    from apeGmsh.viewers.model_viewer import _decl_record_view

    r = GravityLoadDef(
        target="Body", target_source="auto",
        g=(0.0, 0.0, -9.81), density=2500.0,
    )
    out = _decl_record_view(r, with_pattern=True)

    assert out["target_tuple"] is None
    assert out["target"] == "Body"
    assert out["type"] == "gravity"
