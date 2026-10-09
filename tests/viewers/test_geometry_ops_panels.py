"""BooleanPanel / TransformPanel — callback-wiring (headless Qt).

The panels are pure UI: they capture operands + params from the
form and fire one callback; ``model_viewer`` owns the library call.
These tests exercise that contract without VTK or a gmsh model.
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


@pytest.fixture(autouse=True)
def _disposes_its_widgets(request):
    """Each panel's lambdas keep it alive; delete it before exit does."""
    from tests.viewers._qt_teardown import disposing_new_top_levels

    yield from disposing_new_top_levels(request.node.nodeid)


# =====================================================================
# BooleanPanel
# =====================================================================


def test_boolean_capture_and_apply(qapp):
    from apeGmsh.viewers.ui._boolean_panel import BooleanPanel

    rec: list = []
    p = BooleanPanel(
        get_selection=lambda: [(3, 1), (3, 2)],
        on_apply=lambda *a: rec.append(a),
    )
    p._capture("_objects")          # Set Objects from selection
    p._apply("fuse")
    op, objs, tools, opts = rec[-1]
    assert op == "fuse"
    assert objs == [(3, 1), (3, 2)]
    assert tools == []              # tools slot left empty
    assert set(opts) == {
        "label", "remove_object", "remove_tool", "cleanup_free"
    }
    assert opts["remove_object"] is True and opts["remove_tool"] is True


def test_boolean_clear_operands(qapp):
    from apeGmsh.viewers.ui._boolean_panel import BooleanPanel

    p = BooleanPanel(get_selection=lambda: [(3, 9)], on_apply=lambda *a: None)
    p._capture("_objects")
    p._capture("_tools")
    assert p._objects and p._tools
    p.clear_operands()
    assert p._objects == [] and p._tools == []


def test_boolean_label_passed_through(qapp):
    from apeGmsh.viewers.ui._boolean_panel import BooleanPanel

    rec: list = []
    p = BooleanPanel(
        get_selection=lambda: [(2, 4)],
        on_apply=lambda *a: rec.append(a),
    )
    p._ed_label.setText("  weld  ")
    p._capture("_objects")
    p._apply("cut")
    assert rec[-1][0] == "cut"
    assert rec[-1][3]["label"] == "weld"   # trimmed


# =====================================================================
# TransformPanel
# =====================================================================


def test_transform_has_all_ops(qapp):
    from apeGmsh.viewers.ui._transform_panel import TransformPanel, _OPS

    p = TransformPanel(get_selection=lambda: [], on_apply=lambda *a: None)
    assert p._combo.count() == len(_OPS) == 9
    assert p._stack.count() == len(_OPS)


def test_transform_translate_payload(qapp):
    from apeGmsh.viewers.ui._transform_panel import TransformPanel

    rec: list = []
    p = TransformPanel(
        get_selection=lambda: [(3, 1)],
        on_apply=lambda *a: rec.append(a),
    )
    p._combo.setCurrentIndex(0)                 # translate
    p._fields["translate"]["dx"].setValue(2.5)
    p._fields["translate"]["dz"].setValue(-1.0)
    p._apply()
    op, params, dup = rec[-1]
    assert op == "translate"
    assert params["dx"] == pytest.approx(2.5)
    assert params["dz"] == pytest.approx(-1.0)
    assert dup is False


def test_transform_duplicate_only_for_inplace(qapp):
    from apeGmsh.viewers.ui._transform_panel import TransformPanel, _OPS

    p = TransformPanel(get_selection=lambda: [], on_apply=lambda *a: None)
    p._combo.setCurrentIndex(_OPS.index("rotate"))     # in-place
    assert p._cb_dup.isEnabled() is True
    p._combo.setCurrentIndex(_OPS.index("extrude"))    # generative
    assert p._cb_dup.isEnabled() is False
    assert p._cb_dup.isChecked() is False


def test_transform_sweep_captures_path_curves_only(qapp):
    from apeGmsh.viewers.ui._transform_panel import TransformPanel, _OPS

    rec: list = []
    # selection mixes a surface (dim 2) and curves (dim 1); only the
    # curves should be captured for the sweep path.
    p = TransformPanel(
        get_selection=lambda: [(2, 7), (1, 5), (1, 6)],
        on_apply=lambda *a: rec.append(a),
    )
    p._combo.setCurrentIndex(_OPS.index("sweep"))
    p._capture_path()
    p._apply()
    op, params, _ = rec[-1]
    assert op == "sweep"
    assert params["path_curves"] == [5, 6]
    assert params["trihedron"] == "DiscreteTrihedron"
    p.reset_captures()
    assert p._path_curves == []


def test_transform_thru_sections_accumulate(qapp):
    from apeGmsh.viewers.ui._transform_panel import TransformPanel, _OPS

    rec: list = []
    sel = [(1, 1), (1, 2)]
    p = TransformPanel(
        get_selection=lambda: sel,
        on_apply=lambda *a: rec.append(a),
    )
    p._combo.setCurrentIndex(_OPS.index("thru_sections"))
    p._add_section()
    sel = [(1, 3), (1, 4)]
    p._add_section()
    p._apply()
    op, params, _ = rec[-1]
    assert op == "thru_sections"
    assert params["sections"] == [[1, 2], [3, 4]]
    assert params["make_solid"] is True
    p.reset_captures()
    assert p._sections == []
