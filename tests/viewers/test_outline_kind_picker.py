"""Inline 2×4 kind picker — toggle visibility + dialog handoff (B++ §4.1, §8)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def stub_director():
    d = MagicMock()
    d.stages.return_value = []
    d.stage_id = None
    d.registry.diagrams.return_value = []
    d.subscribe_stage.return_value = lambda: None
    d.subscribe_diagrams.return_value = lambda: None
    return d


@pytest.fixture
def tree(qapp, stub_director):
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    return OutlineTree(stub_director)


def test_picker_hidden_by_default(tree):
    assert not tree._kind_picker.isVisible()


def test_picker_shows_when_insert_toggled(tree):
    tree._widget.show()  # make ancestors visible so child show takes
    tree._btn_insert.setChecked(True)
    assert tree._kind_picker.isVisible()


def test_picker_hides_when_insert_untoggled(tree):
    tree._widget.show()
    tree._btn_insert.setChecked(True)
    tree._btn_insert.setChecked(False)
    assert not tree._kind_picker.isVisible()


def test_picker_grid_holds_eight_kinds(tree):
    """Phase 1 ships 8 diagram kinds — each should have a button."""
    from apeGmsh.viewers.ui._add_diagram_dialog import kinds_available
    expected = len(kinds_available())
    grid = tree._kind_picker.layout()
    assert grid.count() == expected
    assert expected == 8


def test_kind_button_opens_dialog_with_initial_kind(tree):
    """Clicking a kind button invokes AddDiagramDialog with the
    matching ``initial_kind`` so the user lands on the right form."""
    from apeGmsh.viewers.ui import _outline_tree
    captured = {}

    class FakeDialog:
        def __init__(self, director, *, parent=None, initial_kind=None):
            captured["initial_kind"] = initial_kind
        def run(self):
            return False

    with patch.object(_outline_tree, "_qt") as mock_qt:
        # Real qt is fine — the patch just keeps existing imports in
        # OutlineTree intact. We only need to override the dialog
        # import, which happens lazily inside _on_kind_chosen.
        from qtpy import QtWidgets, QtCore
        mock_qt.return_value = (QtWidgets, QtCore)
        with patch(
            "apeGmsh.viewers.ui._add_diagram_dialog.AddDiagramDialog",
            FakeDialog,
        ):
            tree._on_kind_chosen("contour")

    assert captured["initial_kind"] == "contour"


def test_choosing_kind_collapses_picker(tree):
    tree._widget.show()
    tree._btn_insert.setChecked(True)
    assert tree._kind_picker.isVisible()
    with patch(
        "apeGmsh.viewers.ui._add_diagram_dialog.AddDiagramDialog",
    ) as fake:
        fake.return_value.run.return_value = False
        tree._on_kind_chosen("deformed_shape")
    assert not tree._btn_insert.isChecked()
    assert not tree._kind_picker.isVisible()
