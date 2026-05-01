"""Tree ↔ plot tab two-way binding (B++ §7).

Exercises ``OutlineTree.bind_plot_pane`` against a real ``PlotPane``
and a stub director — no Qt event loop needed since signals are
fired by direct method calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

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
    # subscribe_* return dummy unsubscribe callables.
    d.subscribe_stage.return_value = lambda: None
    d.subscribe_diagrams.return_value = lambda: None
    return d


@pytest.fixture
def tree_and_pane(qapp, stub_director):
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    from apeGmsh.viewers.ui._plot_pane import PlotPane
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    tree = OutlineTree(stub_director)
    pane = PlotPane()
    tree.bind_plot_pane(pane)
    return tree, pane, QtWidgets


def _plots_children(tree):
    g = tree._group_plots
    return [g.child(i) for i in range(g.childCount())]


def _plot_keys_in_tree(tree):
    from apeGmsh.viewers.ui._outline_tree import _ROLE_PLOT_KEY
    return [
        c.data(0, _ROLE_PLOT_KEY)
        for c in _plots_children(tree)
        if c.data(0, _ROLE_PLOT_KEY) is not None
    ]


# =====================================================================
# Initial state — empty plot pane → placeholder row
# =====================================================================

def test_empty_plot_pane_shows_placeholder(tree_and_pane):
    tree, _, _ = tree_and_pane
    children = _plots_children(tree)
    assert len(children) == 1
    assert "shift-click" in children[0].text(0).lower()


# =====================================================================
# Adding a tab populates the Plots group
# =====================================================================

def test_history_tab_appears_in_plots_group(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("history", 412, "displacement_z"), "u(t) · N412 · uz",
                 QtWidgets.QLabel("plot"), closable=True)
    keys = _plot_keys_in_tree(tree)
    assert keys == [("history", 412, "displacement_z")]


def test_history_tab_label_matches_pane_label(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("history", 1, "ux"), "u(t) · N1 · ux",
                 QtWidgets.QLabel("p"), closable=True)
    children = _plots_children(tree)
    assert children[0].text(0) == "u(t) · N1 · ux"


# =====================================================================
# Diagram side-panel tabs are filtered out of the Plots group
# =====================================================================

def test_diagram_side_panel_does_not_appear_in_plots(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("diagram", 999), "Fiber · B-1",
                 QtWidgets.QLabel("p"), closable=False)
    pane.add_tab(("history", 1, "uz"), "u(t) · N1",
                 QtWidgets.QLabel("p"), closable=True)
    keys = _plot_keys_in_tree(tree)
    assert keys == [("history", 1, "uz")]


# =====================================================================
# Removing a tab updates the tree
# =====================================================================

def test_removing_tab_repopulates_plots_group(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("history", 1, "uz"), "u(t) · N1",
                 QtWidgets.QLabel("p"), closable=True)
    pane.add_tab(("history", 2, "uz"), "u(t) · N2",
                 QtWidgets.QLabel("p"), closable=True)
    pane.remove_tab(("history", 1, "uz"))
    keys = _plot_keys_in_tree(tree)
    assert keys == [("history", 2, "uz")]


def test_last_tab_removed_falls_back_to_placeholder(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    key = ("history", 1, "uz")
    pane.add_tab(key, "u(t)", QtWidgets.QLabel("p"), closable=True)
    pane.remove_tab(key)
    children = _plots_children(tree)
    assert len(children) == 1
    assert _plot_keys_in_tree(tree) == []


# =====================================================================
# Active indicator: pane → tree
# =====================================================================

def test_active_tab_in_pane_marks_tree_row_bold(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("history", 1, "uz"), "A", QtWidgets.QLabel("a"), closable=True)
    pane.add_tab(("history", 2, "uz"), "B", QtWidgets.QLabel("b"), closable=True)
    pane.set_active(("history", 1, "uz"))
    children = _plots_children(tree)
    fonts = [c.font(0).bold() for c in children]
    assert sum(fonts) == 1   # exactly one row marked active


# =====================================================================
# Active indicator: tree click → pane.set_active
# =====================================================================

def test_clicking_plot_row_activates_tab(tree_and_pane):
    tree, pane, QtWidgets = tree_and_pane
    pane.add_tab(("history", 1, "uz"), "A", QtWidgets.QLabel("a"), closable=True)
    pane.add_tab(("history", 2, "uz"), "B", QtWidgets.QLabel("b"), closable=True)
    # B is active by virtue of being added last.
    assert pane.active_key() == ("history", 2, "uz")
    # Click the first row (key = N1).
    children = _plots_children(tree)
    target = next(
        c for c in children
        if c.data(0, 0x103) == ("history", 1, "uz")
    )
    tree._on_item_clicked(target, 0)
    assert pane.active_key() == ("history", 1, "uz")
