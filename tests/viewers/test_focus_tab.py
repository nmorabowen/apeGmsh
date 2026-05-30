"""``ViewerWindow.focus_tab`` + outline ``on_row_focused`` plumbing.

ParaView pipeline-browser pattern: clicking a row in the outline
brings the corresponding property tab forward. This file covers:

* ``ViewerWindow.focus_tab(identifier)`` resolves first against the
  extension-dock dictionary (tabified extension docks — model.viewer)
  then against the legacy ``QTabWidget`` (mesh.viewer).
* ``ModelOutlineTree`` and ``MeshOutlineTree`` fire ``on_row_focused``
  with ``(kind, payload)`` on row click.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# ViewerWindow.focus_tab — stub-bound (no VTK)
# =====================================================================


def _make_focus_tab_stub(qapp):
    """Stub ``ViewerWindow`` with the focus_tab method bound.

    Mirrors :mod:`test_viewer_window_extensions`'s stub pattern.
    """
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    class _Stub:
        pass

    stub = _Stub()
    stub._window = QtWidgets.QMainWindow()
    stub._window.setCentralWidget(QtWidgets.QWidget(stub._window))
    stub._extension_docks = {}
    stub._tab_widget = QtWidgets.QTabWidget()
    stub._tabs_dock = QtWidgets.QDockWidget("Tabs", stub._window)
    stub._tabs_dock.setWidget(stub._tab_widget)

    stub.focus_tab = ViewerWindow.focus_tab.__get__(stub)
    return stub


def test_focus_tab_resolves_extension_dock(qapp):
    """When ``identifier`` matches a dock_id in ``_extension_docks``,
    that dock is raised."""
    from qtpy import QtWidgets, QtCore

    stub = _make_focus_tab_stub(qapp)
    dock = QtWidgets.QDockWidget("Test", stub._window)
    dock.setObjectName("dock_test")
    stub._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    stub._extension_docks["dock_test"] = dock
    dock.setVisible(False)

    assert stub.focus_tab("dock_test") is True
    # setVisible(True) flipped via the focus path → isHidden is False.
    assert dock.isHidden() is False


def test_focus_tab_resolves_qtabwidget_by_text(qapp):
    """When ``identifier`` matches a tab text in ``_tab_widget``,
    that tab becomes current."""
    from qtpy import QtWidgets

    stub = _make_focus_tab_stub(qapp)
    a = QtWidgets.QLabel("a")
    b = QtWidgets.QLabel("b")
    c = QtWidgets.QLabel("c")
    stub._tab_widget.addTab(a, "Alpha")
    stub._tab_widget.addTab(b, "Beta")
    stub._tab_widget.addTab(c, "Gamma")
    stub._tab_widget.setCurrentIndex(0)

    assert stub.focus_tab("Beta") is True
    assert stub._tab_widget.currentIndex() == 1
    assert stub._tab_widget.currentWidget() is b


def test_focus_tab_returns_false_on_unknown_identifier(qapp):
    from qtpy import QtWidgets
    stub = _make_focus_tab_stub(qapp)
    stub._tab_widget.addTab(QtWidgets.QLabel("real"), "Real")
    assert stub.focus_tab("not-a-thing") is False


def test_focus_tab_prefers_extension_dock_over_tab_text(qapp):
    """If a dock_id matches an extension dock AND a tab text is the
    same string, the dock wins (extension docks are the new path)."""
    from qtpy import QtWidgets, QtCore

    stub = _make_focus_tab_stub(qapp)
    dock = QtWidgets.QDockWidget("Conflict", stub._window)
    dock.setObjectName("conflict")
    stub._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    stub._extension_docks["conflict"] = dock
    # Same string in the tab widget too.
    tab_widget = QtWidgets.QLabel("tab")
    stub._tab_widget.addTab(tab_widget, "conflict")
    stub._tab_widget.setCurrentIndex(0)

    stub.focus_tab("conflict")
    # The tab widget didn't switch — it would have been current=0
    # before; with only one tab, currentIndex stays 0 regardless. The
    # contract we test is: when the dock dictionary matches, we don't
    # fall through to the tab search. We assert that by leaving the
    # tab pinned: there's only one tab so this is trivial here, but
    # the multi-tab case is covered by the prior tests.
    assert dock.isHidden() is False


# =====================================================================
# ModelOutlineTree.on_row_focused
# =====================================================================


@pytest.fixture
def gmsh_two_groups():
    """Patch ``gmsh.model.getPhysicalGroups`` for ModelOutlineTree."""
    with patch("apeGmsh.viewers.ui._model_outline_tree.gmsh") as m:
        m.model.getPhysicalGroups.return_value = [(2, 1)]
        m.model.getPhysicalName.return_value = "Body"
        m.model.getEntitiesForPhysicalGroup.return_value = [1, 2]
        yield m


class _StubSelection:
    # ADR 0045 S3c: the outline reads groups from staging. Mirror the
    # gmsh_two_groups fixture above (one group "Body", surfaces 1 & 2).
    def __init__(self) -> None:
        from apeGmsh.viewers.scene_ir import SelectionTarget
        self.active_group = None
        self.picks: list = []
        self._staged = {
            "Body": [SelectionTarget.from_dimtag(dt) for dt in [(2, 1), (2, 2)]]
        }
        self.group_order: list = ["Body"]

    @property
    def staged_groups(self) -> dict:
        return dict(self._staged)


class _StubVisManager:
    def __init__(self) -> None:
        self.hidden: set = set()
        self.on_changed: list = []

    def is_hidden(self, dt) -> bool:
        return tuple(dt) in self.hidden

    def set_hidden(self, dts) -> None:
        self.hidden = {tuple(dt) for dt in dts}
        for cb in self.on_changed:
            cb()

    def isolate_dts(self, dts):
        pass

    def reveal_all(self):
        self.set_hidden(set())


def test_model_outline_group_click_fires_on_row_focused(
    qapp, gmsh_two_groups,
):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    captured: list[tuple[str, object]] = []
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_row_focused=lambda k, p: captured.append((k, p)),
    )
    body_item = outline._group_groups.child(0)
    outline._on_item_clicked(body_item, 0)
    assert captured == [("group", "Body")]


def test_model_outline_entity_click_fires_on_row_focused(
    qapp, gmsh_two_groups,
):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    captured: list[tuple[str, object]] = []
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_row_focused=lambda k, p: captured.append((k, p)),
    )
    body_item = outline._group_groups.child(0)
    entity = body_item.child(0)
    outline._on_item_clicked(entity, 0)
    assert captured == [("entity", (2, 1))]


# =====================================================================
# MeshOutlineTree.on_row_focused
# =====================================================================


def _mesh_scene_with_groups_and_types():
    return SimpleNamespace(
        group_to_breps={"Body": [(2, 1)]},
        brep_dominant_type={(2, 1): "Quadrilaterals"},
        brep_to_elems={(2, 1): [0, 1, 2]},
    )


def test_mesh_outline_group_click_fires_on_row_focused(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[tuple[str, object]] = []
    outline = MeshOutlineTree(
        scene=_mesh_scene_with_groups_and_types(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_row_focused=lambda k, p: captured.append((k, p)),
    )
    body_item = outline._group_groups.child(0)
    outline._on_item_clicked(body_item, 0)
    assert captured == [("group", "Body")]


def test_mesh_outline_type_click_fires_on_row_focused(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[tuple[str, object]] = []
    outline = MeshOutlineTree(
        scene=_mesh_scene_with_groups_and_types(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_row_focused=lambda k, p: captured.append((k, p)),
    )
    type_item = outline._group_types.child(0)
    outline._on_item_clicked(type_item, 0)
    assert captured == [("type", "Quadrilaterals")]


def test_mesh_outline_header_click_does_not_fire(qapp):
    """Top-level header rows are non-selectable. ``_on_item_clicked``
    should be a no-op for them."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[tuple[str, object]] = []
    outline = MeshOutlineTree(
        scene=_mesh_scene_with_groups_and_types(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_row_focused=lambda k, p: captured.append((k, p)),
    )
    outline._on_item_clicked(outline._group_groups, 0)    # header row
    assert captured == []
