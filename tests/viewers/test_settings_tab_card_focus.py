"""Tests for plan 04 step 2 cont. — layer-card focus callback.

When the user clicks anywhere inside a layer card, the settings tab
broadcasts the underlying ``Diagram`` via the registered callback.
The owner (``ResultsViewer``) wires this to ``ActiveObjects
.set_active_layer`` so the Color Map Editor follows card-level
navigation.

Uses the same stub-director scaffolding as ``test_settings_tab_auto_apply.py``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.ui._diagram_settings_tab import (
    DiagramSettingsTab,
    _resolve_card_focus_filter_class,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _StubGeometries:
    def subscribe(self, _cb):
        return lambda: None


class _Compositions:
    @property
    def active(self):
        return None


class _StubDirector:
    def __init__(self):
        self.geometries = _StubGeometries()
        self.compositions = _Compositions()
        self.dispatcher = None

    def subscribe_diagrams(self, _cb):
        return lambda: None


@pytest.fixture
def director():
    return _StubDirector()


@pytest.fixture
def tab(qapp, director):
    return DiagramSettingsTab(director)


# =====================================================================
# Callback registration + manual fire
# =====================================================================


def test_default_callback_is_none(tab):
    assert tab._layer_focus_callback is None


def test_on_layer_focused_registers_callback(tab):
    seen: list = []
    tab.on_layer_focused(lambda d: seen.append(d))
    assert tab._layer_focus_callback is not None
    tab._fire_layer_focused("synthetic_diagram")
    assert seen == ["synthetic_diagram"]


def test_on_layer_focused_none_clears_callback(tab):
    tab.on_layer_focused(lambda d: None)
    tab.on_layer_focused(None)
    assert tab._layer_focus_callback is None
    # Firing with no callback is a no-op (doesn't raise).
    tab._fire_layer_focused("anything")


def test_fire_with_no_callback_is_noop(tab):
    # No callback registered; firing must not raise.
    tab._fire_layer_focused("anything")


def test_callback_exception_swallowed(tab):
    """A bad callback must not propagate into the Qt event loop."""
    def _boom(_d):
        raise RuntimeError("synthetic")
    tab.on_layer_focused(_boom)
    # Should not raise.
    tab._fire_layer_focused("d")


# =====================================================================
# Focus filter end-to-end via synthetic widget tree
# =====================================================================


def test_focus_filter_installs_on_card_and_descendants(qapp, tab):
    from qtpy import QtWidgets

    seen: list = []
    tab.on_layer_focused(lambda d: seen.append(d))
    card = QtWidgets.QGroupBox()
    inner = QtWidgets.QPushButton("apply", card)
    QtWidgets.QVBoxLayout(card).addWidget(inner)
    tab._install_card_focus_filter(card, "fake_diagram_a")

    # One filter object retained.
    assert len(tab._card_focus_filters) == 1


def test_focus_filter_fires_on_mouse_press_in_card(qapp, tab):
    """Simulate a mouse-button press on the card itself."""
    from qtpy import QtWidgets, QtCore, QtGui

    seen: list = []
    tab.on_layer_focused(lambda d: seen.append(d))
    card = QtWidgets.QGroupBox()
    tab._install_card_focus_filter(card, "fake_diagram_a")

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(5, 5),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(card, press)
    assert seen == ["fake_diagram_a"]


def test_focus_filter_fires_on_press_in_descendant(qapp, tab):
    """Mouse press on a child widget inside the card also fires."""
    from qtpy import QtWidgets, QtCore, QtGui

    seen: list = []
    tab.on_layer_focused(lambda d: seen.append(d))
    card = QtWidgets.QGroupBox()
    inner = QtWidgets.QPushButton("apply", card)
    QtWidgets.QVBoxLayout(card).addWidget(inner)
    tab._install_card_focus_filter(card, "fake_diagram_b")

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(2, 2),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(inner, press)
    assert seen == ["fake_diagram_b"]


def test_focus_filter_does_not_consume_event(qapp, tab):
    """The filter must return False so widgets still get their event.

    Test by hand-invoking eventFilter; the return value matters.
    """
    from qtpy import QtWidgets, QtCore, QtGui

    cls = _resolve_card_focus_filter_class()
    tab.on_layer_focused(lambda _d: None)
    filt = cls(tab, "fake")
    btn = QtWidgets.QPushButton()
    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(1, 1),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    # eventFilter is the underlying method — returns False (don't consume).
    assert filt.eventFilter(btn, press) is False


def test_rebuild_clears_focus_filters(qapp, tab):
    """A fresh _rebuild drops every prior card's filter so they don't
    accumulate across renders."""
    from qtpy import QtWidgets
    card = QtWidgets.QGroupBox()
    tab._install_card_focus_filter(card, "d")
    assert len(tab._card_focus_filters) == 1
    tab._rebuild()
    assert tab._card_focus_filters == []
