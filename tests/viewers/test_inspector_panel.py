"""Unit tests for :class:`InspectorPanel` (ADR 0088 D2).

The Inspector is the outline-selection-driven context host that
replaced the Diagram / Geometry / Details tab spine. These tests
exercise it in isolation with plain QWidgets standing in for the
hosted panels — no director, no VTK.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.ui._inspector_panel import InspectorPanel


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _label(text):
    from qtpy import QtWidgets
    return QtWidgets.QLabel(text)


@pytest.fixture
def hosted(qapp):
    """(diagram, color, geometry, details) stand-in widgets."""
    return _label("d"), _label("c"), _label("g"), _label("x")


def _make(hosted, **kw):
    d, c, g, x = hosted
    return InspectorPanel(d, c, g, x, **kw)


# =====================================================================
# Boot state (acceptance criterion 2)
# =====================================================================


def test_boots_in_empty_context(hosted):
    panel = _make(hosted)
    assert panel.current_context == "empty"


def test_empty_context_starter_buttons_hidden_without_labels(hosted):
    """INV-2 — no dead controls: buttons only exist once labels are set."""
    panel = _make(hosted)
    assert not panel._btn_primary.isVisibleTo(panel.widget)
    assert not panel._btn_secondary.isVisibleTo(panel.widget)


def test_set_starter_labels_shows_buttons_and_empty_hides(hosted):
    panel = _make(hosted)
    panel.set_starter_labels("Add contour", "")
    assert panel._btn_primary.isVisibleTo(panel.widget)
    assert panel._btn_primary.text() == "Add contour"
    assert not panel._btn_secondary.isVisibleTo(panel.widget)


def test_starter_buttons_fire_callbacks(hosted):
    fired = []
    panel = _make(
        hosted,
        on_primary=lambda: fired.append("p"),
        on_secondary=lambda: fired.append("s"),
    )
    panel.set_starter_labels("A", "B")
    panel._btn_primary.click()
    panel._btn_secondary.click()
    assert fired == ["p", "s"]


# =====================================================================
# Context switching (acceptance criterion 3)
# =====================================================================


def test_show_geometry_switches_context(hosted):
    panel = _make(hosted)
    panel.show_geometry()
    assert panel.current_context == "geometry"


def test_show_diagram_switches_context(hosted):
    panel = _make(hosted)
    panel.show_diagram()
    assert panel.current_context == "diagram"


def test_show_details_switches_context(hosted):
    panel = _make(hosted)
    panel.show_details()
    assert panel.current_context == "details"


def test_show_empty_returns_to_empty(hosted):
    panel = _make(hosted)
    panel.show_diagram()
    panel.show_empty()
    assert panel.current_context == "empty"


def test_contexts_do_not_share_pages(hosted):
    """No context shows another kind's controls — each hosted widget
    lives on exactly one stacked page."""
    d, c, g, x = hosted
    panel = InspectorPanel(d, c, g, x)
    panel.show_diagram()
    page = panel._stack.currentWidget()
    assert page.isAncestorOf(d)
    assert page.isAncestorOf(c)     # Color section rides the diagram page
    assert not page.isAncestorOf(g)
    assert not page.isAncestorOf(x)
    panel.show_geometry()
    page = panel._stack.currentWidget()
    assert page.isAncestorOf(g)
    assert not page.isAncestorOf(d)


# =====================================================================
# Stage context
# =====================================================================


def test_show_stage_renders_provider_rows(hosted):
    rows = [("Stage", "grav"), ("Kind", "static"), ("Steps", "3")]
    panel = _make(hosted, stage_info_provider=lambda sid: rows)
    panel.show_stage("s1")
    assert panel.current_context == "stage"
    assert panel._stage_form.rowCount() == 3
    assert not panel._stage_hint.isVisibleTo(panel.widget)


def test_show_stage_without_provider_shows_hint(hosted):
    panel = _make(hosted)
    panel.show_stage("s1")
    assert panel._stage_form.rowCount() == 0
    assert panel._stage_hint.isVisibleTo(panel.widget)


def test_show_stage_rebuilds_rows_on_reselect(hosted):
    calls = {"n": 0}

    def provider(sid):
        calls["n"] += 1
        return [("Stage", sid)]

    panel = _make(hosted, stage_info_provider=provider)
    panel.show_stage("a")
    panel.show_stage("b")
    assert calls["n"] == 2
    assert panel._stage_form.rowCount() == 1


def test_show_stage_provider_failure_is_hint_not_raise(hosted):
    def provider(sid):
        raise RuntimeError("boom")

    panel = _make(hosted, stage_info_provider=provider)
    panel.show_stage("s1")    # must not raise
    assert panel._stage_hint.isVisibleTo(panel.widget)


# =====================================================================
# Plot context
# =====================================================================


def test_show_plot_sets_label_and_context(hosted):
    panel = _make(hosted)
    panel.show_plot(("hist", 1), "Node 4 — displacement_x")
    assert panel.current_context == "plot"
    assert "Node 4" in panel._plot_label.text()


def test_show_in_plots_fires_with_key(hosted):
    got = []
    panel = _make(hosted, on_show_plot=got.append)
    panel.show_plot(("hist", 7), "H")
    panel._btn_show_plot.click()
    assert got == [("hist", 7)]
