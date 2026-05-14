"""Unit tests for :class:`ColorMapEditor` (plan 06 step 3).

Tests the dock widget in isolation against a bare ``LUT`` instance —
no diagram, no plotter, no VTK. The widget's contract is: when bound
to a LUT, user edits propagate to LUT setters; external LUT mutations
repopulate the widgets.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.core._lut_manager import LUT
from apeGmsh.viewers.ui._color_map_editor import (
    ColorMapEditor,
    make_color_map_editor_dock,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def editor(qapp):
    return ColorMapEditor()


@pytest.fixture
def lut(qapp):
    return LUT("stress_vm", preset="viridis", vmin=-10.0, vmax=10.0)


# =====================================================================
# Unbound state
# =====================================================================


def test_editor_constructs_unbound(editor):
    assert editor.widget is not None
    # All controls disabled until bound.
    assert not editor._preset_combo.isEnabled()
    assert not editor._vmin_spin.isEnabled()
    assert not editor._vmax_spin.isEnabled()
    assert not editor._log_cb.isEnabled()
    assert not editor._bar_cb.isEnabled()
    assert not editor._fit_btn.isEnabled()


def test_unbound_header_is_empty_state(editor):
    assert "No diagram" in editor._header.text()


def test_bind_layer_none_is_noop(editor):
    # Binding None when already unbound shouldn't raise.
    editor.bind_layer(None)
    assert editor._lut is None


# =====================================================================
# Bind via bind_lut (direct LUT — no diagram)
# =====================================================================


def test_bind_lut_populates_widgets(editor, lut):
    editor.bind_lut(lut)
    assert editor._preset_combo.currentText() == "viridis"
    assert editor._vmin_spin.value() == pytest.approx(-10.0)
    assert editor._vmax_spin.value() == pytest.approx(10.0)
    assert editor._log_cb.isChecked() is False
    assert editor._bar_cb.isChecked() is True


def test_bind_lut_enables_controls(editor, lut):
    editor.bind_lut(lut)
    assert editor._preset_combo.isEnabled()
    assert editor._vmin_spin.isEnabled()
    assert editor._vmax_spin.isEnabled()
    assert editor._log_cb.isEnabled()
    assert editor._bar_cb.isEnabled()


def test_bind_lut_disables_fit_button_when_no_diagram(editor, lut):
    """Fit-to-data needs a diagram with autofit_clim_at_current_step —
    pure-LUT binding can't do it."""
    editor.bind_lut(lut)
    assert not editor._fit_btn.isEnabled()


def test_bind_lut_updates_header(editor, lut):
    editor.bind_lut(lut)
    assert "stress_vm" in editor._header.text()


# =====================================================================
# User edits → LUT
# =====================================================================


def test_user_changes_preset_via_combo_updates_lut(editor, lut):
    editor.bind_lut(lut)
    editor._preset_combo.setCurrentText("plasma")
    assert lut.preset == "plasma"


def test_user_changes_vmin_via_spinbox_updates_lut(editor, lut):
    editor.bind_lut(lut)
    editor._vmin_spin.setValue(-50.0)
    assert lut.vmin == pytest.approx(-50.0)
    # vmax untouched.
    assert lut.vmax == pytest.approx(10.0)


def test_user_changes_vmax_via_spinbox_updates_lut(editor, lut):
    editor.bind_lut(lut)
    editor._vmax_spin.setValue(99.0)
    assert lut.vmax == pytest.approx(99.0)


def test_user_toggles_log_scale_updates_lut(editor, lut):
    editor.bind_lut(lut)
    editor._log_cb.setChecked(True)
    assert lut.log_scale is True


def test_user_toggles_show_scalar_bar_updates_lut(editor, lut):
    editor.bind_lut(lut)
    editor._bar_cb.setChecked(False)
    assert lut.show_scalar_bar is False


# =====================================================================
# External LUT mutation → widgets
# =====================================================================


def test_external_preset_change_refreshes_combo(editor, lut):
    editor.bind_lut(lut)
    lut.set_preset("turbo")
    assert editor._preset_combo.currentText() == "turbo"


def test_external_range_change_refreshes_spinboxes(editor, lut):
    editor.bind_lut(lut)
    lut.set_range(0.0, 1000.0)
    assert editor._vmin_spin.value() == pytest.approx(0.0)
    assert editor._vmax_spin.value() == pytest.approx(1000.0)


def test_external_log_scale_change_refreshes_checkbox(editor, lut):
    editor.bind_lut(lut)
    lut.set_log_scale(True)
    assert editor._log_cb.isChecked() is True


# =====================================================================
# Feedback-loop suppression
# =====================================================================


def test_user_edit_does_not_cause_double_emission(editor, lut):
    """Setting a value via the editor must fire LUT.changed exactly
    once — the editor's self-setting guard must not re-trigger the
    setter inside _refresh_from_lut."""
    editor.bind_lut(lut)
    emissions: list = []
    lut.changed.connect(lambda: emissions.append(None))
    editor._preset_combo.setCurrentText("magma")
    assert len(emissions) == 1


# =====================================================================
# Rebinding
# =====================================================================


def test_rebind_to_new_lut_disconnects_old(editor, qapp):
    a = LUT("a", preset="viridis", vmin=0.0, vmax=1.0)
    b = LUT("b", preset="plasma", vmin=10.0, vmax=20.0)
    editor.bind_lut(a)
    editor.bind_lut(b)
    # Editor now reflects b.
    assert editor._preset_combo.currentText() == "plasma"
    assert "b" in editor._header.text()
    # Mutating a should NOT touch the editor.
    a.set_preset("turbo")
    assert editor._preset_combo.currentText() == "plasma"
    # Mutating b should.
    b.set_preset("inferno")
    assert editor._preset_combo.currentText() == "inferno"


def test_rebind_to_none_disables_controls(editor, lut):
    editor.bind_lut(lut)
    assert editor._preset_combo.isEnabled()
    editor.bind_lut(None)
    assert not editor._preset_combo.isEnabled()
    assert "No diagram" in editor._header.text()


# =====================================================================
# bind_layer adapter (diagram.lut)
# =====================================================================


class _FakeDiagram:
    """Stand-in for a Diagram exposing .lut + autofit_clim_at_current_step."""
    def __init__(self, lut):
        self.lut = lut
        self.autofit_called = False
        self.show_bar_called: bool | None = None
    def autofit_clim_at_current_step(self):
        self.autofit_called = True
        # Real implementation calls set_clim → routes through LUT.
        self.lut.set_range(-99.0, 99.0)
    def set_show_scalar_bar(self, on):
        self.show_bar_called = on


def test_bind_layer_reads_lut_from_diagram(editor, lut):
    diagram = _FakeDiagram(lut)
    editor.bind_layer(diagram)
    assert editor._lut is lut
    assert "stress_vm" in editor._header.text()


def test_bind_layer_with_autofit_capable_enables_fit_btn(editor, lut):
    diagram = _FakeDiagram(lut)
    editor.bind_layer(diagram)
    assert editor._fit_btn.isEnabled()


def test_fit_button_calls_diagram_autofit(editor, lut):
    diagram = _FakeDiagram(lut)
    editor.bind_layer(diagram)
    editor._fit_btn.click()
    assert diagram.autofit_called is True
    # Spinboxes should pick up the new range via LUT.changed.
    assert editor._vmin_spin.value() == pytest.approx(-99.0)
    assert editor._vmax_spin.value() == pytest.approx(99.0)


def test_bar_toggle_calls_diagram_set_show_scalar_bar(editor, lut):
    diagram = _FakeDiagram(lut)
    editor.bind_layer(diagram)
    editor._bar_cb.setChecked(False)
    assert diagram.show_bar_called is False


def test_bind_layer_none_clears_diagram(editor, lut):
    diagram = _FakeDiagram(lut)
    editor.bind_layer(diagram)
    editor.bind_layer(None)
    assert editor._diagram is None
    assert editor._lut is None


def test_bind_layer_diagram_without_lut_unbinds(editor):
    class _NoLut:
        lut = None
    editor.bind_layer(_NoLut())
    assert editor._lut is None


# =====================================================================
# Dock spec factory
# =====================================================================


def test_make_color_map_editor_dock_returns_spec(qapp):
    editor, spec = make_color_map_editor_dock()
    assert editor is not None
    assert spec.dock_id == "dock_color_map_editor"
    assert spec.title == "Color Mapping"
    assert spec.default_area == "right"
    # Factory returns the editor's widget.
    assert spec.factory(None) is editor.widget
