"""SectionCutDiagram settings-card panel — checkbox dispatch + Apply path.

Verifies that ``DiagramSettingsTab._dispatch_kind_panel`` recognizes the
``section_cut`` kind and renders a ``QCheckBox`` whose Apply stages
``Diagram.set_show_filter(checked)``. Uses an offscreen Qt platform and
a stub diagram so the test stays scoped to the panel layout — the
runtime semantics of ``set_show_filter`` are covered in
``test_section_cut_diagram.py``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock


# Force offscreen Qt before importing qtpy.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _build_settings_tab():
    """Construct a DiagramSettingsTab against a stub director."""
    from qtpy import QtWidgets

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    geometries = GeometryManager()

    class _RegistryStub:
        def diagrams(self):
            return []

    class _CompMgrStub:
        active = None

    class _Director:
        def __init__(self, geoms):
            self.geometries = geoms
            self.stage_id = None

        def stages(self):
            return []

        def subscribe_stage(self, _cb):
            return lambda: None

        def subscribe_diagrams(self, _cb):
            return lambda: None

        @property
        def registry(self):
            return _RegistryStub()

        @property
        def compositions(self):
            return _CompMgrStub()

    director = _Director(geometries)
    from apeGmsh.viewers.ui._diagram_settings_tab import DiagramSettingsTab
    return DiagramSettingsTab(director)


def _stub_section_cut_diagram(
    *,
    initial: bool = False,
    side: str = "positive",
    label: str = "story 3",
) -> SimpleNamespace:
    """A fake diagram exposing only what the panel builder needs.

    The panel reads ``kind`` and ``show_filter``, calls
    ``set_show_filter`` from the Apply lambda. Side and label come
    from ``spec.style.cut`` via the panel's ``_section_cut_def``
    helper — we mirror that surface with a frozen-like SimpleNamespace.
    """
    cut = SimpleNamespace(side=side, label=label)
    style = SimpleNamespace(cut=cut)
    spec = SimpleNamespace(style=style, label=label)
    return SimpleNamespace(
        kind="section_cut",
        show_filter=initial,
        set_show_filter=MagicMock(),
        spec=spec,
    )


def _dispatch_into_fresh_card(tab, stub):
    """Run ``_dispatch_kind_panel`` with the per-card state the real
    card builder normally sets up around it.

    The production path is ``_build_card`` → swap ``_content_layout``
    to the card's QVBoxLayout, init ``_pending_appliers = []``, call
    ``_dispatch_kind_panel``. We replicate just the bits the panel
    builder reads, then leave the layout in place so test asserts can
    walk it.
    """
    tab._pending_appliers = []
    tab._dispatch_kind_panel(stub)


def _find_checkbox(tab):
    """Return the section-cut 'Show filter elements' checkbox.

    Filters by text so the tab-level Auto-Apply checkbox (added in
    plan 05) doesn't collide with this lookup — both are children
    of ``tab._widget`` but only one belongs to the section-cut card.
    """
    from qtpy import QtWidgets
    for cb in tab._widget.findChildren(QtWidgets.QCheckBox):
        if cb.text() == "Show filter elements":
            return cb
    return None


def test_dispatch_kind_panel_recognizes_section_cut() -> None:
    """``section_cut`` must not fall through to the fallback label."""
    from qtpy import QtWidgets
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(tab, _stub_section_cut_diagram())
    # No fallback label should be present.
    labels = tab._widget.findChildren(QtWidgets.QLabel)
    fallback_seen = any(
        "No settings UI for kind" in label.text()
        for label in labels
    )
    assert not fallback_seen


def test_panel_renders_show_filter_checkbox() -> None:
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(tab, _stub_section_cut_diagram())
    chk = _find_checkbox(tab)
    assert chk is not None
    assert chk.text() == "Show filter elements"


def test_checkbox_reflects_initial_show_filter_state() -> None:
    """Initial ``show_filter`` value seeds the checkbox state."""
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(tab, _stub_section_cut_diagram(initial=True))
    chk = _find_checkbox(tab)
    assert chk is not None
    assert chk.isChecked() is True


def test_apply_propagates_checkbox_to_set_show_filter() -> None:
    """User checks the box, hits Apply → ``set_show_filter(True)`` fires."""
    tab = _build_settings_tab()
    stub = _stub_section_cut_diagram(initial=False)
    _dispatch_into_fresh_card(tab, stub)
    chk = _find_checkbox(tab)
    chk.setChecked(True)

    # Replay the Apply path — the tab stages every per-card mutation in
    # ``_pending_appliers``; the Apply button just walks the list.
    for applier in tab._pending_appliers:
        applier()

    stub.set_show_filter.assert_called_once_with(True)


def test_apply_propagates_unchecking_too() -> None:
    """Unchecking from an initially-on state stages ``set_show_filter(False)``."""
    tab = _build_settings_tab()
    stub = _stub_section_cut_diagram(initial=True)
    _dispatch_into_fresh_card(tab, stub)
    chk = _find_checkbox(tab)
    assert chk.isChecked() is True
    chk.setChecked(False)
    for applier in tab._pending_appliers:
        applier()
    stub.set_show_filter.assert_called_once_with(False)


# ---------------------------------------------------------------------
# Phase D — side + label live edit wiring
# ---------------------------------------------------------------------

def _find_combobox(tab):
    from qtpy import QtWidgets
    return tab._widget.findChild(QtWidgets.QComboBox)


def _find_line_edit(tab):
    from qtpy import QtWidgets
    return tab._widget.findChild(QtWidgets.QLineEdit)


def test_panel_renders_side_combobox() -> None:
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(tab, _stub_section_cut_diagram())
    combo = _find_combobox(tab)
    assert combo is not None
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["positive", "negative"]


def test_side_combobox_reflects_current_value() -> None:
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(tab, _stub_section_cut_diagram(side="negative"))
    combo = _find_combobox(tab)
    assert combo.currentText() == "negative"


def test_panel_renders_label_line_edit_with_current_label() -> None:
    tab = _build_settings_tab()
    _dispatch_into_fresh_card(
        tab, _stub_section_cut_diagram(label="custom label"),
    )
    line = _find_line_edit(tab)
    assert line is not None
    assert line.text() == "custom label"


def test_side_change_invokes_rebuild() -> None:
    """Flipping the combobox fires ``_on_section_cut_rebuild(d, side=...)``."""
    tab = _build_settings_tab()
    stub = _stub_section_cut_diagram(side="positive")
    tab._on_section_cut_rebuild = MagicMock()
    _dispatch_into_fresh_card(tab, stub)
    combo = _find_combobox(tab)
    combo.setCurrentText("negative")
    tab._on_section_cut_rebuild.assert_called_with(stub, side="negative")


def test_label_edit_finished_invokes_rebuild() -> None:
    """``editingFinished`` (Enter / focus loss) fires the rebuild path."""
    tab = _build_settings_tab()
    stub = _stub_section_cut_diagram(label="original")
    tab._on_section_cut_rebuild = MagicMock()
    _dispatch_into_fresh_card(tab, stub)
    line = _find_line_edit(tab)
    line.setText("renamed")
    line.editingFinished.emit()
    tab._on_section_cut_rebuild.assert_called_with(stub, label="renamed")
