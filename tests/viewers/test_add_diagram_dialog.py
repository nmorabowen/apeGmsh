"""Regression coverage for ``AddDiagramDialog``.

The dialog was silently broken at import time (``from ..diagrams._selectors
import normalize_selector`` — the function is exported from the package as
``normalize_selector`` but inside ``_selectors.py`` it's just ``normalize``).
Clicking the Diagrams-tab Add button raised ImportError silently inside Qt's
signal-slot machinery — the user just saw nothing happen.

These tests construct the dialog directly so any import-time / construct-time
breakage surfaces in CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def director():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    from apeGmsh.results import Results
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector(Results.from_mpco(_FIXTURE))


# =====================================================================
# Module-level import — the ImportError was here
# =====================================================================

def test_dialog_module_imports_cleanly():
    """Import surface — would catch a renamed symbol from _selectors.py."""
    import apeGmsh.viewers.ui._add_diagram_dialog as mod
    assert hasattr(mod, "AddDiagramDialog")


# =====================================================================
# Construction — would catch missing director or stages API
# =====================================================================

def test_dialog_constructs_with_director(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._dlg is not None


def test_dialog_kind_combo_populated(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    # Phase 1 ships 8 kinds (contour / deformed / line force / fiber /
    # layer / vector glyph / gauss marker / spring force).
    assert dlg._kind_combo.count() == 8


def test_dialog_stage_combo_populated_from_director(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    # elasticFrame.mpco carries two transient stages.
    assert dlg._stage_combo.count() == len(list(director.stages()))
    assert dlg._stage_combo.count() >= 1


def test_dialog_default_selector_is_all_nodes(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._selector_kind.currentData() == "all"
    assert not dlg._selector_name.isEnabled()


# =====================================================================
# Selector-kind switching enables / disables the name input
# =====================================================================

def test_selector_change_to_pg_enables_name_field(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    # Find the "pg" item index.
    pg_idx = next(
        i for i in range(dlg._selector_kind.count())
        if dlg._selector_kind.itemData(i) == "pg"
    )
    dlg._selector_kind.setCurrentIndex(pg_idx)
    assert dlg._selector_name.isEnabled()


# =====================================================================
# Component combo — populated from kind + stage
# =====================================================================

def _set_kind(dlg, kind_id):
    for i in range(dlg._kind_combo.count()):
        entry = dlg._kind_combo.itemData(i)
        if entry.kind_id == kind_id:
            dlg._kind_combo.setCurrentIndex(i)
            return
    raise AssertionError(f"kind {kind_id} not found in combo")


def test_component_combo_lists_nodal_components_for_contour(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "contour")
    items = [dlg._component_combo.itemText(i)
             for i in range(dlg._component_combo.count())]
    # elasticFrame.mpco has nodal displacements
    assert "displacement_x" in items
    assert "displacement_z" in items


def test_component_combo_default_prefers_displacement_z(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "contour")
    assert dlg._component_combo.currentText() == "displacement_z"


def test_component_combo_updates_when_kind_changes(qapp, director):
    """Switching kind from contour (nodes) to spring_force (springs)
    should swap the component list; the elasticFrame fixture has no
    springs so the list becomes empty.
    """
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "contour")
    assert dlg._component_combo.count() > 0

    _set_kind(dlg, "spring_force")
    items = [dlg._component_combo.itemText(i)
             for i in range(dlg._component_combo.count())]
    # No springs in elasticFrame → empty list
    assert items == [] or all("spring" in c for c in items)


def test_component_combo_lists_spring_components_on_springs_fixture(qapp):
    """zl_springs.mpco carries spring force/deformation."""
    spring_fixture = Path("tests/fixtures/results/zl_springs.mpco")
    if not spring_fixture.exists():
        pytest.skip(f"Missing fixture: {spring_fixture}")
    from apeGmsh.results import Results
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    d = ResultsDirector(Results.from_mpco(spring_fixture))
    dlg = AddDiagramDialog(d, parent=None)
    _set_kind(dlg, "spring_force")
    items = [dlg._component_combo.itemText(i)
             for i in range(dlg._component_combo.count())]
    assert "spring_force_0" in items


def test_component_combo_remains_editable_for_custom_names(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._component_combo.isEditable()
    dlg._component_combo.setEditText("totally_custom_thing")
    assert dlg._component_combo.currentText() == "totally_custom_thing"
