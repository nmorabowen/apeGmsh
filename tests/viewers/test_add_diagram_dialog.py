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


# =====================================================================
# Default style — flip_sign for bending moments by structural convention
# =====================================================================

def test_line_force_default_flips_sign_for_bending_moment():
    """Bending-moment line-force diagrams default to flip_sign=True
    (engineering convention: draw moment on the tension side)."""
    from apeGmsh.viewers.ui._add_diagram_dialog import (
        _line_force_default_style,
    )
    style = _line_force_default_style("bending_moment_z")
    assert style.flip_sign is True
    style = _line_force_default_style("bending_moment_y")
    assert style.flip_sign is True


def test_line_force_default_does_not_flip_for_axial_shear_torsion():
    from apeGmsh.viewers.ui._add_diagram_dialog import (
        _line_force_default_style,
    )
    for component in (
        "axial_force", "shear_y", "shear_z", "torsion",
    ):
        style = _line_force_default_style(component)
        assert style.flip_sign is False, (
            f"{component} should not default to flipped sign"
        )


# =====================================================================
# Stale-text bug: switching to a kind with no available components
# must clear the field, not preserve the previous kind's default.
# =====================================================================

def test_combo_clears_when_switching_to_kind_with_no_components(
    qapp, director,
):
    """Regression: switching from contour (has displacement_z) to
    gauss_marker (empty for elasticFrame.mpco) used to leave
    'displacement_z' in the edit field — wrong since the gauss diagram
    would then try to read a nodal component.
    """
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)

    _set_kind(dlg, "contour")
    assert dlg._component_combo.currentText() == "displacement_z"

    _set_kind(dlg, "gauss_marker")
    assert dlg._component_combo.count() == 0
    assert dlg._component_combo.currentText() == "", (
        "edit text must clear when the new kind has no components"
    )


def test_combo_clears_for_spring_force_when_no_spring_data(qapp, director):
    """Same regression for spring_force on a fixture without springs."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)

    _set_kind(dlg, "line_force")
    assert dlg._component_combo.count() > 0
    assert dlg._component_combo.currentText()  # non-empty default

    _set_kind(dlg, "spring_force")
    assert dlg._component_combo.count() == 0
    assert dlg._component_combo.currentText() == ""


def test_combo_repopulates_when_switching_back(qapp, director):
    """After clearing, switching back to a kind with components must
    refill the dropdown and select the appropriate default again."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)

    _set_kind(dlg, "gauss_marker")
    assert dlg._component_combo.currentText() == ""

    _set_kind(dlg, "contour")
    assert dlg._component_combo.count() > 0
    assert dlg._component_combo.currentText() == "displacement_z"


# =====================================================================
# Empty-state UX: kinds with no data anywhere are flagged in the label,
# and the Component combo placeholder explains why it's empty.
# =====================================================================

def test_kind_label_marks_topologies_with_no_data_in_file(qapp, director):
    """elasticFrame.mpco has nodal displacements but no springs / fibers /
    layers / line-stations. Those kinds should carry a 'no data' suffix
    in the visible label so the user isn't surprised by an empty combo."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    labels_by_kind = {
        dlg._kind_combo.itemData(i).kind_id: dlg._kind_combo.itemText(i)
        for i in range(dlg._kind_combo.count())
    }
    # Contour has nodal data → no suffix.
    assert "no data" not in labels_by_kind["contour"]
    # Spring force has no spring data in this fixture → suffix present.
    assert "no data" in labels_by_kind["spring_force"]


def test_kinds_without_data_set_excludes_kinds_that_have_data(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    # elasticFrame has nodal displacement_z → contour has data.
    assert "contour" not in dlg._kinds_without_data
    # ...but no springs.
    assert "spring_force" in dlg._kinds_without_data


def test_empty_combo_placeholder_says_no_data_in_file(qapp, director):
    """When a kind has no data anywhere in the file, the placeholder
    must say 'in file' (the user needs a different recorder), not
    'in selected stage' (which would imply switching stages helps)."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "spring_force")
    placeholder = dlg._component_combo.lineEdit().placeholderText()
    assert "no springs data in file" in placeholder


def test_populated_combo_restores_generic_placeholder(qapp, director):
    """Switching from an empty kind back to a populated one must clear
    the empty-state placeholder so the original example hint comes back."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "spring_force")
    assert "no" in dlg._component_combo.lineEdit().placeholderText()
    _set_kind(dlg, "contour")
    placeholder = dlg._component_combo.lineEdit().placeholderText()
    assert "displacement_z" in placeholder
