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
