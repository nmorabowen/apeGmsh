"""Coverage for the ``section_cut`` path through ``AddDiagramDialog``.

Phase 4 — file-picker ingress for pickled ``SectionCutDef`` /
``SectionSweepDef``. The dialog gains a new ``_KindEntry`` whose UI
layout swaps every Results-data row for a file picker, a model.h5
picker, and a preflight report panel; OK is gated on a clean preflight.

These tests construct the dialog directly with a real
``ResultsDirector`` fixture and patch ``QFileDialog.getOpenFileName``,
``SectionCutDef.preflight``, and ``director.add_section_cut*`` so we
verify the UI logic without touching real h5 files or the registry.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


def _set_kind(dlg, kind_id: str) -> None:
    for i in range(dlg._kind_combo.count()):
        entry = dlg._kind_combo.itemData(i)
        if entry is not None and entry.kind_id == kind_id:
            dlg._kind_combo.setCurrentIndex(i)
            return
    raise AssertionError(f"kind {kind_id} not found in combo")


def _clean_report(label: str = "story 1"):
    """Build a clean ``PreflightReport`` with no issues for mocking."""
    from apeGmsh.cuts import PreflightReport
    return PreflightReport(cut_label=label, issues=())


def _error_report(label: str = "bad"):
    """Build a ``PreflightReport`` carrying one error for mocking."""
    from apeGmsh.cuts import PreflightIssue, PreflightReport
    issue = PreflightIssue(
        code="E1",
        severity="error",
        message="OpenSees tag 999 not in tag map.",
    )
    return PreflightReport(cut_label=label, issues=(issue,))


def _warning_report(label: str = "edge"):
    """Build a ``PreflightReport`` carrying one warning for mocking."""
    from apeGmsh.cuts import PreflightIssue, PreflightReport
    issue = PreflightIssue(
        code="W1",
        severity="warning",
        message="Filter nodes all on positive side of plane.",
    )
    return PreflightReport(cut_label=label, issues=(issue,))


def _stub_cut():
    """Build a minimal SectionCutDef for tests."""
    from apeGmsh.cuts import SectionCutDef
    return SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
        label="story 1",
    )


def _stub_sweep():
    """Build a minimal SectionSweepDef for tests."""
    from apeGmsh.cuts import SectionSweepDef
    return SectionSweepDef(cuts=(_stub_cut(), _stub_cut()))


# =====================================================================
# Kind combo contains section_cut
# =====================================================================

def test_kind_combo_includes_section_cut(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import (
        SECTION_CUT_KIND_ID, AddDiagramDialog,
    )
    dlg = AddDiagramDialog(director, parent=None)
    ids = [
        dlg._kind_combo.itemData(i).kind_id
        for i in range(dlg._kind_combo.count())
    ]
    assert SECTION_CUT_KIND_ID in ids


# =====================================================================
# Layout switching
# =====================================================================

def test_picking_section_cut_hides_results_rows(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    # Results-data rows are hidden when section_cut is chosen.
    # ``isHidden()`` reflects the explicit set-visible state regardless
    # of whether the parent dialog has been ``.show()``n.
    assert dlg._stage_combo.isHidden()
    assert dlg._component_combo.isHidden()
    assert dlg._preset_combo.isHidden()
    assert dlg._selector_kind.isHidden()
    assert dlg._selector_name.isHidden()
    # Section-cut rows are shown.
    assert not dlg._cut_file_row.isHidden()
    assert not dlg._cut_model_h5_row.isHidden()
    assert not dlg._cut_preflight_status.isHidden()
    assert not dlg._cut_preflight_summary.isHidden()


def test_switching_back_restores_results_rows(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_kind(dlg, "contour")
    assert not dlg._stage_combo.isHidden()
    assert not dlg._component_combo.isHidden()
    # Section-cut rows hidden again.
    assert dlg._cut_file_row.isHidden()
    assert dlg._cut_model_h5_row.isHidden()


# =====================================================================
# model_h5 autofill
# =====================================================================

def test_model_h5_autofills_from_director(qapp, director, tmp_path):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    fake = tmp_path / "model.h5"
    fake.write_bytes(b"")          # touch — content irrelevant for the prefill check
    director.set_model_h5(fake)
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._cut_model_h5_edit.text() == str(fake)


def test_model_h5_empty_when_director_unset(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._cut_model_h5_edit.text() == ""


# =====================================================================
# OK gating on preflight result
# =====================================================================

def test_ok_disabled_before_file_loaded(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    # Nothing loaded — OK must be disabled.
    assert not dlg._ok_button.isEnabled()


def test_ok_enabled_after_clean_preflight(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "cut.pkl"
    _stub_cut().save_pickle(pkl)
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    with patch.object(
        SectionCutDef, "preflight", return_value=_clean_report(),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    assert dlg._ok_button.isEnabled()
    # Status label shows OK.
    assert "OK" in dlg._cut_preflight_status.text()


def test_ok_disabled_when_preflight_errors(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "cut.pkl"
    _stub_cut().save_pickle(pkl)
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    with patch.object(
        SectionCutDef, "preflight", return_value=_error_report(),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    assert not dlg._ok_button.isEnabled()
    assert "ERROR" in dlg._cut_preflight_status.text()


def test_ok_enabled_when_preflight_warnings_only(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "cut.pkl"
    _stub_cut().save_pickle(pkl)
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    with patch.object(
        SectionCutDef, "preflight", return_value=_warning_report(),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    assert dlg._ok_button.isEnabled()
    assert "WARNING" in dlg._cut_preflight_status.text()


# =====================================================================
# OK dispatch — single cut vs sweep
# =====================================================================

def test_ok_dispatches_to_add_section_cut(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "cut.pkl"
    cut = _stub_cut()
    cut.save_pickle(pkl)
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    with patch.object(
        SectionCutDef, "preflight", return_value=_clean_report(),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    with patch.object(director, "add_section_cut") as add_cut:
        with patch.object(SectionCutDef, "preflight", return_value=_clean_report()):
            ok = dlg._run_section_cut()
    assert ok
    add_cut.assert_called_once()
    call = add_cut.call_args
    assert isinstance(call.args[0], SectionCutDef)


def test_ok_dispatches_to_add_section_cut_sweep(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionSweepDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "sweep.pkl"
    sweep = _stub_sweep()
    sweep.save_pickle(pkl)
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    with patch.object(
        SectionSweepDef, "preflight",
        return_value=(_clean_report("a"), _clean_report("b")),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    with patch.object(director, "add_section_cut_sweep") as add_sweep:
        with patch.object(
            SectionSweepDef, "preflight",
            return_value=(_clean_report("a"), _clean_report("b")),
        ):
            ok = dlg._run_section_cut()
    assert ok
    add_sweep.assert_called_once()
    call = add_sweep.call_args
    assert isinstance(call.args[0], SectionSweepDef)


# =====================================================================
# File picker dialog wiring
# =====================================================================

def test_browse_sets_file_path(qapp, director, tmp_path):
    """The Browse button delegates to ``QFileDialog.getOpenFileName``."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    pkl = tmp_path / "cut.pkl"
    pkl.write_bytes(b"")
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    with patch.object(
        QtWidgets.QFileDialog, "getOpenFileName",
        return_value=(str(pkl), "Pickled cut (*.pkl *.pkl.gz)"),
    ):
        dlg._on_cut_file_browse()
    assert dlg._cut_file_edit.text() == str(pkl)


# =====================================================================
# Bad file handling
# =====================================================================

def test_unloadable_pickle_marks_error(qapp, director, tmp_path):
    """A path that doesn't deserialize as a Cut or Sweep keeps OK disabled."""
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    bogus = tmp_path / "not_a_cut.pkl"
    # Pickle of a plain dict — fails both SectionCutDef.load_pickle
    # (TypeError, wrong class) and SectionSweepDef.load_pickle (same).
    import pickle
    bogus.write_bytes(pickle.dumps({"hello": "world"}))
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    dlg._cut_file_edit.setText(str(bogus))
    assert dlg._cut_loaded is None
    assert dlg._cut_load_error is not None
    assert not dlg._ok_button.isEnabled()


def test_missing_file_marks_error(qapp, director, tmp_path):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    missing = tmp_path / "nope.pkl"
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    dlg._cut_file_edit.setText(str(missing))
    assert dlg._cut_loaded is None
    assert dlg._cut_load_error is not None
    assert "not found" in dlg._cut_load_error.lower()
    assert not dlg._ok_button.isEnabled()
