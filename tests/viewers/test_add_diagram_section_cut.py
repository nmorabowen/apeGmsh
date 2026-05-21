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

from tests.conftest import _stub_model_h5_path


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
    return ResultsDirector(Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path()))


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
    """Phase 8 — the deprecated ``set_model_h5`` verb is gone.

    The director's internal ``_bind_model_h5`` helper is the only
    remaining path-binder; it carries the same autofill semantics.
    """
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    fake = tmp_path / "model.h5"
    fake.write_bytes(b"")          # touch — content irrelevant for the prefill check
    director._bind_model_h5(fake)
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


# =====================================================================
# v4-5 — Source toggle (file vs h5) + h5-cut dropdown
# =====================================================================

def _make_minimal_h5(path, *, schema_version="2.6.0"):
    import h5py
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs.create(
            "schema_version", schema_version,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _set_source(dlg, source: str) -> None:
    """Pick the section-cut Source combo entry by data key."""
    for i in range(dlg._cut_source_combo.count()):
        if dlg._cut_source_combo.itemData(i) == source:
            dlg._cut_source_combo.setCurrentIndex(i)
            return
    raise AssertionError(f"source {source!r} not in combo")


def test_source_combo_has_two_entries(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    keys = [
        dlg._cut_source_combo.itemData(i)
        for i in range(dlg._cut_source_combo.count())
    ]
    assert keys == ["file", "h5"]


def test_default_source_is_file(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    assert dlg._cut_source_combo.currentData() == "file"


def test_switching_to_h5_hides_file_shows_dropdown(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    # Default state — file visible, dropdown hidden.
    assert not dlg._cut_file_row.isHidden()
    assert dlg._cut_h5_dropdown.isHidden()

    _set_source(dlg, "h5")
    assert dlg._cut_file_row.isHidden()
    assert not dlg._cut_h5_dropdown.isHidden()


def test_switching_back_to_file_hides_dropdown(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    _set_source(dlg, "file")
    assert not dlg._cut_file_row.isHidden()
    assert dlg._cut_h5_dropdown.isHidden()


def test_source_combo_hidden_when_kind_is_not_section_cut(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "contour")
    assert dlg._cut_source_combo.isHidden()
    _set_kind(dlg, "section_cut")
    assert not dlg._cut_source_combo.isHidden()


# --- Dropdown population --------------------------------------------------

def test_dropdown_placeholder_when_no_model_h5(qapp, director):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog
    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    assert dlg._cut_h5_dropdown.count() == 1
    assert "set Model.h5" in dlg._cut_h5_dropdown.itemText(0)
    assert dlg._cut_h5_dropdown.itemData(0) is None


def test_dropdown_populates_from_h5_with_cuts(qapp, director, tmp_path):
    from apeGmsh.cuts import (
        SectionCutDef, SectionSweepDef, persist_to_h5,
    )
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    cut_a = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="story 1",
    )
    cut_b = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="story 2",
    )
    sweep = SectionSweepDef(cuts=(cut_a, cut_b))
    path = tmp_path / "model.h5"
    _make_minimal_h5(path)
    persist_to_h5(path, cuts=[cut_a, cut_b], sweeps=[sweep])

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    dlg._cut_model_h5_edit.setText(str(path))

    # 2 standalone cuts + 1 sweep = 3 entries
    assert dlg._cut_h5_dropdown.count() == 3
    # Standalone cuts come first, in writer order, with labels.
    assert "story 1" in dlg._cut_h5_dropdown.itemText(0)
    assert "story 2" in dlg._cut_h5_dropdown.itemText(1)
    # Sweep entry mentions the count.
    assert "sweep" in dlg._cut_h5_dropdown.itemText(2).lower()
    assert "2 cuts" in dlg._cut_h5_dropdown.itemText(2)


def test_dropdown_empty_message_for_h5_without_cuts(
    qapp, director, tmp_path,
):
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    path = tmp_path / "no_cuts.h5"
    _make_minimal_h5(path)  # no /opensees/cuts/, no /opensees/sweeps/

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    dlg._cut_model_h5_edit.setText(str(path))

    assert dlg._cut_h5_dropdown.count() == 1
    assert "no cuts persisted" in dlg._cut_h5_dropdown.itemText(0)
    assert dlg._cut_h5_dropdown.itemData(0) is None
    assert dlg._cut_loaded is None


# --- Dropdown selection drives loaded state -------------------------------

def test_dropdown_selection_loads_cut(qapp, director, tmp_path):
    from apeGmsh.cuts import SectionCutDef, persist_to_h5
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 1.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(7, 8),
        label="picked",
    )
    path = tmp_path / "model.h5"
    _make_minimal_h5(path)
    persist_to_h5(path, cuts=[cut])

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    with patch.object(
        SectionCutDef, "preflight", return_value=_clean_report(),
    ):
        dlg._cut_model_h5_edit.setText(str(path))
    # Auto-selected first entry → _cut_loaded matches the persisted cut.
    assert dlg._cut_loaded == cut
    assert dlg._cut_load_error is None


def test_h5_source_dispatches_to_add_section_cut(
    qapp, director, tmp_path,
):
    """OK in h5 mode → director.add_section_cut(loaded_cut)."""
    from apeGmsh.cuts import SectionCutDef, persist_to_h5
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(42,),
        label="from h5",
    )
    path = tmp_path / "model.h5"
    _make_minimal_h5(path)
    persist_to_h5(path, cuts=[cut])

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    with patch.object(
        SectionCutDef, "preflight", return_value=_clean_report(),
    ):
        dlg._cut_model_h5_edit.setText(str(path))

    with patch.object(director, "add_section_cut") as add_cut, \
         patch.object(SectionCutDef, "preflight", return_value=_clean_report()):
        ok = dlg._run_section_cut()
    assert ok
    add_cut.assert_called_once()
    call = add_cut.call_args
    assert isinstance(call.args[0], SectionCutDef)


def test_h5_source_dispatches_to_add_section_cut_sweep(
    qapp, director, tmp_path,
):
    """OK in h5 mode with a sweep entry selected → add_section_cut_sweep."""
    from apeGmsh.cuts import (
        SectionCutDef, SectionSweepDef, persist_to_h5,
    )
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    cut0 = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="sweep cut 0",
    )
    cut1 = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="sweep cut 1",
    )
    sweep = SectionSweepDef(cuts=(cut0, cut1))
    path = tmp_path / "model.h5"
    _make_minimal_h5(path)
    persist_to_h5(path, sweeps=[sweep])

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    _set_source(dlg, "h5")
    with patch.object(
        SectionSweepDef, "preflight",
        return_value=(_clean_report("0"), _clean_report("1")),
    ):
        dlg._cut_model_h5_edit.setText(str(path))

    # Only entry is the sweep — auto-selected.
    with patch.object(
        director, "add_section_cut_sweep",
    ) as add_sweep, patch.object(
        SectionSweepDef, "preflight",
        return_value=(_clean_report("0"), _clean_report("1")),
    ):
        ok = dlg._run_section_cut()
    assert ok
    add_sweep.assert_called_once()
    call = add_sweep.call_args
    assert isinstance(call.args[0], SectionSweepDef)


def test_switching_to_h5_resets_loaded_state(qapp, director, tmp_path):
    """Source switch clears prior loaded state (file → h5 transition)."""
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.viewers.ui._add_diagram_dialog import AddDiagramDialog

    pkl = tmp_path / "cut.pkl"
    _stub_cut().save_pickle(pkl)

    dlg = AddDiagramDialog(director, parent=None)
    _set_kind(dlg, "section_cut")
    # Load via file mode first.
    with patch.object(
        SectionCutDef, "preflight", return_value=_clean_report(),
    ):
        dlg._cut_file_edit.setText(str(pkl))
    assert dlg._cut_loaded is not None
    # Switch to h5 mode with no model.h5 path → loaded state cleared.
    _set_source(dlg, "h5")
    assert dlg._cut_loaded is None
