"""Tests — ADR 0080 B7: the builder GUI's three extras.

Offscreen widget tests (the S6 pattern). Each extra is checked for the
property that makes it worth having:

* **catalog picker** — the prefilled form produces the same document a
  script would write from ``catalog_shape_params`` (the parity law
  reaches the picker too), and the picker vanishes without apeSteel;
* **M–κ** — the button is fiber-lane only, greys with guidance when no
  backend is installed, and its curve equals the headless
  ``moment_curvature`` call;
* **handoff** — ``copy_handoff`` returns exactly ``handoff_snippet`` of
  the live document, including the path the window has open.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apeGmsh.sections import SectionDocument, handoff_snippet  # noqa: E402
from apeGmsh.sections._catalog import (  # noqa: E402
    catalog_available,
    catalog_shape_params,
)
from apeGmsh.sections._mc import backend_available  # noqa: E402


requires_apesteel = pytest.mark.skipif(
    not catalog_available(), reason="apeSteel not installed"
)
def requires_backend(fn):
    """``live`` (deselected by the curated suite) + an honest per-machine
    skip — see ``test_mc_b7.py`` for why both."""
    fn = pytest.mark.live(fn)
    return pytest.mark.skipif(
        not backend_available(), reason="no OpenSees backend installed",
    )(fn)


def _qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    pytest.importorskip("matplotlib")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if app.platformName().lower() != "offscreen":
        pytest.skip("another test created a non-offscreen QApplication")
    return app


def _win(doc, **kwargs):
    from apeGmsh.sections._builder_gui import SectionBuilderWindow

    return SectionBuilderWindow(doc, **kwargs)


def _blank_continuum():
    return SectionDocument.new(name="t", kind="continuum")


def _fiber_doc():
    doc = SectionDocument.new(name="pts", kind="fiber")
    doc.set_material("st", uniaxial=("ElasticMaterial", {"E": 200e3}))
    for y in (-100.0, 100.0):
        for z in (-50.0, 50.0):
            doc.add_point(material="st", y=y, z=z, area=100.0)
    doc.set_GJ(1.0e12)
    return doc


# ─────────────────────────────────────────────────────────────────────
# catalog picker
# ─────────────────────────────────────────────────────────────────────

@requires_apesteel
def test_prefill_then_add_equals_the_document_api():
    """Parity: prefill + "Add shape" writes the same document a script
    writes with ``add_shape(**catalog_shape_params(...))``."""
    _qapp()
    win = _win(_blank_continuum())
    try:
        win._catalog_combo.setCurrentText("W14X90")
        assert win.prefill_from_catalog() is True
        assert win._shape_combo.currentText() == "W_face"
        assert win.add_shape_from_form() is True
        gui_doc = win.doc.to_dict()
    finally:
        win.close()

    script_doc = _blank_continuum()
    script_doc.add_shape(
        "W_face", id="W14X90", **catalog_shape_params("W14X90"),
    )
    assert gui_doc["shapes"] == script_doc.to_dict()["shapes"]


@requires_apesteel
def test_prefill_does_not_touch_the_document_or_undo():
    """A prefill fills GUI fields; nothing is committed until "Add
    shape". It must not land on the undo stack."""
    _qapp()
    win = _win(_blank_continuum())
    try:
        before = win.doc.to_dict()
        win._catalog_combo.setCurrentText("W14X90")
        win.prefill_from_catalog()
        assert win.doc.to_dict() == before
        assert win._undo == []
    finally:
        win.close()


@requires_apesteel
def test_unknown_designation_is_a_status_message_not_a_crash():
    _qapp()
    win = _win(_blank_continuum())
    try:
        assert win.prefill_from_catalog("W99X999") is False
        assert "no apeSteel catalog resolves" in win._status.currentMessage()
        assert win._undo == []
    finally:
        win.close()


@requires_apesteel
def test_empty_designation_is_refused():
    _qapp()
    win = _win(_blank_continuum())
    try:
        win._catalog_combo.setCurrentText("   ")
        assert win.prefill_from_catalog() is False
        assert "designation" in win._status.currentMessage()
    finally:
        win.close()


def test_picker_is_absent_without_apesteel(monkeypatch):
    """apeSteel missing → no picker widget at all, and the builder is
    otherwise fully usable."""
    import apeGmsh.sections._builder_gui as gui

    _qapp()
    monkeypatch.setattr(gui, "catalog_available", lambda: False)
    win = _win(_blank_continuum())
    try:
        assert win._catalog_combo is None
        assert win.prefill_from_catalog("W14X90") is False
        # the rest of the shape form still works
        win._shape_combo.setCurrentText("rect_face")
        win._shape_id.setText("r")
        win._shape_param_fields["b"].setText("2")
        win._shape_param_fields["h"].setText("4")
        assert win.add_shape_from_form() is True
    finally:
        win.close()


# ─────────────────────────────────────────────────────────────────────
# moment–curvature
# ─────────────────────────────────────────────────────────────────────

def test_mc_box_is_fiber_lane_only():
    _qapp()
    win = _win(_blank_continuum())
    try:
        assert win._mc_box.isVisibleTo(win.window) is False
        win.doc = _fiber_doc()
        win._refresh()
        assert win._mc_box.isVisibleTo(win.window) is True
    finally:
        win.close()


def test_mc_button_greys_with_guidance_without_a_backend(monkeypatch):
    import apeGmsh.sections._builder_gui as gui

    _qapp()
    monkeypatch.setattr(gui, "backend_available", lambda: False)
    win = _win(_fiber_doc())
    try:
        assert win._mc_button.isEnabled() is False
        assert "openseespy" in win._mc_button.toolTip()
    finally:
        win.close()


@requires_backend
def test_mc_button_result_equals_the_headless_call():
    from apeGmsh.sections import moment_curvature

    _qapp()
    doc = _fiber_doc()
    win = _win(doc)
    try:
        win._mc_kappa.setText("1e-6")
        win._mc_steps.setText("4")
        curve = win.run_moment_curvature()
        assert curve is not None
        ref = moment_curvature(doc, axis="z", kappa_max=1e-6, n_steps=4)
        assert curve.curvature == pytest.approx(ref.curvature)
        assert curve.moment == pytest.approx(ref.moment)
        assert win._mc_dialog is not None
    finally:
        win.close()


@requires_backend
def test_mc_honours_the_axis_selector():
    _qapp()
    win = _win(_fiber_doc())
    try:
        win._mc_kappa.setText("1e-6")
        win._mc_steps.setText("4")
        strong = win.run_moment_curvature()
        win._mc_axis.setCurrentText("y")
        weak = win.run_moment_curvature()
        assert strong.axis == "z" and weak.axis == "y"
        # (±100, ±50) points: the strong axis is 4× stiffer
        assert strong.EI0 == pytest.approx(4.0 * weak.EI0, rel=1e-9)
    finally:
        win.close()


def test_a_curve_with_no_converged_step_still_renders():
    """A first-step failure leaves no secant to report. The dialog must
    show a dash, not propagate ``MomentCurvatureError`` into Qt."""
    from apeGmsh.sections import MomentCurvature
    from apeGmsh.sections._builder_gui import _mc_ei0

    _qapp()
    stillborn = MomentCurvature(
        axis="z", curvature=(0.0,), moment=(0.0,), axial=0.0,
        complete=False,
    )
    assert _mc_ei0(stillborn) == "—"
    win = _win(_fiber_doc())
    try:
        dialog = win._build_mc_dialog(stillborn)   # built, not shown
        assert dialog is not None
    finally:
        win.close()


def test_mc_bad_input_is_a_status_message():
    _qapp()
    win = _win(_fiber_doc())
    try:
        win._mc_kappa.setText("")           # required
        assert win.run_moment_curvature() is None
        assert "κ max" in win._status.currentMessage()
        win._mc_kappa.setText("not a number")
        assert win.run_moment_curvature() is None
        assert "not a number" in win._status.currentMessage()
    finally:
        win.close()


@requires_backend
def test_mc_on_a_continuum_document_is_refused_softly():
    """The box is hidden on continuum documents, but the method is
    public — calling it anyway must flash, not raise."""
    _qapp()
    win = _win(_blank_continuum())
    try:
        win._mc_kappa.setText("1e-6")
        assert win.run_moment_curvature() is None
        assert "fiber-lane operation" in win._status.currentMessage()
    finally:
        win.close()


# ─────────────────────────────────────────────────────────────────────
# handoff
# ─────────────────────────────────────────────────────────────────────

def test_copy_handoff_matches_the_headless_snippet():
    _qapp()
    doc = _fiber_doc()
    win = _win(doc)
    try:
        assert win.copy_handoff() == handoff_snippet(doc, path=None)
        assert "copied" in win._status.currentMessage()
    finally:
        win.close()


def test_handoff_tracks_the_open_document_path(tmp_path):
    """A continuum snippet re-opens the document by path, so the window
    must remember where it came from — and where it was last saved.

    The path lands as a **Python string literal**, so a Windows path
    arrives with its backslashes escaped — pasting it into a script has
    to work verbatim.
    """
    import ast
    import json

    _qapp()
    doc = _blank_continuum()
    doc.set_material("s", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
    doc.set_mesh(lc=1.0)
    path = tmp_path / "t.section.json"
    doc.save(path)
    opened_line = f"doc = SectionDocument.open({json.dumps(str(path))})"

    win = _win(doc)
    try:
        assert opened_line not in win.handoff_text()   # not saved yet
        win.save_document(path)
        assert opened_line in win.handoff_text()
    finally:
        win.close()

    reopened = _win(SectionDocument.open(path), path=path)
    try:
        text = reopened.handoff_text()
        assert opened_line in text
        # the emitted literal is the path Python parses back out
        literal = text.split("SectionDocument.open(", 1)[1].split(")", 1)[0]
        assert ast.literal_eval(literal) == str(path)
    finally:
        reopened.close()
