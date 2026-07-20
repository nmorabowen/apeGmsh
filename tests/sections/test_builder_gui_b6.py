"""Tests — ADR 0080 B6: live properties panel in the builder GUI.

Offscreen widget tests (the S6 pattern): the live properties panel
build+analyzes on a worker thread and marshals the result back to the
UI thread. These tests drive the controller deterministically
(``join`` + ``drain``) rather than relying on the poll timer, and prove
the panel values equal a headless ``build()`` — the no-solve-on-the-
UI-thread law holds through the GUI.
"""
from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apeGmsh.sections import SectionDocument  # noqa: E402
from apeGmsh.sections._properties import BuildResult  # noqa: E402


def _qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    pytest.importorskip("matplotlib")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if app.platformName().lower() != "offscreen":
        pytest.skip("another test created a non-offscreen QApplication")
    return app


def _win(doc):
    from apeGmsh.sections._builder_gui import SectionBuilderWindow

    return SectionBuilderWindow(doc)


def _rect_doc():
    doc = SectionDocument.new(name="t", kind="continuum")
    doc.set_material("s", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
    doc.set_mesh(lc=1.0)
    return doc


# ─────────────────────────────────────────────────────────────────────
# defaults + graying
# ─────────────────────────────────────────────────────────────────────


def test_live_off_by_default_no_controller():
    _qapp()
    win = _win(_rect_doc())
    try:
        assert win._live_enabled is False
        assert win._controller is None      # no worker until opted in
        assert "off" in win._props_status.text()
    finally:
        win.close()


def test_dispatch_greys_until_fresh():
    _qapp()
    win = _win(_rect_doc())
    try:
        win.refresh_properties()
        # right after dispatch, before draining: greyed, "building…"
        assert win._props_body_host.isEnabled() is False
        assert "building" in win._props_status.text()
        win._controller.join(60.0)
        win._controller.drain()
        assert win._props_body_host.isEnabled() is True
        assert "up to date" in win._props_status.text()
    finally:
        win.close()


# ─────────────────────────────────────────────────────────────────────
# panel == headless build() (continuum), off the UI thread
# ─────────────────────────────────────────────────────────────────────


def test_panel_matches_headless_build_off_ui_thread():
    _qapp()
    doc = _rect_doc()
    win = _win(doc)
    try:
        main_id = threading.get_ident()
        win.set_live_properties(True)       # fires a build
        win._controller.join(60.0)
        win._controller.drain()

        res = win._last_result
        assert res is not None and res.error is None
        assert res.kind == "continuum"
        assert res.worker_thread_id != main_id      # solved off UI thread

        # the embedded inspector panel is present in the dock body
        from qtpy import QtWidgets
        assert win._props_body.count() == 1
        panel = win._props_body.itemAt(0).widget()
        assert panel.findChild(QtWidgets.QTabWidget) is not None

        # values equal a headless build
        ref = doc.build()
        ref.analyze()
        assert res.analysis.geometric().area == pytest.approx(
            ref.geometric().area
        )
    finally:
        win.close()


def test_fiber_identities_table_shown():
    _qapp()
    doc = SectionDocument.new(name="f", kind="fiber")
    doc.set_material("c", uniaxial=("Elastic", {"E": 1.0}))
    doc.add_patch_rect(material="c", ny=4, nz=4,
                       yI=-3.0, zI=-3.0, yJ=3.0, zJ=3.0)   # 6×6 = 36
    win = _win(doc)
    try:
        win.refresh_properties()
        win._controller.join(30.0)
        win._controller.drain()
        res = win._last_result
        assert res.kind == "fiber" and res.error is None
        assert res.identities["total_area"] == pytest.approx(36.0)
        from qtpy import QtWidgets
        table = win._props_body.itemAt(0).widget()
        assert isinstance(table, QtWidgets.QTableWidget)
        assert table.rowCount() >= 5
    finally:
        win.close()


def test_error_state_greys_with_message():
    _qapp()
    doc = SectionDocument.new(name="t", kind="continuum")
    doc.add_shape("rect_face", id="r", b=2.0, h=2.0)   # no set_mesh(lc)
    win = _win(doc)
    try:
        win.refresh_properties()
        win._controller.join(30.0)
        win._controller.drain()
        assert win._last_result.error is not None
        assert "unavailable" in win._props_status.text()
    finally:
        win.close()


# ─────────────────────────────────────────────────────────────────────
# live-mode auto-refresh + coalescing through the GUI (stub builder)
# ─────────────────────────────────────────────────────────────────────


def test_live_mutation_requests_build_with_stub():
    _qapp()
    from apeGmsh.sections._properties import PropertiesController

    win = _win(SectionDocument.new(name="t", kind="continuum"))
    try:
        calls = {"n": 0}

        def stub(doc_dict):
            calls["n"] += 1
            return BuildResult(key="", kind="continuum",
                               analysis=None, error="stub")

        # inject a stub controller so no Gmsh runs, then enable live
        # (set the flag directly so the checkbox toggle doesn't fire an
        # extra empty-document build)
        win._controller = PropertiesController(
            builder=stub, on_result=win._on_properties,
            autostart_timer=False,
        )
        win._live_enabled = True

        # a mutation via the document-API seam should trigger a request
        win._mat_name.setText("s")
        win._mat_E.setText("200000")
        win._mat_nu.setText("0.3")
        assert win.apply_material_from_form() is True
        for _ in range(4):     # settle any coalesced follow-up build
            win._controller.join(5.0)
            if win._controller.drain() and win._last_result is not None:
                break
        assert calls["n"] >= 1
        assert win._last_result is not None
    finally:
        win.close()


def test_controller_stopped_on_close():
    _qapp()
    win = _win(_rect_doc())
    win.refresh_properties()
    ctrl = win._controller
    assert ctrl is not None
    win.close()
    # timer stopped; joining returns promptly
    ctrl.join(5.0)
    assert ctrl._timer is None or not ctrl._timer.isActive()
