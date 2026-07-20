"""Tests — ADR 0080 B5: section builder GUI shell (``launch_builder``).

Offscreen widget tests (the S6 inspector pattern): the file pins
``QT_QPA_PLATFORM=offscreen`` and runs unmarked — offscreen widget
construction and ``grab()`` are safe everywhere, and the win32 launch
guard being *raised* is itself under test. No test enters a blocking Qt
event loop.

The load-bearing check is the **parity law**: every GUI mutation must
equal the corresponding :class:`SectionDocument` API call — asserted as
document-dict equality between the widget path and a hand-authored
reference document.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apeGmsh.sections import SectionDocument  # noqa: E402


def _qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    pytest.importorskip("matplotlib")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if app.platformName().lower() != "offscreen":
        pytest.skip("another test created a non-offscreen QApplication")
    return app


def _win(doc=None):
    from apeGmsh.sections._builder_gui import SectionBuilderWindow

    if doc is None:
        doc = SectionDocument.new(name="t", kind="continuum")
    return SectionBuilderWindow(doc)


# ─────────────────────────────────────────────────────────────────────
# guards
# ─────────────────────────────────────────────────────────────────────


def test_qt_absent_import_guard(monkeypatch):
    monkeypatch.setitem(sys.modules, "qtpy", None)
    from apeGmsh.sections import _builder_gui
    with pytest.raises(ImportError, match=r"apeGmsh\[viewer\]"):
        _builder_gui.launch_builder(blocking=False)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the offscreen crash guard is win32-only (inspector parity)",
)
def test_offscreen_guard_raises_on_launch():
    _qapp()
    from apeGmsh.sections._builder_gui import launch_builder
    with pytest.raises(RuntimeError, match="offscreen"):
        launch_builder(blocking=False)


def test_coerce_document_paths(tmp_path):
    from apeGmsh.sections._builder_gui import _coerce_document

    assert _coerce_document(None).kind == "continuum"
    doc = SectionDocument.new(name="x", kind="fiber")
    assert _coerce_document(doc) is doc
    p = tmp_path / "s.section.json"
    doc.save(p)
    assert _coerce_document(str(p)).kind == "fiber"


# ─────────────────────────────────────────────────────────────────────
# parity law — GUI mutation == SectionDocument API call
# ─────────────────────────────────────────────────────────────────────


def test_parity_add_parametric_shape():
    _qapp()
    win = _win()
    win._shape_combo.setCurrentText("rect_face")
    win._shape_id.setText("r")
    win._shape_param_fields["b"].setText("4")
    win._shape_param_fields["h"].setText("2")
    win._shape_tx.setText("1.5")
    win._shape_ty.setText("0")
    assert win.add_shape_from_form() is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.add_shape("rect_face", id="r", b=4.0, h=2.0,
                  translate=(1.5, 0.0), rotate=None)
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_add_polygon_via_commit():
    _qapp()
    win = _win()
    pts = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0)]
    assert win.commit_polygon(pts, id="poly") is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.add_polygon(pts, id="poly")
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_material_continuum_role():
    _qapp()
    win = _win()
    win._mat_name.setText("steel")
    win._mat_E.setText("200000")
    win._mat_nu.setText("0.3")
    assert win.apply_material_from_form() is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.set_material("steel", E=200000.0, nu=0.3)
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_material_uniaxial_role():
    _qapp()
    win = _win(SectionDocument.new(name="t", kind="fiber"))
    win._mat_name.setText("conc")
    win._mat_uni_type.setText("Concrete01")
    win._mat_uni_params.setText("fpc=-30,epsc0=-0.002,fpcu=-6,epsU=-0.01")
    assert win.apply_material_from_form() is True

    ref = SectionDocument.new(name="t", kind="fiber")
    ref.set_material("conc", uniaxial=("Concrete01", {
        "fpc": -30.0, "epsc0": -0.002, "fpcu": -6.0, "epsU": -0.01,
    }))
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_boolean_embed():
    _qapp()
    win = _win()
    for sid, b, h in (("outer", 10, 10), ("inner", 4, 4)):
        win._shape_combo.setCurrentText("rect_face")
        win._shape_id.setText(sid)
        win._shape_param_fields["b"].setText(str(b))
        win._shape_param_fields["h"].setText(str(h))
        win._shape_tx.setText("0")
        win._shape_ty.setText("0")
        assert win.add_shape_from_form()
    win._bool_op.setCurrentText("embed")
    win._bool_a.setText("outer")
    win._bool_b.setText("inner")
    assert win.add_boolean_from_form() is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.add_shape("rect_face", id="outer", b=10.0, h=10.0)
    ref.add_shape("rect_face", id="inner", b=4.0, h=4.0)
    ref.add_embed("outer", "inner")
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_bar_overlay():
    _qapp()
    win = _win()
    win._bar_mat.setText("rebar")
    win._bar_x.setText("1")
    win._bar_y.setText("-2")
    win._bar_area.setText("0.5")
    assert win.add_bar_from_form() is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.add_bar(material="rebar", x=1.0, y=-2.0, area=0.5)
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_mesh_and_policy():
    _qapp()
    win = _win()
    win._mesh_lc.setText("0.5")
    win._mesh_order.setCurrentText("1")
    win._disc.setCurrentText("sum")
    assert win.apply_mesh_from_form() is True

    ref = SectionDocument.new(name="t", kind="continuum")
    ref.set_mesh(lc=0.5, order=1)
    ref.set_disconnected("sum")
    assert win.doc.to_dict() == ref.to_dict()


def test_parity_fiber_items():
    _qapp()
    win = _win(SectionDocument.new(name="t", kind="fiber"))
    # a rect patch
    win._fiber_kind.setCurrentText("rect patch")
    for k, v in (("material", "conc"), ("ny", "8"), ("nz", "8"),
                 ("yI", "-5"), ("zI", "-5"), ("yJ", "5"), ("zJ", "5")):
        win._fiber_fields[k].setText(v)
    assert win.add_fiber_from_form()
    # a bar layer
    win._fiber_kind.setCurrentText("layer")
    for k, v in (("material", "steel"), ("n_bars", "3"), ("area", "0.2"),
                 ("yI", "-4"), ("zI", "-4"), ("yJ", "-4"), ("zJ", "4")):
        win._fiber_fields[k].setText(v)
    assert win.add_fiber_from_form()

    ref = SectionDocument.new(name="t", kind="fiber")
    ref.add_patch_rect(material="conc", ny=8, nz=8,
                       yI=-5.0, zI=-5.0, yJ=5.0, zJ=5.0)
    ref.add_layer_straight(material="steel", n_bars=3, area=0.2,
                           yI=-4.0, zI=-4.0, yJ=-4.0, zJ=4.0)
    assert win.doc.to_dict() == ref.to_dict()


def test_invalid_input_leaves_document_untouched():
    _qapp()
    win = _win()
    before = win.doc.to_dict()
    win._shape_combo.setCurrentText("rect_face")
    win._shape_id.setText("r")
    win._shape_param_fields["b"].setText("not a number")
    win._shape_param_fields["h"].setText("2")
    assert win.add_shape_from_form() is False
    assert win.doc.to_dict() == before        # unchanged
    assert win._undo == []                     # no undo entry recorded


# ─────────────────────────────────────────────────────────────────────
# undo / redo, open / save
# ─────────────────────────────────────────────────────────────────────


def test_undo_redo_roundtrip():
    _qapp()
    win = _win()
    empty = win.doc.to_dict()
    win._shape_combo.setCurrentText("rect_face")
    win._shape_id.setText("r")
    win._shape_param_fields["b"].setText("2")
    win._shape_param_fields["h"].setText("2")
    win.add_shape_from_form()
    after = win.doc.to_dict()
    assert after != empty

    win.undo()
    assert win.doc.to_dict() == empty
    win.redo()
    assert win.doc.to_dict() == after


def test_open_save_roundtrip(tmp_path):
    _qapp()
    win = _win()
    win._shape_combo.setCurrentText("rect_face")
    win._shape_id.setText("r")
    win._shape_param_fields["b"].setText("3")
    win._shape_param_fields["h"].setText("1")
    win.add_shape_from_form()
    p = tmp_path / "sec.section.json"
    win.save_document(p)

    win2 = _win()
    win2.open_document(p)
    assert win2.doc.to_dict() == win.doc.to_dict()


def test_open_switches_lane_and_rebuilds_palette(tmp_path):
    _qapp()
    fiber = SectionDocument.new(name="f", kind="fiber")
    fiber.add_point(material="c", y=0.0, z=0.0, area=1.0)
    fiber.set_material("c", uniaxial=("Elastic", {"E": 1.0}))
    p = tmp_path / "f.section.json"
    fiber.save(p)

    win = _win()                       # starts continuum
    assert hasattr(win, "_shape_combo")
    win.open_document(p)
    assert win.doc.kind == "fiber"
    assert hasattr(win, "_fiber_kind")  # fiber palette built


# ─────────────────────────────────────────────────────────────────────
# drafting-aid toggles + shortcut law
# ─────────────────────────────────────────────────────────────────────


def test_toggles_flip_state_and_status():
    _qapp()
    win = _win()
    assert (win._grid_on, win._snap_on, win._ortho_on) == (False, True, False)
    win.toggle_grid()
    win.toggle_snap()
    win.toggle_ortho()
    assert (win._grid_on, win._snap_on, win._ortho_on) == (True, False, True)
    text = win._status_label.text()
    assert "GRID:ON" in text and "SNAP:off" in text and "ORTHO:ON" in text


def test_shortcuts_use_application_context():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")  # noqa: F841
    _qapp()
    from qtpy import QtCore
    try:
        from qtpy.QtGui import QShortcut
    except ImportError:  # pragma: no cover
        from qtpy.QtWidgets import QShortcut
    win = _win()
    shortcuts = win.window.findChildren(QShortcut)
    keys = {sc.key().toString() for sc in shortcuts}
    assert {"F7", "F8", "F9"} <= keys
    # the canvas-swallow law: F7/F8/F9 must be ApplicationShortcut
    for sc in shortcuts:
        if sc.key().toString() in {"F7", "F8", "F9"}:
            assert sc.context() == QtCore.Qt.ApplicationShortcut


# ─────────────────────────────────────────────────────────────────────
# cursor resolution through the window (snap / ortho / lock composition)
# ─────────────────────────────────────────────────────────────────────


def test_resolve_cursor_snaps_to_existing_vertex():
    _qapp()
    win = _win()
    win.commit_polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], id="p")
    win.start_polygon()
    # cursor near the (10,0) vertex → snaps exactly onto it
    got = win._resolve_cursor(10.05, 0.05)
    assert got == pytest.approx((10.0, 0.0))


def test_resolve_cursor_ortho_and_angle_lock():
    _qapp()
    win = _win()
    win._snap_on = False
    win.start_polygon()
    win._poly_points.append((0.0, 0.0))
    win._ortho_on = True
    assert win._resolve_cursor(5.0, 0.4) == pytest.approx((5.0, 0.0))
    # a locked angle wins over ortho
    win._lock_angle = 90.0
    assert win._resolve_cursor(5.0, 3.0) == pytest.approx((0.0, 3.0))


# ─────────────────────────────────────────────────────────────────────
# screenshot smoke (QTimer)
# ─────────────────────────────────────────────────────────────────────


def test_screenshot_smoke():
    app = _qapp()
    from qtpy import QtCore

    win = _win()
    win._shape_combo.setCurrentText("rect_face")
    win._shape_id.setText("r")
    win._shape_param_fields["b"].setText("4")
    win._shape_param_fields["h"].setText("2")
    win.add_shape_from_form()
    try:
        win.show()
        app.processEvents()
        shot = win.window.grab()
        assert not shot.isNull() and shot.width() > 0

        fired = {"n": 0}

        def _tick() -> None:
            fired["n"] += 1
            win.toggle_grid()

        QtCore.QTimer.singleShot(0, _tick)
        app.processEvents()
        shot2 = win.window.grab()
        assert not shot2.isNull() and shot2.width() > 0
        assert fired["n"] == 1
    finally:
        win.close()


def test_fiber_canvas_renders():
    """A fiber document with items draws without a session (no solve)."""
    app = _qapp()
    doc = SectionDocument.new(name="f", kind="fiber")
    doc.set_material("c", uniaxial=("Elastic", {"E": 1.0}))
    doc.add_patch_rect(material="c", ny=4, nz=4,
                       yI=-2.0, zI=-2.0, yJ=2.0, zJ=2.0)
    win = _win(doc)
    try:
        win.show()
        app.processEvents()
        assert not win.window.grab().isNull()
    finally:
        win.close()
