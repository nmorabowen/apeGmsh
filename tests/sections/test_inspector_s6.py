"""Tests — ADR 0078 S6: section inspector (`sec.viewer()`).

Contract checks from the plan: Qt-absent ``ImportError`` guidance,
the win32 offscreen-platform raise-guard on the launch path,
blend-equals-``stress()`` identity through the panel's code path,
no-solve-on-the-UI-thread, composite ``e_ref`` transformed column, and
an offscreen screenshot smoke test.  No test enters a blocking Qt
event loop.

Qt widgets here are matplotlib-only (no VTK), so — like
``tests/viewers/test_viewer_window_extensions.py`` — the file pins
``QT_QPA_PLATFORM=offscreen`` and runs unmarked: offscreen widget
construction and ``grab()`` are safe on every platform; the win32
crash guard being *raised* by the launch path is itself under test.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apeGmsh.sections import SectionMaterial, SectionProperties  # noqa: E402


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


def _rect_section(g, *, fy: float | None = 250.0) -> SectionProperties:
    tag = g.model.geometry.add_rectangle(-1.0, -2.0, 0.0, 2.0, 4.0)
    g.physical.add(2, [tag], name="bar")
    fem = _mesh(g, lc=0.4)
    return SectionProperties(
        fem,
        materials={"bar": SectionMaterial(E=200e3, nu=0.3, fy=fy)},
        name="bar",
    )


def _composite_section(g) -> SectionProperties:
    """Two conformal 1×2 strips (shared line) with different moduli."""
    geo = g.model.geometry
    pts = {}
    for i in range(3):
        for jy, y in enumerate((0.0, 2.0)):
            pts[(i, jy)] = geo.add_point(float(i), y, 0.0)
    verts = [geo.add_line(pts[(i, 0)], pts[(i, 1)]) for i in range(3)]
    bots = [geo.add_line(pts[(i, 0)], pts[(i + 1, 0)]) for i in range(2)]
    tops = [geo.add_line(pts[(i, 1)], pts[(i + 1, 1)]) for i in range(2)]
    for i in range(2):
        loop = geo.add_curve_loop([bots[i], verts[i + 1], tops[i], verts[i]])
        surf = geo.add_plane_surface([loop])
        g.physical.add(2, [surf], name=("soft", "stiff")[i])
    fem = _mesh(g, lc=0.3)
    return SectionProperties(
        fem,
        materials={
            "soft": SectionMaterial(E=25e3, nu=0.2),
            "stiff": SectionMaterial(E=200e3, nu=0.3),
        },
        name="duo",
    )


def _qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if app.platformName().lower() != "offscreen":
        pytest.skip("another test created a non-offscreen QApplication")
    return app


# ─────────────────────────────────────────────────────────────────────
# guards
# ─────────────────────────────────────────────────────────────────────

def test_qt_absent_import_guard(g, monkeypatch):
    sec = _rect_section(g)
    monkeypatch.setitem(sys.modules, "qtpy", None)
    with pytest.raises(ImportError, match=r"apeGmsh\[viewer\]"):
        sec.viewer()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the offscreen crash guard is win32-only (ViewerWindow parity)",
)
def test_offscreen_guard_raises_on_launch(g):
    _qapp()
    sec = _rect_section(g)
    with pytest.raises(RuntimeError, match="offscreen"):
        sec.viewer(blocking=False)


# ─────────────────────────────────────────────────────────────────────
# panel behavior (window constructed directly — bypasses the guard)
# ─────────────────────────────────────────────────────────────────────

def _make_window(sec):
    from apeGmsh.sections._inspector import SectionInspectorWindow

    sec.analyze()
    sec.stress()
    return SectionInspectorWindow(sec, stress_available=True)


def test_blend_equals_stress_through_panel_path(g):
    pytest.importorskip("matplotlib")
    _qapp()
    import numpy as np

    sec = _rect_section(g)
    win = _make_window(sec)
    try:
        loads = {"N": -800.0, "Vx": 0.0, "Vy": 350.0,
                 "Mxx": 1.9e6, "Myy": 0.0, "Mzz": 2.5e5}
        for k, v in loads.items():
            win._spin[k].setValue(v)
        panel_state = win._stress_state()
        direct = sec.stress(**loads)
        for comp in ("sigma_zz", "tau_zx", "tau_zy", "von_mises"):
            np.testing.assert_allclose(
                panel_state.get(comp), direct.get(comp),
                rtol=0.0, atol=0.0, equal_nan=True,
            )
    finally:
        win.close()


def test_no_solve_on_the_ui_thread(g, monkeypatch):
    """Constructing the panel and driving loads/component edits never
    re-runs a solve — the spinboxes only re-blend cached unit fields."""
    pytest.importorskip("matplotlib")
    _qapp()
    import apeGmsh.sections._analysis as analysis_mod

    sec = _rect_section(g)
    sec.analyze()
    sec.stress()

    calls = {"warping": 0, "unit_fields": 0}
    monkeypatch.setattr(
        analysis_mod, "compute_warping",
        lambda *a, **k: calls.__setitem__("warping", 99),
    )
    monkeypatch.setattr(
        analysis_mod, "compute_unit_fields",
        lambda *a, **k: calls.__setitem__("unit_fields", 99),
    )

    from apeGmsh.sections._inspector import SectionInspectorWindow
    win = SectionInspectorWindow(sec, stress_available=True)
    try:
        win._component.setCurrentText("von_mises")
        win._spin["Mxx"].setValue(1.0e6)
        win._spin["Vy"].setValue(2.0e3)
    finally:
        win.close()
    assert calls == {"warping": 0, "unit_fields": 0}


def test_tabs_and_composite_e_ref_column(g):
    pytest.importorskip("matplotlib")
    _qapp()

    sec = _composite_section(g)   # no fy -> no plastic tab
    win = _make_window(sec)
    try:
        titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert titles == ["Geometric", "Warping"]
        # composite Geometric tab carries the e_ref transformed column
        assert hasattr(win, "_e_ref_input")
        e_ref = win._e_ref_input
        assert e_ref.value() == pytest.approx(200e3)
        page = win._tabs.widget(0)
        from qtpy import QtWidgets
        table = page.findChild(QtWidgets.QTableWidget)
        assert table is not None and table.columnCount() == 3
        # EA row: transformed value == EA / e_ref
        geo = sec.geometric()
        for r in range(table.rowCount()):
            if table.item(r, 0).text() == "EA":
                shown = float(table.item(r, 2).text())
                assert shown == pytest.approx(geo.EA / e_ref.value(),
                                              rel=1e-5)
                break
        else:
            pytest.fail("EA row not found in the Geometric table")
    finally:
        win.close()


def test_plastic_tab_appears_with_fy(g):
    pytest.importorskip("matplotlib")
    _qapp()

    sec = _rect_section(g, fy=250.0)
    win = _make_window(sec)
    try:
        titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert titles == ["Geometric", "Warping", "Plastic"]
    finally:
        win.close()


def test_screenshot_smoke(g):
    """Offscreen render smoke: show, process events, grab a non-empty
    pixmap in both the geometry view and a stress-contour view."""
    pytest.importorskip("matplotlib")
    app = _qapp()

    sec = _rect_section(g)
    win = _make_window(sec)
    try:
        win.show()
        app.processEvents()
        shot = win.window.grab()
        assert not shot.isNull() and shot.width() > 0

        win._spin["Mxx"].setValue(1.0e6)
        win._component.setCurrentText("von_mises")
        app.processEvents()
        shot2 = win.window.grab()
        assert not shot2.isNull() and shot2.width() > 0
    finally:
        win.close()
