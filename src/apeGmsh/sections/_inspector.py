"""Section inspector — standalone Qt + matplotlib panel (ADR 0078 S6).

Deliberately **not** part of the ADR 0014/0042/0056 viewer family: no
``model.h5``, no SceneLayer IR, no render backend, no event dispatcher.
A cross-section is a small static 2-D domain — the right tool is one
window with an embedded matplotlib canvas (left), tabbed read-only
property tables (right top), and six live load inputs re-blending the
precomputed unit stress fields (right bottom).

The display content lives in a widget built by
:func:`SectionInspectorPanel` — so the ADR 0080 B6 builder can **embed**
the same panel (fed by its worker-thread build results) rather than
fork it.  :class:`SectionInspectorWindow` is a thin ``QMainWindow``
wrapper around one panel.  The panel's ``QWidget`` base is bound lazily
(``_panel_class``) so this module imports without Qt.

Contract (mirrors ``results.viewer``):

* ``sec.viewer()`` blocks until the window closes; **notebooks must
  pass** ``blocking=False`` — a blocking Qt loop kills the kernel.
  ``blocking=False`` shows the window and returns immediately (in
  Jupyter, enable the Qt event-loop integration with ``%gui qt``).
* Qt absent → ``ImportError`` with install guidance.
* ``QT_QPA_PLATFORM=offscreen`` on Windows → ``RuntimeError`` from the
  launch path (same guard as ``ViewerWindow``); the window *class*
  itself stays constructible offscreen for screenshot tests.
* **No solve on the UI thread** — every analysis (geometric, warping,
  plastic when available, the unit stress fields) runs in
  :func:`precompute_analyses` before the window is constructed (the
  inspector) or in a worker thread (the B6 builder); the load
  spinboxes only re-blend cached unit fields.
"""
from __future__ import annotations

from dataclasses import fields as _dc_fields
from typing import TYPE_CHECKING, Any, Mapping

from ._errors import SectionAnalysisError

if TYPE_CHECKING:  # pragma: no cover
    from ._analysis import SectionProperties
    from ._stress import SectionStress

#: windows opened with ``blocking=False`` are pinned here so they are
#: not garbage-collected the moment ``launch_inspector`` returns.
_LIVE_INSPECTORS: list["SectionInspectorWindow"] = []

_LOAD_KEYS = ("N", "Vx", "Vy", "Mxx", "Myy", "Mzz")
_COMPONENTS = (
    "geometry", "von_mises", "sigma_zz", "tau", "tau_zx", "tau_zy",
)


def _import_qt() -> Any:
    """Import qtpy.QtWidgets with the Qt-absent guidance message."""
    try:
        from qtpy import QtWidgets
    except ImportError as exc:
        raise ImportError(
            "sec.viewer() needs Qt. Install the viewer extra "
            "(pip install apeGmsh[viewer]) or a Qt binding + qtpy "
            "(e.g. pip install qtpy PySide6). Headless alternatives: "
            "sec.summary(), sec.plot_section(), sec.stress(...).plot()."
        ) from exc
    return QtWidgets


def launch_inspector(
    analysis: "SectionProperties", *, blocking: bool = True
) -> "SectionInspectorWindow":
    """Open the inspector for one analyzer (``sec.viewer()`` backend).

    Runs every available analysis **before** any Qt object exists, then
    constructs and shows the window.  ``blocking=True`` enters the Qt
    event loop; ``blocking=False`` returns immediately with the window
    alive (notebooks: ``%gui qt``).
    """
    QtWidgets = _import_qt()

    from apeGmsh.viewers.ui._qt_env import prepare_qt_environment
    prepare_qt_environment()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    import sys
    if (
        sys.platform == "win32"
        and app.platformName().lower() == "offscreen"
    ):
        raise RuntimeError(
            "Qt is running on the 'offscreen' platform, which cannot "
            "host the section inspector window on Windows. Unset "
            "QT_QPA_PLATFORM (or start a fresh process without it) to "
            "open the inspector."
        )

    # ── all solves happen here, before the window exists ────────────
    stress_available = precompute_analyses(analysis)

    win = SectionInspectorWindow(
        analysis, stress_available=stress_available,
    )
    win.show()
    if blocking:
        app.exec_()
    else:
        _LIVE_INSPECTORS.append(win)
    return win


def precompute_analyses(analysis: "SectionProperties") -> bool:
    """Run every available analysis (geometric/warping/plastic + the
    unit stress fields) on ``analysis`` and report whether stress
    recovery is available.

    This is the **no-solve-on-the-UI-thread** boundary: the inspector
    runs it before constructing Qt; the ADR 0080 B6 builder runs it in a
    worker thread. Returns ``stress_available``.
    """
    analysis.analyze()
    try:
        analysis.stress()          # precompute the unit stress fields
        return True
    except SectionAnalysisError:
        return False               # disconnected="sum" — no recovery


# ─────────────────────────────────────────────────────────────────────
# Panel — all display + live-blend logic, as an embeddable QWidget
# ─────────────────────────────────────────────────────────────────────


class _InspectorPanelMixin:
    """Display content for one analyzed section: matplotlib canvas
    (left) + property tabs and the re-blend loads box (right).

    Construct only after the analyses are cached (via
    :func:`precompute_analyses`) — the constructor reads cached results
    and re-blends; it never solves. Mixed with ``QWidget`` at runtime by
    :func:`_panel_class` so the module imports without Qt.
    """

    def __init__(
        self,
        analysis: "SectionProperties",
        *,
        stress_available: bool = True,
    ) -> None:
        QtWidgets = _import_qt()
        super().__init__()

        self._analysis = analysis
        self._stress_available = stress_available

        layout = QtWidgets.QHBoxLayout(self)

        self._canvas = self._make_canvas()
        layout.addWidget(self._canvas, stretch=3)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, stretch=2)

        self._tabs = QtWidgets.QTabWidget()
        right.addWidget(self._tabs, stretch=3)
        self._build_property_tabs(QtWidgets)

        loads_box = QtWidgets.QGroupBox("Loads (re-blend, no re-solve)")
        form = QtWidgets.QFormLayout(loads_box)
        self._spin: dict[str, Any] = {}
        for key in _LOAD_KEYS:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1e18, 1e18)
            sb.setDecimals(6)
            sb.setValue(0.0)
            sb.setEnabled(stress_available)
            sb.valueChanged.connect(self._redraw)
            form.addRow(key, sb)
            self._spin[key] = sb
        self._component = QtWidgets.QComboBox()
        self._component.addItems(list(_COMPONENTS))
        self._component.setEnabled(True)
        self._component.currentTextChanged.connect(self._redraw)
        form.addRow("component", self._component)
        if not stress_available:
            loads_box.setToolTip(
                "Stress recovery is unavailable for this section "
                "(disconnected='sum') — analyze the parts separately."
            )
        right.addWidget(loads_box, stretch=1)

        self._redraw()

    # ── construction helpers ────────────────────────────────────────

    def _make_canvas(self) -> Any:
        from matplotlib.figure import Figure
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
        except ImportError:  # matplotlib < 3.5
            from matplotlib.backends.backend_qt5agg import (  # type: ignore[no-redef]
                FigureCanvasQTAgg as FigureCanvas,
            )
        self._figure = Figure(figsize=(6.0, 6.0), layout="constrained")
        return FigureCanvas(self._figure)

    def _build_property_tabs(self, QtWidgets: Any) -> None:
        geo = self._analysis.geometric()
        self._add_table_tab(QtWidgets, "Geometric", geo,
                            skip=("e_ref", "material_areas"))
        warp = self._analysis._warping
        if warp is not None:
            self._add_table_tab(QtWidgets, "Warping", warp,
                                skip=("e_ref", "g_ref", "parts"))
        plas = self._analysis._plastic
        if plas is not None:
            self._add_table_tab(QtWidgets, "Plastic", plas, skip=())

    def _add_table_tab(
        self, QtWidgets: Any, title: str, obj: Any, *, skip: tuple
    ) -> None:
        """One read-only (field, value) table; the Geometric tab of a
        composite section gains an ``e_ref`` input driving a
        transformed column (rigidity / e_ref)."""
        rows = [
            (f.name, getattr(obj, f.name))
            for f in _dc_fields(obj)
            if f.name not in skip and getattr(obj, f.name) is not None
        ]
        composite_geo = title == "Geometric" and obj.e_ref is None
        n_cols = 3 if composite_geo else 2
        table = QtWidgets.QTableWidget(len(rows), n_cols)
        headers = ["field", "value"] + (
            ["/ e_ref"] if composite_geo else []
        )
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        for r, (name, value) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            table.setItem(
                r, 1, QtWidgets.QTableWidgetItem(f"{value:.6g}")
            )
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()

        if not composite_geo:
            self._tabs.addTab(table, title)
            return

        # composite Geometric tab: e_ref spinbox + transformed column
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("e_ref"))
        e_ref = QtWidgets.QDoubleSpinBox()
        e_ref.setRange(1e-12, 1e18)
        e_ref.setDecimals(6)
        e_ref.setValue(max(m.E for m in self._analysis.materials.values()))
        ref_row.addWidget(e_ref)
        ref_row.addStretch(1)
        v.addLayout(ref_row)
        v.addWidget(table)

        field_names = [name for name, _ in rows]
        rigidity = set(type(obj)._E_FIELDS)

        def _fill_transformed() -> None:
            t = obj.transformed(e_ref.value())
            for r, name in enumerate(field_names):
                text = (
                    f"{getattr(t, name):.6g}" if name in rigidity else ""
                )
                table.setItem(
                    r, 2, QtWidgets.QTableWidgetItem(text)
                )
            table.resizeColumnsToContents()

        e_ref.valueChanged.connect(_fill_transformed)
        self._e_ref_input = e_ref
        _fill_transformed()
        self._tabs.addTab(page, title)

    # ── live blending ───────────────────────────────────────────────

    def loads(self) -> dict[str, float]:
        """Current spinbox values as a ``stress()`` kwargs dict."""
        return {k: float(sb.value()) for k, sb in self._spin.items()}

    def _stress_state(
        self, loads: Mapping[str, float] | None = None
    ) -> "SectionStress":
        """The panel's stress path: a pure re-blend of the cached unit
        fields via :meth:`SectionProperties.stress` — never a solve."""
        return self._analysis.stress(**(dict(loads) if loads is not None
                                        else self.loads()))

    def _redraw(self, *_args: object) -> None:
        component = self._component.currentText()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        if component == "geometry" or not self._stress_available:
            self._analysis.plot_section(
                ax=ax,
                shear_centre=self._analysis._warping is not None,
            )
        else:
            self._stress_state().plot(component, ax=ax)
        self._canvas.draw_idle()


#: the runtime ``QWidget`` subclass of :class:`_InspectorPanelMixin`,
#: built once on first construction (kept out of import time).
_PANEL_CLASS: Any = None


def _panel_class() -> Any:
    global _PANEL_CLASS
    if _PANEL_CLASS is None:
        QtWidgets = _import_qt()
        _PANEL_CLASS = type(
            "SectionInspectorPanel",
            (_InspectorPanelMixin, QtWidgets.QWidget),
            {},
        )
    return _PANEL_CLASS


def SectionInspectorPanel(
    analysis: "SectionProperties", *, stress_available: bool = True
) -> Any:
    """Construct the embeddable inspector panel (a ``QWidget``) for an
    already-analyzed section. Factory, not a plain class, because the
    ``QWidget`` base is bound lazily to keep this module import-safe
    without Qt."""
    return _panel_class()(analysis, stress_available=stress_available)


# ─────────────────────────────────────────────────────────────────────
# Window (thin QMainWindow wrapper around one panel)
# ─────────────────────────────────────────────────────────────────────


class SectionInspectorWindow:
    """The inspector window.  Construct only after the analyses are
    cached (``analysis.analyze()`` + optional ``analysis.stress()``) —
    :func:`launch_inspector` guarantees that; direct constructors (the
    screenshot tests) must do the same."""

    def __init__(
        self,
        analysis: "SectionProperties",
        *,
        stress_available: bool = True,
    ) -> None:
        QtWidgets = _import_qt()

        self._analysis = analysis
        self._stress_available = stress_available

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(
            f"Section inspector — {analysis.name or 'section'}"
        )
        self._panel = SectionInspectorPanel(
            analysis, stress_available=stress_available,
        )
        self.window.setCentralWidget(self._panel)

        # expose the panel's widgets/methods (the S6 test surface +
        # back-compat with callers that reached into the window)
        self._canvas = self._panel._canvas
        self._figure = self._panel._figure
        self._tabs = self._panel._tabs
        self._spin = self._panel._spin
        self._component = self._panel._component
        if hasattr(self._panel, "_e_ref_input"):
            self._e_ref_input = self._panel._e_ref_input

        self.window.resize(1100, 700)

    # display methods delegate to the panel
    def loads(self) -> dict[str, float]:
        return self._panel.loads()

    def _stress_state(
        self, loads: Mapping[str, float] | None = None
    ) -> "SectionStress":
        return self._panel._stress_state(loads)

    def _redraw(self, *args: object) -> None:
        self._panel._redraw(*args)

    # ── window plumbing ─────────────────────────────────────────────

    def show(self) -> None:
        self.window.show()

    def close(self) -> None:
        self.window.close()
        if self in _LIVE_INSPECTORS:
            _LIVE_INSPECTORS.remove(self)


__all__ = [
    "SectionInspectorPanel",
    "SectionInspectorWindow",
    "launch_inspector",
    "precompute_analyses",
]
