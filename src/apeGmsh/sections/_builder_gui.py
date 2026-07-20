"""Section builder GUI — standalone Qt + matplotlib shell (ADR 0080 B5).

An **editor for a** :class:`~apeGmsh.sections.SectionDocument`. The
document is the source of truth and the whole headless API; this window
is one client of it. The **parity law** governs every mutation: each
GUI action calls the matching ``SectionDocument`` method, so a script
can do anything the GUI can — the offscreen widget tests assert
document-dict equality between the two paths.

Inspector mold (ADR 0078 S6), deliberately **not** the ADR
0014/0042/0056 viewer family: one ``QMainWindow`` with a matplotlib
canvas (left) and a palette of shape/material/boolean/mesh forms
(right); the freehand polygon tool with AutoCAD-style drafting aids
(grid + object snap, ortho, typed length/angle input) driven by the
Qt-free :mod:`._drafting` primitives; undo/redo as document JSON
snapshots; open/save ``.section.json``.

**No solves in B5** — the window only edits the document (the live
properties panel and worker-thread builds are B6). The canvas draws
``plot_faces``-style outlines for the continuum lane and
patch/layer/point glyphs for the fiber lane, both resolved without a
Gmsh session.

Contract (mirrors ``sec.viewer()``):

* ``launch_builder()`` blocks until the window closes; **notebooks
  must pass** ``blocking=False``.
* Qt absent → ``ImportError`` with install guidance.
* ``QT_QPA_PLATFORM=offscreen`` on Windows → ``RuntimeError`` from the
  launch path; the window *class* stays offscreen-constructible for
  the screenshot tests.
* Status-bar ``GRID`` / ``SNAP`` / ``ORTHO`` toggles on F7 / F9 / F8,
  registered with ``Qt.ApplicationShortcut`` context — a canvas-focused
  widget swallows ``WindowShortcut`` (the established Qt-canvas
  shortcut law).
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._document import _SHAPE_PARAMS, SectionDocument, SectionDocumentError
from ._properties import PropertiesController
from ._drafting import (
    GridSpec,
    constrain_segment,
    ortho_project,
    resolve_snap,
    shape_outlines,
    snap_candidates,
)

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

#: windows opened with ``blocking=False`` are pinned so they survive
#: past :func:`launch_builder` returning.
_LIVE_BUILDERS: list["SectionBuilderWindow"] = []

#: fiber-lane snap/quadrant marker glyphs (kind → matplotlib marker).
_SNAP_MARKER = {
    "endpoint": "s", "midpoint": "^", "center": "o",
    "quadrant": "D", "intersection": "X", "grid": "+",
}


def _import_qt() -> "tuple[Any, Any, Any]":
    """Import ``(QtWidgets, QtCore, QtGui)`` with Qt-absent guidance."""
    try:
        from qtpy import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise ImportError(
            "launch_builder() needs Qt. Install the viewer extra "
            "(pip install apeGmsh[viewer]) or a Qt binding + qtpy "
            "(e.g. pip install qtpy PySide6). Headless alternative: "
            "author the SectionDocument in code (doc.add_shape(...), "
            "doc.save(...))."
        ) from exc
    return QtWidgets, QtCore, QtGui


def _qshortcut() -> Any:
    """``QShortcut`` moved QtWidgets → QtGui in Qt6; qtpy exposes both."""
    try:
        from qtpy.QtGui import QShortcut
    except ImportError:  # pragma: no cover - very old qtpy
        from qtpy.QtWidgets import QShortcut  # type: ignore[no-redef]
    return QShortcut


def launch_builder(
    path_or_doc: "str | Path | SectionDocument | None" = None,
    *,
    blocking: bool = True,
) -> "SectionBuilderWindow":
    """Open the section builder.

    ``path_or_doc`` is a ``.section.json`` path, an existing
    :class:`SectionDocument`, or ``None`` for a blank continuum
    document. ``blocking=True`` enters the Qt event loop;
    ``blocking=False`` returns immediately with the window alive
    (notebooks: ``%gui qt``).
    """
    QtWidgets, _QtCore, _QtGui = _import_qt()

    from apeGmsh.viewers.ui._qt_env import prepare_qt_environment
    prepare_qt_environment()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    if sys.platform == "win32" and app.platformName().lower() == "offscreen":
        raise RuntimeError(
            "Qt is running on the 'offscreen' platform, which cannot "
            "host the section builder window on Windows. Unset "
            "QT_QPA_PLATFORM (or start a fresh process without it) to "
            "open the builder."
        )

    doc = _coerce_document(path_or_doc)
    win = SectionBuilderWindow(doc)
    win.set_live_properties(True)   # real launches solve live (B6)
    win.show()
    if blocking:
        app.exec_()
    else:
        _LIVE_BUILDERS.append(win)
    return win


def _coerce_document(
    path_or_doc: "str | Path | SectionDocument | None",
) -> SectionDocument:
    if path_or_doc is None:
        return SectionDocument.new(name="section", kind="continuum")
    if isinstance(path_or_doc, SectionDocument):
        return path_or_doc
    return SectionDocument.open(path_or_doc)


# ─────────────────────────────────────────────────────────────────────
# Window
# ─────────────────────────────────────────────────────────────────────


class SectionBuilderWindow:
    """The builder window. Offscreen-constructible (the screenshot
    tests build it directly, bypassing the win32 launch guard)."""

    def __init__(self, doc: SectionDocument) -> None:
        QtWidgets, QtCore, _QtGui = _import_qt()
        self._qt = (QtWidgets, QtCore, _QtGui)

        self.doc = doc
        self._undo: list[dict[str, Any]] = []
        self._redo: list[dict[str, Any]] = []

        # drafting-aid state (GUI-layer only — no document state)
        self._grid_on = False
        self._snap_on = True
        self._ortho_on = False
        self._grid_spec = GridSpec(spacing=1.0)
        self._snap_tol = 0.25
        # in-progress polygon tool
        self._poly_active = False
        self._poly_points: list[tuple[float, float]] = []
        self._poly_seq = 0
        self._lock_length: float | None = None
        self._lock_angle: float | None = None
        self._last_snap: Any = None

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(
            f"Section builder — {doc.name or 'section'} [{doc.kind}]"
        )
        central = QtWidgets.QWidget()
        self._root = QtWidgets.QHBoxLayout(central)
        self.window.setCentralWidget(central)

        self._canvas = self._make_canvas()
        self._root.addWidget(self._canvas, stretch=3)

        self._palette_host = QtWidgets.QWidget()
        self._palette_layout = QtWidgets.QVBoxLayout(self._palette_host)
        self._root.addWidget(self._palette_host, stretch=2)
        self._build_palette()

        # live properties (ADR 0080 B6) — off until the user opts in, so
        # no worker build (and no Gmsh session) runs unless requested.
        self._live_enabled = False
        self._controller: PropertiesController | None = None
        self._last_result: Any = None
        self._build_properties_dock()

        self._make_toolbar()
        self._status = self.window.statusBar()
        self._status_label = QtWidgets.QLabel()
        self._status.addPermanentWidget(self._status_label)
        self._register_shortcuts()

        self.window.resize(1200, 760)
        self._refresh()

    # ── canvas ───────────────────────────────────────────────────────

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
        self._figure = Figure(figsize=(6.5, 6.5), layout="constrained")
        canvas = FigureCanvas(self._figure)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        return canvas

    def _redraw(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_aspect("equal")
        try:
            if self.doc.kind == "continuum":
                self._draw_continuum(ax)
            else:
                self._draw_fiber(ax)
        except Exception as exc:  # never let a repaint kill the window
            ax.set_title(f"(preview unavailable: {exc})", fontsize=8)
        ax.set_title(
            f"{self.doc.name or 'section'} — {self.doc.kind}",
            fontsize=9,
        )
        self._canvas.draw_idle()

    def _draw_continuum(self, ax: Any) -> None:
        import numpy as np

        for sh in self.doc.to_dict().get("shapes", []):
            out = shape_outlines(sh)
            for loop in out.polylines:
                xs = [p[0] for p in loop] + [loop[0][0]]
                ys = [p[1] for p in loop] + [loop[0][1]]
                ax.plot(xs, ys, "-", color="0.2", linewidth=1.1)
            for cx, cy, r in out.circles:
                th = np.linspace(0, 2 * np.pi, 96)
                ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                        "-", color="0.2", linewidth=1.1)
        # bars overlay
        for b in self.doc._expand_bars():
            ax.plot(b["x"], b["y"], "o", color="tab:red", markersize=4)
        # in-progress polygon
        if self._poly_points:
            xs = [p[0] for p in self._poly_points]
            ys = [p[1] for p in self._poly_points]
            ax.plot(xs, ys, "-o", color="tab:green", markersize=4)
        if self._last_snap is not None:
            ax.plot(
                self._last_snap.x, self._last_snap.y,
                _SNAP_MARKER.get(self._last_snap.kind, "o"),
                mfc="none", mec="tab:orange", markersize=10,
            )

    def _draw_fiber(self, ax: Any) -> None:
        import numpy as np

        try:
            recipe = self.doc.build()
        except SectionDocumentError:
            return
        import matplotlib.patches as mpatches

        for p in recipe.patches:
            if p["kind"] == "rect":
                ax.add_patch(mpatches.Rectangle(
                    (min(p["zI"], p["zJ"]), min(p["yI"], p["yJ"])),
                    abs(p["zJ"] - p["zI"]), abs(p["yJ"] - p["yI"]),
                    facecolor="0.85", edgecolor="0.4",
                ))
            else:
                th = np.linspace(np.radians(p.get("start_ang", 0.0)),
                                 np.radians(p.get("end_ang", 360.0)), 96)
                for rr in (p["int_rad"], p["ext_rad"]):
                    ax.plot(p["zC"] + rr * np.cos(th),
                            p["yC"] + rr * np.sin(th), "-", color="0.4")
        for la in recipe.layers:
            n = la["n_bars"]
            ys = np.linspace(la["yI"], la["yJ"], n)
            zs = np.linspace(la["zI"], la["zJ"], n)
            ax.plot(zs, ys, "o", color="tab:red", markersize=4)
        for pt in recipe.points:
            ax.plot(pt["z"], pt["y"], "o", color="tab:red", markersize=4)
        ax.set_xlabel("z")
        ax.set_ylabel("y")

    # ── palette ──────────────────────────────────────────────────────

    def _build_palette(self) -> None:
        QtWidgets, _QtCore, _QtGui = self._qt
        # clear any previous palette (lane switch on open)
        while self._palette_layout.count():
            item = self._palette_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        if self.doc.kind == "continuum":
            self._build_shape_group(QtWidgets)
            self._build_boolean_group(QtWidgets)
            self._build_bars_group(QtWidgets)
            self._build_mesh_group(QtWidgets)
        else:
            self._build_fiber_group(QtWidgets)
        self._build_material_group(QtWidgets)
        self._palette_layout.addStretch(1)

    def _line(self, text: str = "") -> Any:
        QtWidgets = self._qt[0]
        le = QtWidgets.QLineEdit()
        le.setText(text)
        return le

    def _build_shape_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Shape")
        v = QtWidgets.QVBoxLayout(box)
        self._shape_combo = QtWidgets.QComboBox()
        self._shape_combo.addItems([*_SHAPE_PARAMS.keys(), "polygon"])
        v.addWidget(self._shape_combo)
        self._shape_form_host = QtWidgets.QWidget()
        self._shape_form = QtWidgets.QFormLayout(self._shape_form_host)
        v.addWidget(self._shape_form_host)
        self._shape_param_fields: dict[str, Any] = {}
        self._shape_combo.currentTextChanged.connect(self._rebuild_shape_form)
        add = QtWidgets.QPushButton("Add shape")
        add.clicked.connect(self.add_shape_from_form)
        v.addWidget(add)
        self._palette_layout.addWidget(box)
        self._rebuild_shape_form()

    def _rebuild_shape_form(self, *_a: object) -> None:
        # QFormLayout.removeRow deletes the row's widgets, so every field
        # (shared and per-shape) is recreated fresh here — reusing a
        # stored QLineEdit across rebuilds would touch a deleted C++
        # object.
        QtWidgets = self._qt[0]
        while self._shape_form.rowCount():
            self._shape_form.removeRow(0)
        self._shape_param_fields = {}
        shape = self._shape_combo.currentText()
        self._shape_id = self._line()
        self._shape_form.addRow("id", self._shape_id)
        if shape == "polygon":
            note = QtWidgets.QLabel(
                "click the canvas to place vertices;\nEnter commits, "
                "Esc cancels"
            )
            note.setWordWrap(True)
            self._shape_form.addRow(note)
        else:
            for k in _SHAPE_PARAMS[shape]:
                f = self._line()
                self._shape_param_fields[k] = f
                self._shape_form.addRow(k, f)
        self._shape_mat = self._line()
        self._shape_tx = self._line("0")
        self._shape_ty = self._line("0")
        self._shape_rot = self._line()
        self._shape_form.addRow("material", self._shape_mat)
        self._shape_form.addRow("translate x", self._shape_tx)
        self._shape_form.addRow("translate y", self._shape_ty)
        self._shape_form.addRow("rotate°", self._shape_rot)

    def _build_boolean_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Boolean")
        f = QtWidgets.QFormLayout(box)
        self._bool_op = QtWidgets.QComboBox()
        self._bool_op.addItems(["embed", "cut", "fragment_pair"])
        self._bool_a = self._line()
        self._bool_b = self._line()
        self._bool_remove = QtWidgets.QCheckBox("remove tool (cut)")
        self._bool_remove.setChecked(True)
        f.addRow("op", self._bool_op)
        f.addRow("outer / target / a", self._bool_a)
        f.addRow("inner / tool / b", self._bool_b)
        f.addRow(self._bool_remove)
        btn = QtWidgets.QPushButton("Add boolean")
        btn.clicked.connect(self.add_boolean_from_form)
        f.addRow(btn)
        self._palette_layout.addWidget(box)

    def _build_bars_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Bar (overlay)")
        f = QtWidgets.QFormLayout(box)
        self._bar_mat = self._line()
        self._bar_x = self._line("0")
        self._bar_y = self._line("0")
        self._bar_area = self._line("1")
        f.addRow("material", self._bar_mat)
        f.addRow("x", self._bar_x)
        f.addRow("y", self._bar_y)
        f.addRow("area", self._bar_area)
        btn = QtWidgets.QPushButton("Add bar")
        btn.clicked.connect(self.add_bar_from_form)
        f.addRow(btn)
        self._palette_layout.addWidget(box)

    def _build_mesh_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Mesh / policy")
        f = QtWidgets.QFormLayout(box)
        self._mesh_lc = self._line("")
        self._mesh_order = QtWidgets.QComboBox()
        self._mesh_order.addItems(["2", "1"])
        self._disc = QtWidgets.QComboBox()
        self._disc.addItems(["raise", "sum"])
        f.addRow("lc", self._mesh_lc)
        f.addRow("order", self._mesh_order)
        f.addRow("disconnected", self._disc)
        btn = QtWidgets.QPushButton("Apply mesh/policy")
        btn.clicked.connect(self.apply_mesh_from_form)
        f.addRow(btn)
        self._palette_layout.addWidget(box)

    def _build_fiber_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Fiber item")
        v = QtWidgets.QVBoxLayout(box)
        self._fiber_kind = QtWidgets.QComboBox()
        self._fiber_kind.addItems(
            ["rect patch", "circ patch", "layer", "point"]
        )
        v.addWidget(self._fiber_kind)
        self._fiber_form_host = QtWidgets.QWidget()
        self._fiber_form = QtWidgets.QFormLayout(self._fiber_form_host)
        v.addWidget(self._fiber_form_host)
        self._fiber_fields: dict[str, Any] = {}
        self._fiber_kind.currentTextChanged.connect(self._rebuild_fiber_form)
        add = QtWidgets.QPushButton("Add fiber item")
        add.clicked.connect(self.add_fiber_from_form)
        v.addWidget(add)
        self._palette_layout.addWidget(box)
        self._rebuild_fiber_form()

    _FIBER_SPECS = {
        "rect patch": ("material", "ny", "nz", "yI", "zI", "yJ", "zJ"),
        "circ patch": ("material", "n_circ", "n_rad", "yC", "zC",
                       "int_rad", "ext_rad", "start_ang", "end_ang"),
        "layer": ("material", "n_bars", "area", "yI", "zI", "yJ", "zJ"),
        "point": ("material", "y", "z", "area"),
    }

    def _rebuild_fiber_form(self, *_a: object) -> None:
        while self._fiber_form.rowCount():
            self._fiber_form.removeRow(0)
        self._fiber_fields = {}
        for k in self._FIBER_SPECS[self._fiber_kind.currentText()]:
            f = self._line()
            self._fiber_fields[k] = f
            self._fiber_form.addRow(k, f)

    def _build_material_group(self, QtWidgets: Any) -> None:
        box = QtWidgets.QGroupBox("Material")
        f = QtWidgets.QFormLayout(box)
        self._mat_name = self._line()
        self._mat_E = self._line()
        self._mat_nu = self._line()
        self._mat_G = self._line()
        self._mat_fy = self._line()
        self._mat_density = self._line()
        self._mat_uni_type = self._line()
        self._mat_uni_params = self._line()
        for label, w in (
            ("name", self._mat_name), ("E", self._mat_E),
            ("nu", self._mat_nu), ("G", self._mat_G),
            ("fy", self._mat_fy), ("density", self._mat_density),
            ("uniaxial type", self._mat_uni_type),
            ("uniaxial params (k=v,…)", self._mat_uni_params),
        ):
            f.addRow(label, w)
        btn = QtWidgets.QPushButton("Set material")
        btn.clicked.connect(self.apply_material_from_form)
        f.addRow(btn)
        self._palette_layout.addWidget(box)

    # ── toolbar / shortcuts ──────────────────────────────────────────

    def _make_toolbar(self) -> None:
        QtWidgets, _QtCore, _QtGui = self._qt
        tb = self.window.addToolBar("main")
        for text, slot in (
            ("Open", self._open_dialog), ("Save", self._save_dialog),
            ("Undo", self.undo), ("Redo", self.redo),
        ):
            act = tb.addAction(text)
            act.triggered.connect(slot)

    def _register_shortcuts(self) -> None:
        QtWidgets, QtCore, _QtGui = self._qt
        QShortcut = _qshortcut()
        ctx = QtCore.Qt.ApplicationShortcut
        for key, slot in (
            ("F7", self.toggle_grid), ("F9", self.toggle_snap),
            ("F8", self.toggle_ortho),
        ):
            sc = QShortcut(_key_sequence(key), self.window)
            sc.setContext(ctx)
            sc.activated.connect(slot)
        for key, slot in (
            ("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo),
        ):
            sc = QShortcut(_key_sequence(key), self.window)
            sc.setContext(ctx)
            sc.activated.connect(slot)

    # ── mutation plumbing (the parity seam) ──────────────────────────

    def _apply(self, fn: "Any") -> bool:
        """Run one document mutation with undo capture. Returns True on
        success; a :class:`SectionDocumentError` is caught and surfaced
        in the status bar (the document is left untouched)."""
        snapshot = self.doc.to_dict()
        try:
            fn()
        except SectionDocumentError as exc:
            self._flash(f"✗ {exc}")
            return False
        except ValueError as exc:
            self._flash(f"✗ {exc}")
            return False
        self._undo.append(snapshot)
        self._redo.clear()
        self._refresh()
        return True

    def _refresh(self) -> None:
        self._redraw()
        self._update_status()
        self._maybe_request_properties()

    def _flash(self, msg: str) -> None:
        self._status.showMessage(msg, 6000)

    _REQUIRED: Any = object()

    def _read_float(self, w: Any, name: str, *, default: Any = _REQUIRED
                    ) -> "float | None":
        """Read a numeric field. Empty text raises when ``default`` is
        left at the ``_REQUIRED`` sentinel; otherwise ``default`` (which
        may be ``None`` for an optional field) is returned."""
        text = w.text().strip()
        if not text:
            if default is self._REQUIRED:
                raise ValueError(f"{name} is required.")
            return default
        try:
            return float(text)
        except ValueError as e:
            raise ValueError(f"{name}: {text!r} is not a number.") from e

    # ── mutation actions (each == a SectionDocument API call) ────────

    def add_shape_from_form(self) -> bool:
        shape = self._shape_combo.currentText()
        sid = self._shape_id.text().strip()
        mat = self._shape_mat.text().strip() or None
        rot_text = self._shape_rot.text().strip()

        def _do() -> None:
            tx = self._read_float(self._shape_tx, "translate x", default=0.0)
            ty = self._read_float(self._shape_ty, "translate y", default=0.0)
            rotate = float(rot_text) if rot_text else None
            if shape == "polygon":
                raise ValueError(
                    "use the canvas polygon tool (click vertices, Enter) "
                    "for polygons."
                )
            params = {
                k: self._read_float(w, k)
                for k, w in self._shape_param_fields.items()
            }
            self.doc.add_shape(
                shape, id=sid, material=mat,
                translate=(tx, ty), rotate=rotate, **params,
            )
        return self._apply(_do)

    def add_boolean_from_form(self) -> bool:
        op = self._bool_op.currentText()
        a = self._bool_a.text().strip()
        b = self._bool_b.text().strip()
        remove = self._bool_remove.isChecked()

        def _do() -> None:
            if op == "embed":
                self.doc.add_embed(a, b)
            elif op == "cut":
                self.doc.add_cut(a, b, remove_tool=remove)
            else:
                self.doc.add_fragment_pair(a, b)
        return self._apply(_do)

    def add_bar_from_form(self) -> bool:
        mat = self._bar_mat.text().strip()

        def _do() -> None:
            self.doc.add_bar(
                material=mat,
                x=self._read_float(self._bar_x, "x"),
                y=self._read_float(self._bar_y, "y"),
                area=self._read_float(self._bar_area, "area"),
            )
        return self._apply(_do)

    def apply_mesh_from_form(self) -> bool:
        order = int(self._mesh_order.currentText())
        policy = self._disc.currentText()

        def _do() -> None:
            lc = self._read_float(self._mesh_lc, "lc", default=None)
            if lc is not None:
                self.doc.set_mesh(lc=lc, order=order)
            self.doc.set_disconnected(policy)
        return self._apply(_do)

    def add_fiber_from_form(self) -> bool:
        kind = self._fiber_kind.currentText()
        fields = self._fiber_fields
        mat = fields["material"].text().strip()

        def _f(name: str) -> float:
            return self._read_float(fields[name], name)

        def _do() -> None:
            if kind == "rect patch":
                self.doc.add_patch_rect(
                    material=mat, ny=int(_f("ny")), nz=int(_f("nz")),
                    yI=_f("yI"), zI=_f("zI"), yJ=_f("yJ"), zJ=_f("zJ"),
                )
            elif kind == "circ patch":
                self.doc.add_patch_circ(
                    material=mat, n_circ=int(_f("n_circ")),
                    n_rad=int(_f("n_rad")), yC=_f("yC"), zC=_f("zC"),
                    int_rad=_f("int_rad"), ext_rad=_f("ext_rad"),
                    start_ang=_f("start_ang"), end_ang=_f("end_ang"),
                )
            elif kind == "layer":
                self.doc.add_layer_straight(
                    material=mat, n_bars=int(_f("n_bars")), area=_f("area"),
                    yI=_f("yI"), zI=_f("zI"), yJ=_f("yJ"), zJ=_f("zJ"),
                )
            else:
                self.doc.add_point(
                    material=mat, y=_f("y"), z=_f("z"), area=_f("area"),
                )
        return self._apply(_do)

    def apply_material_from_form(self) -> bool:
        name = self._mat_name.text().strip()
        uni_type = self._mat_uni_type.text().strip()
        uni_params_text = self._mat_uni_params.text().strip()

        def _opt(w: Any, label: str) -> "float | None":
            return self._read_float(w, label, default=None)

        def _do() -> None:
            uniaxial = None
            if uni_type:
                params: dict[str, Any] = {}
                if uni_params_text:
                    for pair in uni_params_text.split(","):
                        k, _, v = pair.partition("=")
                        params[k.strip()] = float(v)
                uniaxial = (uni_type, params)
            self.doc.set_material(
                name,
                E=_opt(self._mat_E, "E"), nu=_opt(self._mat_nu, "nu"),
                G=_opt(self._mat_G, "G"), fy=_opt(self._mat_fy, "fy"),
                density=_opt(self._mat_density, "density"),
                uniaxial=uniaxial,
            )
        return self._apply(_do)

    # ── polygon tool (canvas) ────────────────────────────────────────

    def start_polygon(self) -> None:
        self._poly_active = True
        self._poly_points = []
        self._lock_length = self._lock_angle = None
        self._flash("polygon: click vertices, Enter commits, Esc cancels")

    def _resolve_cursor(
        self, x: float, y: float
    ) -> tuple[float, float]:
        """Apply ortho / lock resolver / snap to a raw cursor position,
        honouring the composition law: a locked angle wins over ortho,
        and snap adjusts only the free component (the vertex is
        re-projected onto its lock afterwards)."""
        pt = (x, y)
        anchor = self._poly_points[-1] if self._poly_points else None
        locked = self._lock_length is not None or self._lock_angle is not None
        if anchor is not None:
            if locked:
                pt = constrain_segment(
                    anchor, pt,
                    length=self._lock_length, angle=self._lock_angle,
                )
            elif self._ortho_on:
                pt = ortho_project(anchor, pt)
        if self._snap_on or self._grid_on:
            cands = snap_candidates(self.doc, extra_points=self._poly_points)
            grid = self._grid_spec if self._grid_on else None
            snapped = resolve_snap(pt, cands, grid, self._snap_tol)
            self._last_snap = snapped
            if snapped is not None:
                pt = (snapped.x, snapped.y)
                if anchor is not None and locked:  # keep the lock
                    pt = constrain_segment(
                        anchor, pt,
                        length=self._lock_length, angle=self._lock_angle,
                    )
        else:
            self._last_snap = None
        return pt

    def _on_click(self, event: Any) -> None:  # pragma: no cover - Qt event
        if not self._poly_active or event.xdata is None:
            return
        self._poly_points.append(self._resolve_cursor(event.xdata, event.ydata))
        self._redraw()

    def _on_move(self, event: Any) -> None:  # pragma: no cover - Qt event
        if not self._poly_active or event.xdata is None:
            return
        self._resolve_cursor(event.xdata, event.ydata)
        self._redraw()

    def commit_polygon(
        self, points: "list[tuple[float, float]] | None" = None,
        *, id: "str | None" = None,
    ) -> bool:
        """Commit the in-progress (or supplied) polygon into the
        document — the parity image of :meth:`SectionDocument.add_polygon`.
        Auto-names ``polygon_N`` when ``id`` is omitted."""
        pts = points if points is not None else list(self._poly_points)
        if id is None:
            self._poly_seq += 1
            id = f"polygon_{self._poly_seq}"

        def _do() -> None:
            self.doc.add_polygon(pts, id=id)
        ok = self._apply(_do)
        if ok:
            self._poly_active = False
            self._poly_points = []
            self._last_snap = None
        return ok

    def cancel_polygon(self) -> None:
        self._poly_active = False
        self._poly_points = []
        self._last_snap = None
        self._redraw()

    # ── undo / redo ──────────────────────────────────────────────────

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self.doc.to_dict())
        self.doc = SectionDocument(self._undo.pop())
        self._build_palette()   # lane may have changed
        self._refresh()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self.doc.to_dict())
        self.doc = SectionDocument(self._redo.pop())
        self._build_palette()
        self._refresh()

    # ── open / save ──────────────────────────────────────────────────

    def open_document(self, path: "str | Path") -> None:
        self.doc = SectionDocument.open(path)
        self._undo.clear()
        self._redo.clear()
        self._poly_active = False
        self._poly_points = []
        self.window.setWindowTitle(
            f"Section builder — {self.doc.name or 'section'} "
            f"[{self.doc.kind}]"
        )
        self._build_palette()
        self._refresh()

    def save_document(self, path: "str | Path") -> None:
        self.doc.save(path)
        self._flash(f"saved → {path}")

    def _open_dialog(self) -> None:  # pragma: no cover - Qt dialog
        QtWidgets = self._qt[0]
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Open section document", "",
            "Section documents (*.section.json *.json)",
        )
        if path:
            self.open_document(path)

    def _save_dialog(self) -> None:  # pragma: no cover - Qt dialog
        QtWidgets = self._qt[0]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Save section document", "",
            "Section documents (*.section.json)",
        )
        if path:
            self.save_document(path)

    # ── drafting-aid toggles ─────────────────────────────────────────

    def toggle_grid(self) -> None:
        self._grid_on = not self._grid_on
        self._update_status()

    def toggle_snap(self) -> None:
        self._snap_on = not self._snap_on
        self._update_status()

    def toggle_ortho(self) -> None:
        self._ortho_on = not self._ortho_on
        self._update_status()

    def _update_status(self) -> None:
        def _fmt(name: str, on: bool) -> str:
            mark = "ON" if on else "off"
            return f"{name}:{mark}"
        self._status_label.setText(
            "   ".join((
                _fmt("GRID", self._grid_on),
                _fmt("SNAP", self._snap_on),
                _fmt("ORTHO", self._ortho_on),
            ))
        )

    # ── live properties panel (ADR 0080 B6) ─────────────────────────

    def _build_properties_dock(self) -> None:
        QtWidgets, _QtCore, _QtGui = self._qt
        dock = QtWidgets.QDockWidget("Properties", self.window)
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host)

        controls = QtWidgets.QHBoxLayout()
        self._live_checkbox = QtWidgets.QCheckBox("Live")
        self._live_checkbox.setToolTip(
            "Build + analyze the document on a worker thread after each "
            "edit (never on the UI thread)."
        )
        self._live_checkbox.toggled.connect(self.set_live_properties)
        controls.addWidget(self._live_checkbox)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_properties)
        controls.addWidget(refresh)
        controls.addStretch(1)
        v.addLayout(controls)

        self._props_status = QtWidgets.QLabel("(properties off)")
        v.addWidget(self._props_status)
        self._props_body_host = QtWidgets.QWidget()
        self._props_body = QtWidgets.QVBoxLayout(self._props_body_host)
        v.addWidget(self._props_body_host, stretch=1)

        dock.setWidget(host)
        self.window.addDockWidget(
            _QtCore.Qt.RightDockWidgetArea, dock,
        )
        self._props_dock = dock

    def _ensure_controller(self) -> PropertiesController:
        if self._controller is None:
            self._controller = PropertiesController(
                on_result=self._on_properties,
            )
        return self._controller

    def set_live_properties(self, enabled: bool) -> None:
        """Turn the live properties panel on/off. Enabling fires an
        immediate build of the current document state."""
        self._live_enabled = bool(enabled)
        if self._live_checkbox.isChecked() != self._live_enabled:
            self._live_checkbox.setChecked(self._live_enabled)
        if self._live_enabled:
            self.refresh_properties()
        else:
            self._props_status.setText("(properties off)")

    def _maybe_request_properties(self) -> None:
        if self._live_enabled:
            self._dispatch_properties()

    def refresh_properties(self) -> None:
        """Force a one-shot properties build of the current document
        (independent of the Live toggle)."""
        self._dispatch_properties()

    def _dispatch_properties(self) -> None:
        self._ensure_controller()
        self._props_status.setText("building…")
        self._props_body_host.setEnabled(False)   # grey until fresh
        self._controller.request(self.doc.to_dict())

    def _on_properties(self, result: Any) -> None:
        """UI-thread callback with a fresh build result — repopulate the
        panel. (Runs on the UI thread: the controller marshals here from
        its worker via the drain timer.)"""
        self._last_result = result
        while self._props_body.count():
            item = self._props_body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._props_body_host.setEnabled(True)
        if result.error is not None:
            self._props_status.setText(f"unavailable: {result.error}")
            return
        self._props_status.setText("up to date")
        if result.kind == "continuum":
            from ._inspector import SectionInspectorPanel
            panel = SectionInspectorPanel(
                result.analysis,
                stress_available=result.stress_available,
            )
            self._props_body.addWidget(panel)
        else:
            self._props_body.addWidget(
                self._fiber_identities_table(result.identities)
            )

    def _fiber_identities_table(self, identities: "dict[str, Any]") -> Any:
        QtWidgets = self._qt[0]
        rows: list[tuple[str, str]] = [
            ("total area", f"{identities['total_area']:.6g}"),
            ("patches", str(identities["n_patches"])),
            ("layers", str(identities["n_layers"])),
            ("points", str(identities["n_points"])),
            ("GJ", "—" if identities["GJ"] is None
             else f"{identities['GJ']:.6g}"),
        ]
        for mat, area in sorted(identities["areas_by_material"].items()):
            rows.append((f"area[{mat}]", f"{area:.6g}"))
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["property", "value"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for r, (k, val) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(val))
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        return table

    # ── window plumbing ──────────────────────────────────────────────

    def show(self) -> None:
        self.window.show()

    def close(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller.join(2.0)
        self.window.close()
        if self in _LIVE_BUILDERS:
            _LIVE_BUILDERS.remove(self)


def _key_sequence(key: str) -> Any:
    """A ``QKeySequence`` for ``key`` (kept a free function so the class
    body stays binding-agnostic)."""
    from qtpy.QtGui import QKeySequence
    return QKeySequence(key)


__all__ = ["SectionBuilderWindow", "launch_builder"]
