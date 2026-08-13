"""
GraphicsColorsDialog — non-modal viewport colour editor.

The full Theme editor is modal, so hover / pick cannot be tried while
it is open. This window is a Tool dialog: it stays up next to the
viewport, applies every edit immediately via ``THEME.update_current``,
and exposes only the graphics roles (idle, hover, pick, overlays).

Okabe–Ito (protanopia) is a one-click preset on the blue/yellow axis.

Usage::

    from apeGmsh.viewers.ui.graphics_colors_dialog import open_graphics_colors
    open_graphics_colors(parent=window)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .theme import PALETTES, THEME, ThemeManager, _BUILTIN_THEME_IDS
from .theme_editor_dialog import (
    _ColorButton,
    _hex_to_rgb,
    _rgb_to_hex,
    _slugify,
)


# Okabe–Ito / Wong palette — remaining axis for protanopia.
# Idle greys stay as-is; these four are the interaction/overlay roles.
OKABE_ITO_GRAPHICS: dict[str, object] = {
    "hover_rgb": (240, 228, 66),       # #F0E442 yellow
    "pick_rgb": (0, 114, 178),         # #0072B2 blue
    "origin_marker_color": "#56B4E9",  # sky blue
    "measure_color": "#56B4E9",
}


def _qt():
    from qtpy import QtWidgets, QtGui, QtCore
    return QtWidgets, QtGui, QtCore


# (field, label, kind) — kind is "rgb" or "hex"
_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("hover_rgb", "Hover", "rgb"),
    ("pick_rgb", "Pick", "rgb"),
    ("hidden_rgb", "Hidden", "rgb"),
    ("dim_pt", "Idle points", "rgb"),
    ("dim_crv", "Idle curves", "rgb"),
    ("dim_srf", "Idle surfaces", "rgb"),
    ("dim_vol", "Idle volumes", "rgb"),
    ("origin_marker_color", "Origin marker", "hex"),
    ("measure_color", "Measure probe", "hex"),
)

_SELECTION = ("hover_rgb", "pick_rgb", "hidden_rgb")
_IDLE = ("dim_pt", "dim_crv", "dim_srf", "dim_vol")
_OVERLAYS = ("origin_marker_color", "measure_color")


class GraphicsColorsDialog:
    """Non-modal Tool window; live-applies via ``THEME.update_current``."""

    def __init__(self, parent: Any = None) -> None:
        QtWidgets, _, QtCore = _qt()

        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("apeGmsh — Graphics colors")
        self.dialog.setModal(False)
        try:
            flags = (
                QtCore.Qt.WindowType.Tool
                | QtCore.Qt.WindowType.WindowCloseButtonHint
            )
            self.dialog.setWindowFlags(flags)
        except Exception:
            pass
        self.dialog.resize(380, 560)

        self._syncing = False
        self._widgets: dict[str, _ColorButton] = {}

        root = QtWidgets.QVBoxLayout(self.dialog)
        hint = QtWidgets.QLabel(
            "Edits apply immediately — hover and pick in the viewport "
            "while this is open. Session-only until you Save as theme."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        groups = (
            ("Selection", _SELECTION),
            ("Idle geometry", _IDLE),
            ("Overlays", _OVERLAYS),
        )
        labels = {name: label for name, label, _kind in _FIELDS}
        kinds = {name: kind for name, _label, kind in _FIELDS}

        for title, names in groups:
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setSpacing(4)
            for name in names:
                initial = self._field_hex(name, kinds[name])
                btn = _ColorButton(
                    initial, lambda _v, n=name: self._on_field_change(n),
                )
                self._widgets[name] = btn
                form.addRow(labels[name], btn.button)
            root.addWidget(box)

        self._swatches = QtWidgets.QLabel()
        self._swatches.setWordWrap(False)
        root.addWidget(self._swatches)
        self._refresh_swatches()

        btns = QtWidgets.QHBoxLayout()
        preset = QtWidgets.QPushButton("Okabe–Ito (protanopia)")
        preset.setToolTip(
            "Yellow hover + blue pick + sky-blue overlays. "
            "Safe remaining axis when red/green collapse."
        )
        preset.clicked.connect(self._apply_okabe_ito)
        btns.addWidget(preset)
        reset = QtWidgets.QPushButton("Reset theme")
        reset.setToolTip("Restore this theme's stock graphics colours.")
        reset.clicked.connect(self._reset_theme)
        btns.addWidget(reset)
        root.addLayout(btns)

        save_row = QtWidgets.QHBoxLayout()
        save_row.addStretch(1)
        save = QtWidgets.QPushButton("Save as theme…")
        save.clicked.connect(self._save_as_theme)
        save_row.addWidget(save)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.dialog.close)
        save_row.addWidget(close)
        root.addLayout(save_row)

        self._unsub = THEME.subscribe(self._on_external_theme)
        self.dialog.finished.connect(self._cleanup)

    def show(self) -> None:
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    # ── populate / apply ─────────────────────────────────────────────

    @staticmethod
    def _field_hex(name: str, kind: str, palette: Any = None) -> str:
        pal = palette if palette is not None else THEME.current
        value = getattr(pal, name)
        if kind == "rgb":
            return _rgb_to_hex(tuple(int(c) for c in value))
        return str(value)

    def _refresh_swatches(self) -> None:
        p = THEME.current
        idle = _rgb_to_hex(tuple(int(c) for c in p.dim_srf))
        hover = _rgb_to_hex(tuple(int(c) for c in p.hover_rgb))
        pick = _rgb_to_hex(tuple(int(c) for c in p.pick_rgb))
        self._swatches.setText(
            f'<span style="background:{idle};color:#000;padding:2px 8px;">Idle</span> '
            f'<span style="background:{hover};color:#000;padding:2px 8px;">Hover</span> '
            f'<span style="background:{pick};color:#fff;padding:2px 8px;">Pick</span>'
        )

    def _populate(self, palette: Any) -> None:
        kinds = {name: kind for name, _label, kind in _FIELDS}
        self._syncing = True
        try:
            for name, btn in self._widgets.items():
                btn.set_value(self._field_hex(name, kinds[name], palette))
        finally:
            self._syncing = False
        self._refresh_swatches()

    def _collect_overrides(self) -> dict[str, object]:
        kinds = {name: kind for name, _label, kind in _FIELDS}
        out: dict[str, object] = {}
        for name, btn in self._widgets.items():
            hex_str = btn.value()
            if kinds[name] == "rgb":
                out[name] = _hex_to_rgb(hex_str)
            else:
                out[name] = hex_str
        return out

    def _on_field_change(self, _name: str) -> None:
        if self._syncing:
            return
        THEME.update_current(**self._collect_overrides())
        self._refresh_swatches()

    def _apply_okabe_ito(self) -> None:
        THEME.update_current(**OKABE_ITO_GRAPHICS)
        self._populate(THEME.current)

    def _reset_theme(self) -> None:
        name = THEME.current.name
        stock = PALETTES.get(name)
        if stock is None:
            return
        # ``set_theme`` no-ops on identity; the live palette is a
        # replace() copy, so this restores the stock entry.
        THEME.set_theme(name)
        # If the radio was already on this name and PALETTES still
        # holds the original, set_theme always assigns it. If a custom
        # theme *is* the live object, re-apply the stock graphics
        # fields from PALETTES anyway.
        if THEME.current is not stock:
            THEME.update_current(**{
                n: getattr(stock, n) for n, _l, _k in _FIELDS
            })
        self._populate(THEME.current)

    def _save_as_theme(self) -> None:
        QtWidgets, _, _ = _qt()
        name, ok = QtWidgets.QInputDialog.getText(
            self.dialog, "Save as theme",
            "Theme name:",
            text=f"my_{THEME.current.name}",
        )
        if not ok:
            return
        slug = _slugify(name)
        if slug in _BUILTIN_THEME_IDS:
            QtWidgets.QMessageBox.warning(
                self.dialog, "Reserved name",
                f"{slug!r} is a built-in theme. Pick a different name.",
            )
            return
        pal = replace(THEME.current, name=slug)
        try:
            path = ThemeManager.save_custom_theme(pal)
            THEME.set_theme(slug)
            QtWidgets.QMessageBox.information(
                self.dialog, "Saved",
                f"Theme saved to:\n{path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self.dialog, "Save failed", str(exc),
            )

    def _on_external_theme(self, palette: Any) -> None:
        if self._syncing:
            return
        self._populate(palette)

    def _cleanup(self, _result: int = 0) -> None:
        try:
            self._unsub()
        except Exception:
            pass


def open_graphics_colors(parent: Any = None) -> GraphicsColorsDialog:
    """Show the Graphics colors window (creates QApplication if needed)."""
    from ._qt_env import prepare_qt_environment
    prepare_qt_environment()

    QtWidgets, _, _ = _qt()
    app = QtWidgets.QApplication.instance()
    if app is None:
        QtWidgets.QApplication([])
    dlg = GraphicsColorsDialog(parent=parent)
    dlg.show()
    return dlg
