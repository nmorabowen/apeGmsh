"""
ThemeEditorDialog — modal palette editor with live preview.

Lets users author custom themes by copying an existing theme, editing
any Palette field (chrome colors, viewport colors, outlines, backgrounds,
etc.), and saving to the per-user themes directory.

Workflow:

1. Pick a **base** theme from the list (built-ins + installed customs).
2. Edit the **draft** palette — color buttons open ``QColorDialog``,
   numbers use spinboxes, enums use combos.
3. Every edit fires ``THEME.set_theme(draft_name)`` so open viewers
   re-render immediately.
4. **Save**: writes JSON to ``ThemeManager.themes_dir()`` and registers
   in ``PALETTES``.
5. **Delete**: removes a custom theme (built-ins are protected).

Usage::

    from apeGmsh.viewers.ui.theme_editor_dialog import open_theme_editor
    open_theme_editor()
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from .theme import (
    PALETTES,
    Palette,
    THEME,
    ThemeManager,
    _BUILTIN_THEME_IDS,
)


def _qt():
    from qtpy import QtWidgets, QtGui, QtCore
    return QtWidgets, QtGui, QtCore


# Fields grouped by domain for the editor layout. Missing fields fall
# back to "Other" at the bottom.
_FIELD_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Chrome — surfaces", (
        "base", "mantle", "surface0", "surface1", "surface2",
    )),
    ("Chrome — text", ("text", "subtext", "overlay")),
    ("Chrome — accents", (
        "accent", "icon", "success", "warning", "error", "info",
    )),
    ("Background", (
        "background_mode", "bg_top", "bg_bottom",
    )),
    ("Viewport — per-dimension idle (RGB)", (
        "dim_pt", "dim_crv", "dim_srf", "dim_vol",
    )),
    ("Viewport — interaction (RGB)", (
        "hover_rgb", "pick_rgb", "hidden_rgb",
    )),
    ("Viewport — outlines", (
        "outline_color", "outline_silhouette_px", "outline_feature_px",
    )),
    ("Viewport — mesh edges & nodes", (
        "mesh_edge_color", "node_accent", "origin_marker_color",
    )),
    ("Axis scene", (
        "grid_major", "grid_minor", "bbox_color", "bbox_line_px",
    )),
    ("Results colormaps", ("cmap_seq", "cmap_div")),
    ("Rendering", ("ao_intensity", "corner_triad_default")),
]

_BACKGROUND_MODES = ("radial", "linear", "flat_corner")
_AO_CHOICES = ("none", "light", "moderate")


def _is_hex_field(name: str) -> bool:
    """Palette hex-color fields (``str`` starting with ``#``)."""
    return name in {
        "base", "mantle", "surface0", "surface1", "surface2",
        "text", "subtext", "overlay",
        "accent", "icon", "success", "warning", "error", "info",
        "bg_top", "bg_bottom",
        "outline_color", "mesh_edge_color", "node_accent",
        "origin_marker_color",
        "grid_major", "grid_minor", "bbox_color",
    }


def _is_rgb_field(name: str) -> bool:
    """Palette tuple-RGB fields."""
    return name in {
        "dim_pt", "dim_crv", "dim_srf", "dim_vol",
        "hover_rgb", "pick_rgb", "hidden_rgb",
    }


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _slugify(name: str) -> str:
    """Convert a free-text name into a safe theme id (lowercase, underscores)."""
    import re
    s = re.sub(r"\s+", "_", name.strip().lower())
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "custom_theme"


class _ColorButton:
    """QPushButton that opens QColorDialog and shows the picked color as swatch."""

    def __init__(self, initial: str, on_change: Any) -> None:
        QtWidgets, QtGui, _ = _qt()
        self._on_change = on_change
        self._hex = initial
        self.button = QtWidgets.QPushButton()
        self.button.setFixedHeight(24)
        self.button.setMinimumWidth(140)
        self.button.clicked.connect(self._open_picker)
        self._sync()

    def value(self) -> str:
        return self._hex

    def set_value(self, hex_str: str) -> None:
        self._hex = hex_str
        self._sync()

    def _sync(self) -> None:
        QtWidgets, QtGui, _ = _qt()
        self.button.setText(self._hex)
        # Compute readable text color for contrast
        r, g, b = _hex_to_rgb(self._hex)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        fg = "#000000" if brightness > 128 else "#ffffff"
        self.button.setStyleSheet(
            f"QPushButton {{ background-color: {self._hex}; color: {fg}; "
            f"border: 1px solid #555; padding: 2px 6px; }}"
        )

    def _open_picker(self) -> None:
        QtWidgets, QtGui, _ = _qt()
        initial = QtGui.QColor(self._hex)
        picked = QtWidgets.QColorDialog.getColor(initial, self.button)
        if picked.isValid():
            self.set_value(picked.name())
            self._on_change(self._hex)


class ThemeEditorDialog:
    """Modal editor with live preview via ``THEME.set_theme``."""

    def __init__(self, parent: Any = None) -> None:
        QtWidgets, _, _ = _qt()

        # Remember the theme that was active when we opened, so Cancel
        # can restore it.
        self._original_theme_name = THEME.current.name

        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("apeGmsh — Theme editor")
        self.dialog.setModal(True)
        self.dialog.resize(720, 760)

        root = QtWidgets.QVBoxLayout(self.dialog)

        # ── Top bar: base theme + draft name ────────────────────────
        top = QtWidgets.QHBoxLayout()

        top.addWidget(QtWidgets.QLabel("Base:"))
        self._cmb_base = QtWidgets.QComboBox()
        self._cmb_base.addItems(sorted(PALETTES.keys()))
        self._cmb_base.setCurrentText(self._original_theme_name)
        self._cmb_base.currentTextChanged.connect(self._reload_base)
        top.addWidget(self._cmb_base, 1)

        top.addWidget(QtWidgets.QLabel("Draft name:"))
        self._le_name = QtWidgets.QLineEdit()
        self._le_name.setPlaceholderText("my_theme")
        top.addWidget(self._le_name, 1)

        root.addLayout(top)

        # ── Field editors (scrollable) ──────────────────────────────
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setSpacing(6)

        self._widgets: dict[str, Any] = {}
        shown: set[str] = set()

        for title, field_names in _FIELD_GROUPS:
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            form.setSpacing(4)
            for name in field_names:
                w = self._make_field_widget(name)
                if w is not None:
                    form.addRow(name, w)
                    shown.add(name)
            inner_layout.addWidget(group)

        # "Other" — anything we didn't group above (e.g. body_palette)
        other_names = [f.name for f in fields(Palette) if f.name not in shown and f.name != "name"]
        if other_names:
            group = QtWidgets.QGroupBox("Other")
            form = QtWidgets.QFormLayout(group)
            for name in other_names:
                w = self._make_field_widget(name)
                if w is not None:
                    form.addRow(name, w)
            inner_layout.addWidget(group)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # ── Footer: hint + buttons ──────────────────────────────────
        hint = QtWidgets.QLabel(
            "Edits preview live on open viewers. "
            "Save writes to the themes folder; Cancel reverts."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        file_label = QtWidgets.QLabel(f"Themes: {ThemeManager.themes_dir()}")
        file_label.setWordWrap(True)
        file_label.setStyleSheet("color: gray; font-size: 10px;")
        root.addWidget(file_label)

        btns = QtWidgets.QHBoxLayout()
        self._btn_delete = QtWidgets.QPushButton("Delete custom…")
        self._btn_delete.clicked.connect(self._on_delete)
        btns.addWidget(self._btn_delete)
        btns.addStretch(1)
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        btns.addWidget(self._btn_cancel)
        self._btn_save = QtWidgets.QPushButton("Save")
        self._btn_save.setDefault(True)
        self._btn_save.clicked.connect(self._on_save)
        btns.addWidget(self._btn_save)
        root.addLayout(btns)

        # Populate widgets from the initial base
        self._reload_base(self._original_theme_name)

    # ── widget factory ───────────────────────────────────────────────

    def _make_field_widget(self, name: str) -> Any:
        QtWidgets, _, _ = _qt()
        if name == "name":
            return None

        field = next((f for f in fields(Palette) if f.name == name), None)
        if field is None:
            return None

        if _is_hex_field(name):
            btn = _ColorButton("#000000", lambda _v, n=name: self._on_field_change())
            self._widgets[name] = btn
            return btn.button

        if _is_rgb_field(name):
            btn = _ColorButton("#000000", lambda _v, n=name: self._on_field_change())
            self._widgets[name] = btn
            return btn.button

        if name == "background_mode":
            cmb = QtWidgets.QComboBox()
            cmb.addItems(list(_BACKGROUND_MODES))
            cmb.currentTextChanged.connect(lambda _v: self._on_field_change())
            self._widgets[name] = cmb
            return cmb

        if name == "ao_intensity":
            cmb = QtWidgets.QComboBox()
            cmb.addItems(list(_AO_CHOICES))
            cmb.currentTextChanged.connect(lambda _v: self._on_field_change())
            self._widgets[name] = cmb
            return cmb

        if field.type in (float, "float") or name.endswith("_px"):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 20.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.valueChanged.connect(lambda _v: self._on_field_change())
            self._widgets[name] = spin
            return spin

        if field.type in (bool, "bool"):
            cb = QtWidgets.QCheckBox()
            cb.toggled.connect(lambda _v: self._on_field_change())
            self._widgets[name] = cb
            return cb

        if name == "body_palette":
            # Comma-separated hex list — simple line edit
            le = QtWidgets.QLineEdit()
            le.textChanged.connect(lambda _v: self._on_field_change())
            self._widgets[name] = le
            return le

        # Fallback: free-text (cmap names)
        le = QtWidgets.QLineEdit()
        le.textChanged.connect(lambda _v: self._on_field_change())
        self._widgets[name] = le
        return le

    # ── populate / read ──────────────────────────────────────────────

    def _reload_base(self, base_name: str) -> None:
        pal = PALETTES.get(base_name)
        if pal is None:
            return
        for fname, widget in self._widgets.items():
            value = getattr(pal, fname)
            if _is_hex_field(fname):
                widget.set_value(value)
            elif _is_rgb_field(fname):
                widget.set_value(_rgb_to_hex(value))
            elif isinstance(widget, type(widget)) and hasattr(widget, "setCurrentText") and not hasattr(widget, "setText"):
                # QComboBox
                widget.setCurrentText(str(value))
            elif hasattr(widget, "setValue"):
                widget.setValue(float(value))
            elif hasattr(widget, "setChecked"):
                widget.setChecked(bool(value))
            elif fname == "body_palette":
                widget.setText(", ".join(value))
            else:
                widget.setText(str(value))

        # Draft name — prefix "my_" on built-ins to discourage direct overwrite
        if base_name in _BUILTIN_THEME_IDS:
            self._le_name.setText(f"my_{base_name}")
        else:
            self._le_name.setText(base_name)
        # Live-preview the base theme
        THEME.set_theme(base_name)

    def _collect_palette(self) -> Palette | None:
        """Assemble a Palette from current widget values. Returns None on invalid input."""
        base_name = self._cmb_base.currentText()
        base = PALETTES.get(base_name)
        if base is None:
            return None
        draft_id = _slugify(self._le_name.text()) or base.name
        overrides: dict[str, Any] = {"name": draft_id}

        for fname, widget in self._widgets.items():
            field = next((f for f in fields(Palette) if f.name == fname), None)
            if field is None:
                continue
            if _is_hex_field(fname):
                overrides[fname] = widget.value()
            elif _is_rgb_field(fname):
                overrides[fname] = _hex_to_rgb(widget.value())
            elif hasattr(widget, "currentText") and not hasattr(widget, "text"):
                overrides[fname] = widget.currentText()
            elif hasattr(widget, "isChecked") and field.type in (bool, "bool"):
                overrides[fname] = widget.isChecked()
            elif hasattr(widget, "value"):
                overrides[fname] = widget.value()
            elif fname == "body_palette":
                text = widget.text()
                parts = tuple(
                    p.strip() for p in text.split(",") if p.strip()
                )
                overrides[fname] = parts
            else:
                overrides[fname] = widget.text()

        try:
            return replace(base, **overrides)
        except Exception:
            import logging
            logging.getLogger("apeGmsh.viewer.theme.editor").exception(
                "failed to build draft palette",
            )
            return None

    # ── handlers ─────────────────────────────────────────────────────

    def _on_field_change(self) -> None:
        pal = self._collect_palette()
        if pal is None:
            return
        # Register as live preview under the draft name
        PALETTES[pal.name] = pal
        THEME.set_theme(pal.name)

    def _on_save(self) -> None:
        QtWidgets, _, _ = _qt()
        pal = self._collect_palette()
        if pal is None:
            QtWidgets.QMessageBox.warning(
                self.dialog, "Invalid theme",
                "Could not build a valid palette from the current fields.",
            )
            return
        if pal.name in _BUILTIN_THEME_IDS:
            QtWidgets.QMessageBox.warning(
                self.dialog, "Reserved name",
                f"{pal.name!r} is a built-in theme. Pick a different draft name.",
            )
            return
        try:
            path = ThemeManager.save_custom_theme(pal)
            QtWidgets.QMessageBox.information(
                self.dialog, "Saved",
                f"Theme saved to:\n{path}",
            )
            THEME.set_theme(pal.name)
            self.dialog.accept()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self.dialog, "Save failed", str(exc),
            )

    def _on_cancel(self) -> None:
        # Revert any preview-only draft palette
        draft_name = _slugify(self._le_name.text())
        if draft_name and draft_name not in _BUILTIN_THEME_IDS:
            # If the draft wasn't saved, drop it from PALETTES (but keep
            # truly-installed custom themes that were already on disk).
            themes_dir = ThemeManager.themes_dir()
            disk_path = themes_dir / f"{draft_name}.json"  # type: ignore[union-attr, operator]
            try:
                exists_on_disk = disk_path.exists()
            except Exception:
                exists_on_disk = False
            if not exists_on_disk:
                PALETTES.pop(draft_name, None)
        THEME.set_theme(self._original_theme_name)
        self.dialog.reject()

    def _on_delete(self) -> None:
        QtWidgets, _, _ = _qt()
        name = self._cmb_base.currentText()
        if name in _BUILTIN_THEME_IDS:
            QtWidgets.QMessageBox.information(
                self.dialog, "Built-in theme",
                f"{name!r} is a built-in theme and cannot be deleted.",
            )
            return
        reply = QtWidgets.QMessageBox.question(
            self.dialog, "Delete theme",
            f"Delete custom theme {name!r} from disk?",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        ThemeManager.delete_custom_theme(name)
        # Repopulate base list and switch to mocha as a safe fallback
        self._cmb_base.blockSignals(True)
        self._cmb_base.clear()
        self._cmb_base.addItems(sorted(PALETTES.keys()))
        self._cmb_base.setCurrentText("catppuccin_mocha")
        self._cmb_base.blockSignals(False)
        self._reload_base("catppuccin_mocha")

    def exec(self) -> int:
        return self.dialog.exec()


def open_theme_editor(parent: Any = None) -> int:
    """Open the theme editor (spins up a QApplication if none exists).

    Returns the dialog result code (``QDialog.Accepted`` / ``Rejected``).
    """
    from ._qt_env import prepare_qt_environment
    prepare_qt_environment()

    QtWidgets, _, _ = _qt()
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])
    dlg = ThemeEditorDialog(parent=parent)
    return int(dlg.exec())
