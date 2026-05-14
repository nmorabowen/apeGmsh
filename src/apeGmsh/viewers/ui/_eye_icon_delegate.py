"""EyeIconDelegate — ParaView-style visibility eye column for tree widgets.

Paints a small eye-glyph icon in the left margin of column 0 and
intercepts clicks on the icon's hit area. Tree-widget owners stash
the current visibility flag on each item via a custom Qt role; the
delegate reads that role to pick the open / closed glyph, and emits
:attr:`icon_clicked` (carrying the row's ``QTreeWidgetItem``) when
the user clicks on the icon region.

Plan 03 v1: visibility icons on Composition + Geometry rows in
``results.viewer`` outline. Layer-row support is a follow-up — the
outline doesn't currently render Layer rows; the per-card visibility
toggle in :class:`DiagramSettingsTab` is the existing path.

Glyph rendering uses Qt's ``QPainter`` against a small pixmap so we
don't need an icon-file dependency (same pattern as
:meth:`ViewerWindow._make_icon`). Two Unicode glyphs:

* visible: ``"●"`` — filled disc, reads as "the eye is on"
* hidden:  ``"○"`` — empty circle, reads as "the eye is off"

The disc/circle pair is universally available across Qt's text
rendering backends — no font-fallback risk on minimal Windows / Linux
shells. A future iteration may swap in proper eye glyphs / SVG icons.
"""
from __future__ import annotations

from typing import Any, Optional


def _qt():
    from qtpy import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


# Custom Qt item role for visibility state. Items that should show the
# eye icon set this role to a bool; items without it render no icon.
ROLE_VISIBLE = 0x106

# Icon column hit-area width (px). Matches the typical decoration
# offset Qt's default style uses, plus a small padding for an easy
# click target.
_ICON_HIT_WIDTH = 22
_ICON_GLYPH_SIZE = 16


def _build_delegate_class():
    """Lazy-construct ``EyeIconDelegate`` — avoids pulling qtpy at
    module import time, mirrors the pattern used by
    :mod:`apeGmsh.viewers.core._active_objects` and
    :mod:`._diagram_settings_tab`.
    """
    QtCore, QtGui, QtWidgets = _qt()

    class EyeIconDelegate(QtWidgets.QStyledItemDelegate):
        """Paint an eye-glyph in column 0; emit on icon click.

        Parameters
        ----------
        parent
            The owning tree widget. The delegate is parented here so
            Qt's GC tears them down together.

        Signals
        -------
        icon_clicked(QTreeWidgetItem)
            Fires when the user clicks anywhere in the left
            ``_ICON_HIT_WIDTH`` pixels of an item that has
            ``ROLE_VISIBLE`` set. The payload is the underlying
            ``QTreeWidgetItem``; callers read its current role value
            to decide which way to toggle.
        """

        icon_clicked = QtCore.Signal(object)

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            # Cache the two pixmaps once — re-rendering them on every
            # paint call would burn CPU on large trees.
            self._pix_visible: Any = None
            self._pix_hidden: Any = None
            self._pix_color: str = ""

        # ──────────────────────────────────────────────────────────
        # Painting
        # ──────────────────────────────────────────────────────────

        def paint(self, painter, option, index) -> None:    # noqa: D401
            # Delegate the standard paint first so selection / hover
            # highlight + text render exactly as Qt would by default.
            # Then paint the eye glyph on top of the left margin.
            super().paint(painter, option, index)
            if index.column() != 0:
                return
            role_val = index.data(ROLE_VISIBLE)
            if role_val is None:
                return
            is_visible = bool(role_val)
            color = self._foreground_color(option)
            pix = self._glyph(is_visible, color)
            if pix is None:
                return
            x = option.rect.x() + (_ICON_HIT_WIDTH - pix.width()) // 2
            y = option.rect.y() + (option.rect.height() - pix.height()) // 2
            painter.drawPixmap(x, y, pix)

        # ──────────────────────────────────────────────────────────
        # Click handling
        # ──────────────────────────────────────────────────────────

        def editorEvent(self, event, model, option, index) -> bool:    # noqa: D401
            # MouseButtonPress + left button + within icon column +
            # the row exposes ROLE_VISIBLE → emit and consume.
            if (event.type() != QtCore.QEvent.MouseButtonPress
                    or event.button() != QtCore.Qt.LeftButton
                    or index.column() != 0
                    or index.data(ROLE_VISIBLE) is None):
                return False
            x_local = event.pos().x() - option.rect.x()
            if 0 <= x_local <= _ICON_HIT_WIDTH:
                tree = self.parent()
                item = None
                if tree is not None:
                    try:
                        item = tree.itemFromIndex(index)
                    except Exception:
                        item = None
                self.icon_clicked.emit(item)
                return True
            return False

        # ──────────────────────────────────────────────────────────
        # Geometry hint — make room for the icon in column 0
        # ──────────────────────────────────────────────────────────

        def sizeHint(self, option, index) -> Any:    # noqa: D401
            hint = super().sizeHint(option, index)
            if index.column() == 0 and index.data(ROLE_VISIBLE) is not None:
                hint.setWidth(hint.width() + _ICON_HIT_WIDTH)
            return hint

        # ──────────────────────────────────────────────────────────
        # Internals — glyph rendering + theming
        # ──────────────────────────────────────────────────────────

        def _glyph(self, is_visible: bool, color: str) -> Any:
            """Return the cached pixmap for the requested state, or
            re-render if the color has changed (theme switch)."""
            if color != self._pix_color:
                self._pix_visible = None
                self._pix_hidden = None
                self._pix_color = color
            cache_attr = "_pix_visible" if is_visible else "_pix_hidden"
            pix = getattr(self, cache_attr)
            if pix is not None:
                return pix
            pix = self._render_glyph("●" if is_visible else "○", color)
            setattr(self, cache_attr, pix)
            return pix

        @staticmethod
        def _render_glyph(text: str, color: str) -> Any:
            pix = QtGui.QPixmap(_ICON_GLYPH_SIZE, _ICON_GLYPH_SIZE)
            pix.fill(QtGui.QColor(0, 0, 0, 0))
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setPen(QtGui.QColor(color))
            font = QtGui.QFont("Segoe UI", 11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QtCore.QRect(0, 0, _ICON_GLYPH_SIZE, _ICON_GLYPH_SIZE),
                QtCore.Qt.AlignCenter, text,
            )
            painter.end()
            return pix

        @staticmethod
        def _foreground_color(option: Any) -> str:
            """Resolve the foreground color for the icon: matches the
            tree's text color so the icon reads with the theme. Falls
            back to a neutral mid-gray if the palette isn't queryable.
            """
            try:
                palette = option.palette
                col = palette.color(palette.Text)
                return col.name()
            except Exception:
                return "#A0A0A0"

    return EyeIconDelegate


_EyeIconDelegateClass: Optional[type] = None


def resolve_delegate_class() -> type:
    """Build :class:`EyeIconDelegate` on first call; cache thereafter."""
    global _EyeIconDelegateClass
    if _EyeIconDelegateClass is None:
        _EyeIconDelegateClass = _build_delegate_class()
    return _EyeIconDelegateClass


def __getattr__(name: str) -> Any:
    """Public lazy-attribute access so callers can ``from … import
    EyeIconDelegate`` without paying the Qt import cost up front."""
    if name == "EyeIconDelegate":
        return resolve_delegate_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
