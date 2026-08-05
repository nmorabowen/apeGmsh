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
* gated:   the filled disc at reduced alpha — intent says visible
  (``ROLE_VISIBLE`` True) but the row is not actually on screen
  (``ROLE_EFFECTIVE`` False): hidden by the composition gate or its
  geometry's eye. Rows that never set ``ROLE_EFFECTIVE`` behave
  exactly as before (treated as effective).

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
# EFFECTIVE visibility (ADR 0056 INV-2 — intent vs effective are
# distinct channels). Optional: absent / None reads as True, so only
# rows that mirror a gate-controlled object (layer rows) set it. When
# ``ROLE_VISIBLE`` is True but this is False, the eye paints the third
# (dimmed) state instead of the filled dot.
ROLE_EFFECTIVE = 0x107

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
            # Cache the three pixmaps once — re-rendering them on every
            # paint call would burn CPU on large trees.
            self._pix_visible: Any = None
            self._pix_hidden: Any = None
            self._pix_dimmed: Any = None
            self._pix_color: str = ""

        # ──────────────────────────────────────────────────────────
        # Painting
        # ──────────────────────────────────────────────────────────

        def paint(self, painter, option, index) -> None:    # noqa: D401
            role_val = (
                index.data(ROLE_VISIBLE) if index.column() == 0 else None
            )
            if role_val is None:
                super().paint(painter, option, index)
                return
            # Reserve a left gutter for the eye and inset the text into
            # the remaining area, so the glyph never paints over the
            # label (that over-paint was the "eye overlaps text" bug).
            # Pre-fill the selection highlight full-width so the gutter
            # still reads as selected.
            if option.state & QtWidgets.QStyle.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())
            opt = QtWidgets.QStyleOptionViewItem(option)
            opt.rect = QtCore.QRect(option.rect)
            opt.rect.setLeft(option.rect.left() + _ICON_HIT_WIDTH)
            super().paint(painter, opt, index)
            dpr = self._dpr(option)
            color = self._foreground_color(option)
            # Third state: intent-visible but not effectively on screen
            # (composition gate / geometry hidden) → dimmed filled dot.
            # Absent role → effective, so two-state rows are unchanged.
            eff_val = index.data(ROLE_EFFECTIVE)
            dimmed = (
                bool(role_val) and eff_val is not None and not bool(eff_val)
            )
            pix = self._glyph(bool(role_val), color, dpr, dimmed=dimmed)
            if pix is None:
                return
            pw = pix.width() / dpr
            ph = pix.height() / dpr
            x = option.rect.x() + (_ICON_HIT_WIDTH - pw) / 2.0
            y = option.rect.y() + (option.rect.height() - ph) / 2.0
            painter.drawPixmap(int(x), int(y), pix)

        @staticmethod
        def _dpr(option: Any) -> float:
            """View device-pixel-ratio so the glyph renders crisp on
            HiDPI (the pixmap is built at this ratio, never rescaled)."""
            try:
                w = option.widget
                if w is not None:
                    return float(w.devicePixelRatioF()) or 1.0
            except Exception:
                pass
            return 1.0

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

        def _glyph(
            self, is_visible: bool, color: str, dpr: float = 1.0,
            *, dimmed: bool = False,
        ) -> Any:
            """Cached crisp dot pixmap; re-render on theme (colour) or
            display (dpr) change. ``dimmed`` selects the third (gated)
            state — only meaningful with ``is_visible=True``."""
            key = f"{color}|{dpr:.3f}"
            if key != self._pix_color:
                self._pix_visible = None
                self._pix_hidden = None
                self._pix_dimmed = None
                self._pix_color = key
            if is_visible and dimmed:
                cache_attr = "_pix_dimmed"
            elif is_visible:
                cache_attr = "_pix_visible"
            else:
                cache_attr = "_pix_hidden"
            pix = getattr(self, cache_attr)
            if pix is not None:
                return pix
            pix = self._render_glyph(is_visible, color, dpr, dimmed=dimmed)
            setattr(self, cache_attr, pix)
            return pix

        @staticmethod
        def _render_glyph(
            is_visible: bool, color: str, dpr: float = 1.0,
            *, dimmed: bool = False,
        ) -> Any:
            # Vector ellipse (not a font glyph) at the view dpr so it
            # stays a crisp dot instead of a rescaled fuzzy oval —
            # same treatment as the model.viewer outline.
            s = _ICON_GLYPH_SIZE
            pix = QtGui.QPixmap(round(s * dpr), round(s * dpr))
            pix.setDevicePixelRatio(dpr)
            pix.fill(QtGui.QColor(0, 0, 0, 0))
            p = QtGui.QPainter(pix)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            qc = QtGui.QColor(color)
            if dimmed:
                # Gated state: the filled dot at 40% alpha. The colour
                # is the palette text colour in BOTH themes (dark text
                # on light, light text on dark), so a 40% fade reads as
                # "present but muted" against either background while
                # staying clearly distinct from the hollow ring.
                qc.setAlphaF(qc.alphaF() * 0.40)
            d = s * 0.55
            off = (s - d) / 2.0
            rect = QtCore.QRectF(off, off, d, d)
            if is_visible:
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(qc)               # filled dot = visible
            else:
                pen = QtGui.QPen(qc)
                pen.setWidthF(max(1.0, s * 0.09))
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)  # hollow ring = hidden
            p.drawEllipse(rect)
            p.end()
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
