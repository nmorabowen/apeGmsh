"""Render icon-factory contact sheets — one PNG per canonical palette.

Shows every factory glyph at 1x (22 px) and 2x (44 px) with its name,
on the palette's ``base`` background, stroked in ``palette.icon`` —
exactly what the toolbar renders (ADR 0087 INV-4 / INV-6).

Deliberately does NOT call ``THEME.set_theme`` — palettes are read
straight from ``PALETTES``, so the user's persisted QSettings theme is
never touched. Belt-and-braces: the saved theme value is snapshotted
and restored in a ``finally`` anyway, in case an import path mutates it.

Run (worktree src on PYTHONPATH)::

    python scratchpad_shots/render_contact_sheet.py

No window is ever shown — everything paints into a QImage. The
native platform (not ``offscreen``) is used because Qt's offscreen
plugin ships an empty font database on Windows, which turns every
label into tofu.
"""
from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

CANONICAL = ("catppuccin_mocha", "neutral_studio", "paper")
COLS = 6
CELL_W, CELL_H = 150, 108
PAD = 24


def _saved_theme():
    from qtpy.QtCore import QSettings
    from apeGmsh.viewers.ui.theme import ThemeManager
    s = QSettings(ThemeManager._settings_org, ThemeManager._settings_app)
    return s.value("theme", None)


def _restore_theme(value) -> None:
    from qtpy.QtCore import QSettings
    from apeGmsh.viewers.ui.theme import ThemeManager
    s = QSettings(ThemeManager._settings_org, ThemeManager._settings_app)
    if value is None:
        s.remove("theme")
    else:
        s.setValue("theme", value)


def render_sheet(palette, out_path: pathlib.Path) -> None:
    from qtpy import QtCore, QtGui
    from apeGmsh.viewers.ui._icon_factory import glyph_names, toolbar_icon

    names = glyph_names()
    rows = math.ceil(len(names) / COLS)
    w = PAD * 2 + COLS * CELL_W
    h = PAD * 2 + 40 + rows * CELL_H

    img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(QtGui.QColor(palette.base))
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)

    title_font = QtGui.QFont("Segoe UI", 11)
    title_font.setWeight(QtGui.QFont.DemiBold)
    p.setFont(title_font)
    p.setPen(QtGui.QColor(palette.text))
    p.drawText(
        QtCore.QRectF(PAD, PAD - 8, w - 2 * PAD, 30),
        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
        f"apeGmsh icon factory — {palette.name}  "
        f"({len(names)} glyphs, 1x/2x)",
    )

    name_font = QtGui.QFont("Consolas", 8)
    for i, name in enumerate(names):
        r, c = divmod(i, COLS)
        x0 = PAD + c * CELL_W
        y0 = PAD + 40 + r * CELL_H

        # Hit-area frames so the 22/44 px boxes are visible.
        frame = QtGui.QColor(palette.surface0)
        p.setPen(QtGui.QPen(frame, 1))
        p.setBrush(QtCore.Qt.NoBrush)
        box1 = QtCore.QRectF(x0 + 14, y0 + 18, 22, 22)
        box2 = QtCore.QRectF(x0 + 54, y0 + 7, 44, 44)
        p.drawRect(box1)
        p.drawRect(box2)

        icon = toolbar_icon(name, palette.icon, dpr=1.0)
        pix1 = icon.pixmap(22, 22)
        # 2x: render fresh at dpr=2 so it is genuinely re-rasterised.
        icon2 = toolbar_icon(name, palette.icon, size=44, dpr=1.0)
        pix2 = icon2.pixmap(44, 44)
        p.drawPixmap(box1.topLeft(), pix1)
        p.drawPixmap(box2.topLeft(), pix2)

        p.setFont(name_font)
        p.setPen(QtGui.QColor(palette.subtext))
        p.drawText(
            QtCore.QRectF(x0, y0 + 58, CELL_W - 10, 40),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            name,
        )

    p.end()
    img.save(str(out_path))
    print(f"wrote {out_path}")


def main() -> int:
    saved = None
    have_snapshot = False
    try:
        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            sys.argv,
        )
        saved = _saved_theme()
        have_snapshot = True

        from apeGmsh.viewers.ui.theme import PALETTES
        for name in CANONICAL:
            palette = PALETTES[name]
            render_sheet(palette, HERE / f"contact_sheet_{name}.png")
        del app
        return 0
    finally:
        if have_snapshot:
            _restore_theme(saved)


if __name__ == "__main__":
    raise SystemExit(main())
