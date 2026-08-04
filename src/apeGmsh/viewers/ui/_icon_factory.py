"""Icon factory — the ONE way viewer chrome gets an icon (ADR 0087 INV-4).

Generalizes the ``_eye_icon_delegate.py`` technique: every glyph is
vector-drawn with ``QPainter`` into a pixmap at the caller's
device-pixel-ratio, cached per ``(name, color, size, dpr)``, and
colored from the palette so a theme switch re-renders it crisp. No
icon files, no icon-font dependency, no emoji.

Glyph metrics (ADR 0087 D1.4 / INV-4):

* design size 16 px inside a 22 px hit box (callers may pass another
  ``size``; the glyph keeps the same 16/22 proportion),
* stroke width ``max(1.0, glyph_size * 0.09)`` — the shipped
  eye-glyph value,
* round caps / joins, no fills except state dots and face highlights
  (drawn at reduced alpha so they read as state, not decoration).

Usage::

    from ._icon_factory import toolbar_icon
    act.setIcon(toolbar_icon("view_top", palette.icon, dpr=dpr))

Adding a glyph = adding one ``_draw_<name>`` function and its entry in
``_GLYPHS``. Coordinates are expressed in the 16 px design box
(0..16); the painter is pre-transformed to center that box in the hit
area.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def _qt():
    from qtpy import QtCore, QtGui
    return QtCore, QtGui


# Design-box size (px) the draw functions are authored against, and
# the default hit-area the glyph is centered in (ADR 0087 INV-4).
_DESIGN = 16.0
_HIT = 22

# Cache: (name, color, size, round(dpr*1000)) -> QIcon
_CACHE: Dict[Tuple[str, str, int, int], Any] = {}


def _stroke_pen(QtGui, QtCore, color: str, glyph: float):
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(max(1.0, glyph * 0.09))
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    return pen


def _fill_brush(QtGui, color: str, alpha: float = 0.50):
    qc = QtGui.QColor(color)
    qc.setAlphaF(qc.alphaF() * alpha)
    return QtGui.QBrush(qc)


# ──────────────────────────────────────────────────────────────────────
# Draw functions — all in the 16 px design box.
# ──────────────────────────────────────────────────────────────────────

def _iso_cube_faces(QtCore):
    """The three visible faces of the house iso cube: (top, left, right).

    Each face is a list of QPointF corners in the design box. The cube
    is the shared base of every camera-preset glyph.
    """
    P = QtCore.QPointF
    top = [P(8, 1.5), P(14, 4.75), P(8, 8), P(2, 4.75)]
    left = [P(2, 4.75), P(8, 8), P(8, 14.5), P(2, 11.25)]
    right = [P(8, 8), P(14, 4.75), P(14, 11.25), P(8, 14.5)]
    return top, left, right


def _draw_cube(p, QtCore, QtGui, color, *, fill=None, flip_v=False,
               flip_h=False):
    """Iso cube outline; ``fill`` picks the highlighted face
    (``"top" | "left" | "right" | None``). Flips mirror the whole
    drawing so the highlight can sit on any of six positions while the
    cube stays one shape (top / bottom / front / back / left / right
    map to six distinct fill positions)."""
    if flip_h or flip_v:
        p.save()
        p.translate(_DESIGN / 2.0, _DESIGN / 2.0)
        p.scale(-1.0 if flip_h else 1.0, -1.0 if flip_v else 1.0)
        p.translate(-_DESIGN / 2.0, -_DESIGN / 2.0)
    top, left, right = _iso_cube_faces(QtCore)
    faces = {"top": top, "left": left, "right": right}
    if fill is not None:
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(_fill_brush(QtGui, color))
        p.drawPolygon(QtGui.QPolygonF(faces[fill]))
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    for face in (top, left, right):
        p.drawPolygon(QtGui.QPolygonF(face))
    if flip_h or flip_v:
        p.restore()


def _draw_view_top(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="top")


def _draw_view_bottom(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="top", flip_v=True)


def _draw_view_front(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="left")


def _draw_view_back(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="left", flip_h=True,
               flip_v=True)


def _draw_view_right(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="right")


def _draw_view_left(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill="right", flip_h=True)


def _draw_view_iso(p, QtCore, QtGui, color):
    _draw_cube(p, QtCore, QtGui, color, fill=None)


def _draw_ortho(p, QtCore, QtGui, color):
    """Parallel projection: front square + offset rear square, joined."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(2, 5, 9, 9))    # front square
    # Rear square: only the two edges not hidden by the front one.
    p.drawLine(QtCore.QPointF(5, 2), QtCore.QPointF(14, 2))
    p.drawLine(QtCore.QPointF(14, 2), QtCore.QPointF(14, 11))
    # Connectors.
    p.drawLine(QtCore.QPointF(2, 5), QtCore.QPointF(5, 2))
    p.drawLine(QtCore.QPointF(11, 5), QtCore.QPointF(14, 2))
    p.drawLine(QtCore.QPointF(11, 14), QtCore.QPointF(14, 11))


def _draw_fit(p, QtCore, QtGui, color):
    """Fit view: four corner brackets framing a center dot."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    L = 4.0
    for x, y, dx, dy in (
        (1, 1, 1, 1), (15, 1, -1, 1), (1, 15, 1, -1), (15, 15, -1, -1),
    ):
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx * L, y))
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + dy * L))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(6.75, 6.75, 2.5, 2.5))


def _draw_camera(p, QtCore, QtGui, color):
    """Camera body + lens (screenshot to clipboard)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRoundedRect(QtCore.QRectF(1.5, 4.5, 13, 9), 1.5, 1.5)
    # Viewfinder bump.
    p.drawLine(QtCore.QPointF(5.5, 4.5), QtCore.QPointF(6.5, 2.5))
    p.drawLine(QtCore.QPointF(6.5, 2.5), QtCore.QPointF(9.5, 2.5))
    p.drawLine(QtCore.QPointF(9.5, 2.5), QtCore.QPointF(10.5, 4.5))
    p.drawEllipse(QtCore.QRectF(5.5, 6.5, 5, 5))


def _draw_save(p, QtCore, QtGui, color):
    """Down arrow into a tray (save to file)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(8, 2), QtCore.QPointF(8, 10))
    p.drawLine(QtCore.QPointF(4.5, 6.5), QtCore.QPointF(8, 10))
    p.drawLine(QtCore.QPointF(11.5, 6.5), QtCore.QPointF(8, 10))
    p.drawLine(QtCore.QPointF(2, 13.5), QtCore.QPointF(14, 13.5))


def _draw_image(p, QtCore, QtGui, color):
    """Picture frame + mountain + sun dot (save image)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(1.5, 2.5, 13, 11))
    p.drawLine(QtCore.QPointF(3, 11.5), QtCore.QPointF(7, 6.5))
    p.drawLine(QtCore.QPointF(7, 6.5), QtCore.QPointF(10, 10))
    p.drawLine(QtCore.QPointF(10, 10), QtCore.QPointF(12.5, 7.5))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(10.2, 4.2, 2.2, 2.2))


def _eye_path(QtCore, QtGui):
    """Pointed-oval eye outline as a QPainterPath."""
    path = QtGui.QPainterPath()
    path.moveTo(1.5, 8)
    path.quadTo(8, 2.0, 14.5, 8)
    path.quadTo(8, 14.0, 1.5, 8)
    return path


def _draw_eye_off(p, QtCore, QtGui, color):
    """Eye + slash (hide selected)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPath(_eye_path(QtCore, QtGui))
    p.drawLine(QtCore.QPointF(3, 14), QtCore.QPointF(13, 2))


def _draw_isolate(p, QtCore, QtGui, color):
    """Center dot in four inward corner brackets (isolate selected)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    L = 3.5
    for x, y, dx, dy in (
        (2, 2, 1, 1), (14, 2, -1, 1), (2, 14, 1, -1), (14, 14, -1, -1),
    ):
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx * L, y))
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + dy * L))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(6, 6, 4, 4))


def _draw_reveal(p, QtCore, QtGui, color):
    """Four outward arrows from center (reveal all)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    A = 2.6
    for x, y, dx, dy in (
        (2, 2, 1, 1), (14, 2, -1, 1), (2, 14, 1, -1), (14, 14, -1, -1),
    ):
        # Shaft from near-center out to the corner + arrow head.
        p.drawLine(
            QtCore.QPointF(8 - dx * 1.5, 8 - dy * 1.5), QtCore.QPointF(x, y),
        )
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx * A, y))
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + dy * A))


def _draw_palette(p, QtCore, QtGui, color):
    """Circle + three state dots (color by group)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawEllipse(QtCore.QRectF(1.5, 1.5, 13, 13))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    # Diagonal triad — NOT two-eyes-plus-mouth, which reads as a
    # smiley at 16 px (contact-sheet finding, 2026-08-04).
    for cx, cy in ((4.9, 6.6), (9.0, 4.6), (10.6, 9.8)):
        p.drawEllipse(QtCore.QRectF(cx - 1.1, cy - 1.1, 2.2, 2.2))


def _draw_colormap(p, QtCore, QtGui, color):
    """Horizontal bar with step dividers, densities rising (color map)."""
    # State-dot-style fills of increasing alpha — reads as a ramp.
    p.setPen(QtCore.Qt.NoPen)
    for x0, alpha in ((1.5, 0.2), (5.83, 0.45), (10.17, 0.75)):
        qc = QtGui.QColor(color)
        qc.setAlphaF(qc.alphaF() * alpha)
        p.setBrush(qc)
        p.drawRect(QtCore.QRectF(x0, 5, 4.33, 6))
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(1.5, 5, 13, 6))


def _draw_section(p, QtCore, QtGui, color):
    """Square sliced by a plane line, cut half highlighted (sections)."""
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(_fill_brush(QtGui, color))
    tri = [QtCore.QPointF(2, 14), QtCore.QPointF(14, 2),
           QtCore.QPointF(14, 14)]
    p.drawPolygon(QtGui.QPolygonF(tri))
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(2, 2, 12, 12))
    p.drawLine(QtCore.QPointF(2, 14), QtCore.QPointF(14, 2))


def _draw_axes(p, QtCore, QtGui, color):
    """Origin triad (local axes)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    o = QtCore.QPointF(4, 12)
    p.drawLine(o, QtCore.QPointF(4, 2))       # up
    p.drawLine(o, QtCore.QPointF(14, 12))     # right
    p.drawLine(o, QtCore.QPointF(9.5, 6.5))   # diagonal (out of plane)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(2.9, 10.9, 2.2, 2.2))


def _draw_stages(p, QtCore, QtGui, color):
    """Two overlapping squares (stage activation)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(2, 5, 9, 9))
    p.drawRect(QtCore.QRectF(5, 2, 9, 9))


# ── Visibility ────────────────────────────────────────────────────────

def _draw_eye(p, QtCore, QtGui, color):
    """Open eye + pupil dot (show / visible)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPath(_eye_path(QtCore, QtGui))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(6.2, 6.2, 3.6, 3.6))


# ── Row / panel actions ───────────────────────────────────────────────

def _draw_close(p, QtCore, QtGui, color):
    """Diagonal cross (dismiss / delete row). Deliberately smaller
    than viewport glyphs — close targets read lighter."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(4.2, 4.2), QtCore.QPointF(11.8, 11.8))
    p.drawLine(QtCore.QPointF(11.8, 4.2), QtCore.QPointF(4.2, 11.8))


def _draw_add(p, QtCore, QtGui, color):
    """Plus (create / add)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(8, 3), QtCore.QPointF(8, 13))
    p.drawLine(QtCore.QPointF(3, 8), QtCore.QPointF(13, 8))


def _draw_gear(p, QtCore, QtGui, color):
    """Gear — hub dot + ring + eight radial teeth (settings). The
    filled hub is what keeps it reading as a gear (not a sun) at
    16 px."""
    import math
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawEllipse(QtCore.QRectF(4.6, 4.6, 6.8, 6.8))
    for i in range(8):
        a = math.tau * i / 8.0
        c, s = math.cos(a), math.sin(a)
        p.drawLine(
            QtCore.QPointF(8 + 4.6 * c, 8 + 4.6 * s),
            QtCore.QPointF(8 + 6.7 * c, 8 + 6.7 * s),
        )
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(6.8, 6.8, 2.4, 2.4))


def _draw_flip(p, QtCore, QtGui, color):
    """Two opposing horizontal arrows (reverse direction)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    # Top arrow → right.
    p.drawLine(QtCore.QPointF(2.5, 5.2), QtCore.QPointF(13.5, 5.2))
    p.drawLine(QtCore.QPointF(10.5, 2.4), QtCore.QPointF(13.5, 5.2))
    p.drawLine(QtCore.QPointF(10.5, 8.0), QtCore.QPointF(13.5, 5.2))
    # Bottom arrow ← left.
    p.drawLine(QtCore.QPointF(13.5, 10.8), QtCore.QPointF(2.5, 10.8))
    p.drawLine(QtCore.QPointF(5.5, 8.0), QtCore.QPointF(2.5, 10.8))
    p.drawLine(QtCore.QPointF(5.5, 13.6), QtCore.QPointF(2.5, 10.8))


def _draw_move_up(p, QtCore, QtGui, color):
    """Full-shaft arrow up (reorder earlier)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(8, 13.5), QtCore.QPointF(8, 2.5))
    p.drawLine(QtCore.QPointF(3.8, 6.7), QtCore.QPointF(8, 2.5))
    p.drawLine(QtCore.QPointF(12.2, 6.7), QtCore.QPointF(8, 2.5))


def _draw_move_down(p, QtCore, QtGui, color):
    """Full-shaft arrow down (reorder later)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(8, 2.5), QtCore.QPointF(8, 13.5))
    p.drawLine(QtCore.QPointF(3.8, 9.3), QtCore.QPointF(8, 13.5))
    p.drawLine(QtCore.QPointF(12.2, 9.3), QtCore.QPointF(8, 13.5))


def _draw_dot(p, QtCore, QtGui, color):
    """Filled state dot (series swatch, active marker)."""
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(5, 5, 6, 6))


# ── Transport (time scrubber) ────────────────────────────────────────
# Language: triangle = play, twin bars = pause, chevron = step ±1,
# chevron + terminal bar = jump to first/last. Never size-only
# distinctions — each control keeps its own silhouette at 16 px.

def _tri(QtCore, x0, x1):
    P = QtCore.QPointF
    return [P(x0, 3.2), P(x1, 8), P(x0, 12.8)]


def _draw_play(p, QtCore, QtGui, color):
    """Right-pointing triangle, face-highlight fill (play)."""
    tri = _tri(QtCore, 4.6, 12.4)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(_fill_brush(QtGui, color))
    p.drawPolygon(QtGui.QPolygonF(tri))
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPolygon(QtGui.QPolygonF(tri))


def _draw_pause(p, QtCore, QtGui, color):
    """Two vertical bars (pause)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(_fill_brush(QtGui, color))
    p.drawRect(QtCore.QRectF(4.4, 3.4, 2.4, 9.2))
    p.drawRect(QtCore.QRectF(9.2, 3.4, 2.4, 9.2))


def _chevron(p, QtCore, tip_x, base_x):
    p.drawLine(QtCore.QPointF(base_x, 3.2), QtCore.QPointF(tip_x, 8))
    p.drawLine(QtCore.QPointF(base_x, 12.8), QtCore.QPointF(tip_x, 8))


def _draw_step_back(p, QtCore, QtGui, color):
    """Left chevron (step -1)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    _chevron(p, QtCore, 5.4, 10.6)


def _draw_step_forward(p, QtCore, QtGui, color):
    """Right chevron (step +1)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    _chevron(p, QtCore, 10.6, 5.4)


def _draw_skip_first(p, QtCore, QtGui, color):
    """Left chevron + terminal bar (jump to first step)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(3.6, 3.2), QtCore.QPointF(3.6, 12.8))
    _chevron(p, QtCore, 6.8, 12.4)


def _draw_skip_last(p, QtCore, QtGui, color):
    """Right chevron + terminal bar (jump to last step)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.drawLine(QtCore.QPointF(12.4, 3.2), QtCore.QPointF(12.4, 12.8))
    _chevron(p, QtCore, 9.2, 3.6)


def _draw_movie(p, QtCore, QtGui, color):
    """Video frame + play triangle (export animation)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRoundedRect(QtCore.QRectF(1.5, 3, 13, 10), 1.5, 1.5)
    P = QtCore.QPointF
    tri = [P(6.4, 5.6), P(10.6, 8), P(6.4, 10.4)]
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(_fill_brush(QtGui, color))
    p.drawPolygon(QtGui.QPolygonF(tri))
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPolygon(QtGui.QPolygonF(tri))


# ── Probes (viewport HUD) ────────────────────────────────────────────

def _draw_probe_point(p, QtCore, QtGui, color):
    """Circled dot (point probe)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawEllipse(QtCore.QRectF(2.5, 2.5, 11, 11))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(6.2, 6.2, 3.6, 3.6))


def _draw_probe_line(p, QtCore, QtGui, color):
    """Sampled arc — curve between two endpoint dots (line probe)."""
    path = QtGui.QPainterPath()
    path.moveTo(2.5, 11.5)
    path.quadTo(8, 2.0, 13.5, 6.5)
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPath(path)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(QtCore.QRectF(1.3, 10.3, 2.4, 2.4))
    p.drawEllipse(QtCore.QRectF(12.3, 5.3, 2.4, 2.4))


def _draw_probe_slice(p, QtCore, QtGui, color):
    """Sampling grid — square + inner grid lines (slice probe)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(QtCore.QRectF(2, 2, 12, 12))
    p.drawLine(QtCore.QPointF(6, 2), QtCore.QPointF(6, 14))
    p.drawLine(QtCore.QPointF(10, 2), QtCore.QPointF(10, 14))
    p.drawLine(QtCore.QPointF(2, 6), QtCore.QPointF(14, 6))
    p.drawLine(QtCore.QPointF(2, 10), QtCore.QPointF(14, 10))


def _draw_stop(p, QtCore, QtGui, color):
    """Rounded square, face-highlight fill (stop the active probe)."""
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(_fill_brush(QtGui, color))
    p.drawRoundedRect(QtCore.QRectF(4, 4, 8, 8), 1.5, 1.5)


def _draw_clear(p, QtCore, QtGui, color):
    """Backspace key — pentagon + inner cross (clear all)."""
    P = QtCore.QPointF
    poly = [P(5.6, 3), P(14, 3), P(14, 13), P(5.6, 13), P(1.8, 8)]
    p.setPen(_stroke_pen(QtGui, QtCore, color, _DESIGN))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawPolygon(QtGui.QPolygonF(poly))
    p.drawLine(P(7.6, 6), P(11.6, 10))
    p.drawLine(P(11.6, 6), P(7.6, 10))


_GLYPHS: Dict[str, Callable[..., None]] = {
    "view_top": _draw_view_top,
    "view_bottom": _draw_view_bottom,
    "view_front": _draw_view_front,
    "view_back": _draw_view_back,
    "view_left": _draw_view_left,
    "view_right": _draw_view_right,
    "view_iso": _draw_view_iso,
    "ortho": _draw_ortho,
    "fit": _draw_fit,
    "camera": _draw_camera,
    "save": _draw_save,
    "image": _draw_image,
    "eye_off": _draw_eye_off,
    "isolate": _draw_isolate,
    "reveal": _draw_reveal,
    "palette": _draw_palette,
    "colormap": _draw_colormap,
    "section": _draw_section,
    "axes": _draw_axes,
    "stages": _draw_stages,
    # Visibility.
    "eye": _draw_eye,
    # Row / panel actions.
    "close": _draw_close,
    "add": _draw_add,
    "gear": _draw_gear,
    "flip": _draw_flip,
    "move_up": _draw_move_up,
    "move_down": _draw_move_down,
    "dot": _draw_dot,
    # Transport (time scrubber).
    "play": _draw_play,
    "pause": _draw_pause,
    "step_back": _draw_step_back,
    "step_forward": _draw_step_forward,
    "skip_first": _draw_skip_first,
    "skip_last": _draw_skip_last,
    "movie": _draw_movie,
    # Probes (viewport HUD).
    "probe_point": _draw_probe_point,
    "probe_line": _draw_probe_line,
    "probe_slice": _draw_probe_slice,
    "stop": _draw_stop,
    "clear": _draw_clear,
}


def glyph_names() -> "tuple[str, ...]":
    """Every glyph the factory can draw (stable, for tests)."""
    return tuple(_GLYPHS)


def toolbar_icon(name: str, color: str, *, size: int = _HIT,
                 dpr: float = 1.0) -> Any:
    """A themed ``QIcon`` for glyph ``name``, cached per look.

    Parameters
    ----------
    name
        Key in the glyph registry — see :func:`glyph_names`.
    color
        Stroke color (normally ``palette.icon``).
    size
        Hit-area edge in logical px (default 22 per ADR 0087 INV-4);
        the glyph occupies a centered 16/22 fraction of it.
    dpr
        Device-pixel-ratio of the target widget so the pixmap renders
        crisp on HiDPI — never rescaled.
    """
    draw = _GLYPHS.get(name)
    if draw is None:
        raise KeyError(
            f"unknown glyph {name!r} — known: {sorted(_GLYPHS)}",
        )
    key = (name, color, int(size), int(round(dpr * 1000)))
    icon = _CACHE.get(key)
    if icon is not None:
        return icon
    QtCore, QtGui = _qt()
    pix = QtGui.QPixmap(round(size * dpr), round(size * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(QtGui.QColor(0, 0, 0, 0))
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    # Center the 16-px design box in the hit area, scaled with size.
    scale = (size / float(_HIT)) * 1.0
    glyph_px = _DESIGN * scale
    off = (size - glyph_px) / 2.0
    p.translate(off, off)
    p.scale(scale, scale)
    draw(p, QtCore, QtGui, color)
    p.end()
    icon = QtGui.QIcon(pix)
    _CACHE[key] = icon
    return icon
