"""Plan 03 — EyeIconDelegate unit tests.

Tests the delegate in isolation against a minimal ``QTreeWidget``:

* paint() draws something only when the row exposes ``ROLE_VISIBLE``
* editorEvent() fires ``icon_clicked`` on left-click in the icon
  hit area and consumes the event
* clicks outside the icon area / on rows without ``ROLE_VISIBLE`` are
  pass-through (return False, never emit)
* glyphs are cached and re-rendered when the theme color changes
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.ui._eye_icon_delegate import (
    ROLE_EFFECTIVE,
    ROLE_VISIBLE,
    resolve_delegate_class,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def tree(qapp):
    from qtpy import QtWidgets
    t = QtWidgets.QTreeWidget()
    t.setColumnCount(1)
    t.setHeaderHidden(True)
    yield t
    try:
        t.close()
    except Exception:
        pass


@pytest.fixture
def delegate(tree):
    cls = resolve_delegate_class()
    d = cls(tree)
    tree.setItemDelegateForColumn(0, d)
    return d


def _add_row(tree, label: str, *, visible=None):
    """Add a root-level tree row optionally tagged with ROLE_VISIBLE."""
    from qtpy import QtWidgets
    item = QtWidgets.QTreeWidgetItem([label])
    tree.addTopLevelItem(item)
    if visible is not None:
        item.setData(0, ROLE_VISIBLE, bool(visible))
    return item


# =====================================================================
# ROLE_VISIBLE contract
# =====================================================================


def test_role_visible_constant_distinct():
    """ROLE_VISIBLE must not collide with existing outline roles."""
    from apeGmsh.viewers.ui._outline_tree import (
        _ROLE_STAGE_ID, _ROLE_DIAGRAM_OBJ, _ROLE_GROUP_KEY,
        _ROLE_PLOT_KEY, _ROLE_COMPOSITION_KEY, _ROLE_GEOMETRY_KEY,
    )
    other_roles = {
        _ROLE_STAGE_ID, _ROLE_DIAGRAM_OBJ, _ROLE_GROUP_KEY,
        _ROLE_PLOT_KEY, _ROLE_COMPOSITION_KEY, _ROLE_GEOMETRY_KEY,
    }
    assert ROLE_VISIBLE not in other_roles


# =====================================================================
# Click handling
# =====================================================================


def _mouse_press(x, y):
    """Build a QMouseEvent for a left-button press at (x, y)."""
    from qtpy import QtCore, QtGui
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(x, y),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )


def _editor_event(delegate, tree, item, x_local):
    """Manually drive editorEvent for the row hosting ``item``.

    Returns the boolean ``editorEvent`` returns (True = consumed).
    """
    from qtpy import QtWidgets
    index = tree.indexFromItem(item, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = tree.visualRect(index)
    # If the tree hasn't been shown, visualRect is empty. Stub it to
    # a known rect so the relative-x math is well-defined.
    if option.rect.width() == 0:
        from qtpy import QtCore as _QtCore
        option.rect = _QtCore.QRect(0, 0, 200, 24)
    event = _mouse_press(option.rect.x() + x_local, option.rect.y() + 5)
    return delegate.editorEvent(event, tree.model(), option, index)


def test_click_in_icon_area_emits_and_consumes(qapp, tree, delegate):
    """Left-click within the icon hit area on a visibility row emits
    ``icon_clicked`` carrying the item and consumes the event."""
    item = _add_row(tree, "Visible row", visible=True)
    seen: list = []
    delegate.icon_clicked.connect(lambda it: seen.append(it))

    consumed = _editor_event(delegate, tree, item, x_local=5)

    assert consumed is True
    assert seen == [item]


def test_click_outside_icon_area_is_passthrough(qapp, tree, delegate):
    """Click outside the left icon strip does NOT emit and does NOT
    consume — the standard item-view click handling still runs."""
    item = _add_row(tree, "Visible row", visible=True)
    seen: list = []
    delegate.icon_clicked.connect(lambda it: seen.append(it))

    consumed = _editor_event(delegate, tree, item, x_local=120)

    assert consumed is False
    assert seen == []


def test_click_on_row_without_role_is_passthrough(qapp, tree, delegate):
    """Rows that did not opt into ROLE_VISIBLE never fire the signal."""
    item = _add_row(tree, "Plain row")    # no ROLE_VISIBLE
    seen: list = []
    delegate.icon_clicked.connect(lambda it: seen.append(it))

    consumed = _editor_event(delegate, tree, item, x_local=5)
    assert consumed is False
    assert seen == []


def test_emits_for_hidden_row_too(qapp, tree, delegate):
    """Both 'visible=True' and 'visible=False' rows respond to clicks
    — the toggle decision is the caller's, not the delegate's."""
    item = _add_row(tree, "Hidden row", visible=False)
    seen: list = []
    delegate.icon_clicked.connect(lambda it: seen.append(it))

    consumed = _editor_event(delegate, tree, item, x_local=5)
    assert consumed is True
    assert seen == [item]


def test_non_left_button_click_is_passthrough(qapp, tree, delegate):
    from qtpy import QtCore, QtGui, QtWidgets
    item = _add_row(tree, "Visible row", visible=True)
    seen: list = []
    delegate.icon_clicked.connect(lambda it: seen.append(it))

    index = tree.indexFromItem(item, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = QtCore.QRect(0, 0, 200, 24)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(5, 5),
        QtCore.Qt.RightButton,
        QtCore.Qt.RightButton,
        QtCore.Qt.NoModifier,
    )
    consumed = delegate.editorEvent(event, tree.model(), option, index)
    assert consumed is False
    assert seen == []


# =====================================================================
# Glyph rendering
# =====================================================================


def test_glyph_cache_returns_same_instance(qapp, delegate):
    """The cached pixmap is returned without re-rendering when neither
    visibility nor color has changed."""
    a = delegate._glyph(True, "#FFFFFF")
    b = delegate._glyph(True, "#FFFFFF")
    assert a is b


def test_glyph_cache_invalidates_on_color_change(qapp, delegate):
    """A different color flushes the cache so the next call repaints
    against the new theme."""
    a = delegate._glyph(True, "#FFFFFF")
    b = delegate._glyph(True, "#000000")
    assert a is not b


def test_visible_and_hidden_glyphs_are_different(qapp, delegate):
    """Sanity — the two states render different pixmaps."""
    on = delegate._glyph(True, "#FFFFFF")
    off = delegate._glyph(False, "#FFFFFF")
    assert on is not off


# =====================================================================
# Tri-state — ROLE_EFFECTIVE / the dimmed (gated) glyph
# =====================================================================


def test_role_effective_constant_distinct():
    """ROLE_EFFECTIVE must not collide with ROLE_VISIBLE or the
    outline's existing roles."""
    from apeGmsh.viewers.ui._outline_tree import (
        _ROLE_STAGE_ID, _ROLE_DIAGRAM_OBJ, _ROLE_GROUP_KEY,
        _ROLE_PLOT_KEY, _ROLE_COMPOSITION_KEY, _ROLE_GEOMETRY_KEY,
    )
    other_roles = {
        ROLE_VISIBLE,
        _ROLE_STAGE_ID, _ROLE_DIAGRAM_OBJ, _ROLE_GROUP_KEY,
        _ROLE_PLOT_KEY, _ROLE_COMPOSITION_KEY, _ROLE_GEOMETRY_KEY,
    }
    assert ROLE_EFFECTIVE not in other_roles


def test_dimmed_glyph_differs_from_both_binary_states(qapp, delegate):
    """The gated state renders its own pixmap, pixel-distinct from
    the filled dot and the hollow ring, in a light AND a dark theme
    colour (the palette text colour in each)."""
    for color in ("#FFFFFF", "#000000"):    # dark theme / light theme
        on = delegate._glyph(True, color)
        dim = delegate._glyph(True, color, dimmed=True)
        off = delegate._glyph(False, color)
        assert dim is not on and dim is not off
        assert dim.toImage() != on.toImage()
        assert dim.toImage() != off.toImage()


def test_dimmed_glyph_is_cached(qapp, delegate):
    a = delegate._glyph(True, "#FFFFFF", dimmed=True)
    b = delegate._glyph(True, "#FFFFFF", dimmed=True)
    assert a is b
    # …and the cache flushes with the rest on a theme change.
    c = delegate._glyph(True, "#000000", dimmed=True)
    assert c is not a


def _paint_row(delegate, tree, item):
    """Drive delegate.paint() for ``item`` against a scratch pixmap."""
    from qtpy import QtCore, QtGui, QtWidgets
    index = tree.indexFromItem(item, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = QtCore.QRect(0, 0, 200, 24)
    canvas = QtGui.QPixmap(200, 24)
    canvas.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(canvas)
    try:
        delegate.paint(painter, option, index)
    finally:
        painter.end()


def test_paint_uses_dimmed_glyph_when_gated(qapp, tree, delegate):
    """ROLE_VISIBLE True + ROLE_EFFECTIVE False must paint the dimmed
    state — asserted through the glyph cache the paint populates."""
    item = _add_row(tree, "Gated row", visible=True)
    item.setData(0, ROLE_EFFECTIVE, False)

    _paint_row(delegate, tree, item)

    assert delegate._pix_dimmed is not None
    assert delegate._pix_visible is None
    assert delegate._pix_hidden is None


def test_paint_absent_effective_role_behaves_as_before(qapp, tree, delegate):
    """Rows that never set ROLE_EFFECTIVE keep the two-state contract
    (geometry / composition rows are unchanged)."""
    item = _add_row(tree, "Plain visible row", visible=True)

    _paint_row(delegate, tree, item)

    assert delegate._pix_visible is not None
    assert delegate._pix_dimmed is None


def test_paint_effective_true_paints_normal_dot(qapp, tree, delegate):
    item = _add_row(tree, "Effective row", visible=True)
    item.setData(0, ROLE_EFFECTIVE, True)

    _paint_row(delegate, tree, item)

    assert delegate._pix_visible is not None
    assert delegate._pix_dimmed is None


def test_paint_hidden_intent_wins_over_effective(qapp, tree, delegate):
    """ROLE_VISIBLE False renders the hollow ring regardless of the
    effective role — intent-off is already the stronger statement."""
    item = _add_row(tree, "Hidden row", visible=False)
    item.setData(0, ROLE_EFFECTIVE, False)

    _paint_row(delegate, tree, item)

    assert delegate._pix_hidden is not None
    assert delegate._pix_dimmed is None
