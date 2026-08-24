"""ADR 0098 A7 (R1) — section planes are reachable from the window.

The *reach* half of R1, and the reason it is in R1 at all. Restoring
the gizmo gives back the direct-manipulation gesture, but a plane that
only ``view.add_clip(...)`` can create is precisely the A6.1
"works from Python, unreachable in the window" row the parity gate was
built to end: ``make_clip_planes_dock`` has exactly ONE caller,
``results_viewer.py:975``, so the session window could not make a
section plane at all.

These tests drive the inspector page the way a person does — press the
Add button, toggle the controls — and assert on the session record,
never on the widget's own state. A row that showed a checkbox and
wrote nothing would pass an is-it-checked test and fail every one of
these.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.results.session import ResultsSession
from apeGmsh.viewers.session._inspector import _CLIP_AXES, MeshInspectorPage


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeRealized:
    """What ``set_realized`` reads: the pane id and the reference box."""

    def __init__(self, pane_id, bounds=(0.0, 0.0, 0.0, 10.0, 20.0, 30.0)):
        self.pane_id = pane_id
        self.reference_bounds = bounds
        self.layers = ()


@pytest.fixture
def page(qapp):
    """An inspector page over a bare session — no results needed: the
    clip section reads the VIEW, not the data."""
    session = ResultsSession()
    view = session.add_view(name="mesh")
    page = MeshInspectorPage(session, view)
    yield page, view
    page.dispose()


# ---------------------------------------------------------------------
# Empty state — 0087 INV-2
# ---------------------------------------------------------------------

def test_an_empty_section_offers_add_and_no_plane_controls(page):
    """INV-2: no offset field for a plane that does not exist.

    Not "a disabled offset field" — none at all. There is nothing for
    it to be about, and a dead control is the thing INV-2 forbids.
    """
    p, view = page
    assert view.clips == ()
    assert p._clip_add.isEnabled(), "Add must be live — a view can always cut"
    assert p._clip_rows == {}
    assert not p._clip_rows_host.isVisibleTo(p.widget)


def test_the_axis_choices_are_the_three_world_axes(page):
    p, _view = page
    assert [p._clip_axis.itemText(i) for i in range(p._clip_axis.count())] == [
        label for label, _n in _CLIP_AXES
    ]


# ---------------------------------------------------------------------
# Add — the gesture R1 exists to restore
# ---------------------------------------------------------------------

def test_add_plane_creates_a_clip_on_the_VIEW(page):
    """The whole point: no Python, and the record changes."""
    p, view = page
    p._clip_add.click()
    assert len(view.clips) == 1
    assert view.clips[0].active is True
    assert view.clips[0].gizmo_visible is True, (
        "a plane added by hand must come with its handle showing, or the "
        "gesture R1 restored is still unreachable"
    )


@pytest.mark.parametrize("index,expected", list(enumerate(
    [n for _l, n in _CLIP_AXES])))
def test_the_axis_choice_sets_the_normal(page, index, expected):
    p, view = page
    p._clip_axis.setCurrentIndex(index)
    p._clip_add.click()
    assert view.clips[0].normal == pytest.approx(expected)


def test_a_new_plane_lands_through_the_middle_of_the_model(page):
    """A plane that appears outside the mesh reads as "Add did
    nothing" — the offset is seeded from the realized extent."""
    p, view = page
    p.set_realized(FakeRealized(view.id, (0.0, 0.0, 0.0, 10.0, 20.0, 30.0)))
    p._clip_axis.setCurrentIndex(1)          # Y, spanning 0..20
    p._clip_add.click()
    assert view.clips[0].offset == pytest.approx(10.0)


def test_adding_a_plane_with_no_realization_still_works(page):
    """The section must not depend on a realize having happened —
    otherwise the first thing a user tries on a fresh pane refuses."""
    p, view = page
    p._clip_add.click()
    assert len(view.clips) == 1
    assert view.clips[0].offset == pytest.approx(0.0)


def test_two_adds_make_two_planes_with_distinct_ids(page):
    p, view = page
    p._clip_add.click()
    p._clip_add.click()
    assert len({c.plane_id for c in view.clips}) == 2


# ---------------------------------------------------------------------
# The row edits the record
# ---------------------------------------------------------------------

def _row(p, view):
    return p._clip_rows[view.clips[0].plane_id]


def test_the_row_appears_and_restates_the_record(page):
    p, view = page
    p._clip_add.click()
    row = _row(p, view)
    assert p._clip_rows_host.isVisibleTo(p.widget)
    assert row.cut.isChecked() is True
    assert row.eye.isChecked() is True
    assert row.name.text() == view.clips[0].name


def test_unchecking_cut_deactivates_the_plane_without_deleting_it(page):
    p, view = page
    p._clip_add.click()
    _row(p, view).cut.setChecked(False)
    assert view.clips[0].active is False
    assert len(view.clips) == 1, "un-cutting must not remove the plane"


def test_the_gizmo_toggle_writes_gizmo_visible(page):
    p, view = page
    p._clip_add.click()
    _row(p, view).eye.setChecked(False)
    assert view.clips[0].gizmo_visible is False
    assert view.clips[0].active is True, "hiding the handle unset the cut"


def test_reverse_flips_which_half_survives(page):
    p, view = page
    p._clip_add.click()
    row = _row(p, view)
    assert view.clips[0].flipped is False
    row.flip.click()
    assert view.clips[0].flipped is True
    row.flip.click()
    assert view.clips[0].flipped is False


def test_the_offset_spinbox_moves_the_plane(page):
    p, view = page
    p._clip_add.click()
    _row(p, view).offset.setValue(4.25)
    assert view.clips[0].offset == pytest.approx(4.25)


def test_remove_deletes_the_plane_and_its_row(page):
    p, view = page
    p._clip_add.click()
    plane_id = view.clips[0].plane_id
    p._clip_rows[plane_id].remove.click()
    assert view.clips == ()
    assert p._clip_rows == {}
    assert not p._clip_rows_host.isVisibleTo(p.widget)


# ---------------------------------------------------------------------
# Projection — the page restates, it does not own
# ---------------------------------------------------------------------

def test_a_python_edit_shows_up_in_the_row(page):
    """§9: the page is a projection, so a scripted edit reads back the
    same as a click would."""
    p, view = page
    p._clip_add.click()
    view.set_clip(view.clips[0].plane_id, offset=7.5, active=False)
    p.refresh()
    row = _row(p, view)
    assert row.offset.value() == pytest.approx(7.5)
    assert row.cut.isChecked() is False


def test_a_plane_added_in_python_grows_a_row(page):
    p, view = page
    view.add_clip((0.0, 0.0, 1.0), offset=1.0)
    p.refresh()
    assert len(p._clip_rows) == 1


# ---------------------------------------------------------------------
# Density discipline — the drag hazard
# ---------------------------------------------------------------------

def test_rows_are_not_rebuilt_when_only_the_offset_moves(page):
    """A gizmo drag ticks the session on EVERY mouse-move, and the page
    refreshes on every tick. Rebuilding the rows at that rate would
    destroy and re-create the widget under the user's cursor — the
    focus-flicker hazard ``_outline.py`` documents for pane rows.

    Identity of the row widget is the assertion, because "it looks the
    same" is exactly what a rebuilt row does.
    """
    p, view = page
    p._clip_add.click()
    plane_id = view.clips[0].plane_id
    row = p._clip_rows[plane_id]
    widget = row.widget

    for i in range(20):                       # a drag
        view.set_clip(plane_id, offset=0.1 * i)
        p.refresh()

    assert p._clip_rows[plane_id] is row, "the row object was rebuilt"
    assert p._clip_rows[plane_id].widget is widget, "the widget was rebuilt"
    assert row.offset.value() == pytest.approx(1.9)


def test_the_ROW_has_its_own_sync_guard(page):
    """Belt and braces, tested separately.

    The page guards ``_on_clip_changed`` with its own ``_syncing``
    flag, which masks the row's. But the row is what Qt calls back
    when ``setChecked``/``setValue`` fire during a sync, so it needs a
    guard of its own — and an untested guard is one a refactor
    silently deletes. Drive ``row.sync`` directly, outside any page
    refresh, and require no tick.
    """
    p, view = page
    p._clip_add.click()
    clip = view.clips[0]
    row = p._clip_rows[clip.plane_id]

    ticks: list = []
    view._notify = lambda: ticks.append(1)
    # Values that DIFFER from the widgets, so every setter fires.
    view._clips[0] = type(clip)(
        plane_id=clip.plane_id, name="Renamed", normal=clip.normal,
        offset=12.5, active=False, flipped=True, gizmo_visible=False,
    )
    row.sync(view.clips[0], 0.1)

    assert ticks == [], "restating the row wrote back to the record"
    assert row.offset.value() == pytest.approx(12.5)
    assert row.cut.isChecked() is False
    assert row.flipped is True


def test_syncing_a_row_does_not_write_back(page):
    """The sync guard: restating the record must not look like a
    gesture, or a refresh mid-drag would fight the drag."""
    p, view = page
    p._clip_add.click()
    plane_id = view.clips[0].plane_id
    view.set_clip(plane_id, offset=3.0)

    ticks: list = []
    view._notify = lambda: ticks.append(1)
    p.refresh()
    assert ticks == [], "a projection wrote back to the record"
