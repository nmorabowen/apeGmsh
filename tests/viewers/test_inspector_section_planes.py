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
from apeGmsh.viewers.session._inspector import (
    _CLIP_AXES,
    MAX_ACTIVE_PLANES,
    MeshInspectorPage,
)


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


# ---------------------------------------------------------------------
# The six-active-plane GL cap
# ---------------------------------------------------------------------
#
# `gl_ClipDistance` guarantees six. Past that the driver is FREE to
# ignore the extras — so a seventh active plane does not fail, it draws
# a model that looks cut and is not. Render enforces the cap (§3 puts it
# where planes meet a backend); this section is the half that keeps a
# person from ever creating the situation without being told.

def _fill_to_cap(p, view, n=MAX_ACTIVE_PLANES):
    for _ in range(n):
        p._clip_add.click()
    assert sum(1 for c in view.clips if c.active) == n


def test_no_note_while_the_cap_is_far_away(page):
    """0087 INV-2: a permanent "at most 6" caption is chrome nobody
    reads, and then the once it matters it is invisible."""
    p, view = page
    p._clip_add.click()
    assert not p._clip_note.isVisibleTo(p.widget)


def test_the_note_appears_when_the_cap_is_reached(page):
    """Said BEFORE the seventh is tried, so the refusal is expected
    rather than mysterious."""
    p, view = page
    _fill_to_cap(p, view)
    assert p._clip_note.isVisibleTo(p.widget)
    assert str(MAX_ACTIVE_PLANES) in p._clip_note.text()


def test_a_seventh_plane_is_added_INACTIVE_not_refused(page):
    """``ClipPlaneSetController.add``'s behaviour, and the friendlier
    half of the rule: the plane exists, ready to be swapped in for one
    of the six, rather than a button that appears to do nothing."""
    p, view = page
    _fill_to_cap(p, view)
    p._clip_add.click()

    assert len(view.clips) == MAX_ACTIVE_PLANES + 1, "the plane was refused"
    assert view.clips[-1].active is False, "a seventh plane is cutting"
    assert sum(1 for c in view.clips if c.active) == MAX_ACTIVE_PLANES
    assert p._clip_note.isVisibleTo(p.widget)
    assert "inactive" in p._clip_note.text().lower()


def test_checking_cut_on_a_seventh_plane_is_refused_and_says_so(page):
    """The defect one layer up: a silently-ignored checkbox would leave
    the user looking at a ticked box and an uncut model."""
    p, view = page
    _fill_to_cap(p, view)
    p._clip_add.click()
    spare = view.clips[-1].plane_id

    p._clip_rows[spare].cut.setChecked(True)

    assert view.clips[-1].active is False, "the seventh plane started cutting"
    assert p._clip_rows[spare].cut.isChecked() is False, (
        "the checkbox stayed ticked while the record said inactive — "
        "exactly the lie the cap exists to prevent"
    )
    assert p._clip_note.isVisibleTo(p.widget)


def test_freeing_a_slot_lets_the_spare_plane_cut(page):
    """The cap must be a queue, not a wall: un-check one, check the
    other, and it works."""
    p, view = page
    _fill_to_cap(p, view)
    p._clip_add.click()
    first, spare = view.clips[0].plane_id, view.clips[-1].plane_id

    p._clip_rows[first].cut.setChecked(False)
    p._clip_rows[spare].cut.setChecked(True)

    by_id = {c.plane_id: c for c in view.clips}
    assert by_id[first].active is False
    assert by_id[spare].active is True
    assert sum(1 for c in view.clips if c.active) == MAX_ACTIVE_PLANES


def test_editing_an_ALREADY_active_plane_at_the_cap_still_works(page):
    """The trap in the predicate.

    The row writes all four fields at once, so every offset nudge on a
    plane that is already cutting re-asserts ``active=True``. Counting
    all actives instead of the OTHERS would read that as a seventh
    activation and refuse — freezing every control on a pane that has
    exactly six planes.
    """
    p, view = page
    _fill_to_cap(p, view)
    first = view.clips[0].plane_id

    p._clip_rows[first].offset.setValue(3.5)

    by_id = {c.plane_id: c for c in view.clips}
    assert by_id[first].offset == pytest.approx(3.5), (
        "an offset edit was refused as if it were a seventh activation"
    )
    assert by_id[first].active is True


def test_a_record_that_already_exceeds_the_cap_says_so(page):
    """Python and snapshot restore can both author seven actives — §3
    keeps the cap out of the IR on purpose. Realize then draws only
    six, and this note is the only place a person would find out.
    """
    p, view = page
    for i in range(MAX_ACTIVE_PLANES + 2):
        view.add_clip((1.0, 0.0, 0.0), offset=float(i))
    p.refresh()

    assert sum(1 for c in view.clips if c.active) == MAX_ACTIVE_PLANES + 2
    assert p._clip_note.isVisibleTo(p.widget)
    text = p._clip_note.text()
    assert str(MAX_ACTIVE_PLANES + 2) in text and str(MAX_ACTIVE_PLANES) in text


def test_the_note_clears_when_the_pane_drops_back_under_the_cap(page):
    p, view = page
    _fill_to_cap(p, view)
    assert p._clip_note.isVisibleTo(p.widget)
    p._clip_rows[view.clips[0].plane_id].remove.click()
    assert not p._clip_note.isVisibleTo(p.widget)
