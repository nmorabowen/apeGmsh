"""ADR 0098 A5.3 — where a colour scale was dragged is session state.

The pure half of Amendment 5: the view record and the snapshot. The Qt
half (interactor install, controller re-bind, write-back) is in
``tests/viewers/test_legend_binding.py``.

The law being pinned is a split: legend EXISTENCE stays derived from
the occupied colour-mapped slots (§5, unchanged), while legend
PLACEMENT is remembered. Every test here is about that seam holding in
both directions — placement cannot conjure a legend, and clearing a
slot cannot leave a placement behind.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.session import (
    Contour,
    Deform,
    Gauss,
    LegendPlacement,
    ResultsSession,
    restore_snapshot,
    snapshot,
)


@pytest.fixture
def view():
    return ResultsSession().add_view()


# ----------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------

def test_unplaced_legend_has_no_placement(view):
    view.contour = Contour("stress_zz")
    assert view.legend_placement("stress_zz") is None
    assert view.legend_placements() == {}


def test_set_and_read_back(view):
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.25, 0.4), 1.5)
    placement = view.legend_placement("stress_zz")
    assert placement == LegendPlacement(anchor=(0.25, 0.4), font_scale=1.5)


def test_font_scale_is_optional(view):
    """Dragged but never resized — the common case."""
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.1, 0.2))
    assert view.legend_placement("stress_zz").font_scale is None


def test_placement_coerces_to_float(view):
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0, 1), 2)
    placement = view.legend_placement("stress_zz")
    assert placement.anchor == (0.0, 1.0)
    assert isinstance(placement.anchor[0], float)
    assert placement.font_scale == 2.0


def test_placements_returns_a_copy(view):
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.3, 0.3))
    view.legend_placements().clear()
    assert view.legend_placement("stress_zz") is not None


def test_refuses_a_field_no_slot_causes(view):
    """Placement describes a legend; it cannot invent one (§5)."""
    view.deform = Deform("displacement", 10.0)
    with pytest.raises(ValueError, match="No legend for field"):
        view.set_legend_placement("stress_zz", (0.1, 0.1))


def test_clear_returns_it_to_the_stack(view):
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.25, 0.4))
    view.clear_legend_placement("stress_zz")
    assert view.legend_placement("stress_zz") is None


def test_clear_of_an_unplaced_field_is_a_noop(view):
    view.contour = Contour("stress_zz")
    view.clear_legend_placement("stress_zz")
    view.clear_legend_placement("never_existed")


def test_clearing_the_slot_prunes_the_placement(view):
    """INV-LEGEND-2 — the legend dies with the slot, and so does
    everything said about it."""
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.25, 0.4))
    view.contour = None
    assert view.legend_placements() == {}


def test_two_slots_one_field_share_one_placement(view):
    """A contour and a Gauss slot of the same field are ONE legend
    (§5), so they cannot hold two placements — which is why the record
    is keyed by field, not by slot."""
    view.contour = Contour("stress_zz")
    view.gauss = Gauss("stress_zz")
    assert len(view.legends()) == 1
    view.set_legend_placement("stress_zz", (0.5, 0.5))
    view.contour = None
    # The Gauss slot still causes the legend, so the placement lives.
    assert view.legend_placement("stress_zz") is not None


# ----------------------------------------------------------------------
# A5.6-9 — placement must not enter the reconciler's structure signature
# ----------------------------------------------------------------------

def test_placement_is_not_on_the_legend_record(view):
    """The signature term. ``_pane_signature`` includes
    ``pane.legends()``; if placement rode on that record every
    mouse-move of a drag would compare unequal and cost a full
    realize."""
    view.contour = Contour("stress_zz")
    before = view.legends()
    view.set_legend_placement("stress_zz", (0.25, 0.4), 1.5)
    assert view.legends() == before


# ----------------------------------------------------------------------
# A5.6-6 — the snapshot
# ----------------------------------------------------------------------

def _round_trip(session):
    return restore_snapshot(snapshot(session))


def test_snapshot_round_trip_preserves_placement():
    session = ResultsSession()
    view = session.add_view()
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.25, 0.4), 1.5)

    restored = _round_trip(session)
    pane = restored.session.panes[0]
    assert pane.legend_placement("stress_zz") == LegendPlacement(
        anchor=(0.25, 0.4), font_scale=1.5,
    )
    assert not restored.notices


def test_snapshot_round_trip_without_font_scale():
    session = ResultsSession()
    view = session.add_view()
    view.contour = Contour("stress_zz")
    view.set_legend_placement("stress_zz", (0.1, 0.2))

    pane = _round_trip(session).session.panes[0]
    assert pane.legend_placement("stress_zz").font_scale is None


def test_an_unplaced_session_writes_no_placement():
    session = ResultsSession()
    session.add_view().contour = Contour("stress_zz")
    raw = snapshot(session)
    assert raw["panes"][0]["legend_placement"] == {}


def test_a_v1_file_without_the_key_restores_to_the_stack():
    """Backward compatibility: the key is optional, absence means the
    automatic stack — which is the pre-amendment behaviour."""
    session = ResultsSession()
    session.add_view().contour = Contour("stress_zz")
    raw = snapshot(session)
    del raw["panes"][0]["legend_placement"]

    restored = restore_snapshot(raw)
    assert restored.session.panes[0].legend_placements() == {}
    assert not restored.notices


def test_placement_for_a_dead_legend_is_dropped_loudly():
    """A hand-edited file must not resurrect a legend no slot causes —
    the rule ``legend_hidden`` already uses."""
    session = ResultsSession()
    session.add_view().contour = Contour("stress_zz")
    raw = snapshot(session)
    raw["panes"][0]["legend_placement"]["not_a_field"] = {
        "anchor": [0.1, 0.2], "font_scale": None,
    }

    restored = restore_snapshot(raw)
    assert restored.session.panes[0].legend_placement("not_a_field") is None
    assert any("not_a_field" in n for n in restored.notices)


def test_a_malformed_anchor_is_dropped_loudly():
    session = ResultsSession()
    session.add_view().contour = Contour("stress_zz")
    raw = snapshot(session)
    raw["panes"][0]["legend_placement"]["stress_zz"] = {"anchor": [0.1]}

    restored = restore_snapshot(raw)
    assert restored.session.panes[0].legend_placement("stress_zz") is None
    assert any("anchor" in n for n in restored.notices)
