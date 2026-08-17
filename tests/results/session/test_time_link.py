"""§7 time laws — one instant when linked, per-pane when not, and a
mode pose has no instant at all."""
from __future__ import annotations

import pytest

from apeGmsh.results.session import Deform, Instant, ResultsSession


# ----------------------------------------------------------------------
# Instant is a strict value
# ----------------------------------------------------------------------

def test_instant_is_a_value() -> None:
    assert Instant("push", 7) == Instant("push", 7)
    assert Instant("push", 7) != Instant("push", 8)
    assert Instant("push", 7) != Instant("grav", 7)


def test_instant_validation_is_loud() -> None:
    with pytest.raises(ValueError, match="stage"):
        Instant("", 0)
    with pytest.raises(ValueError, match="no negative aliases"):
        Instant("push", -1)
    with pytest.raises(TypeError, match="step"):
        Instant("push", 1.5)
    with pytest.raises(TypeError, match="step"):
        Instant("push", True)


# ----------------------------------------------------------------------
# Linked: one instant — scrubber, every mesh view, every plot cursor
# ----------------------------------------------------------------------

def test_linked_session_instant_governs_every_pane() -> None:
    session = ResultsSession()
    v1 = session.add_view()
    v2 = session.add_view()
    plot = session.add_plot()
    assert session.time_linked is True  # linked is the default
    session.time = Instant("push", 7)
    for pane in (v1, v2, plot):
        assert session.effective_instant(pane) == Instant("push", 7)


def test_link_ignores_pane_time() -> None:
    # §9: pane time — set it here; the link ignores it.
    session = ResultsSession()
    view = session.add_view()
    session.time = Instant("push", 7)
    view.time = Instant("push", 3)
    assert session.effective_instant(view) == Instant("push", 7)


def test_link_ignores_plot_cursor() -> None:
    session = ResultsSession()
    plot = session.add_plot()
    session.time = Instant("push", 7)
    plot.cursor = Instant("push", 1)
    assert session.effective_instant(plot) == Instant("push", 7)


# ----------------------------------------------------------------------
# Unlinked: each pane keeps its own instant
# ----------------------------------------------------------------------

def test_unlinked_panes_keep_their_own_instants() -> None:
    session = ResultsSession()
    v1 = session.add_view()
    v2 = session.add_view()
    plot = session.add_plot()
    session.time = Instant("push", 7)
    v1.time = Instant("push", 3)
    plot.cursor = Instant("grav", 0)
    session.time_linked = False
    assert session.effective_instant(v1) == Instant("push", 3)
    assert session.effective_instant(v2) is None  # never given one
    assert session.effective_instant(plot) == Instant("grav", 0)


def test_relinking_snaps_every_pane_back_to_the_session_instant() -> None:
    session = ResultsSession()
    view = session.add_view()
    session.time = Instant("push", 7)
    session.time_linked = False
    view.time = Instant("push", 2)
    session.time_linked = True
    assert session.effective_instant(view) == Instant("push", 7)


# ----------------------------------------------------------------------
# A mode pose has no instant — frozen under the link (§4/§7)
# ----------------------------------------------------------------------

def test_mode_posed_view_is_frozen_under_the_link() -> None:
    session = ResultsSession()
    view = session.add_view()
    session.time = Instant("push", 7)
    view.deform = Deform(mode=2)
    assert view.is_mode_posed
    assert session.effective_instant(view) is None  # linked
    session.time_linked = False
    view.time = Instant("push", 3)
    assert session.effective_instant(view) is None  # unlinked too


def test_unmoded_pose_has_an_instant_again() -> None:
    session = ResultsSession()
    view = session.add_view()
    session.time = Instant("push", 7)
    view.deform = Deform(mode=2)
    view.deform = Deform()  # plain displacement pose
    assert not view.is_mode_posed
    assert session.effective_instant(view) == Instant("push", 7)


# ----------------------------------------------------------------------
# Addressing + validation
# ----------------------------------------------------------------------

def test_effective_instant_accepts_a_pane_id() -> None:
    session = ResultsSession()
    view = session.add_view()
    session.time = Instant("push", 4)
    assert session.effective_instant(view.id) == Instant("push", 4)


def test_session_time_type_is_checked() -> None:
    session = ResultsSession()
    with pytest.raises(TypeError, match="Instant"):
        session.time = ("push", 4)
    view = session.add_view()
    with pytest.raises(TypeError, match="Instant"):
        view.time = 3
