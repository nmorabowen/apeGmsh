"""§8 selection laws — nodes XOR Gauss, last writer wins, ONE store.

Proven through the REAL ``SelectionState`` (ADR 0045), not a mock:
the session surface writes real ``SelectionTarget`` values into the
real log, and the cross-kind replacement is one undoable gesture.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.session import ResultsSession, gauss_target, node_target
from apeGmsh.viewers.core.selection import SelectionState
from apeGmsh.viewers.scene_ir import SelectionTarget, Substrate


@pytest.fixture
def session():
    return ResultsSession()


# ----------------------------------------------------------------------
# The one store (INV-5)
# ----------------------------------------------------------------------

def test_the_surface_wraps_a_real_selection_state(session) -> None:
    assert isinstance(session.selection.state, SelectionState)


def test_targets_land_in_the_real_store(session) -> None:
    session.selection.set_nodes([4, 9])
    assert session.selection.state.targets == [
        node_target(4), node_target(9),
    ]


# ----------------------------------------------------------------------
# The S0-decided target encoding (one-way door into the S5 snapshot)
# ----------------------------------------------------------------------

def test_node_target_encoding() -> None:
    t = node_target(42)
    assert t == SelectionTarget(Substrate.RESULTS_TOPO, dim=0, key=42)
    assert t.sub is None and t.parent is None


def test_gauss_target_encoding() -> None:
    t = gauss_target(7, 3)
    assert t.substrate is Substrate.RESULTS_TOPO
    assert (t.dim, t.key, t.sub, t.parent) == (0, 7, 3, None)


# ----------------------------------------------------------------------
# XOR — the set's kind follows the last writer
# ----------------------------------------------------------------------

def test_write_of_the_other_kind_replaces_the_set(session) -> None:
    sel = session.selection
    sel.add_nodes([1, 2, 3])
    assert sel.kind == "nodes" and sel.nodes == (1, 2, 3)

    sel.add_gauss([(10, 0), (10, 1)])  # an ADDITIVE write, other kind
    assert sel.kind == "gauss"
    assert sel.gauss == ((10, 0), (10, 1))  # replaced, not merged
    assert sel.nodes == ()


def test_replacement_works_both_directions(session) -> None:
    sel = session.selection
    sel.add_gauss([(5, 0)])
    sel.add_nodes([8])
    assert sel.kind == "nodes" and sel.nodes == (8,)


def test_the_set_stays_homogeneous(session) -> None:
    sel = session.selection
    sel.add_nodes([1, 2])
    sel.add_gauss([(3, 0)])
    sel.add_gauss([(3, 1)])
    subs = [t.sub for t in sel.state.targets]
    assert all(s is not None for s in subs)  # no node leaked through


def test_cross_kind_replace_is_one_undoable_gesture(session) -> None:
    # The law is recorded as one SET gesture in the real log: a single
    # undo restores the previous kind's whole set exactly.
    sel = session.selection
    sel.set_nodes([1, 2, 3])
    sel.add_gauss([(10, 0)])
    assert sel.kind == "gauss"
    assert sel.state.undo()
    assert sel.kind == "nodes" and sel.nodes == (1, 2, 3)
    assert sel.state.redo()
    assert sel.kind == "gauss" and sel.gauss == ((10, 0),)


# ----------------------------------------------------------------------
# Same-kind writes extend / replace as asked
# ----------------------------------------------------------------------

def test_same_kind_add_extends(session) -> None:
    sel = session.selection
    sel.add_nodes([1, 2])
    sel.add_nodes([3])
    assert sel.nodes == (1, 2, 3)


def test_set_replaces_within_the_same_kind(session) -> None:
    sel = session.selection
    sel.set_nodes([1, 2])
    sel.set_nodes([9])
    assert sel.nodes == (9,)


def test_clear_empties_the_set(session) -> None:
    sel = session.selection
    sel.set_gauss([(1, 0)])
    sel.clear()
    assert sel.kind is None
    assert len(sel) == 0
    assert sel.nodes == () and sel.gauss == ()


# ----------------------------------------------------------------------
# The per-view radio only AIMS clicks — it never touches the set
# ----------------------------------------------------------------------

def test_pick_target_radio_does_not_own_or_clear_the_set(session) -> None:
    view = session.add_view()
    session.selection.set_nodes([1, 2])
    view.pick_target = "gauss"
    assert session.selection.kind == "nodes"
    assert session.selection.nodes == (1, 2)


def test_two_views_may_aim_at_different_targets_over_one_set(
    session,
) -> None:
    a = session.add_view()
    b = session.add_view()
    session.selection.set_gauss([(4, 0)])
    a.pick_target = "nodes"
    b.pick_target = "gauss"
    assert (a.pick_target, b.pick_target) == ("nodes", "gauss")
    assert session.selection.kind == "gauss"  # the one set, untouched


def test_pick_target_vocabulary_is_closed(session) -> None:
    view = session.add_view()
    with pytest.raises(ValueError, match="pick_target"):
        view.pick_target = "elements"  # retired in this window (§8)
