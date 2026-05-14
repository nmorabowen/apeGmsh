"""Plan 04 step 4 — model.viewer ActiveObjects bridge.

ModelViewer.show() requires a live Gmsh model + QApplication — too
heavy for unit tests. This file exercises the *selection bridge*
closure that show() installs by reconstructing the same closure
against real ``ActiveObjects`` + ``SelectionState`` instances and
asserting the end-to-end signal flow.

The bridge under test is the lambda registered at::

    sel.on_changed.append(
        lambda: _active_ref.set_selection(tuple(sel.picks)),
    )

If the real closure drifts (different payload shape, different
ActiveObjects setter), these tests have to be updated — which is the
intended trigger for thinking about whether the drift was deliberate.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from apeGmsh.viewers.core._active_objects import ActiveObjects
from apeGmsh.viewers.core.selection import SelectionState


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _collect(signal):
    bucket: list = []
    signal.connect(lambda payload: bucket.append(payload))
    return bucket


def _install_selection_bridge(sel: SelectionState, active: ActiveObjects):
    """Reproduce the closure installed by ``ModelViewer.show()``."""
    sel.on_changed.append(
        lambda: active.set_selection(tuple(sel.picks)),
    )


# =====================================================================
# Bridge end-to-end
# =====================================================================


def test_bridge_emits_tuple_of_picks(qapp):
    sel = SelectionState()
    active = ActiveObjects()
    _install_selection_bridge(sel, active)

    emitted = _collect(active.selectionChanged)
    sel.toggle((2, 1))
    assert emitted[-1] == ((2, 1),)

    sel.toggle((3, 5))
    assert emitted[-1] == ((2, 1), (3, 5))


def test_bridge_coexists_with_legacy_handlers(qapp):
    """The ModelViewer cascade has 5 ``sel.on_changed`` handlers
    (recolor, sel_tree, browser, parts_tree, commit_active_group)
    before the bridge. All handlers must fire on every mutation, in
    registration order, and the bridge's emit must reach the
    ActiveObjects subscriber regardless of where it sits in the list.
    """
    sel = SelectionState()
    active = ActiveObjects()
    log: list = []

    sel.on_changed.append(lambda: log.append("recolor"))
    sel.on_changed.append(lambda: log.append("sel_tree"))
    sel.on_changed.append(lambda: log.append("browser"))
    sel.on_changed.append(lambda: log.append("parts_tree"))
    sel.on_changed.append(lambda: log.append("commit_group"))
    _install_selection_bridge(sel, active)
    sel.on_changed.append(lambda: log.append("after_bridge"))

    emitted = _collect(active.selectionChanged)

    sel.toggle((2, 7))

    assert log == [
        "recolor", "sel_tree", "browser", "parts_tree", "commit_group",
        "after_bridge",
    ]
    assert emitted == [((2, 7),)]


def test_bridge_fires_on_clear(qapp):
    sel = SelectionState()
    active = ActiveObjects()
    _install_selection_bridge(sel, active)

    emitted = _collect(active.selectionChanged)
    sel.toggle((1, 1))
    sel.clear()
    assert emitted[-1] == ()


def test_bridge_fresh_payload_per_emit(qapp):
    """Each emit must produce a distinct tuple instance — otherwise
    ActiveObjects' identity check would suppress in-place changes that
    happen to produce the same content."""
    sel = SelectionState()
    active = ActiveObjects()
    _install_selection_bridge(sel, active)

    emitted = _collect(active.selectionChanged)
    sel.toggle((1, 1))
    sel.toggle((1, 2))
    sel.toggle((1, 2))    # remove
    assert len(emitted) == 3
    # Non-empty tuples are freshly allocated per emit.
    assert emitted[0] is not emitted[1]


def test_bridge_does_not_short_circuit_with_active_state(qapp):
    """When other code paths drive ``set_selection`` directly with a
    non-tuple payload, the bridge's next emit (carrying a tuple) still
    fires because the payload identity differs."""
    sel = SelectionState()
    active = ActiveObjects()
    _install_selection_bridge(sel, active)

    # Simulate an out-of-band setter (e.g., a future panel pushing a
    # synthetic selection state).
    active.set_selection("synthetic")

    emitted = _collect(active.selectionChanged)
    sel.toggle((2, 9))
    assert emitted == [((2, 9),)]
