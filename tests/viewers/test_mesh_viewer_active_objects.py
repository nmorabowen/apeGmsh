"""Plan 04 step 3 — mesh.viewer ActiveObjects bridges.

The full ``MeshViewer.show()`` path requires a live gmsh session +
QApplication, which is heavy for unit tests. These tests poke just
the two bridges added by step 3 (pick mode + selection) by partially
constructing the surface that ``show()`` would install — the
canonical wiring is then exercised end-to-end without spinning a
window.
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


# =====================================================================
# Pick-mode bridge — set_pick_mode → ActiveObjects → cache + status
# =====================================================================


class _StubMeshViewer:
    """The minimal surface MeshViewer's pick-mode bridge needs.

    Mirrors the real ``_set_pick_mode`` / ``_on_active_pick_mode``
    methods that landed in plan 04 step 3 — copy/pasting them here
    would test nothing. Instead this stub *imports the real ones*
    from ``MeshViewer`` via direct attribute access, so any drift in
    the real implementation surfaces immediately.
    """

    def __init__(self) -> None:
        from apeGmsh.viewers.mesh_viewer import MeshViewer
        self._pick_mode = ["brep"]
        self._hover_label = None
        self._win = _StubWin()
        self._active: ActiveObjects = ActiveObjects()
        self._active.activePickModeChanged.connect(
            lambda mode: MeshViewer._on_active_pick_mode(self, mode),
        )
        # Seed the active mode to match the cache so first
        # ``_set_pick_mode`` call actually transitions.
        self._active.set_active_pick_mode(self._pick_mode[0])
        # Bind the real instance methods to this stub for direct calls.
        self._set_pick_mode = lambda m: MeshViewer._set_pick_mode(self, m)


class _StubWin:
    def __init__(self) -> None:
        self.status: str = ""

    def set_status(self, text: str) -> None:
        self.status = text


def test_set_pick_mode_routes_through_active_objects(qapp):
    viewer = _StubMeshViewer()
    emitted = _collect(viewer._active.activePickModeChanged)

    viewer._set_pick_mode("element")
    assert viewer._active.active_pick_mode == "element"
    assert viewer._pick_mode[0] == "element"    # cache stays in sync
    assert "ELEMENT" in viewer._win.status
    assert emitted == ["element"]


def test_set_pick_mode_idempotent(qapp):
    viewer = _StubMeshViewer()
    viewer._set_pick_mode("node")
    emitted = _collect(viewer._active.activePickModeChanged)
    # Setting to the same mode again must not re-fire.
    viewer._set_pick_mode("node")
    assert emitted == []


def test_set_pick_mode_sequence(qapp):
    viewer = _StubMeshViewer()
    for mode in ("element", "node", "brep", "element"):
        viewer._set_pick_mode(mode)
        assert viewer._active.active_pick_mode == mode
        assert viewer._pick_mode[0] == mode
        assert mode.upper() in viewer._win.status


def test_pre_show_set_pick_mode_falls_back_to_cache(qapp):
    """When ``_active`` is None (before show() wires it), the legacy
    cache + status-bar update path still runs."""
    from apeGmsh.viewers.mesh_viewer import MeshViewer
    stub = type("S", (), {})()
    stub._pick_mode = ["brep"]
    stub._hover_label = None
    stub._win = _StubWin()
    stub._active = None

    MeshViewer._set_pick_mode(stub, "element")
    assert stub._pick_mode[0] == "element"
    assert "ELEMENT" in stub._win.status


# =====================================================================
# Selection bridge — sel.on_changed → ActiveObjects.selectionChanged
# =====================================================================


def test_selection_bridge_emits_picks_tuple(qapp):
    """The bridge installs a callback on ``sel.on_changed`` that pushes
    a tuple snapshot of picks into ActiveObjects. Verifies the same
    bridge logic that ``MeshViewer.show()`` installs."""
    sel = SelectionState()
    active = ActiveObjects()
    # Mirror MeshViewer.show()'s closure.
    def _sel_bridge() -> None:
        active.set_selection(tuple(sel.picks))
    sel.on_changed.append(_sel_bridge)

    emitted = _collect(active.selectionChanged)

    # toggle picks → bridge fires → tuple payload reaches subscriber.
    sel.toggle((2, 7))
    assert emitted[-1] == ((2, 7),)

    sel.toggle((2, 8))
    assert emitted[-1] == ((2, 7), (2, 8))

    sel.toggle((2, 7))     # remove
    assert emitted[-1] == ((2, 8),)

    sel.clear()
    assert emitted[-1] == ()


def test_selection_bridge_fresh_payload_each_emit(qapp):
    """Each emit must be a fresh tuple — the identity check in
    ``ActiveObjects.set_selection`` would suppress duplicate-instance
    emits if the bridge ever returned the same object twice."""
    sel = SelectionState()
    active = ActiveObjects()
    def _sel_bridge() -> None:
        active.set_selection(tuple(sel.picks))
    sel.on_changed.append(_sel_bridge)

    emitted = _collect(active.selectionChanged)
    sel.toggle((1, 3))
    sel.toggle((1, 4))
    assert len(emitted) == 2
    # Tuples have content-equal hashes but should not be the same
    # instance across emits.
    assert emitted[0] is not emitted[1]


def test_selection_bridge_repeat_empty_short_circuits(qapp):
    """Setting selection to () twice should only emit once — Python
    interns the empty tuple, so the ``is`` check in ActiveObjects
    correctly suppresses the redundant emit. This documents the
    behaviour rather than mandating it."""
    sel = SelectionState()
    active = ActiveObjects()
    def _sel_bridge() -> None:
        active.set_selection(tuple(sel.picks))
    sel.on_changed.append(_sel_bridge)

    emitted = _collect(active.selectionChanged)
    sel.toggle((1, 1))
    sel.clear()
    sel.clear()    # no-op on already-empty selection (no on_changed fire)
    # Two emits: the toggle, the clear. The second clear doesn't fire
    # sel.on_changed at all in the current SelectionState impl, so the
    # bridge is never invoked the third time.
    assert len(emitted) == 2
    assert emitted[-1] == ()
