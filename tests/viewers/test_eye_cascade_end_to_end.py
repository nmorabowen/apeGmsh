"""N-layer eye cascade, end to end through the outline tree.

ADR 0084 D7 (PR 5). ``test_dispatcher_contract.py`` locks the collapse
at the *dispatcher* level (N fires inside ``gesture_batch`` → one gate
pump + one render). This is the same gate one level up, driven by the
real UI surface that users actually click: ``OutlineTree._on_eye_clicked``
on a composition row, which wraps ``_apply_composition_visibility`` in
``dispatcher.gesture_batch()`` (``_outline_tree.py:574``).

Headless and GL-free: a real ``QApplication`` on the offscreen platform
plugin, a real ``OutlineTree``, a real ``DiagramRegistry`` (whose
``set_visible`` owner-fires ``LAYER_VISIBILITY_CHANGED``) and a real
``Dispatcher`` with counting pumps. No window is shown, no render
backend is bound. Not marked ``qt`` — it runs in default CI.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N_LAYERS = 5


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _FakeLayer:
    """Minimal Diagram stand-in — the cascade only reads/writes intent."""

    def __init__(self, label: str) -> None:
        self._label = label
        self.is_visible = True
        self.kind = "fake"
        self.is_attached = False

    def display_label(self) -> str:
        return self._label

    def set_visible(self, v: bool) -> None:
        self.is_visible = bool(v)


class _Counts:
    """Counting pump bindings — the test_dispatcher_contract idiom."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def count(self, name: str) -> int:
        return self.calls.count(name)


def _build(qapp):
    """OutlineTree over one geometry / one composition / N layers, wired
    to a real registry + a real dispatcher with counting pumps.

    Returns ``(tree, comp, layers, counts)``.
    """
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    geometries = GeometryManager()
    geom = geometries.active
    comp = geom.compositions.add(name="Composition", make_active=True)
    layers = [_FakeLayer(f"layer-{i}") for i in range(N_LAYERS)]
    for layer in layers:
        geom.compositions.add_layer(comp.id, layer)

    registry = DiagramRegistry()
    for layer in layers:
        registry.add(layer)

    counts = _Counts()
    dispatcher = Dispatcher(
        None,
        pump_step=lambda layer: counts.calls.append("step"),
        pump_deform=lambda layer: counts.calls.append("deform"),
        pump_gate=lambda: counts.calls.append("gate"),
        pump_restack=lambda: counts.calls.append("restack"),
        render=lambda: counts.calls.append("render"),
        defer_fn=lambda fn: fn(),
    )
    registry.dispatcher = dispatcher

    class _Director:
        def __init__(self_inner):
            self_inner.geometries = geometries
            self_inner.registry = registry
            self_inner.dispatcher = dispatcher
            self_inner.stage_id = None

        def stages(self_inner):
            return []

        def subscribe_stage(self_inner, _cb):
            return lambda: None

        def subscribe_diagrams(self_inner, _cb):
            return lambda: None

    tree = OutlineTree(_Director())
    return tree, comp, layers, counts


def _composition_item(tree):
    """The composition row under Diagrams → Geometry 1."""
    return tree._group_diagrams.child(0).child(0)


def test_composition_eye_cascade_costs_one_gate_and_one_render(qapp):
    """Clicking one composition eye over N layers writes N intents but
    pumps the gate once and renders once — the ADR 0056 V1 perf gate,
    asserted through the real click path rather than raw ``fire``."""
    tree, _, layers, counts = _build(qapp)
    comp_item = _composition_item(tree)
    assert all(L.is_visible for L in layers)

    tree._on_eye_clicked(comp_item)

    assert all(L.is_visible is False for L in layers)
    assert counts.count("gate") == 1
    assert counts.count("render") == 1
    # The eye is a visibility gesture only: no step/deform work.
    assert counts.count("step") == 0
    assert counts.count("deform") == 0

    # NOTE (ADR 0084 D3 / PR 9): the substrate-occlusion resync that
    # should follow this cascade is NOT asserted here — batch exit
    # replays primitives + the UI lane only, so RENDER-lane subscribers
    # (including the occlusion sync that restores a hidden contour's
    # substrate fill) are dropped inside a batch. Add the "hide the
    # contour → the grey fill comes back" assertion here once PR 9's
    # batch-exit RENDER-lane replay lands.


def test_composition_eye_round_trip_stays_one_gate_per_click(qapp):
    """Hide → show. Each click is its own gesture: one gate + one render
    each, and the snapshot restores every layer's prior intent."""
    tree, comp, layers, counts = _build(qapp)
    comp_item = _composition_item(tree)

    tree._on_eye_clicked(comp_item)          # hide
    assert counts.count("gate") == 1
    assert counts.count("render") == 1
    assert comp.saved_visibility is not None

    tree._on_eye_clicked(comp_item)          # show
    assert all(L.is_visible is True for L in layers)
    assert comp.saved_visibility is None
    assert counts.count("gate") == 2
    assert counts.count("render") == 2


def test_partially_hidden_composition_restores_from_snapshot_in_one_pump(
    qapp,
):
    """A composition whose layers disagree: hiding snapshots the mixed
    state, showing restores it — still one gate pump per click, and the
    layer the user had already hidden stays hidden."""
    tree, _, layers, counts = _build(qapp)
    comp_item = _composition_item(tree)

    # User hides one layer first (layer row eye — its own gesture).
    tree._on_eye_clicked(comp_item.child(2))
    assert layers[2].is_visible is False
    baseline_gate = counts.count("gate")
    baseline_render = counts.count("render")

    tree._on_eye_clicked(comp_item)          # composition hide
    assert all(L.is_visible is False for L in layers)
    assert counts.count("gate") == baseline_gate + 1
    assert counts.count("render") == baseline_render + 1

    tree._on_eye_clicked(comp_item)          # composition show
    assert [L.is_visible for L in layers] == [
        True, True, False, True, True,
    ]
    assert counts.count("gate") == baseline_gate + 2
    assert counts.count("render") == baseline_render + 2
