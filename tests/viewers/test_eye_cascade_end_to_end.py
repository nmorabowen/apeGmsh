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


class _OccludingLayer(_FakeLayer):
    """A contour-shaped layer: attached, and it paints over the fill.

    ``occludes_substrate`` is the class flag ``ContourDiagram`` /
    ``IsochroneMapDiagram`` / ``SandDiagram`` set, and it is the only
    thing ``_geometries_occluded_by_diagrams`` reads off a diagram
    besides ``is_attached`` / ``is_visible``.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.is_attached = True
        self.kind = "contour"
        self.occludes_substrate = True


class _FakeActor:
    """The one vtkActor call the substrate fill sync makes."""

    def __init__(self) -> None:
        self.visibility = 1

    def SetVisibility(self, v) -> None:      # noqa: N802 (VTK name)
        self.visibility = int(v)


class _Counts:
    """Counting pump bindings — the test_dispatcher_contract idiom."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def count(self, name: str) -> int:
        return self.calls.count(name)


def _build(qapp, layer_factory=_FakeLayer, n_layers=N_LAYERS):
    """OutlineTree over one geometry / one composition / N layers, wired
    to a real registry + a real dispatcher with counting pumps.

    Returns ``(tree, comp, layers, counts, director)``.
    """
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    geometries = GeometryManager()
    geom = geometries.active
    comp = geom.compositions.add(name="Composition", make_active=True)
    layers = [layer_factory(f"layer-{i}") for i in range(n_layers)]
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

    director = _Director()
    tree = OutlineTree(director)
    return tree, comp, layers, counts, director


def _composition_item(tree):
    """The composition row under Diagrams → Geometry 1."""
    return tree._group_diagrams.child(0).child(0)


def test_composition_eye_cascade_costs_one_gate_and_one_render(qapp):
    """Clicking one composition eye over N layers writes N intents but
    pumps the gate once and renders once — the ADR 0056 V1 perf gate,
    asserted through the real click path rather than raw ``fire``."""
    tree, _, layers, counts, _director = _build(qapp)
    comp_item = _composition_item(tree)
    assert all(L.is_visible for L in layers)

    tree._on_eye_clicked(comp_item)

    assert all(L.is_visible is False for L in layers)
    assert counts.count("gate") == 1
    assert counts.count("render") == 1
    # The eye is a visibility gesture only: no step/deform work.
    assert counts.count("step") == 0
    assert counts.count("deform") == 0


def test_composition_eye_round_trip_stays_one_gate_per_click(qapp):
    """Hide → show. Each click is its own gesture: one gate + one render
    each, and the snapshot restores every layer's prior intent."""
    tree, comp, layers, counts, _director = _build(qapp)
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
    tree, _, layers, counts, _director = _build(qapp)
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


# =====================================================================
# The ghost mesh — ADR 0084 D3 / PR 9, fork F-A
# =====================================================================


def _wire_occlusion_sync(director, fill, counts):
    """Subscribe the substrate-occlusion sync on the RENDER lane.

    This is the headless slice of the shell's ``_sync_substrate_visibility``
    → ``_apply_geometry_display`` pair: the fill's visibility is
    ``0`` where an occluding diagram covers the geometry and ``1``
    otherwise (``results_viewer.py`` — ``fill_v = 0 if gid in occluding
    else mesh_v``). The occlusion set itself is the REAL
    ``_geometries_occluded_by_diagrams``, so the truth table under test
    is production code; only the actor and the surrounding closure
    (opacity, node cloud, labels — none of which this bug touches) are
    stand-ins. The shell subscribes the same handler to the same four
    kinds it does at ``show()``.
    """
    from apeGmsh.viewers.diagrams._dispatch import (
        DIAGRAM_ATTACHED,
        DIAGRAM_DETACHED,
        GEOMETRY_VISIBILITY_CHANGED,
        LAYER_VISIBILITY_CHANGED,
        Lane,
    )
    from apeGmsh.viewers.results_viewer import (
        _geometries_occluded_by_diagrams,
    )

    gid = director.geometries.active.id

    def _sync(_kind=None, _payload=None) -> None:
        counts.calls.append("occlusion_sync")
        occluding = _geometries_occluded_by_diagrams(director)
        fill.SetVisibility(0 if gid in occluding else 1)

    director.dispatcher.subscribe(
        (
            GEOMETRY_VISIBILITY_CHANGED, DIAGRAM_ATTACHED,
            DIAGRAM_DETACHED, LAYER_VISIBILITY_CHANGED,
        ),
        _sync,
        lane=Lane.RENDER,
    )
    _sync()
    counts.calls.clear()
    return _sync


def test_hiding_an_occluding_composition_brings_the_substrate_back(qapp):
    """THE ghost-mesh regression (ADR 0084 D3).

    A visible contour hides the grey substrate fill so the two
    coincident opaque surfaces don't z-fight. Hide that contour through
    the composition eye — a ``gesture_batch`` — and the fill must come
    back. Before the batch-exit RENDER-lane replay it did not: the
    occlusion sync is a RENDER-lane subscriber, ``fire()`` returned
    early under suppression, and the user was left staring at a
    viewport with neither contour nor mesh in it.
    """
    tree, _, layers, counts, director = _build(
        qapp, layer_factory=_OccludingLayer, n_layers=3,
    )
    fill = _FakeActor()
    _wire_occlusion_sync(director, fill, counts)
    # Baseline: the contours are visible, so the grey fill is hidden.
    assert fill.visibility == 0

    tree._on_eye_clicked(_composition_item(tree))     # eye off

    assert all(L.is_visible is False for L in layers)
    assert fill.visibility == 1, (
        "batch exit left the substrate fill hidden — the mesh vanished "
        "along with the contour"
    )
    # Still one gate + one render: the replay must not cost a frame.
    assert counts.count("gate") == 1
    assert counts.count("render") == 1
    # …and the sync ran ONCE for the whole 3-layer cascade.
    assert counts.count("occlusion_sync") == 1

    tree._on_eye_clicked(_composition_item(tree))     # eye back on
    assert all(L.is_visible is True for L in layers)
    assert fill.visibility == 0
    assert counts.count("render") == 2
    assert counts.count("occlusion_sync") == 2


def test_replayed_sync_lands_before_the_batch_render(qapp):
    """Ordering contract: primitives → RENDER lane → render.

    The fill flag has to be written into the frame the batch then
    paints, not after it — otherwise the correct state shows up one
    frame late (or never, in a viewer that only renders on demand).
    """
    tree, _, _layers, counts, director = _build(
        qapp, layer_factory=_OccludingLayer, n_layers=3,
    )
    _wire_occlusion_sync(director, _FakeActor(), counts)

    tree._on_eye_clicked(_composition_item(tree))

    order = [c for c in counts.calls if c in ("gate", "occlusion_sync", "render")]
    assert order == ["gate", "occlusion_sync", "render"]
