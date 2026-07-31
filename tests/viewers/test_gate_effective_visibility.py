"""Gate-channel visibility — ``Diagram.apply_effective_visibility``.

The composition gate computes ``effective = is_visible AND
in_active_composition`` and must push it onto the rendered artifacts
of *every* diagram — including the backend-routed (ADR 0042 R-B) ones
that hold layer handles instead of raw actors — without clobbering
``is_visible`` (the user-intent flag the gate itself reads).

Regression context: ``pump_gate`` used to iterate ``d._actors``
directly, which is empty for every migrated diagram, so the
composition gate was a silent no-op post-R-B migration.

Also covers the outline tree's eye-toggle routing: it must fire
``LAYER_VISIBILITY_CHANGED`` through the dispatcher (same path as the
settings-tab visibility checkbox) so the gate re-runs, falling back to
a raw render only when no dispatcher is installed.

ADR 0084 D7 (PR 5) adds the *real-diagram* half: the same gate math run
over real ``ContourDiagram`` instances attached to the no-GL
``RecordingBackend``, so the assertion lands on actual backend layer
visibility rather than a stub's actor list. PR 7 then swaps the local
copy of the pump body for the extracted
:class:`~apeGmsh.viewers._pump_set.PumpSet` — see ``_run_gate`` below.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._base import Diagram, DiagramSpec
from apeGmsh.viewers.diagrams._selectors import SlabSelector
from apeGmsh.viewers.diagrams._styles import DiagramStyle


# =====================================================================
# Stub diagram — legacy actor path
# =====================================================================


class _FakeActor:
    def __init__(self) -> None:
        self.visible = True

    def SetVisibility(self, v: bool) -> None:
        self.visible = bool(v)


class _StubDiagram(Diagram):
    kind = "stub"
    topology = "nodes"

    def update_to_step(self, step_index: int) -> None:
        pass


def _make_stub(visible: bool = True) -> _StubDiagram:
    spec = DiagramSpec(
        kind="stub",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
        visible=visible,
    )
    d = _StubDiagram(spec, MagicMock())
    d._actors = [_FakeActor(), _FakeActor()]
    return d


# =====================================================================
# apply_effective_visibility — contract
# =====================================================================


def test_gate_off_hides_actors_but_preserves_intent():
    d = _make_stub(visible=True)
    d.apply_effective_visibility(False)
    assert all(a.visible is False for a in d._actors)
    # User intent untouched — the next gate run recomputes from it.
    assert d.is_visible is True


def test_gate_on_shows_actors_without_flipping_intent():
    d = _make_stub(visible=False)
    # A gate should never be asked to show a user-hidden layer
    # (desired = is_visible AND in_active), but the channel contract
    # holds regardless: artifacts follow ``effective``, intent doesn't.
    d.apply_effective_visibility(True)
    assert all(a.visible is True for a in d._actors)
    assert d.is_visible is False


def test_set_visible_still_owns_the_intent_flag():
    d = _make_stub(visible=True)
    d.set_visible(False)
    assert d.is_visible is False
    assert all(a.visible is False for a in d._actors)


def test_gate_routes_subclass_override():
    """The gate channel reuses the subclass's set_visible artifact
    path — a backend-routed override sees the effective value."""
    seen: list[bool] = []

    class _Routed(_StubDiagram):
        kind = "stub"

        def set_visible(self, visible: bool) -> None:
            self._visible = visible
            seen.append(bool(visible))

    spec = DiagramSpec(
        kind="stub",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    d = _Routed(spec, MagicMock())
    d.apply_effective_visibility(False)
    assert seen == [False]
    assert d.is_visible is True


# =====================================================================
# Composition gate over REAL diagrams (ADR 0084 D7 / PR 5)
# =====================================================================
#
# The stub coverage above proves the *channel*; this section proves the
# *composition*: ``_gate_visible_layer_ids`` (the real module-level
# truth table) feeding ``apply_effective_visibility`` on real
# ``ContourDiagram`` instances, landing on real backend layer
# visibility. No GL: the diagrams attach to ``RecordingBackend``, whose
# ``set_layer_visible`` writes the handle each contour routes through.


@pytest.fixture
def gate_results(g, tmp_path: Path):
    """Tiny native Results with one nodal component — enough for a
    contour to build a real submesh and emit a real layer."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "gate.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_z": np.vstack([node_ids, node_ids * 2.0]),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _real_contour(results, backend):
    """A real, attached ``ContourDiagram`` over ``backend``."""
    from apeGmsh.viewers.diagrams import ContourDiagram, ContourStyle
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    d = ContourDiagram(spec, results)
    d.attach(backend, results.fem, build_fem_scene(results.fem))
    return d


def _backend_visible(diagram) -> bool:
    """The contour's *rendered* state — the backend layer handle it
    routes ``set_visible`` through (ADR 0042 R-B)."""
    handle = diagram._handle           # noqa: SLF001
    assert handle is not None, "contour did not emit a backend layer"
    return bool(handle.visible)


def _run_gate(geom_mgr, diagrams) -> None:
    """The REAL GATE pump.

    This used to be a verbatim copy of the ``_pump_gate`` body, because
    the pump was a ``show()`` closure. ADR 0084 D7 / PR 7 extracted it
    into :class:`~apeGmsh.viewers._pump_set.PumpSet`, so the assertions
    below now run production code — a divergence between the pump and
    this test is no longer possible.
    """
    from apeGmsh.viewers._pump_set import PumpSet

    PumpSet(
        director=SimpleNamespace(
            geometries=geom_mgr,
            registry=SimpleNamespace(diagrams=lambda: list(diagrams)),
        ),
        # DEFORM-lane collaborators — untouched by GATE, but the
        # constructor is deliberately explicit (frozen interface), so
        # stub them rather than default them away.
        scene=None,
        read_deform_field=lambda *a, **k: None,
        render_geometries=list,
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
    ).pump_gate()


@pytest.fixture
def real_gate_tree(gate_results):
    """One geometry, two compositions: ``comp`` holds two real contour
    layers, ``other`` holds one so ``set_active`` has somewhere to go.

    Returns ``(geom_mgr, comp, other, layers, all_layers)``.
    """
    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    from tests.viewers.conftest import RecordingBackend

    backend = RecordingBackend()
    geom_mgr = GeometryManager()
    geom = geom_mgr.active

    comp = geom.compositions.add(name="Composition", make_active=False)
    layers = [_real_contour(gate_results, backend) for _ in range(2)]
    for layer in layers:
        geom.compositions.add_layer(comp.id, layer)

    other = geom.compositions.add(name="Other", make_active=False)
    lone = _real_contour(gate_results, backend)
    geom.compositions.add_layer(other.id, lone)

    return geom_mgr, comp, other, layers, [*layers, lone]


def test_real_layers_gate_round_trip_visible_hidden_visible(
    real_gate_tree,
):
    """Composition visible → hidden → visible on real diagrams: the
    backend layer follows the gate, user intent never moves."""
    geom_mgr, comp, other, layers, all_layers = real_gate_tree

    # No active composition → every composition's layers pass the gate.
    _run_gate(geom_mgr, all_layers)
    assert [_backend_visible(d) for d in layers] == [True, True]

    # Activate the *other* composition → these two are gated out.
    geom_mgr.active.compositions.set_active(other.id)
    _run_gate(geom_mgr, all_layers)
    assert [_backend_visible(d) for d in layers] == [False, False]
    assert [d.is_visible for d in layers] == [True, True]

    # Back → both return, still on real backend state.
    geom_mgr.active.compositions.set_active(comp.id)
    _run_gate(geom_mgr, all_layers)
    assert [_backend_visible(d) for d in layers] == [True, True]
    assert [d.is_visible for d in layers] == [True, True]


def test_real_layer_intent_off_inside_visible_composition(
    real_gate_tree,
):
    """Mixed case: the composition passes the gate but one layer's own
    intent is off — only that layer goes dark, and re-running the gate
    is idempotent (intent is not consumed)."""
    geom_mgr, comp, _, layers, all_layers = real_gate_tree
    geom_mgr.active.compositions.set_active(comp.id)

    layers[0].set_visible(False)
    _run_gate(geom_mgr, all_layers)
    assert _backend_visible(layers[0]) is False
    assert _backend_visible(layers[1]) is True

    # The gate reads intent, it never writes it — a second run agrees.
    _run_gate(geom_mgr, all_layers)
    assert layers[0].is_visible is False
    assert layers[1].is_visible is True
    assert _backend_visible(layers[0]) is False
    assert _backend_visible(layers[1]) is True

    # Restoring intent restores the rendered layer.
    layers[0].set_visible(True)
    _run_gate(geom_mgr, all_layers)
    assert _backend_visible(layers[0]) is True


def test_real_layers_survive_composition_eye_cascade(real_gate_tree):
    """The composition-eye cascade (per-layer intent writes) composed
    with the gate: hiding then showing a 2-layer composition round-trips
    real backend visibility."""
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    geom_mgr, comp, _, layers, all_layers = real_gate_tree
    geom_mgr.active.compositions.set_active(comp.id)
    registry = DiagramRegistry()
    for d in all_layers:
        registry.add(d)

    for d in layers:                       # eye off
        registry.set_visible(d, False)
    _run_gate(geom_mgr, all_layers)
    assert [_backend_visible(d) for d in layers] == [False, False]

    for d in layers:                       # eye on
        registry.set_visible(d, True)
    _run_gate(geom_mgr, all_layers)
    assert [_backend_visible(d) for d in layers] == [True, True]


# =====================================================================
# Outline eye-toggle — dispatcher routing (ADR 0056 V1)
# =====================================================================
#
# V0 routed eye-toggles through an OutlineTree._fire_layer_visibility
# helper (with a raw-render fallback for headless contexts). V1
# deleted both: the registry mutator owner-fires
# LAYER_VISIBILITY_CHANGED itself and bulk cascades wrap in
# dispatcher.gesture_batch() — see test_dispatcher_contract.py for the
# owner-fire and batch coverage. Here we lock that the deleted bypass
# helpers stay deleted (a re-introduced fallback is a dispatcher
# bypass, INV-4/INV-5).


def test_outline_has_no_raw_render_helpers():
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    assert not hasattr(OutlineTree, "_fire_layer_visibility")
    assert not hasattr(OutlineTree, "_fire_render")
