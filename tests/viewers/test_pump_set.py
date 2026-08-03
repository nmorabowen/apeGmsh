"""``PumpSet`` — the dispatcher pumps, driven without a window.

ADR 0084 D7 / PR 7. The point of the extraction is exactly what this
file does: build the STEP / DEFORM / GATE primitives from explicit
collaborators and call them, with no ``QApplication``, no ``show()``,
no plotter. Before PR 7 these were closures inside
``ResultsViewer._show_impl`` and none of this was reachable.

The collaborators here are the *real* owner objects wherever the pump
reads owner state (``GeometryManager``, ``DiagramRegistry``) and thin
fakes only where the pump talks to the shell (the deform-field reader,
the node-cloud sync, the diagram fan-out) — those are callbacks by
design, so the fake IS the contract.

PR 8 builds the full scrub-values test (contour + line_force + deform,
asserting values and points across a scrub) on this same seam; keep the
constructor signature honest here so that test has something stable to
stand on.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.viewers._pump_set import PumpSet
from apeGmsh.viewers.diagrams._base import Diagram, DiagramSpec
from apeGmsh.viewers.diagrams._geometries import GeometryManager
from apeGmsh.viewers.diagrams._registry import DiagramRegistry
from apeGmsh.viewers.diagrams._selectors import SlabSelector
from apeGmsh.viewers.diagrams._styles import DiagramStyle


# =====================================================================
# Fakes
# =====================================================================


class _RecordingDiagram(Diagram):
    """Records what the pumps push at it. Raises on demand."""

    kind = "stub"
    topology = "nodes"

    def __init__(self, results, *, raise_on: str = "") -> None:
        super().__init__(
            DiagramSpec(
                kind="stub",
                selector=SlabSelector(component="displacement_x"),
                style=DiagramStyle(),
            ),
            results,
        )
        self.steps: list[int] = []
        self.effective: list[bool] = []
        self.substrate_calls: list[tuple] = []
        self._raise_on = raise_on
        self._attached = True

    def update_to_step(self, step_index: int) -> None:
        if self._raise_on == "step":
            raise RuntimeError("bad step")
        self.steps.append(int(step_index))

    def apply_effective_visibility(self, effective: bool) -> None:
        if self._raise_on == "gate":
            raise RuntimeError("bad gate")
        self.effective.append(bool(effective))

    def sync_substrate_points(self, pts, scene) -> None:
        if self._raise_on == "deform":
            raise RuntimeError("bad deform")
        self.substrate_calls.append((pts, scene))


class _FakeScene:
    """The two attributes the DEFORM pump reads off a scene."""

    def __init__(self, n: int = 4) -> None:
        self.reference_points = np.zeros((n, 3), dtype=np.float64)
        self.grid = SimpleNamespace(points=self.reference_points.copy())


def _director(
    geom_mgr, registry, *, step_index=0, local_steps=None, active_step=None,
):
    """The director surface the pumps actually use.

    ``local_step_for_active_stage`` joined the surface in ADR 0084 PR 10:
    the pin-aware STEP path now resolves UNPINNED layers through it too,
    so that combined mode's global→local translation lives in the one
    surviving implementation. Outside combined mode it returns the
    global cursor, which is why it defaults to ``step_index`` here.
    """
    local_steps = local_steps or {}
    return SimpleNamespace(
        geometries=geom_mgr,
        registry=registry,
        step_index=step_index,
        local_step_for_stage=lambda sid: local_steps[sid],
        local_step_for_active_stage=lambda: (
            step_index if active_step is None else active_step
        ),
        scene_for=lambda geom: None,      # pumps fall back to ``scene``
    )


def _pumps(director, scene=None, **overrides):
    """A PumpSet over ``director`` with inert shell collaborators.

    Every constructor argument is named here on purpose — this call is
    the readable form of the frozen interface PR 7 froze.
    """
    kwargs = dict(
        director=director,
        scene=scene,
        read_deform_field=lambda *a, **k: None,
        render_geometries=lambda: list(director.geometries.geometries),
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
    )
    kwargs.update(overrides)
    return PumpSet(**kwargs)


@pytest.fixture
def tree():
    """Real GeometryManager + DiagramRegistry holding two layers in one
    composition of the active geometry."""
    geom_mgr = GeometryManager()
    registry = DiagramRegistry()
    results = MagicMock()
    layers = [_RecordingDiagram(results) for _ in range(2)]
    for d in layers:
        registry.add(d)
    comp = geom_mgr.active.compositions.add(name="C", make_active=True)
    for d in layers:
        geom_mgr.active.compositions.add_layer(comp.id, d)
    return geom_mgr, registry, layers


# =====================================================================
# STEP
# =====================================================================


def test_pump_step_pushes_the_global_step_to_every_live_diagram(tree):
    geom_mgr, registry, layers = tree
    _pumps(_director(geom_mgr, registry, step_index=7)).pump_step(None)
    assert [d.steps for d in layers] == [[7], [7]]


def test_pump_step_skips_detached_and_hidden_diagrams(tree):
    geom_mgr, registry, layers = tree
    layers[0]._attached = False
    layers[1].set_visible(False)
    _pumps(_director(geom_mgr, registry, step_index=3)).pump_step(None)
    assert [d.steps for d in layers] == [[], []]


def test_pump_step_scoped_to_one_layer_leaves_the_others_alone(tree):
    geom_mgr, registry, layers = tree
    _pumps(_director(geom_mgr, registry, step_index=2)).pump_step(layers[1])
    assert layers[0].steps == []
    assert layers[1].steps == [2]


def test_pump_step_is_pin_aware(tree):
    """ADR 0058 S3b — a layer owned by a stage-pinned geometry steps
    through the PINNED stage's clamped local step, not the raw cursor."""
    geom_mgr, registry, layers = tree
    geom_mgr.set_stage_pin(geom_mgr.active_id, "stage-a")
    director = _director(
        geom_mgr, registry, step_index=99, local_steps={"stage-a": 4},
    )
    _pumps(director).pump_step(None)
    assert [d.steps for d in layers] == [[4], [4]]


# =====================================================================
# GATE
# =====================================================================


def test_pump_gate_composes_intent_with_the_composition(tree):
    geom_mgr, registry, layers = tree
    other = geom_mgr.active.compositions.add(name="Other", make_active=True)
    assert other is not None

    pumps = _pumps(_director(geom_mgr, registry))
    pumps.pump_gate()                       # layers are in the OTHER comp
    assert [d.effective[-1] for d in layers] == [False, False]
    # Intent is read, never written.
    assert [d.is_visible for d in layers] == [True, True]

    geom_mgr.active.compositions.set_active(
        geom_mgr.active.compositions.compositions[0].id,
    )
    layers[0].set_visible(False)
    pumps.pump_gate()
    assert [d.effective[-1] for d in layers] == [False, True]


def test_pump_gate_hides_layers_of_a_hidden_geometry(tree):
    """ADR 0058 S2b — geometry ``visible=False`` drops its layers."""
    geom_mgr, registry, layers = tree
    geom_mgr.set_visible(geom_mgr.active_id, False)
    _pumps(_director(geom_mgr, registry)).pump_gate()
    assert [d.effective[-1] for d in layers] == [False, False]


# =====================================================================
# DEFORM — the collaborator contract PR 8 leans on
# =====================================================================


def test_pump_deform_composes_points_and_fans_out(tree):
    """The full DEFORM pump reads the field through the injected
    reader, writes the scene grid, and hands the composed points to the
    node cloud and the diagram fan-out."""
    geom_mgr, registry, _layers = tree
    scene = _FakeScene()
    field = np.ones((4, 3), dtype=np.float64)
    geom_mgr.set_deformation(
        geom_mgr.active_id, enabled=True, field="displacement", scale=2.0,
    )

    cloud: list = []
    fanout: list = []
    pumps = _pumps(
        _director(geom_mgr, registry, step_index=5),
        scene=scene,
        read_deform_field=lambda name, step, stage_id=None: field,
        sync_node_cloud=cloud.append,
        sync_diagram_substrate_points=(
            lambda pts, **kw: fanout.append((pts, kw))
        ),
    )
    pumps.pump_deform(None)

    assert np.allclose(scene.grid.points, 2.0)          # 0 + 2.0 * 1
    assert len(cloud) == 1 and np.allclose(cloud[0], 2.0)
    assert len(fanout) == 1
    pts, kwargs = fanout[0]
    assert np.allclose(pts, 2.0)
    assert kwargs["geometry"] is geom_mgr.active
    assert kwargs["scene"] is scene


def test_pump_deform_with_no_field_resets_to_reference(tree):
    """The legacy fast path: no deformation and no offset composes to
    ``None``, and the pump restores the reference points."""
    geom_mgr, registry, _layers = tree
    scene = _FakeScene()
    scene.grid.points = np.full((4, 3), 9.0)
    fanout: list = []
    _pumps(
        _director(geom_mgr, registry),
        scene=scene,
        sync_diagram_substrate_points=(
            lambda pts, **kw: fanout.append(pts)
        ),
    ).pump_deform(None)

    assert np.allclose(scene.grid.points, 0.0)
    assert fanout == [None]


# =====================================================================
# Loud failures (ADR 0084 D4)
# =====================================================================


@pytest.mark.allow_pump_failures
def test_a_raising_diagram_routes_through_the_loud_path(tree, pump_failures):
    """A pump catch site must report through the ``_failures`` registry
    — the ``pump_failures`` fixture is the collector — and must keep
    pumping the rest of the registry."""
    geom_mgr, registry, layers = tree
    bad = _RecordingDiagram(MagicMock(), raise_on="step")
    registry.add(bad)
    geom_mgr.active.compositions.add_layer(
        geom_mgr.active.compositions.active.id, bad,
    )

    _pumps(_director(geom_mgr, registry, step_index=1)).pump_step(None)

    assert [name for name, _exc in pump_failures.failures] == ["pump.step"]
    # The good layers still advanced — one bad diagram does not strand
    # the registry at the old step.
    assert [d.steps for d in layers] == [[1], [1]]


@pytest.mark.allow_pump_failures
def test_a_raising_gate_is_loud_and_keeps_gating(tree, pump_failures):
    geom_mgr, registry, layers = tree
    bad = _RecordingDiagram(MagicMock(), raise_on="gate")
    registry.add(bad)

    _pumps(_director(geom_mgr, registry)).pump_gate()

    assert [name for name, _exc in pump_failures.failures] == ["pump.gate"]
    assert [d.effective[-1] for d in layers] == [True, True]


@pytest.mark.allow_pump_failures
def test_a_raising_scoped_deform_is_loud(tree, pump_failures):
    geom_mgr, registry, _layers = tree
    bad = _RecordingDiagram(MagicMock(), raise_on="deform")
    registry.add(bad)

    _pumps(
        _director(geom_mgr, registry), scene=_FakeScene(),
    ).pump_deform(bad)

    assert [name for name, _exc in pump_failures.failures] == ["pump.deform"]


def test_on_failure_is_injectable(tree):
    """The loud sink is a constructor argument, so a caller can route
    pump failures somewhere other than the global registry."""
    geom_mgr, registry, _layers = tree
    bad = _RecordingDiagram(MagicMock(), raise_on="step")
    registry.add(bad)
    seen: list = []

    _pumps(
        _director(geom_mgr, registry),
        on_failure=lambda action, exc, **kw: seen.append((action, kw)),
    ).pump_step(None)

    assert [action for action, _kw in seen] == ["step"]
    assert seen[0][1] == {"diagram": id(bad)}
