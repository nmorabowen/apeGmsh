"""ADR 0058 S1 — geometry→scene resolution seam.

S1 plumbs every scene access through ``director.scene_for(geometry)``
and the registry's per-diagram ``scene_resolver`` while the viewport
still renders one substrate: every geometry resolves to the single
bound scene. S2 swaps the internals of ``scene_for`` for real
per-geometry ``FEMSceneData`` instances — these tests pin the seam's
contract so that swap needs no caller changes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import _stub_model_h5_path
from tests.viewers.conftest import RecordingBackend


_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")


class _StubDiagram:
    """Duck-typed minimal layer — records what attach() received."""

    def __init__(self) -> None:
        self.attached_scene = None
        self._attached = False
        self.kind = "stub"

    @property
    def is_attached(self) -> bool:
        return self._attached

    def attach(self, backend, view, scene=None) -> None:
        self.attached_scene = scene
        self._attached = True

    def detach(self) -> None:
        self._attached = False


# =====================================================================
# Registry: per-diagram scene resolution
# =====================================================================

def test_registry_attach_uses_scene_resolver():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    default_scene = object()
    resolved_scene = object()
    reg = DiagramRegistry()
    reg.bind(
        RecordingBackend(), view=object(), scene=default_scene,
        scene_resolver=lambda d: resolved_scene,
    )
    d = _StubDiagram()
    reg.add(d)
    assert d.attached_scene is resolved_scene


def test_registry_resolver_none_falls_back_to_bound_scene():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    default_scene = object()
    reg = DiagramRegistry()
    reg.bind(
        RecordingBackend(), view=object(), scene=default_scene,
        scene_resolver=lambda d: None,
    )
    d = _StubDiagram()
    reg.add(d)
    assert d.attached_scene is default_scene


def test_registry_without_resolver_keeps_legacy_behaviour():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    default_scene = object()
    reg = DiagramRegistry()
    reg.bind(RecordingBackend(), view=object(), scene=default_scene)
    d = _StubDiagram()
    reg.add(d)
    assert d.attached_scene is default_scene


def test_registry_reattach_all_resolves_per_diagram():
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    scenes = {}
    d1, d2 = _StubDiagram(), _StubDiagram()
    scenes[id(d1)] = object()
    scenes[id(d2)] = object()
    reg = DiagramRegistry()
    reg.bind(
        RecordingBackend(), view=object(), scene=object(),
        scene_resolver=lambda d: scenes[id(d)],
    )
    reg.add(d1)
    reg.add(d2)
    reg.reattach_all()
    assert d1.attached_scene is scenes[id(d1)]
    assert d2.attached_scene is scenes[id(d2)]


# =====================================================================
# Director: scene_for / _scene_for_diagram
# =====================================================================

@pytest.fixture
def director():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    from apeGmsh.results import Results
    from apeGmsh.viewers.diagrams._director import ResultsDirector
    return ResultsDirector(
        Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path()),
    )


def test_director_scene_for_resolves_every_geometry_to_bound_scene(director):
    scene = object()
    director.bind_plotter(RecordingBackend(), scene=scene)
    active = director.geometries.active
    other = director.geometries.add("Geometry B", make_active=False)
    # S1 contract: one bound scene, every geometry maps to it. S2
    # changes the internals (per-geometry scenes), not this call shape.
    assert director.scene_for(active) is scene
    assert director.scene_for(other) is scene


def test_director_scene_for_none_before_bind(director):
    assert director.scene_for(director.geometries.active) is None


def test_director_resolves_unowned_diagram_to_active_geometry(director):
    scene = object()
    director.bind_plotter(RecordingBackend(), scene=scene)
    # A diagram with no composition membership (freshly built, not yet
    # placed) resolves through the active geometry.
    assert director._scene_for_diagram(_StubDiagram()) is scene
