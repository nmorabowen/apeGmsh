"""Batch-exit RENDER-lane replay at the real call sites (ADR 0084 D3).

``test_dispatcher_contract.py`` locks the protocol itself (record while
suppressed, replay once per handler on exit, between the pumps and the
render). ``test_eye_cascade_end_to_end.py`` carries the product-level
ghost-mesh repro. This file covers the *other* batching call sites the
verification sweep found stale, plus the one that used to hand-patch:

* the reference-ghost preset — ``_outline_tree.py`` wraps
  ``GeometryManager.add_reference_ghost`` in ``gesture_batch``;
* ``ResultsDirector.duplicate_geometry`` — its own ``session_batch``;
* ``SessionController.apply`` — the restore batch that used to re-run
  the substrate sync, the off-seam clip re-cut, the gizmos, the cut
  faces and the plane panel by hand once the batch had flushed.

Headless: real owner objects (``GeometryManager``, ``ResultsDirector``,
``SessionController``), a real ``Dispatcher`` with counting pumps, the
shell's own observer→dispatcher bridges, and a recording RENDER-lane
subscriber standing in for ``_sync_substrate_visibility``. No plotter,
no ``QApplication``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._dispatch import (
    COMP_ACTIVE_CHANGED,
    DIAGRAM_ATTACHED,
    DIAGRAM_DETACHED,
    GEOMETRIES_CHANGED,
    GEOMETRY_ACTIVE_CHANGED,
    GEOMETRY_ADDED,
    GEOMETRY_REMOVED,
    GEOMETRY_VISIBILITY_CHANGED,
    LAYER_VISIBILITY_CHANGED,
    Dispatcher,
    Lane,
)

# The kinds the shell subscribes ``_sync_substrate_visibility`` to on
# the RENDER lane (``results_viewer.py`` — the GEOMETRY_ACTIVE /
# GEOMETRY_VISIBILITY / GEOMETRY_REMOVED subscriptions plus the
# three-kind occlusion loop), and ``GEOMETRY_ADDED``, which the sync
# also rides so a geometry that appears inside a batch gets its
# born-hidden actor pair turned on.
SUBSTRATE_SYNC_KINDS = (
    GEOMETRY_ACTIVE_CHANGED, GEOMETRY_VISIBILITY_CHANGED,
    GEOMETRY_ADDED, GEOMETRY_REMOVED,
    DIAGRAM_ATTACHED, DIAGRAM_DETACHED, LAYER_VISIBILITY_CHANGED,
)


class _Spy:
    """Counting pumps + a RENDER-lane subscriber that records its calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.replays: list[tuple] = []

    def bind(self, dispatcher: Dispatcher, kinds=SUBSTRATE_SYNC_KINDS) -> None:
        dispatcher.bind(
            pump_step=lambda _l: self.calls.append("step"),
            pump_deform=lambda _l: self.calls.append("deform"),
            pump_gate=lambda: self.calls.append("gate"),
            pump_restack=lambda: self.calls.append("restack"),
            render=lambda: self.calls.append("render"),
            defer_fn=lambda fn: fn(),
        )
        dispatcher.subscribe(kinds, self._on_event, lane=Lane.RENDER)

    def _on_event(self, kind, payload) -> None:
        self.calls.append("sync")
        self.replays.append((kind, payload))

    def count(self, name: str) -> int:
        return self.calls.count(name)


def _bridge(geometries, registry, dispatcher: Dispatcher) -> None:
    """The shell's observer→dispatcher wiring (``results_viewer.py``).

    Neither owner holds a dispatcher: the viewer bridges the typed and
    omnibus observer chains into ``fire``. Reproduced here so the call
    sites under test emit exactly the events they emit in the app.
    """
    geometries.subscribe_typed(
        lambda kind, payload: dispatcher.fire(kind, payload=payload),
    )
    geometries.subscribe(lambda: dispatcher.fire(GEOMETRIES_CHANGED))
    if registry is not None:
        registry.dispatcher = dispatcher
        registry.subscribe(lambda: dispatcher.fire(COMP_ACTIVE_CHANGED))


# =====================================================================
# Reference ghost — gesture_batch (_outline_tree.py)
# =====================================================================

def test_reference_ghost_batch_replays_the_render_lane():
    """The ghost preset composes N owner mutators inside one
    ``gesture_batch``. Before ADR 0084 D3 the RENDER-lane substrate sync
    never ran, so the new ghost geometry kept the born-hidden actor pair
    ``_materialize_scene`` gave it."""
    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    gm = GeometryManager()
    dispatcher = Dispatcher(None)
    spy = _Spy()
    spy.bind(dispatcher)
    _bridge(gm, None, dispatcher)

    with dispatcher.gesture_batch():
        ghost = gm.add_reference_ghost(gm.active_id)

    assert ghost is not None
    # Replayed exactly once for the whole preset, and the gesture still
    # costs a single render.
    assert spy.count("sync") == 1
    assert spy.count("render") == 1
    # …after the pumps, before the render (the fire() ordering).
    assert spy.calls[-2:] == ["sync", "render"]


def test_reference_ghost_without_a_batch_syncs_per_event():
    """Sanity floor: the replay is what a batch owes, not a new
    requirement on unbatched call sites — and the coalescing is worth
    having, because the unbatched preset syncs more than once."""
    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    gm = GeometryManager()
    dispatcher = Dispatcher(None)
    spy = _Spy()
    spy.bind(dispatcher)
    _bridge(gm, None, dispatcher)

    gm.add_reference_ghost(gm.active_id)

    assert spy.count("sync") > 1


# =====================================================================
# duplicate_geometry — session_batch (_director.py)
# =====================================================================

def _native_results(g, tmp_path: Path):
    """Tiny single-stage native Results (the S3c fixture recipe)."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    n_nodes = node_ids.size

    path = tmp_path / "pr9.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_x": np.ones((3, n_nodes))},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _contour_spec():
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._selectors import SlabSelector
    from apeGmsh.viewers.diagrams._styles import ContourStyle

    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_x"),
        style=ContourStyle(),
        stage_id="grav",
        label="warp",
    )


@pytest.fixture
def director(g, tmp_path: Path):
    from apeGmsh.viewers.diagrams._director import ResultsDirector

    return ResultsDirector(_native_results(g, tmp_path))


@pytest.fixture
def wired(director):
    """Director + a spy bound to its dispatcher, with the shell bridges."""
    spy = _Spy()
    spy.bind(director.dispatcher)
    _bridge(director.geometries, director.registry, director.dispatcher)
    return director, spy


def test_duplicate_geometry_batch_replays_the_render_lane(wired):
    """``duplicate_geometry`` runs its whole rebuild inside a
    ``session_batch``. It appends a geometry and makes it active — both
    of which the RENDER-lane substrate sync exists to react to — and
    neither reached it before ADR 0084 D3."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    director, spy = wired
    geom = director.geometries.active
    comp = geom.compositions.add(name="Warp", make_active=True)
    for _ in range(2):
        d = ContourDiagram(_contour_spec(), director.results)
        director.registry.add(d)
        geom.compositions.add_layer(comp.id, d)
    spy.calls.clear()

    clone = director.duplicate_geometry(geom.id)

    assert clone is not None
    # Two rebuilt layers + a new active geometry, one replay.
    assert spy.count("sync") == 1
    assert spy.count("render") == 1
    assert spy.calls[-2:] == ["sync", "render"]


# =====================================================================
# Session restore — the retired hand patch (_session_apply.py)
# =====================================================================

def _session(tmp_path: Path, *, n_geometries=1, specs=None):
    from apeGmsh.viewers.diagrams._session import (
        CompositionSnapshot,
        GeometrySnapshot,
        deserialize_session,
        serialize_session,
    )

    specs = specs if specs is not None else [_contour_spec()]
    geometries = [
        GeometrySnapshot(
            id=f"g{i}",
            name=f"Restored {i}",
            active_composition_id="c0",
            visible=True,
            compositions=(
                CompositionSnapshot(
                    id="c0", name="Warp",
                    layer_indices=tuple(range(len(specs))) if i == 0 else (),
                ),
            ),
        )
        for i in range(n_geometries)
    ]
    payload = serialize_session(
        specs=specs,
        results_path=tmp_path / "pr9.h5",
        fem_snapshot_id=None,
        geometries=geometries,
    )
    return deserialize_session(payload)


def _apply(director, session) -> None:
    from apeGmsh.viewers._session_apply import SessionController

    SessionController(
        director=director, results=director.results,
    ).apply(session, SimpleNamespace(set_status=lambda *a, **k: None))


def test_single_geometry_restore_lands_without_a_hand_patch(
    wired, tmp_path: Path,
):
    """PR 9 deleted ``SessionController``'s post-batch compensations.

    A one-geometry session reuses the bootstrap geometry, so nothing is
    added, activated or shown: the only substrate work the restore owes
    is the occlusion recompute for the contour it just attached — and
    that rides ``_apply_geometry_display`` on the plain ``geometries``
    observer chain, which a dispatcher batch does not suppress. Hence
    no compensation is needed and none is asserted; the restore just
    has to land, in one render.
    """
    director, spy = wired
    observed: list[int] = []
    director.geometries.subscribe(
        lambda: observed.append(
            len(director.geometries.active.compositions.compositions),
        ),
    )

    _apply(director, _session(tmp_path))

    geom = director.geometries.geometries[0]
    assert geom.name == "Restored 0"
    assert [c.name for c in geom.compositions.compositions] == ["Warp"]
    assert len(geom.compositions.compositions[0].layers) == 1
    assert spy.count("render") == 1
    # The unsuppressed observer chain ran with the composition in
    # place — the occlusion recompute sees the restored layers.
    assert observed and observed[-1] == 1


def test_multi_geometry_restore_syncs_the_geometries_it_added(
    wired, tmp_path: Path,
):
    """The case the hand patch was written for (ADR 0058 S2a/S2b): a
    session with more than one geometry. Each extra geometry is
    ``geom_mgr.add``-ed inside the suppressed batch and its actor pair
    is born hidden, so the substrate sync — which rides GEOMETRY_ADDED —
    must be replayed on exit or the restore lands invisible geometries.
    """
    director, spy = wired

    _apply(director, _session(tmp_path, n_geometries=3))

    names = [g.name for g in director.geometries.geometries]
    assert names == ["Restored 0", "Restored 1", "Restored 2"]
    # Two GEOMETRY_ADDED fires inside the batch, one replayed sync.
    assert spy.count("sync") == 1
    assert spy.count("render") == 1
    assert spy.calls[-2:] == ["sync", "render"]


def test_session_controller_no_longer_takes_compensation_collaborators():
    """The hand-patch interface is gone, not merely unused — a future
    call site cannot quietly re-introduce a per-site compensation by
    passing one in."""
    import inspect

    from apeGmsh.viewers._session_apply import SessionController

    params = set(inspect.signature(SessionController).parameters)
    assert params == {"director", "results", "legends", "clip_planes"}
