"""Unit tests for ``VisibilityManager._rebuild_node_cloud``.

Covers the regression where hiding a Physical Group left the per-dim
sphere-glyph node cloud (``registry.dim_node_actors``) on screen
because the actor was never rebuilt by the hide path.

Tests exercise ``_rebuild_node_cloud`` directly so the full
``_rebuild_actors`` machinery (extract_cells, silhouettes) isn't
required — the only dependency is an off-screen pyvista plotter for
the underlying ``build_node_cloud`` call.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.viewers.core.entity_registry import EntityRegistry
from apeGmsh.viewers.core.visibility import VisibilityManager


@pytest.fixture
def plotter():
    p = pv.Plotter(off_screen=True)
    yield p
    p.close()


def _make_registry_with_node_data(
    plotter: pv.Plotter,
    *,
    dim: int = 1,
    coords: np.ndarray,
    pairs: np.ndarray,
) -> EntityRegistry:
    """Build a minimal registry populated with node-cloud data + an
    initial dim_node_actors[dim] entry, matching what mesh_scene.py
    leaves behind."""
    from apeGmsh.viewers.scene.glyph_points import build_node_cloud

    reg = EntityRegistry()
    reg.register_node_cloud_data(
        node_coords=coords,
        dim_node_entity_pairs={dim: pairs},
        kwargs={"model_diagonal": 1.0, "marker_size": 6.0, "color": "white"},
    )
    cloud, actor = build_node_cloud(
        plotter, coords,
        model_diagonal=1.0, marker_size=6.0, color="white",
    )
    reg.register_node_cloud(dim, cloud, actor)
    return reg


class _StubColorManager:
    def __init__(self):
        self._idle_fn = lambda dt: np.array([200, 200, 200], dtype=np.uint8)


class _StubSelection:
    pass


def _make_vis_mgr(plotter, registry) -> VisibilityManager:
    return VisibilityManager(
        registry, _StubColorManager(), _StubSelection(), plotter,
    )


def test_rebuild_node_cloud_filters_hidden_entity(plotter):
    """Hiding entity (1, 5) drops only the nodes owned exclusively
    by entity 5; nodes owned by another entity stay visible."""
    # 4 nodes: node 0 in entity 5, node 1 in entity 5, node 2 in
    # entity 7, node 3 shared between 5 and 7.
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    pairs = np.array([
        [0, 5],
        [1, 5],
        [2, 7],
        [3, 5],
        [3, 7],
    ], dtype=np.int64)
    reg = _make_registry_with_node_data(
        plotter, dim=1, coords=coords, pairs=pairs,
    )
    vm = _make_vis_mgr(plotter, reg)

    vm._rebuild_node_cloud(1, {(1, 5)})

    new_cloud = reg.dim_node_clouds[1]
    # Visible nodes: node 2 (owned only by entity 7) + node 3
    # (shared with still-visible entity 7) = 2 points.
    assert new_cloud.n_points == 2
    visible_xy = sorted(new_cloud.points[:, 0].tolist())
    assert visible_xy == pytest.approx([2.0, 3.0])


def test_rebuild_node_cloud_blanks_when_all_hidden(plotter):
    """When every node is owned solely by hidden entities, the
    actor is hidden (not rebuilt) so reveal_all can resurrect it."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pairs = np.array([[0, 5], [1, 5]], dtype=np.int64)
    reg = _make_registry_with_node_data(
        plotter, dim=1, coords=coords, pairs=pairs,
    )
    vm = _make_vis_mgr(plotter, reg)
    initial_actor = reg.dim_node_actors[1]

    vm._rebuild_node_cloud(1, {(1, 5)})

    # Same actor, just set invisible — the registry entry is
    # preserved so reveal_all has something to flip back on.
    assert reg.dim_node_actors[1] is initial_actor
    assert initial_actor.GetVisibility() == 0


def test_rebuild_node_cloud_no_hides_keeps_full_set(plotter):
    """An effective set with no entries of *dim* leaves every node
    visible (the rebuild loop calls us once per affected dim even
    when no hide intersects this dim)."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pairs = np.array([[0, 5], [1, 5]], dtype=np.int64)
    reg = _make_registry_with_node_data(
        plotter, dim=1, coords=coords, pairs=pairs,
    )
    vm = _make_vis_mgr(plotter, reg)

    # Hide a dim=3 entity — irrelevant to dim=1's pairs.
    vm._rebuild_node_cloud(1, {(3, 99)})

    new_cloud = reg.dim_node_clouds[1]
    assert new_cloud.n_points == 2


def test_rebuild_node_cloud_skips_when_no_pair_data(plotter):
    """Dims that fell through to mesh_scene's gmsh-fallback path
    (no entity tags cached) have no pairs registered.  The rebuild
    must be a no-op so the previous all-nodes-visible behaviour
    is preserved instead of accidentally wiping the cloud."""
    coords = np.array([[0.0, 0.0, 0.0]])
    # Register with no pairs for dim=2 at all.
    reg = _make_registry_with_node_data(
        plotter, dim=1,
        coords=coords,
        pairs=np.empty((0, 2), dtype=np.int64),
    )
    # Manually attach a dim=2 actor with an arbitrary cloud.
    from apeGmsh.viewers.scene.glyph_points import build_node_cloud
    cloud2, actor2 = build_node_cloud(
        plotter, np.array([[5.0, 0.0, 0.0]]),
        model_diagonal=1.0, marker_size=6.0, color="white",
    )
    reg.register_node_cloud(2, cloud2, actor2)
    vm = _make_vis_mgr(plotter, reg)

    vm._rebuild_node_cloud(2, {(2, 99)})

    # Untouched.
    assert reg.dim_node_actors[2] is actor2
    assert reg.dim_node_clouds[2] is cloud2


def test_rebuild_node_cloud_warns_on_no_pair_fallback(plotter, monkeypatch):
    """The all-nodes-visible fallback is preserved, but when a hide
    actually intersects the dim it must be LOUD — silent ghost nodes
    read as a visibility bug."""
    coords = np.array([[0.0, 0.0, 0.0]])
    reg = _make_registry_with_node_data(
        plotter, dim=1,
        coords=coords,
        pairs=np.empty((0, 2), dtype=np.int64),
    )
    vm = _make_vis_mgr(plotter, reg)

    calls: list[tuple] = []
    monkeypatch.setattr(
        "apeGmsh.viewers._log.log_action",
        lambda *a, **k: calls.append((a, k)),
    )

    # Hide intersects dim 2 (which has no pair data) → warn.
    vm._rebuild_node_cloud(2, {(2, 99)})
    assert any(a[1] == "node_cloud_no_ownership_data" for a, _ in calls)

    # No hide of that dim → fallback stays silent.
    calls.clear()
    vm._rebuild_node_cloud(2, {(3, 7)})
    assert calls == []
