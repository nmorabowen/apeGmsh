"""PR1 / ADR 0027 — ColorMode "Partition" idle-fn dispatch.

PR0 plumbed ``ViewerElements.partition_for(eid)`` (FEM eid -> OpenSeesMP
rank). PR1 wires the per-entity dispatch on
:class:`apeGmsh.viewers.core.color_mode_controller.ColorModeController`
so the mesh viewer can color BRep entities by their dominant rank.

These tests exercise ``_partition_idle`` directly with minimal mocks for
the controller's GPU-bearing dependencies (no plotter, no VTK).
End-to-end rendering can't be exercised in this CI environment (no
GPU); user-facing parity is verified by eyeballing in a real session.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.color_mode_controller import (
    _FALLBACK_RGB,
    _GROUP_PALETTE_RGB,
    ColorModeController,
)
from apeGmsh.viewers.data._elements import (
    ElementLoadView,
    SurfaceConstraintView,
    ViewerElements,
)
from apeGmsh.viewers.data._nodes import (
    MassView,
    NodalLoadView,
    NodeConstraintView,
    SPView,
    ViewerNodes,
    _NamedNodeSelection,
)


# =====================================================================
# Minimal dependency stand-ins
# =====================================================================


class _NoopColorMgr:
    def __init__(self) -> None:
        self.idle_fn = None

    def set_idle_fn(self, fn) -> None:
        self.idle_fn = fn

    def reset_idle_fn(self) -> None:
        self.idle_fn = None

    def recolor_all(self, *, picks, hidden) -> None:
        pass


class _NoopRegistry:
    dims: list[int] = []
    dim_meshes: dict = {}
    dim_actors: dict = {}


class _NoopSel:
    picks: set = set()


class _NoopVisMgr:
    hidden: set = set()


class _NoopPlotter:
    def render(self) -> None:
        pass


class _FakeScene:
    """Minimal MeshSceneData stand-in — the controller only reads
    ``brep_to_elems`` / ``brep_to_group`` / ``brep_dominant_type`` /
    ``quality`` / ``batch_cell_to_elem`` on the paths we exercise.
    """
    def __init__(self, brep_to_elems: "dict") -> None:
        self.brep_to_elems = brep_to_elems
        self.brep_to_group: dict = {}
        self.brep_dominant_type: dict = {}
        self.quality: dict = {}
        self.batch_cell_to_elem: dict = {}


def _make_viewer_data(
    *,
    partition_by_eid: "dict[int, int] | None" = None,
) -> "tuple[ViewerNodes, ViewerElements]":
    """Build a minimal pair the controller can wear."""
    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    nodes = ViewerNodes(
        ids=np.array([], dtype=np.int64),
        coords=np.zeros((0, 3), dtype=np.float64),
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=NodalLoadView([]), sp=SPView([]),
        masses=MassView([]), constraints=NodeConstraintView([]),
    )
    elements = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        partition_by_eid=partition_by_eid,
    )
    return nodes, elements


class _MiniView:
    """Stand-in for ViewerData carrying only ``elements``."""
    def __init__(self, elements: ViewerElements) -> None:
        self.elements = elements


def _make_controller(
    *,
    scene: _FakeScene,
    view: "_MiniView | None",
) -> ColorModeController:
    return ColorModeController(
        color_mgr=_NoopColorMgr(),
        registry=_NoopRegistry(),
        scene=scene,
        sel=_NoopSel(),
        vis_mgr=_NoopVisMgr(),
        plotter=_NoopPlotter(),
        view=view,  # type: ignore[arg-type]
    )


# =====================================================================
# _partition_idle — happy path
# =====================================================================


def test_partition_idle_returns_palette_color_for_single_rank_entity() -> None:
    """Entity (2, 10) owns elements 100 and 101, both on rank 1.
    Dominant rank = 1, color = palette index 1."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101]})
    _, elements = _make_viewer_data(partition_by_eid={100: 1, 101: 1})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(rgb, _GROUP_PALETTE_RGB[1])


def test_partition_idle_distinct_colors_for_distinct_ranks() -> None:
    """Two entities on different ranks get different palette colors."""
    scene = _FakeScene(brep_to_elems={
        (2, 10): [100, 101],
        (2, 20): [200, 201],
    })
    _, elements = _make_viewer_data(partition_by_eid={
        100: 0, 101: 0, 200: 1, 201: 1,
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_0 = ctrl._partition_idle((2, 10))
    rgb_1 = ctrl._partition_idle((2, 20))
    np.testing.assert_array_equal(rgb_0, _GROUP_PALETTE_RGB[0])
    np.testing.assert_array_equal(rgb_1, _GROUP_PALETTE_RGB[1])
    assert not np.array_equal(rgb_0, rgb_1)


def test_partition_idle_dominant_rank_wins_in_mixed_entity() -> None:
    """Entity owns 3 elements: 2 on rank 0, 1 on rank 1.
    Dominant rank = 0; mixed-rank case degrades to dominant color."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101, 102]})
    _, elements = _make_viewer_data(partition_by_eid={
        100: 0, 101: 0, 102: 1,
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(rgb, _GROUP_PALETTE_RGB[0])


def test_partition_idle_rank_wraps_palette_length() -> None:
    """Rank N >= len(palette) wraps modulo so high-rank-count models
    still render without crashing — colors collide, which is acceptable
    visual degradation."""
    high_rank = len(_GROUP_PALETTE_RGB) + 3
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(partition_by_eid={100: high_rank})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(
        rgb, _GROUP_PALETTE_RGB[high_rank % len(_GROUP_PALETTE_RGB)],
    )


# =====================================================================
# _partition_idle — degraded paths
# =====================================================================


def test_partition_idle_no_view_returns_fallback() -> None:
    """No ViewerData bound (e.g. live from_fem with no h5 source) —
    every entity gets the fallback color."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    ctrl = _make_controller(scene=scene, view=None)

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_partition_idle_view_without_partitions_returns_fallback() -> None:
    """Single-partition models / pre-2.10.0 archives produce a view
    with ``has_partitions == False``. All entities uniform fallback."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(partition_by_eid=None)
    assert elements.has_partitions is False
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_partition_idle_entity_without_elements_returns_fallback() -> None:
    """A DimTag with no elements in scene.brep_to_elems gets fallback."""
    scene = _FakeScene(brep_to_elems={})  # entity not registered
    _, elements = _make_viewer_data(partition_by_eid={100: 0})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 99))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_partition_idle_entity_with_only_unranked_elements() -> None:
    """Entity owns elements that aren't in the partition map (their
    partition_for returns None — e.g. elements emitted outside any
    partition bracket). Fallback color, not crash."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101]})
    # partition_by_eid maps a different element (50), not 100/101
    _, elements = _make_viewer_data(partition_by_eid={50: 0})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._partition_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


# =====================================================================
# set_mode dispatch
# =====================================================================


def test_set_mode_partition_installs_partition_idle_fn() -> None:
    """``set_mode("Partition")`` swaps the ColorManager's idle-fn to
    ``_partition_idle`` — same dispatch shape as Element Type / Physical
    Group, so hover/pick/visibility layering all keep working."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(partition_by_eid={100: 0})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    ctrl.set_mode("Partition")
    assert ctrl.mode == "Partition"
    # Bound-method identity: ``is`` would compare the per-access
    # bound-method object; ``==`` checks ``__func__`` + ``__self__``.
    assert ctrl._color_mgr.idle_fn == ctrl._partition_idle


def test_set_mode_default_after_partition_resets_idle_fn() -> None:
    """Switching from Partition to Default goes through reset_idle_fn,
    not just a no-op — consistent with the other idle-fn modes."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(partition_by_eid={100: 0})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    ctrl.set_mode("Partition")
    ctrl.set_mode("Default")
    assert ctrl.mode == "Default"
    assert ctrl._color_mgr.idle_fn is None


# =====================================================================
# DisplayTab dropdown entry
# =====================================================================


def test_color_modes_list_includes_partition() -> None:
    """``COLOR_MODES`` in mesh_tabs feeds the DisplayTab dropdown.
    ``Partition`` must appear so the user can select it from the UI."""
    from apeGmsh.viewers.ui.mesh_tabs import COLOR_MODES
    assert "Partition" in COLOR_MODES
    # Ordering convention: append between Physical Group and Quality
    # (Quality is the scalar-mapper outlier; categorical modes group
    # together).
    idx_pg = COLOR_MODES.index("Physical Group")
    idx_part = COLOR_MODES.index("Partition")
    idx_qual = COLOR_MODES.index("Quality")
    assert idx_pg < idx_part < idx_qual
