"""ExplodeController — unit tests (no Qt, no GPU).

Tests cover:
- _clamp01 bounds
- _NO_EXPLODE_MODES guard
- _build_surf_elem_colors mapping
- _apply_block_colors uniform color (including grey-face fix)
- Zero magnitude guard → no explosion
- Single group guard → no explosion
- Two-group PG explode → 2 actors added
- Actor hiding of dim=3 during explosion
- reset via _clear_explode
- set_value backward compat
- Per-axis isolation (x only, x+y)
- Partition grouping
- Partition no-view fallback
- Color restoration from surface mesh
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.explode_controller import (
    ExplodeController,
    _NO_EXPLODE_MODES,
    _apply_block_colors,
    _build_surf_elem_colors,
    _clamp01,
)


# ======================================================================
# Minimal stubs
# ======================================================================


class _FakeActor:
    def __init__(self) -> None:
        self._visible = True

    def GetVisibility(self) -> bool:
        return self._visible

    def SetVisibility(self, v) -> None:
        self._visible = bool(v)


class _FakeRegistry:
    def __init__(self) -> None:
        self.actor3 = _FakeActor()
        self.dim_actors = {3: self.actor3}
        self.dim_meshes: dict = {}


class _FakeSurfMesh:
    def __init__(self, colors, c2e) -> None:
        self.cell_data = {"colors": np.asarray(colors, dtype=np.uint8)}
        self._c2e = np.asarray(c2e, dtype=np.int64)


class _FakeVol:
    """Minimal pyvista UnstructuredGrid stand-in."""

    def __init__(self, n_cells: int) -> None:
        self.n_cells = n_cells
        self._centers = np.zeros((n_cells, 3), dtype=np.float64)
        for i in range(n_cells):
            self._centers[i] = [float(i), 0.0, 0.0]

    def extract_cells(self, idxs) -> "_FakeVol":
        sub = _FakeVol(len(idxs))
        sub._centers = self._centers[idxs]
        return sub

    def copy(self) -> "_FakeVol":
        v = _FakeVol(self.n_cells)
        v._centers = self._centers.copy()
        return v

    def translate(self, offset, inplace=False) -> "_FakeVol":
        self._centers += np.asarray(offset)
        return self

    def cell_centers(self) -> "_FakeCenters":
        return _FakeCenters(self._centers)

    @property
    def cell_data(self) -> dict:
        if not hasattr(self, "_cell_data"):
            self._cell_data: dict = {}
        return self._cell_data


class _FakeCenters:
    def __init__(self, pts) -> None:
        self.points = pts


class _FakePlotter:
    def __init__(self) -> None:
        self._added: list = []
        self._removed: list = []

    def add_mesh(self, block, **kw) -> object:
        actor = object()
        self._added.append(actor)
        return actor

    def remove_actor(self, actor) -> None:
        self._removed.append(actor)


class _FakeScene:
    def __init__(
        self,
        *,
        vol=None,
        vol_to_elem=None,
        elem_to_brep=None,
        brep_to_group=None,
        elem_data=None,
        surf_colors=None,
        c2e=None,
    ) -> None:
        self.registry = _FakeRegistry()
        self.vol_grids: dict = {3: vol} if vol is not None else {}
        self.vol_cell_to_elem: dict = (
            {3: np.asarray(vol_to_elem, dtype=np.int64)}
            if vol_to_elem is not None else {}
        )
        self.elem_to_brep: dict = elem_to_brep or {}
        self.brep_to_group: dict = brep_to_group or {}
        self.elem_data: dict = elem_data or {}
        self.batch_cell_to_elem: dict = (
            {3: np.asarray(c2e, dtype=np.int64)} if c2e is not None else {}
        )
        self.model_diagonal: float = 10.0

        if surf_colors is not None and c2e is not None:
            self.registry.dim_meshes[3] = _FakeSurfMesh(surf_colors, c2e)


class _FakeElements:
    def __init__(self, partition_by_eid=None) -> None:
        self._partition_by_eid: dict[int, int] = partition_by_eid or {}

    @property
    def has_partitions(self) -> bool:
        return bool(self._partition_by_eid)

    def partition_for(self, eid: int) -> "int | None":
        return self._partition_by_eid.get(int(eid))

    @property
    def has_modules(self) -> bool:
        return False

    def module_for(self, eid: int) -> "str | None":
        return None


class _FakeView:
    def __init__(self, partition_by_eid=None) -> None:
        self.elements = _FakeElements(partition_by_eid)


def _make_ctrl(scene, view=None) -> ExplodeController:
    plotter = _FakePlotter()
    ctrl = ExplodeController(
        registry=scene.registry,
        scene=scene,
        plotter=plotter,
        view=view,
    )
    return ctrl


# ======================================================================
# _clamp01
# ======================================================================


def test_clamp01_below_zero() -> None:
    assert _clamp01(-1.0) == 0.0


def test_clamp01_above_one() -> None:
    assert _clamp01(2.5) == 1.0


def test_clamp01_in_range() -> None:
    assert _clamp01(0.5) == pytest.approx(0.5)


# ======================================================================
# _NO_EXPLODE_MODES
# ======================================================================


def test_no_explode_modes_contains_default_and_quality() -> None:
    assert "Default" in _NO_EXPLODE_MODES
    assert "Quality" in _NO_EXPLODE_MODES


# ======================================================================
# _build_surf_elem_colors
# ======================================================================


def test_build_surf_elem_colors_happy_path() -> None:
    colors = [[255, 0, 0], [0, 255, 0]]
    c2e = [100, 200]
    scene = _FakeScene(surf_colors=colors, c2e=c2e)
    result = _build_surf_elem_colors(scene)
    assert 100 in result
    assert 200 in result
    np.testing.assert_array_equal(result[100], [255, 0, 0])
    np.testing.assert_array_equal(result[200], [0, 255, 0])


def test_build_surf_elem_colors_no_surface_mesh() -> None:
    scene = _FakeScene()
    result = _build_surf_elem_colors(scene)
    assert result == {}


# ======================================================================
# _apply_block_colors
# ======================================================================


class _FakeBlock:
    def __init__(self, n) -> None:
        self.n_cells = n
        self.cell_data: dict = {}


def test_apply_block_colors_uniform_first_hit() -> None:
    block = _FakeBlock(4)
    vol_to_elem = np.array([100, 200, 300], dtype=np.int64)
    surf_colors = {100: np.array([255, 0, 0], dtype=np.uint8)}
    cell_indices = [0, 1, 2]
    _apply_block_colors(block, cell_indices, vol_to_elem, surf_colors)
    expected = np.tile([255, 0, 0], (4, 1))
    np.testing.assert_array_equal(block.cell_data["colors"], expected)


def test_apply_block_colors_fallback_grey_when_no_surf_color() -> None:
    block = _FakeBlock(3)
    vol_to_elem = np.array([500, 501], dtype=np.int64)
    surf_colors: dict = {}
    cell_indices = [0, 1]
    _apply_block_colors(block, cell_indices, vol_to_elem, surf_colors)
    expected = np.tile([136, 136, 136], (3, 1))
    np.testing.assert_array_equal(block.cell_data["colors"], expected)


# ======================================================================
# Zero magnitude guard
# ======================================================================


def test_apply_zero_magnitude_does_not_explode() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(vol=vol, vol_to_elem=[10, 10, 20, 20])
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    # all magnitudes are 0.0 by default
    ctrl.apply()
    assert ctrl._plotter._added == []
    assert not ctrl._active


# ======================================================================
# _NO_EXPLODE_MODES guard
# ======================================================================


def test_apply_no_explode_for_default_mode() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(vol=vol, vol_to_elem=[10, 10, 20, 20])
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Default"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert ctrl._plotter._added == []


def test_apply_no_explode_for_quality_mode() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(vol=vol, vol_to_elem=[10, 10, 20, 20])
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Quality"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert ctrl._plotter._added == []


# ======================================================================
# Single group guard
# ======================================================================


def test_apply_single_group_does_not_explode() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 10, 10],
        elem_to_brep={10: (3, 1)},
        brep_to_group={(3, 1): "A"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert ctrl._plotter._added == []


# ======================================================================
# Two-group Physical Group explode → 2 actors
# ======================================================================


def test_apply_two_pg_groups_adds_two_actors() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert len(ctrl._plotter._added) == 2
    assert ctrl._active


# ======================================================================
# Actor hiding during explosion
# ======================================================================


def test_apply_hides_dim3_actor_during_explosion() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert not scene.registry.actor3.GetVisibility()


# ======================================================================
# _clear_explode restores actor
# ======================================================================


def test_clear_explode_restores_dim3_actor() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    ctrl._magnitudes["x"] = 0.0
    ctrl.apply()
    assert scene.registry.actor3.GetVisibility()
    assert not ctrl._active


# ======================================================================
# set_value backward compat
# ======================================================================


def test_set_value_sets_all_axes() -> None:
    scene = _FakeScene()
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Default"  # won't explode, but axes are set
    ctrl.set_value(0.7)
    assert ctrl._magnitudes["x"] == pytest.approx(0.7)
    assert ctrl._magnitudes["y"] == pytest.approx(0.7)
    assert ctrl._magnitudes["z"] == pytest.approx(0.7)


# ======================================================================
# X-only offset (y/z should be zero)
# ======================================================================


def test_x_only_offset_leaves_y_z_zero() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 0.5
    ctrl._magnitudes["y"] = 0.0
    ctrl._magnitudes["z"] = 0.0
    groups = ctrl._cell_groups_vol()
    offsets = ctrl._group_offsets(vol, groups)
    for off in offsets.values():
        assert off[1] == pytest.approx(0.0)
        assert off[2] == pytest.approx(0.0)


# ======================================================================
# X+Y offset both non-zero
# ======================================================================


def test_xy_offset_both_nonzero() -> None:
    vol = _FakeVol(4)
    for i in range(4):
        vol._centers[i] = [float(i % 2) * 5.0, float(i // 2) * 5.0, 0.0]
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl._magnitudes["y"] = 1.0
    groups = ctrl._cell_groups_vol()
    offsets = ctrl._group_offsets(vol, groups)
    for off in offsets.values():
        # At least one axis non-zero when both mags are set
        assert np.any(np.abs(off[:2]) > 0.0)


# ======================================================================
# Partition grouping
# ======================================================================


def test_partition_grouping_produces_two_groups() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(vol=vol, vol_to_elem=[10, 10, 20, 20])
    view = _FakeView(partition_by_eid={10: 0, 20: 1})
    ctrl = _make_ctrl(scene, view=view)
    ctrl._mode = "Partition"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert len(ctrl._plotter._added) == 2


# ======================================================================
# Partition no-view fallback → no explosion
# ======================================================================


def test_partition_no_view_no_explosion() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(vol=vol, vol_to_elem=[10, 10, 20, 20])
    ctrl = _make_ctrl(scene, view=None)
    ctrl._mode = "Partition"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert ctrl._plotter._added == []


# ======================================================================
# Color restoration from surface mesh
# ======================================================================


def test_block_colors_taken_from_surface_mesh() -> None:
    colors = [[255, 0, 0], [0, 0, 255]]
    c2e = [10, 20]
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
        surf_colors=colors,
        c2e=c2e,
    )
    # Verify that the surf_colors lookup resolves correctly
    surf_map = _build_surf_elem_colors(scene)
    assert int(surf_map[10][0]) == 255
    assert int(surf_map[20][2]) == 255
