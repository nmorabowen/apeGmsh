"""ExplodeController — unit tests (no Qt, no GPU).

Tests cover:
- _clamp01 bounds
- _NO_EXPLODE_MODES guard
- _build_surf_elem_colors mapping
- _apply_block_colors uniform color (including grey-face fix)
- Zero magnitude guard → no explosion
- Single group guard → no explosion
- Two-group PG explode → 2 actors added
- Pure 1D / pure 2D explode → grouped actors added
- Mixed 1D+2D explode → every displayed dimension participates
- Actor hiding/restoration for dimensions 1, 2, and 3
- Per-dimension render style and group colour preservation
- Hidden-entity filtering uses the exploded dimension
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
    def __init__(self, *, representation=2, edge_visibility=True) -> None:
        self._visible = True
        self._property = _FakeProperty(representation, edge_visibility)

    def GetVisibility(self) -> bool:
        return self._visible

    def SetVisibility(self, v) -> None:
        self._visible = bool(v)

    def GetProperty(self):
        return self._property


class _FakeProperty:
    def __init__(self, representation, edge_visibility) -> None:
        self._representation = representation
        self._edge_visibility = edge_visibility

    def GetRepresentation(self):
        return self._representation

    def GetEdgeVisibility(self):
        return self._edge_visibility


class _FakeRegistry:
    def __init__(self, dims=(3,)) -> None:
        self.dim_actors = {dim: _FakeActor() for dim in dims}
        self.actor3 = self.dim_actors.get(3)
        self.dim_meshes: dict = {}
        self.dim_wire_actors: dict = {}
        self.dim_node_actors: dict = {}
        self.dim_silhouette_actors: dict = {}
        self._add_mesh_kwargs = {
            1: {
                "line_width": 3.0,
                "render_lines_as_tubes": True,
                "show_edges": False,
                "opacity": 1.0,
            },
            2: {
                "line_width": 0.5,
                "render_lines_as_tubes": False,
                "show_edges": True,
                "opacity": 0.65,
            },
            3: {
                "line_width": 0.5,
                "render_lines_as_tubes": False,
                "show_edges": True,
                "opacity": 1.0,
            },
        }

    @property
    def dims(self):
        return sorted(self.dim_meshes)


class _FakeSurfMesh:
    def __init__(self, colors, c2e) -> None:
        self.cell_data = {"colors": np.asarray(colors, dtype=np.uint8)}
        self._c2e = np.asarray(c2e, dtype=np.int64)


class _FakeVol:
    """Minimal pyvista UnstructuredGrid stand-in."""

    def __init__(self, n_cells: int, *, dim: int = 3) -> None:
        self.n_cells = n_cells
        self.dim = dim
        self._centers = np.zeros((n_cells, 3), dtype=np.float64)
        self._cell_data: dict = {}
        for i in range(n_cells):
            self._centers[i] = [float(i), 0.0, 0.0]

    def extract_cells(self, idxs) -> "_FakeVol":
        idxs = np.asarray(idxs, dtype=np.int64)
        sub = _FakeVol(len(idxs), dim=self.dim)
        sub._centers = self._centers[idxs]
        for name, values in self._cell_data.items():
            values_arr = np.asarray(values)
            if len(values_arr) == self.n_cells:
                sub._cell_data[name] = values_arr[idxs].copy()
        return sub

    def copy(self) -> "_FakeVol":
        v = _FakeVol(self.n_cells, dim=self.dim)
        v._centers = self._centers.copy()
        v._cell_data = {
            name: np.asarray(values).copy()
            for name, values in self._cell_data.items()
        }
        return v

    def translate(self, offset, inplace=False) -> "_FakeVol":
        self._centers += np.asarray(offset)
        return self

    def cell_centers(self) -> "_FakeCenters":
        return _FakeCenters(self._centers)

    @property
    def cell_data(self) -> dict:
        return self._cell_data


class _FakeCenters:
    def __init__(self, pts) -> None:
        self.points = pts


class _FakePlotter:
    def __init__(self) -> None:
        self._added: list = []
        self._add_calls: list[tuple[_FakeVol, dict]] = []
        self._removed: list = []

    def add_mesh(self, block, **kw) -> object:
        actor = _FakeActor()
        self._added.append(actor)
        self._add_calls.append((block, kw))
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
        grids_by_dim=None,
        cell_to_elem_by_dim=None,
    ) -> None:
        grids_by_dim = dict(grids_by_dim or {})
        dims = tuple(sorted(grids_by_dim)) if grids_by_dim else (3,)
        self.registry = _FakeRegistry(dims=dims)
        self.registry.dim_meshes.update(grids_by_dim)
        self.vol_grids: dict = {3: vol} if vol is not None else {}
        self.vol_cell_to_elem: dict = (
            {3: np.asarray(vol_to_elem, dtype=np.int64)}
            if vol_to_elem is not None else {}
        )
        self.elem_to_brep: dict = elem_to_brep or {}
        self.brep_to_group: dict = brep_to_group or {}
        self.elem_data: dict = elem_data or {}
        self.batch_cell_to_elem: dict = {
            int(dim): np.asarray(tags, dtype=np.int64)
            for dim, tags in (cell_to_elem_by_dim or {}).items()
        }
        # A real 3D MeshSceneData always carries both the render mesh in the
        # registry and the pre-surface volume grid. Keep the legacy test setup
        # faithful to that contract so a dimension-generic implementation can
        # discover dim=3 through the same registry path as dim=1/2.
        if vol is not None and 3 not in self.registry.dim_meshes:
            self.registry.dim_meshes[3] = vol.copy()
        if vol_to_elem is not None and 3 not in self.batch_cell_to_elem:
            self.batch_cell_to_elem[3] = np.asarray(vol_to_elem, dtype=np.int64)
        legacy_batch = (
            {3: np.asarray(c2e, dtype=np.int64)} if c2e is not None else {}
        )
        self.batch_cell_to_elem.update(legacy_batch)
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


def _make_ctrl(scene, view=None, vis_mgr=None) -> ExplodeController:
    plotter = _FakePlotter()
    ctrl = ExplodeController(
        registry=scene.registry,
        scene=scene,
        plotter=plotter,
        view=view,
        vis_mgr=vis_mgr,
    )
    return ctrl


def _make_dim_scene(dim: int, *, n_groups: int = 2) -> _FakeScene:
    """Build a pure 1D/2D scene using the same stores as MeshSceneData."""
    cells_per_group = 2
    n_cells = n_groups * cells_per_group
    grid = _FakeVol(n_cells, dim=dim)
    elem_tags: list[int] = []
    elem_to_brep: dict[int, tuple[int, int]] = {}
    elem_data: dict[int, dict] = {}
    brep_to_group: dict[tuple[int, int], str] = {}
    palette = (
        np.array([220, 20, 60], dtype=np.uint8),
        np.array([65, 105, 225], dtype=np.uint8),
        np.array([60, 179, 113], dtype=np.uint8),
    )
    colors: list[np.ndarray] = []

    for group_idx in range(n_groups):
        brep = (dim, group_idx + 1)
        brep_to_group[brep] = f"dim{dim}-group-{group_idx + 1}"
        for local_idx in range(cells_per_group):
            elem_tag = dim * 1000 + group_idx * 100 + local_idx + 1
            elem_tags.append(elem_tag)
            elem_to_brep[elem_tag] = brep
            elem_data[elem_tag] = {"type_name": f"type-{group_idx + 1}"}
            colors.append(palette[group_idx % len(palette)])
            cell_idx = group_idx * cells_per_group + local_idx
            grid._centers[cell_idx] = [group_idx * 10.0 + local_idx, 0.0, 0.0]

    grid.cell_data["colors"] = np.asarray(colors, dtype=np.uint8)
    return _FakeScene(
        grids_by_dim={dim: grid},
        cell_to_elem_by_dim={dim: elem_tags},
        elem_to_brep=elem_to_brep,
        brep_to_group=brep_to_group,
        elem_data=elem_data,
    )


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
# Pure lower-dimensional scenes (regression: explode was dim=3-only)
# ======================================================================


@pytest.mark.parametrize("dim", [1, 2])
def test_apply_two_pg_groups_explodes_pure_lower_dim_scene(dim: int) -> None:
    scene = _make_dim_scene(dim)
    ctrl = _make_ctrl(scene)

    ctrl.set_mode("Physical Group")
    ctrl.set_axis("x", 1.0)

    assert ctrl._active
    assert len(ctrl._plotter._add_calls) == 2
    assert not scene.registry.dim_actors[dim].GetVisibility()

    blocks = [block for block, _ in ctrl._plotter._add_calls]
    assert {block.dim for block in blocks} == {dim}

    # Each physical group keeps its active-mode colour instead of falling
    # back to the volume-only grey path.
    block_colours = set()
    for block in blocks:
        unique = np.unique(block.cell_data["colors"], axis=0)
        assert len(unique) == 1
        block_colours.add(tuple(int(v) for v in unique[0]))
    assert block_colours == {(220, 20, 60), (65, 105, 225)}

    # The two groups must move away from their original centroids (0.5, 10.5).
    exploded_x = sorted(
        float(block.cell_centers().points[:, 0].mean())
        for block in blocks
    )
    assert exploded_x[0] < 0.5
    assert exploded_x[1] > 10.5


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (1, {"line_width": 3.0, "render_lines_as_tubes": True, "show_edges": False}),
        (2, {"opacity": 0.65, "show_edges": True}),
    ],
)
def test_lower_dim_explode_preserves_dimension_render_style(
    dim: int, expected: dict,
) -> None:
    scene = _make_dim_scene(dim)
    ctrl = _make_ctrl(scene)

    ctrl.set_mode("Physical Group")
    ctrl.set_axis("x", 1.0)

    assert len(ctrl._plotter._add_calls) == 2
    for _, kwargs in ctrl._plotter._add_calls:
        for name, value in expected.items():
            assert kwargs[name] == value


@pytest.mark.parametrize("dim", [1, 2])
def test_element_type_mode_explodes_pure_lower_dim_scene(dim: int) -> None:
    scene = _make_dim_scene(dim)
    ctrl = _make_ctrl(scene)

    ctrl.set_mode("Element Type")
    ctrl.set_axis("x", 1.0)

    assert ctrl._active
    assert len(ctrl._plotter._add_calls) == 2


@pytest.mark.parametrize("dim", [1, 2])
def test_partition_mode_explodes_pure_lower_dim_scene(dim: int) -> None:
    scene = _make_dim_scene(dim)
    elem_tags = scene.batch_cell_to_elem[dim]
    partition_by_eid = {
        int(elem_tag): cell_idx // 2
        for cell_idx, elem_tag in enumerate(elem_tags)
    }
    ctrl = _make_ctrl(scene, view=_FakeView(partition_by_eid=partition_by_eid))

    ctrl.set_mode("Partition")
    ctrl.set_axis("x", 1.0)

    assert ctrl._active
    assert len(ctrl._plotter._add_calls) == 2


def test_shared_partition_uses_same_offset_across_dimensions() -> None:
    line_scene = _make_dim_scene(1, n_groups=3)
    surface_scene = _make_dim_scene(2, n_groups=3)
    surface_grid = surface_scene.registry.dim_meshes[2]
    for group_idx, x0 in enumerate((0.0, 30.0, 40.0)):
        for local_idx in range(2):
            cell_idx = group_idx * 2 + local_idx
            surface_grid._centers[cell_idx] = [x0 + local_idx, 0.0, 0.0]

    scene = _FakeScene(
        grids_by_dim={
            1: line_scene.registry.dim_meshes[1],
            2: surface_grid,
        },
        cell_to_elem_by_dim={
            1: line_scene.batch_cell_to_elem[1],
            2: surface_scene.batch_cell_to_elem[2],
        },
        elem_to_brep={
            **line_scene.elem_to_brep,
            **surface_scene.elem_to_brep,
        },
    )
    partition_by_eid = {
        int(elem_tag): cell_idx // 2
        for dim in (1, 2)
        for cell_idx, elem_tag in enumerate(scene.batch_cell_to_elem[dim])
    }
    ctrl = _make_ctrl(scene, view=_FakeView(partition_by_eid=partition_by_eid))

    ctrl.set_mode("Partition")
    ctrl.set_axis("x", 1.0)

    assert len(ctrl._plotter._add_calls) == 6
    original_grids = {
        1: line_scene.registry.dim_meshes[1],
        2: surface_grid,
    }
    offsets_by_partition: dict[int, dict[int, float]] = {}
    for call_idx, (block, _kwargs) in enumerate(ctrl._plotter._add_calls):
        partition = call_idx // 2
        dim = block.dim
        cell_slice = slice(partition * 2, partition * 2 + 2)
        original_x = original_grids[dim]._centers[cell_slice, 0].mean()
        exploded_x = block._centers[:, 0].mean()
        offsets_by_partition.setdefault(partition, {})[dim] = (
            exploded_x - original_x
        )

    for offsets_by_dim in offsets_by_partition.values():
        assert offsets_by_dim[1] == pytest.approx(offsets_by_dim[2])


@pytest.mark.parametrize("dim", [1, 2])
def test_reset_lower_dim_explode_restores_original_actor(dim: int) -> None:
    scene = _make_dim_scene(dim)
    ctrl = _make_ctrl(scene)

    ctrl.set_mode("Physical Group")
    ctrl.set_axis("x", 1.0)
    exploded_actors = list(ctrl._explode_actors)
    assert ctrl._active

    ctrl.set_axis("x", 0.0)

    assert not ctrl._active
    assert ctrl._explode_actors == []
    assert scene.registry.dim_actors[dim].GetVisibility()
    assert ctrl._plotter._removed == exploded_actors


# ======================================================================
# Mixed lower-dimensional scene
# ======================================================================


def test_mixed_dim1_dim2_scene_explodes_both_dimensions() -> None:
    line_scene = _make_dim_scene(1)
    surface_scene = _make_dim_scene(2)
    scene = _FakeScene(
        grids_by_dim={
            1: line_scene.registry.dim_meshes[1],
            2: surface_scene.registry.dim_meshes[2],
        },
        cell_to_elem_by_dim={
            1: line_scene.batch_cell_to_elem[1],
            2: surface_scene.batch_cell_to_elem[2],
        },
        elem_to_brep={
            **line_scene.elem_to_brep,
            **surface_scene.elem_to_brep,
        },
        brep_to_group={
            **line_scene.brep_to_group,
            **surface_scene.brep_to_group,
        },
    )
    ctrl = _make_ctrl(scene)

    ctrl.set_mode("Physical Group")
    ctrl.set_axis("x", 1.0)

    assert ctrl._active
    assert len(ctrl._plotter._add_calls) == 4
    exploded_dims = [block.dim for block, _ in ctrl._plotter._add_calls]
    assert exploded_dims.count(1) == 2
    assert exploded_dims.count(2) == 2
    assert not scene.registry.dim_actors[1].GetVisibility()
    assert not scene.registry.dim_actors[2].GetVisibility()


class _FakeVisibilityManager:
    def __init__(self, hidden) -> None:
        self.hidden = set(hidden)


def test_lower_dim_explode_excludes_hidden_entities_of_same_dimension() -> None:
    scene = _make_dim_scene(2, n_groups=3)
    vis_mgr = _FakeVisibilityManager(hidden={(2, 2)})
    ctrl = _make_ctrl(scene, vis_mgr=vis_mgr)

    ctrl.set_mode("Physical Group")
    ctrl.set_axis("x", 1.0)

    assert ctrl._active
    assert len(ctrl._plotter._add_calls) == 2
    visible_colours = {
        tuple(int(v) for v in np.unique(block.cell_data["colors"], axis=0)[0])
        for block, _ in ctrl._plotter._add_calls
    }
    assert visible_colours == {(220, 20, 60), (60, 179, 113)}


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
# enforce_hiding hides a node actor swapped in mid-explosion
# (regression: point-size rebuild leaked a visible node cloud)
# ======================================================================


def test_enforce_hiding_hides_swapped_in_node_actor() -> None:
    vol = _FakeVol(4)
    scene = _FakeScene(
        vol=vol,
        vol_to_elem=[10, 10, 20, 20],
        elem_to_brep={10: (3, 1), 20: (3, 2)},
        brep_to_group={(3, 1): "A", (3, 2): "B"},
    )
    # A node-cloud actor present before explosion (snapshotted + hidden).
    old_node = _FakeActor()
    scene.registry.dim_node_actors = {0: old_node}
    ctrl = _make_ctrl(scene)
    ctrl._mode = "Physical Group"
    ctrl._magnitudes["x"] = 1.0
    ctrl.apply()
    assert ctrl._active
    assert not old_node.GetVisibility()

    # Simulate a point-size rebuild swapping in a NEW node actor: visible and
    # unknown to the id-keyed _original_visibility snapshot.
    new_node = _FakeActor()
    new_node.SetVisibility(True)
    scene.registry.dim_node_actors = {0: new_node}

    ctrl.enforce_hiding()
    assert not new_node.GetVisibility(), (
        "swapped-in node actor must be hidden while exploded"
    )


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
