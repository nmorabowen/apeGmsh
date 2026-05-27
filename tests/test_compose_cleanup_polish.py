"""Compose-v1 cleanup polish — small follow-ups bundled after Phase 3.

Three items, all minor:

1. ``_phys_group_idle`` uses ``zlib.crc32`` for cross-process color
   determinism (matches the fix applied to ``_module_idle`` in
   #374). Pre-existing wart from before compose work.

2. H5 group-name collision (``"a/b"`` vs ``"a_b"``) is cosmetic-only
   per the inline comment now in ``_write_composed_from``; the
   verbatim ``label`` attribute round-trips correctly. Comment-only
   change — covered by existing 3E.1 round-trip tests.

3. ``compose_inspect(path)`` now returns ``"compose_tree"`` key with
   a derived tree view of the source's nested compose hierarchy —
   sibling to the existing flat ``"composed_from"`` list.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.viewers.core.color_mode_controller import (
    _FALLBACK_RGB,
    _GROUP_PALETTE_RGB,
    ColorModeController,
)


# =====================================================================
# Item 1: _phys_group_idle determinism — uses crc32 not hash()
# =====================================================================


class _FakeSceneForPhysGroup:
    def __init__(self, brep_to_group: dict) -> None:
        self.brep_to_group = brep_to_group


def _make_phys_group_controller(brep_to_group: dict) -> ColorModeController:
    ctrl = ColorModeController.__new__(ColorModeController)
    ctrl._scene = _FakeSceneForPhysGroup(brep_to_group)  # type: ignore[attr-defined]
    return ctrl


def test_phys_group_idle_uses_crc32_not_hash() -> None:
    """The callback must compute palette index via zlib.crc32(name)
    not abs(hash(name)). Anchors the implementation choice so a
    regression to hash() (which is PYTHONHASHSEED-randomized) is
    caught at unit-test time, not on a CI flake."""
    ctrl = _make_phys_group_controller({(2, 10): "wall_steel"})
    rgb = ctrl._phys_group_idle((2, 10))
    expected_idx = zlib.crc32(b"wall_steel") % len(_GROUP_PALETTE_RGB)
    np.testing.assert_array_equal(rgb, _GROUP_PALETTE_RGB[expected_idx])


def test_phys_group_idle_none_returns_fallback() -> None:
    """Group not in mapping -> fallback (unchanged behavior)."""
    ctrl = _make_phys_group_controller({})
    np.testing.assert_array_equal(ctrl._phys_group_idle((2, 10)), _FALLBACK_RGB)


def test_phys_group_idle_deterministic_across_calls() -> None:
    """Same name -> same color, repeated calls."""
    ctrl = _make_phys_group_controller({(2, 10): "wall_steel"})
    rgb_first = ctrl._phys_group_idle((2, 10))
    rgb_again = ctrl._phys_group_idle((2, 10))
    np.testing.assert_array_equal(rgb_first, rgb_again)


# =====================================================================
# Item 3: compose_inspect adds 'compose_tree' key
# =====================================================================
# Fixtures mirror tests/test_compose_tree.py — no Gmsh, no OpenSeesPy;
# pure FEMData → H5 → from_h5 → compose → save chain.


def _make_fem(
    *,
    node_ids: "list[int] | np.ndarray",
    elem_ids: "list[int] | np.ndarray",
) -> FEMData:
    """Tiny FEMData with one Line2 group; empty arrays supported."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    elem_ids = np.asarray(elem_ids, dtype=np.int64)
    n = node_ids.size
    if n > 0:
        node_coords = np.array(
            [[float(i), 0.0, 0.0] for i in range(n)],
            dtype=np.float64,
        )
    else:
        node_coords = np.zeros((0, 3), dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    if elem_ids.size > 0:
        conn = np.array(
            [
                [int(node_ids[i % n]), int(node_ids[(i + 1) % n])]
                for i in range(elem_ids.size)
            ],
            dtype=np.int64,
        )
    else:
        conn = np.zeros((0, 2), dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=n,
        n_elems=elem_ids.size,
        bandwidth=1,
        types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


@pytest.fixture
def empty_h5(tmp_path: Path) -> Path:
    fem = _make_fem(node_ids=[], elem_ids=[])
    p = tmp_path / "empty.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def leaf_h5(tmp_path: Path) -> Path:
    fem = _make_fem(node_ids=[1, 2, 3], elem_ids=[10, 11])
    p = tmp_path / "leaf.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def depth_1_h5(tmp_path: Path, empty_h5: Path, leaf_h5: Path) -> Path:
    """Depth-1 source: composed_from = [partA]."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(leaf_h5, label="partA")
    out = tmp_path / "depth_1.h5"
    g.save(out)
    return out


@pytest.fixture
def depth_2_h5(tmp_path: Path, empty_h5: Path, depth_1_h5: Path) -> Path:
    """Depth-2 source: composed_from = [assemblyM, assemblyM/partA]."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(depth_1_h5, label="assemblyM")
    out = tmp_path / "depth_2.h5"
    g.save(out)
    return out


def test_compose_inspect_uncomposed_returns_empty_tree(leaf_h5: Path) -> None:
    """An uncomposed source: composed_from is empty -> compose_tree is ()."""
    g = apeGmsh.from_h5(leaf_h5)
    result = g.compose_inspect(leaf_h5)

    assert "compose_tree" in result
    assert result["compose_tree"] == ()
    # Cross-check: composed_from is also empty (input to the tree builder)
    assert result["composed_from"] == ()


def test_compose_inspect_depth_1_returns_single_root(depth_1_h5: Path) -> None:
    """A depth-1 composed source: compose_tree has one root, no children."""
    g = apeGmsh.from_h5(depth_1_h5)
    result = g.compose_inspect(depth_1_h5)

    assert "compose_tree" in result
    tree = result["compose_tree"]
    assert len(tree) == 1
    assert tree[0].label == "partA"
    assert tree[0].children == ()
    # tree[0].record identity preserved from composed_from
    composed_from = result["composed_from"]
    assert len(composed_from) == 1
    assert tree[0].record is composed_from[0]


def test_compose_inspect_depth_2_returns_nested(depth_2_h5: Path) -> None:
    """A depth-2 composed source: compose_tree has one root with one child.

    Flat composed_from is [assemblyM, assemblyM/partA]; tree is
    assemblyM -> partA.
    """
    g = apeGmsh.from_h5(depth_2_h5)
    result = g.compose_inspect(depth_2_h5)

    tree = result["compose_tree"]
    assert len(tree) == 1
    assert tree[0].label == "assemblyM"
    assert len(tree[0].children) == 1
    assert tree[0].children[0].label == "partA"
    assert tree[0].children[0].children == ()


def test_compose_inspect_tree_matches_compose_tree_method(depth_1_h5: Path) -> None:
    """Parity: compose_inspect(path)["compose_tree"] equals
    g.compose_tree() on the post-load session. Both call
    _build_compose_tree internally — this is a structural lock."""
    g_inspect = apeGmsh.from_h5(depth_1_h5)
    inspect_tree = g_inspect.compose_inspect(depth_1_h5)["compose_tree"]

    g_load = apeGmsh.from_h5(depth_1_h5)
    method_tree = g_load.compose_tree()

    assert len(inspect_tree) == len(method_tree)
    for a, b in zip(inspect_tree, method_tree):
        assert a.label == b.label
        assert a.children == b.children
        assert a.record.label == b.record.label
