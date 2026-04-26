"""Phase 0 — FEMData snapshot_id content hash."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.mesh._femdata_hash import compute_snapshot_id


# =====================================================================
# Mock FEMData — minimal surface area used by compute_snapshot_id
# =====================================================================

class _MockGroup(SimpleNamespace):
    """Stand-in for ElementGroup in the hash function."""


class _MockPhysical(SimpleNamespace):
    """Stand-in for PhysicalGroupSet."""

    def __init__(self, groups: dict[tuple[int, int], dict]) -> None:
        # groups: {(dim, tag): {"name": str, "node_ids": ndarray}}
        super().__init__()
        self._groups = groups

    def get_all(self) -> list[tuple[int, int]]:
        return sorted(self._groups.keys())

    def get_name(self, dim: int, tag: int) -> str:
        return self._groups[(dim, tag)].get("name", "")

    def node_ids(self, target):
        if isinstance(target, tuple):
            key = target
        else:
            # Resolve by name if string
            for k, info in self._groups.items():
                if info.get("name") == target:
                    key = k
                    break
            else:
                raise KeyError(target)
        return self._groups[key]["node_ids"]


def _make_fem(
    *,
    node_ids,
    coords,
    elem_groups,        # list of (type_name, ids, connectivity)
    physical=None,      # dict of (dim, tag) -> {"name": ..., "node_ids": ...}
):
    """Build a minimal mock FEMData for hash testing."""
    nodes = SimpleNamespace(
        ids=np.asarray(node_ids),
        coords=np.asarray(coords, dtype=np.float64),
        physical=_MockPhysical(physical) if physical else None,
    )
    groups = [
        _MockGroup(
            type_name=name,
            ids=np.asarray(eids),
            connectivity=np.asarray(conn),
        )
        for name, eids, conn in elem_groups
    ]
    elements = SimpleNamespace()
    elements.__iter__ = lambda self: iter(groups)
    # SimpleNamespace doesn't auto-bind; do it manually
    elements_obj = type("MockElements", (), {})()
    elements_obj.__class__.__iter__ = lambda self: iter(groups)
    return SimpleNamespace(nodes=nodes, elements=elements_obj)


# =====================================================================
# Hash properties
# =====================================================================

def test_hash_is_32_char_hex() -> None:
    fem = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
    )
    h = compute_snapshot_id(fem)
    assert isinstance(h, str)
    assert len(h) == 32
    int(h, 16)  # parses as hex


def test_hash_is_deterministic() -> None:
    fem = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
    )
    a = compute_snapshot_id(fem)
    b = compute_snapshot_id(fem)
    assert a == b


def test_hash_unchanged_by_node_id_permutation() -> None:
    """Re-ordering nodes (same IDs / coords) should not change the hash."""
    base = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    permuted = _make_fem(
        node_ids=[3, 1, 2],
        coords=[[0, 1, 0], [0, 0, 0], [1, 0, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    assert compute_snapshot_id(base) == compute_snapshot_id(permuted)


def test_hash_changes_on_coord_edit() -> None:
    base = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    moved = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1.0001, 0, 0], [0, 1, 0]],   # tiny shift
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    assert compute_snapshot_id(base) != compute_snapshot_id(moved)


def test_hash_changes_on_connectivity_edit() -> None:
    base = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
    )
    different = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 4, 3]])],   # winding swap
    )
    assert compute_snapshot_id(base) != compute_snapshot_id(different)


def test_hash_changes_on_node_addition() -> None:
    base = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    extra = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    assert compute_snapshot_id(base) != compute_snapshot_id(extra)


def test_hash_changes_on_element_addition() -> None:
    base = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    extra = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        elem_groups=[("tri3", [1, 2], [[1, 2, 3], [1, 3, 4]])],
    )
    assert compute_snapshot_id(base) != compute_snapshot_id(extra)


def test_hash_changes_on_type_name_change() -> None:
    base = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("tri3", [1], [[1, 2, 3]])],
    )
    renamed = _make_fem(
        node_ids=[1, 2, 3],
        coords=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        elem_groups=[("triangle3", [1], [[1, 2, 3]])],
    )
    assert compute_snapshot_id(base) != compute_snapshot_id(renamed)


# =====================================================================
# Physical groups affect the hash
# =====================================================================

def test_hash_changes_on_pg_addition() -> None:
    no_pg = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
    )
    with_pg = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([1, 2, 3, 4])}},
    )
    assert compute_snapshot_id(no_pg) != compute_snapshot_id(with_pg)


def test_hash_changes_on_pg_rename() -> None:
    a = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([1, 2, 3, 4])}},
    )
    b = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Floor", "node_ids": np.array([1, 2, 3, 4])}},
    )
    assert compute_snapshot_id(a) != compute_snapshot_id(b)


def test_hash_unchanged_by_pg_node_order() -> None:
    a = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([1, 2, 3, 4])}},
    )
    b = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([4, 3, 2, 1])}},
    )
    assert compute_snapshot_id(a) == compute_snapshot_id(b)


def test_hash_changes_on_pg_membership_change() -> None:
    a = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([1, 2, 3])}},
    )
    b = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        elem_groups=[("quad4", [1], [[1, 2, 3, 4]])],
        physical={(2, 1): {"name": "Body", "node_ids": np.array([1, 2, 3, 4])}},
    )
    assert compute_snapshot_id(a) != compute_snapshot_id(b)


# =====================================================================
# FEMData.snapshot_id property — caching behavior
# =====================================================================

def test_snapshot_id_property_is_cached_via_real_fem(g) -> None:
    """Smoke test against a real FEMData built from a tiny gmsh box."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)   # very coarse — fast
    g.mesh.generation.generate(dim=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    h1 = fem.snapshot_id
    h2 = fem.snapshot_id   # second access — should hit the cache

    assert isinstance(h1, str)
    assert len(h1) == 32
    assert h1 == h2

    # Cache lives as ``_snapshot_id_cache`` instance attribute.
    assert getattr(fem, "_snapshot_id_cache", None) == h1
