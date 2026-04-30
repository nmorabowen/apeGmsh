"""SlabSelector — record and resolution tests (Phase 0).

Pure logic; no Qt, no plotter. Uses a small real FEMData built via the
``g`` session fixture so PG / label resolution exercises the actual
broker.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.diagrams import SlabSelector, normalize_selector


# =====================================================================
# Construction validation
# =====================================================================

def test_construction_with_only_component():
    s = SlabSelector(component="displacement_x")
    assert s.component == "displacement_x"
    assert s.pg is None and s.label is None
    assert s.selection is None and s.ids is None


def test_construction_rejects_multiple_named_targets():
    with pytest.raises(ValueError):
        SlabSelector(
            component="displacement_x",
            pg=("A",), label=("B",),
        )


def test_construction_rejects_blank_component():
    with pytest.raises(ValueError):
        SlabSelector(component="")


def test_construction_rejects_pg_plus_ids():
    with pytest.raises(ValueError):
        SlabSelector(
            component="displacement_x",
            pg=("A",), ids=(1, 2),
        )


# =====================================================================
# normalize_selector helper
# =====================================================================

def test_normalize_string_pg():
    s = normalize_selector(component="displacement_x", pg="Top")
    assert s.pg == ("Top",)
    assert s.label is None and s.selection is None and s.ids is None


def test_normalize_list_label():
    s = normalize_selector(component="stress_xx", label=["A", "B"])
    assert s.label == ("A", "B")


def test_normalize_ids_array():
    arr = np.array([5, 7, 11], dtype=np.int64)
    s = normalize_selector(component="displacement_x", ids=arr)
    assert s.ids == (5, 7, 11)


# =====================================================================
# short_label
# =====================================================================

def test_short_label_unrestricted():
    s = SlabSelector(component="displacement_x")
    assert "(all)" in s.short_label()


def test_short_label_pg():
    s = SlabSelector(component="stress_xx", pg=("Body",))
    assert "Body" in s.short_label()
    assert "stress_xx" in s.short_label()


def test_short_label_pg_union():
    s = SlabSelector(component="displacement_z", pg=("Top", "Roof"))
    assert "Top+Roof" in s.short_label()


def test_short_label_ids():
    s = SlabSelector(component="displacement_x", ids=(1, 2, 3))
    assert "[3 ids]" in s.short_label()


# =====================================================================
# Resolution against a real FEMData
# =====================================================================

@pytest.fixture
def fem_with_groups(g):
    """Simple mesh with a known physical group + label for selector tests."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    return g.mesh.queries.get_fem_data(dim=3)


def test_resolve_node_ids_unrestricted_returns_none(fem_with_groups):
    s = SlabSelector(component="displacement_x")
    assert s.resolve_node_ids(fem_with_groups) is None


def test_resolve_node_ids_by_pg(fem_with_groups):
    s = SlabSelector(component="displacement_x", pg=("Body",))
    out = s.resolve_node_ids(fem_with_groups)
    assert out is not None
    assert out.dtype == np.int64
    assert out.size > 0


def test_resolve_element_ids_by_pg(fem_with_groups):
    s = SlabSelector(component="stress_xx", pg=("Body",))
    out = s.resolve_element_ids(fem_with_groups)
    assert out is not None
    assert out.size > 0
    assert out.dtype == np.int64


def test_resolve_node_ids_by_label(fem_with_groups):
    s = SlabSelector(component="displacement_x", label=("cube",))
    out = s.resolve_node_ids(fem_with_groups)
    assert out is not None and out.size > 0


def test_resolve_node_ids_by_explicit_ids(fem_with_groups):
    target = np.asarray(list(fem_with_groups.nodes.ids), dtype=np.int64)[:5]
    s = SlabSelector(
        component="displacement_x",
        ids=tuple(int(x) for x in target),
    )
    out = s.resolve_node_ids(fem_with_groups)
    assert out is not None
    np.testing.assert_array_equal(np.sort(out), np.sort(target))


def test_resolve_unknown_pg_raises(fem_with_groups):
    s = SlabSelector(component="displacement_x", pg=("DoesNotExist",))
    with pytest.raises(Exception):  # KeyError or similar from broker
        s.resolve_node_ids(fem_with_groups)
