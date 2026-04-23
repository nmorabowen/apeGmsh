"""Tests for g.parts.register() — dimtags / label= / pg= input modes."""
import gmsh
import pytest


# ---------------------------------------------------------------------------
# dimtags= (existing behaviour)
# ---------------------------------------------------------------------------

def test_register_positional_dimtags(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    inst = g.parts.register("col", [(3, box)])
    assert inst.entities[3] == [box]


def test_register_keyword_dimtags(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    inst = g.parts.register("col", dimtags=[(3, box)])
    assert inst.entities[3] == [box]


# ---------------------------------------------------------------------------
# label=
# ---------------------------------------------------------------------------

def test_register_by_label_unambiguous(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [box], "shaft")

    inst = g.parts.register("col", label="shaft")
    assert inst.entities[3] == [box]


def test_register_by_label_with_explicit_dim(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [box], "shaft")

    inst = g.parts.register("col", label="shaft", dim=3)
    assert inst.entities[3] == [box]


def test_register_by_label_ambiguous_raises(g):
    # Same label name at two different dimensions → ValueError.
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    pt = g.model.geometry.add_point(5, 5, 5)
    g.labels.add(3, [box], "dup")
    with pytest.warns(UserWarning):
        g.labels.add(0, [pt], "dup")  # cross-dim shadow warning

    with pytest.raises(ValueError, match="multiple dimensions"):
        g.parts.register("col", label="dup")


def test_register_by_label_ambiguous_resolves_with_dim(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    pt = g.model.geometry.add_point(5, 5, 5)
    g.labels.add(3, [box], "dup")
    with pytest.warns(UserWarning):
        g.labels.add(0, [pt], "dup")

    inst = g.parts.register("col", label="dup", dim=3)
    assert inst.entities[3] == [box]
    assert 0 not in inst.entities


# ---------------------------------------------------------------------------
# pg=
# ---------------------------------------------------------------------------

def test_register_by_pg(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.physical.add(3, [box], name="Col_A")

    inst = g.parts.register("col", pg="Col_A")
    assert inst.entities[3] == [box]


# ---------------------------------------------------------------------------
# Mutual-exclusion
# ---------------------------------------------------------------------------

def test_register_no_source_raises(g):
    with pytest.raises(TypeError, match="exactly one"):
        g.parts.register("col")


def test_register_dimtags_and_label_raises(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [box], "shaft")
    with pytest.raises(TypeError, match="exactly one"):
        g.parts.register("col", dimtags=[(3, box)], label="shaft")


def test_register_dimtags_and_pg_raises(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.physical.add(3, [box], name="Col_A")
    with pytest.raises(TypeError, match="exactly one"):
        g.parts.register("col", dimtags=[(3, box)], pg="Col_A")


def test_register_label_and_pg_raises(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [box], "shaft")
    g.physical.add(3, [box], name="Col_A")
    with pytest.raises(TypeError, match="exactly one"):
        g.parts.register("col", label="shaft", pg="Col_A")


# ---------------------------------------------------------------------------
# Ownership collision
# ---------------------------------------------------------------------------

def test_ownership_collision_via_label(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [box], "shaft")

    g.parts.register("first", [(3, box)])
    with pytest.raises(ValueError, match="already belongs"):
        g.parts.register("second", label="shaft")


def test_ownership_collision_via_pg(g):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.physical.add(3, [box], name="Col_A")

    g.parts.register("first", [(3, box)])
    with pytest.raises(ValueError, match="already belongs"):
        g.parts.register("second", pg="Col_A")
