"""Tests for parts fragmentation — fragment_all, fragment_pair, fuse_group."""
import warnings

import gmsh
import pytest


def test_fragment_all_two_parts(g):
    """Two overlapping parts fragmented produces 3 volumes."""
    with g.parts.part("a"):
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
    with g.parts.part("b"):
        g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

    result = g.parts.fragment_all()
    vols = gmsh.model.getEntities(3)
    assert len(vols) == 3
    assert len(result) == 3


def test_fragment_all_updates_instance_entities(g):
    """Instance.entities is remapped in-place after fragmentation."""
    with g.parts.part("a"):
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
    with g.parts.part("b"):
        g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

    g.parts.fragment_all()

    a_tags = g.parts.get("a").entities.get(3, [])
    b_tags = g.parts.get("b").entities.get(3, [])
    # Each part should have at least 1 entity after fragment
    assert len(a_tags) >= 1
    assert len(b_tags) >= 1
    # Together they should cover all volumes (overlap counted once)
    all_tags = set(a_tags) | set(b_tags)
    all_vols = {t for _, t in gmsh.model.getEntities(3)}
    assert all_tags == all_vols


def test_fragment_all_warns_untracked(g):
    """Untracked entities produce a warning during fragmentation."""
    with g.parts.part("a"):
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
    # Create an untracked box directly
    gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)
    gmsh.model.occ.synchronize()

    with pytest.warns(UserWarning, match="not tracked"):
        g.parts.fragment_all()


def test_fragment_all_single_entity_noop(g):
    """Single entity returns unchanged."""
    with g.parts.part("only"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

    result = g.parts.fragment_all()
    assert len(result) == 1


def test_fragment_pair(g):
    """Pairwise fragmentation of two named instances."""
    with g.parts.part("left"):
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
    with g.parts.part("right"):
        g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

    result = g.parts.fragment_pair("left", "right")
    assert len(result) == 3


def test_fragment_pair_no_common_dim_raises(g):
    """No common dimension between instances raises RuntimeError."""
    with g.parts.part("vol"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    # Create a part with only a point (dim=0)
    with g.parts.part("pt"):
        g.model.geometry.add_point(5, 5, 5)

    with pytest.raises(RuntimeError, match="No common dimension"):
        g.parts.fragment_pair("vol", "pt")


def test_fuse_group_two_parts(g):
    """Fuse two instances into one new instance."""
    with g.parts.part("a"):
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
    with g.parts.part("b"):
        g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

    inst = g.parts.fuse_group(["a", "b"], label="merged")
    assert inst.label == "merged"
    assert 3 in inst.entities
    assert len(inst.entities[3]) == 1  # fuse produces 1 volume
    # Old instances removed
    assert "a" not in g.parts.labels()
    assert "b" not in g.parts.labels()
    assert "merged" in g.parts.labels()


def test_fuse_group_inherits_properties(g):
    """New instance inherits first label's properties."""
    with g.parts.part("a"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.parts.get("a").properties["material"] = "steel"

    with g.parts.part("b"):
        g.model.geometry.add_box(0.5, 0, 0, 1, 1, 1)

    inst = g.parts.fuse_group(["a", "b"])
    assert inst.properties.get("material") == "steel"


def test_fuse_group_too_few_raises(g):
    """fuse_group with fewer than 2 labels raises ValueError."""
    with g.parts.part("only"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

    with pytest.raises(ValueError, match="at least 2"):
        g.parts.fuse_group(["only"])
