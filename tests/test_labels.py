"""
Tests -- Labels system (geometry-time entity naming).

Covers the Labels class (g.labels composite) and the module-level
helper functions in ``apeGmsh.core.Labels``.
"""
from __future__ import annotations

import warnings

import gmsh
import pytest

from apeGmsh.core.Labels import (
    Labels,
    is_label_pg,
    strip_prefix,
    add_prefix,
    cleanup_label_pgs,
    reconcile_label_pgs,
    LABEL_PREFIX,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_labels() -> Labels:
    """Create a Labels instance with a minimal stub parent."""
    stub = type("_Stub", (), {"_verbose": False})()
    return Labels(stub)


# ==================================================================
# Module-level utility functions
# ==================================================================


def test_is_label_pg_true():
    assert is_label_pg("_label:beam") is True


def test_is_label_pg_false():
    assert is_label_pg("beam") is False
    assert is_label_pg("") is False
    assert is_label_pg("label:beam") is False


def test_strip_prefix_with_prefix():
    assert strip_prefix("_label:slab") == "slab"


def test_strip_prefix_without_prefix():
    assert strip_prefix("slab") == "slab"


def test_add_prefix_bare_name():
    assert add_prefix("wall") == "_label:wall"


def test_add_prefix_already_prefixed():
    assert add_prefix("_label:wall") == "_label:wall"


# ==================================================================
# Labels.add / Labels.entities / Labels.has / Labels.get_all
# ==================================================================


def test_add_and_entities(gmsh_session):
    """add() creates a label PG; entities() retrieves its tags."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    labels.add(3, [box], "block")
    result = labels.entities("block")
    assert result == [box]


def test_has_returns_true(gmsh_session):
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "block")

    assert labels.has("block") is True


def test_has_returns_false(gmsh_session):
    labels = _make_labels()
    assert labels.has("nonexistent") is False


def test_entities_keyerror_when_missing(gmsh_session):
    labels = _make_labels()
    with pytest.raises(KeyError, match="no label"):
        labels.entities("ghost")


def test_entities_with_dim_filter(gmsh_session):
    """entities(name, dim=D) returns tags only at that dimension."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "solid")

    result = labels.entities("solid", dim=3)
    assert result == [box]

    # Asking for the wrong dimension raises KeyError
    with pytest.raises(KeyError):
        labels.entities("solid", dim=2)


def test_get_all_empty(gmsh_session):
    labels = _make_labels()
    assert labels.get_all() == []


def test_get_all_lists_all_labels(gmsh_session):
    labels = _make_labels()
    b1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b2 = gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [b1], "alpha")
    labels.add(3, [b2], "beta")

    all_labels = labels.get_all()
    assert "alpha" in all_labels
    assert "beta" in all_labels
    assert len(all_labels) == 2


def test_get_all_with_dim_filter(gmsh_session):
    """get_all(dim=D) only returns labels at that dimension."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    # Get a surface tag from the box
    surfaces = gmsh.model.getEntities(2)
    labels.add(3, [box], "vol_label")
    labels.add(2, [surfaces[0][1]], "face_label")

    assert labels.get_all(dim=3) == ["vol_label"]
    assert labels.get_all(dim=2) == ["face_label"]


# ==================================================================
# Labels.add -- merge and cross-dim warning
# ==================================================================


def test_add_merge_same_dim_warns(gmsh_session):
    """Adding the same label name at the same dim merges tags and warns."""
    labels = _make_labels()
    b1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b2 = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [b1], "parts")

    with pytest.warns(UserWarning, match="already exists"):
        labels.add(3, [b2], "parts")

    merged = labels.entities("parts")
    assert b1 in merged
    assert b2 in merged


def test_add_cross_dim_warns(gmsh_session):
    """Creating a label that already exists at a different dim warns."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(2)
    labels.add(3, [box], "thing")

    with pytest.warns(UserWarning, match="already exists at dim="):
        labels.add(2, [surfaces[0][1]], "thing")


def test_entities_valueerror_multi_dim(gmsh_session):
    """entities() without dim= raises ValueError when label spans dims."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(2)

    labels.add(3, [box], "thing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        labels.add(2, [surfaces[0][1]], "thing")

    with pytest.raises(ValueError, match="multiple dimensions"):
        labels.entities("thing")


# ==================================================================
# Labels.remove
# ==================================================================


def test_remove_deletes_label(gmsh_session):
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "temp")

    labels.remove("temp")
    assert labels.has("temp") is False


def test_remove_keyerror_when_missing(gmsh_session):
    labels = _make_labels()
    with pytest.raises(KeyError, match="no label"):
        labels.remove("nothing")


# ==================================================================
# Labels.rename
# ==================================================================


def test_rename_preserves_entities(gmsh_session):
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "old_name")

    labels.rename("old_name", "new_name")

    assert labels.has("old_name") is False
    assert labels.has("new_name") is True
    assert labels.entities("new_name") == [box]


def test_rename_keyerror_when_missing(gmsh_session):
    labels = _make_labels()
    with pytest.raises(KeyError, match="no label"):
        labels.rename("ghost", "anything")


# ==================================================================
# Labels.reverse_map / Labels.labels_for_entity
# ==================================================================


def test_reverse_map(gmsh_session):
    labels = _make_labels()
    b1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b2 = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [b1], "alpha")
    labels.add(3, [b2], "beta")

    rmap = labels.reverse_map()
    assert rmap[(3, b1)] == "alpha"
    assert rmap[(3, b2)] == "beta"


def test_labels_for_entity(gmsh_session):
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "group_a")

    found = labels.labels_for_entity(3, box)
    assert "group_a" in found


def test_labels_for_entity_empty(gmsh_session):
    """Entity not in any label returns empty list."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    found = labels.labels_for_entity(3, box)
    assert found == []


# ==================================================================
# Labels.promote_to_physical
# ==================================================================


def test_promote_to_physical(gmsh_session):
    """promote_to_physical creates a non-label PG with the same entities."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "col")

    pg_tag = labels.promote_to_physical("col")

    # The new PG should be a non-label PG
    pg_name = gmsh.model.getPhysicalName(3, pg_tag)
    assert pg_name == "col"
    assert not is_label_pg(pg_name)

    ents = list(gmsh.model.getEntitiesForPhysicalGroup(3, pg_tag))
    assert int(ents[0]) == box


def test_promote_to_physical_custom_name(gmsh_session):
    """promote_to_physical with pg_name= uses a custom PG name."""
    labels = _make_labels()
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [box], "col")

    pg_tag = labels.promote_to_physical("col", pg_name="column_concrete")
    pg_name = gmsh.model.getPhysicalName(3, pg_tag)
    assert pg_name == "column_concrete"


# ==================================================================
# Module-level cleanup / reconcile
# ==================================================================


def test_cleanup_label_pgs_removes_dead_tags(gmsh_session):
    """cleanup_label_pgs drops tags for removed entities."""
    labels = _make_labels()
    b1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b2 = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [b1, b2], "pair")

    # Remove one box
    gmsh.model.occ.remove([(3, b2)])
    gmsh.model.occ.synchronize()

    cleanup_label_pgs([(3, b2)])
    remaining = labels.entities("pair")
    assert b1 in remaining
    assert b2 not in remaining


def test_reconcile_label_pgs_removes_stale_tags(gmsh_session):
    """reconcile_label_pgs does a full scan and drops stale entries."""
    labels = _make_labels()
    b1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b2 = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    labels.add(3, [b1, b2], "pair")

    # Remove one box without calling cleanup_label_pgs
    gmsh.model.occ.remove([(3, b2)])
    gmsh.model.occ.synchronize()

    reconcile_label_pgs()
    remaining = labels.entities("pair")
    assert b1 in remaining
    assert b2 not in remaining


# ==================================================================
# Full session (g fixture) integration
# ==================================================================


def test_g_labels_add_and_query(g):
    """Labels work through the full g.labels composite."""
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    g.labels.add(3, [box], "foundation")
    assert g.labels.has("foundation")
    assert g.labels.entities("foundation") == [box]
