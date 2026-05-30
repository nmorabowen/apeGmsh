"""
Regression tests for ModelViewer physical-group persistence.

Bugs covered:
  - empty active group not propagated (commit + flush + set_active)
  - rename leaves stale PG in gmsh
  - apply_group missing from _group_order
"""
import gmsh
import pytest

from apeGmsh.viewers.core.selection import (
    SelectionState,
    _load_group_members,
)


@pytest.fixture
def pg_gmsh_session():
    gmsh.initialize()
    gmsh.model.add("test_pg_persistence")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)   # vol 1
    gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)   # vol 2
    gmsh.model.occ.synchronize()
    yield
    gmsh.finalize()


def _pg_names() -> set[str]:
    names = set()
    for d, t in gmsh.model.getPhysicalGroups():
        try:
            names.add(gmsh.model.getPhysicalName(d, t))
        except Exception:
            pass
    return names


# ── Phase 1 ─────────────────────────────────────────────────────────

def test_empty_active_group_deleted_at_flush(pg_gmsh_session):
    # ADR 0045 S3c: writes are deferred to flush (the single freeze
    # boundary). commit_active_group only stages in memory now.
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" not in _pg_names()    # not written yet — staged only
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.flush_to_gmsh()                # empty group dropped at flush
    assert "Foo" not in _pg_names()


def test_empty_group_deleted_on_flush(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" not in _pg_names()


def test_switching_from_empty_group_deletes_it_at_flush(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.set_active_group("Bar")  # switching away stages Foo empty
    sel.flush_to_gmsh()
    assert "Foo" not in _pg_names()


# ── Phase 2 ─────────────────────────────────────────────────────────

def test_rename_removes_old_pg(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()

    sel.rename_group("Foo", "Bar")
    sel.flush_to_gmsh()                # tombstoned Foo dropped, Bar written
    names = _pg_names()
    assert "Bar" in names
    assert "Foo" not in names


def test_rename_updates_group_order(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.rename_group("Foo", "Bar")
    assert "Foo" not in sel.group_order
    assert "Bar" in sel.group_order


def test_rename_active_group_preserves_pointer(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.rename_group("Foo", "Bar")
    assert sel.active_group == "Bar"


def test_apply_group_registers_order(pg_gmsh_session):
    sel = SelectionState()
    sel.pick((3, 1))
    sel.apply_group("X")
    sel.unpick((3, 1))
    sel.pick((3, 2))
    sel.apply_group("Y")
    assert sel.group_order == ["X", "Y"]


# ── Reported repro ──────────────────────────────────────────────────

def test_reported_repro_create_then_query(pg_gmsh_session):
    """Reproduction: create a group via _on_new_group flow, query it after."""
    sel = SelectionState()
    sel.pick((3, 1))
    sel.pick((3, 2))
    # Simulates _on_new_group in model_viewer.py
    sel._staged_groups["Foo"] = list(sel._picks)
    sel.set_active_group("Foo")
    # Simulates close
    sel.flush_to_gmsh()

    assert "Foo" in _pg_names()
    members = _load_group_members("Foo")
    assert set(members) == {(3, 1), (3, 2)}


# ── ADR 0045 S3c — single freeze boundary (deferred writes) ─────────

def test_create_group_not_written_until_flush(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" not in _pg_names()          # staged, not written
    assert "Foo" in sel.staged_groups        # but present in staging
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()              # written at the freeze boundary


def test_delete_defers_gmsh_but_destages_immediately(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()

    sel.delete_group("Foo")
    # Destaged immediately (UI is staged-authoritative)...
    assert "Foo" not in sel.staged_groups
    assert "Foo" not in sel.group_order
    # ...but the gmsh PG is dropped only at the next flush.
    assert "Foo" in _pg_names()
    sel.flush_to_gmsh()
    assert "Foo" not in _pg_names()


def test_seed_from_gmsh_pulls_existing_groups_into_staging(pg_gmsh_session):
    # A pre-existing gmsh PG (not made via the viewer).
    pg = gmsh.model.addPhysicalGroup(3, [1])
    gmsh.model.setPhysicalName(3, pg, "Existing")

    sel = SelectionState()
    sel.seed_from_gmsh()
    assert "Existing" in sel.staged_groups
    assert "Existing" in sel.group_order
    assert {t.dimtag for t in sel.staged_groups["Existing"]} == {(3, 1)}


def test_rename_roundtrip_keeps_live_group(pg_gmsh_session):
    # Foo -> Bar -> Foo within one session: the original name must survive
    # the flush (the tombstone is cleared when the name is reborn).
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.flush_to_gmsh()
    sel.rename_group("Foo", "Bar")
    sel.rename_group("Bar", "Foo")
    sel.flush_to_gmsh()
    assert "Foo" in _pg_names()
