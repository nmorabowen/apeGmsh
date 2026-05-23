"""PR2 / ADR 0027 — Partitions section on the mesh-viewer outline tree.

Same pattern as ``test_mesh_outline_tree``: stub the
``MeshSceneData`` + ``VisibilityManager`` + ``ViewerData`` surfaces the
outline reads, exercise partition row creation, ``_resolve_dts`` for
the new ``"partition"`` kind, the eye-toggle path, and the absent-view
/ no-partitions degradation.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# Stubs
# =====================================================================


def _make_scene_two_ranks():
    """Three BRep entities:

    - (2, 1): elements [10, 11]  (both on rank 0)
    - (2, 2): elements [20, 21]  (both on rank 1)
    - (2, 3): elements [30, 31, 32]  (rank 1 dominant: 2 vs 1)
    """
    return SimpleNamespace(
        group_to_breps={},
        brep_dominant_type={},
        brep_to_elems={
            (2, 1): [10, 11],
            (2, 2): [20, 21],
            (2, 3): [30, 31, 32],
        },
    )


def _make_scene_single_entity_rank_0():
    return SimpleNamespace(
        group_to_breps={},
        brep_dominant_type={},
        brep_to_elems={(2, 1): [10, 11]},
    )


class _StubSelection:
    def __init__(self) -> None:
        self.active_group: str | None = None
        self.picks: list = []


class _StubVisManager:
    def __init__(self) -> None:
        self.hidden: set = set()
        self.on_changed: list = []

    def is_hidden(self, dt) -> bool:
        return tuple(dt) in self.hidden

    def set_hidden(self, dts) -> None:
        self.hidden = {tuple(dt) for dt in dts}
        for cb in self.on_changed:
            cb()

    def isolate_dts(self, dts) -> None:
        pass

    def reveal_all(self) -> None:
        self.set_hidden(set())


def _make_view(*, partition_by_eid):
    """Real :class:`ViewerData`-ish stub carrying only ``elements``."""
    from apeGmsh.viewers.data._elements import (
        ElementLoadView,
        SurfaceConstraintView,
        ViewerElements,
    )
    from apeGmsh.viewers.data._nodes import _NamedNodeSelection

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    elements = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        partition_by_eid=partition_by_eid,
    )
    return SimpleNamespace(elements=elements)


# =====================================================================
# Section visibility — view absent / view without partitions
# =====================================================================


def test_partitions_section_hidden_when_view_absent(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=None,
    )
    assert outline._group_partitions.isHidden() is True
    assert outline._group_partitions.childCount() == 0


def test_partitions_section_hidden_when_view_has_no_partitions(qapp):
    """A from_fem viewer or single-partition / pre-2.10.0 archive
    produces ``view.elements.has_partitions == False``."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=_make_view(partition_by_eid=None),
    )
    assert outline._group_partitions.isHidden() is True


def test_partitions_section_hidden_when_no_brep_carries_a_rank(qapp):
    """View has partitions but none of the scene's elements appear in
    the partition map (e.g. all elements were emitted outside any
    partition bracket and have partition_for == None)."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    # Scene has elements 10/11/... but partition map keys a different
    # element (99); no eid in any brep maps to a rank.
    view = _make_view(partition_by_eid={99: 0})
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
    )
    assert outline._group_partitions.isHidden() is True


# =====================================================================
# Row creation — dominant-rank reduction
# =====================================================================


def test_partitions_section_one_row_per_rank(qapp):
    """Two ranks (0, 1) across three entities → two outline rows.

    Entity (2,3) has rank 1 dominant (2 vs 1) and goes to rank 1's
    row; entity (2,1) is all rank 0; entity (2,2) is all rank 1.
    """
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND, _ROLE_PAYLOAD,
    )
    partition_by_eid = {
        10: 0, 11: 0,         # entity (2, 1) -> rank 0
        20: 1, 21: 1,         # entity (2, 2) -> rank 1
        30: 1, 31: 1, 32: 0,  # entity (2, 3) -> rank 1 dominant (2 vs 1)
    }
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    assert outline._group_partitions.isHidden() is False
    assert outline._group_partitions.childCount() == 2
    rank_rows = [
        outline._group_partitions.child(i)
        for i in range(outline._group_partitions.childCount())
    ]
    payloads = [r.data(0, _ROLE_PAYLOAD) for r in rank_rows]
    kinds = [r.data(0, _ROLE_KIND) for r in rank_rows]
    assert kinds == ["partition", "partition"]
    # Sorted by rank
    assert payloads == [0, 1]
    # Rank 0 owns 1 brep (2,1); rank 1 owns 2 breps (2,2) + (2,3)
    assert outline._rank_to_brep[0] == [(2, 1)]
    assert sorted(outline._rank_to_brep[1]) == [(2, 2), (2, 3)]


def test_partitions_row_element_count_matches_brep_sum(qapp):
    """Each rank row's ``Detail`` text shows the total element count
    across all BReps assigned to that rank."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    partition_by_eid = {
        10: 0, 11: 0,         # (2,1) -> rank 0, 2 elems
        20: 1, 21: 1,         # (2,2) -> rank 1, 2 elems
        30: 1, 31: 1, 32: 0,  # (2,3) -> rank 1 dominant, 3 elems
    }
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    rows = {
        outline._group_partitions.child(i).data(0, 0x0211):
        outline._group_partitions.child(i).text(1)
        for i in range(outline._group_partitions.childCount())
    }
    # Rank 0 row: 2 elems from (2,1). Rank 1 row: 2 + 3 = 5 elems.
    assert rows == {0: "2", 1: "5"}


# =====================================================================
# _resolve_dts dispatch + eye toggle
# =====================================================================


def test_resolve_dts_returns_breps_for_partition_row(qapp):
    """The eye-toggle handler uses _resolve_dts; partition rows must
    return the rank's owning BReps so vis_mgr.set_hidden gets the
    right set."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    partition_by_eid = {10: 0, 11: 0, 20: 1, 21: 1}
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    rows = list(
        outline._group_partitions.child(i)
        for i in range(outline._group_partitions.childCount())
    )
    rank_0_row = rows[0]   # rank 0 is sorted first
    rank_1_row = rows[1]
    assert outline._resolve_dts(rank_0_row) == [(2, 1)]
    # Scene has (2,3) unranked here -> rank 1 owns only (2,2)
    assert outline._resolve_dts(rank_1_row) == [(2, 2)]


def test_eye_click_on_partition_row_hides_rank_breps(qapp):
    """Clicking the eye on Rank 0 adds rank 0's BReps to vis_mgr.hidden."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    partition_by_eid = {10: 0, 11: 0, 20: 1, 21: 1}
    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    rank_0_row = outline._group_partitions.child(0)
    assert rank_0_row.data(0, 0x0211) == 0  # _ROLE_PAYLOAD

    outline._on_eye_clicked(rank_0_row)
    # Rank 0 owns (2, 1); now hidden
    assert (2, 1) in vis_mgr.hidden
    # Rank 1's BRep is untouched
    assert (2, 2) not in vis_mgr.hidden


def test_eye_click_again_reveals_rank_breps(qapp):
    """A second click on the same row toggles the rank back to visible."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    partition_by_eid = {10: 0, 11: 0}
    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene_single_entity_rank_0(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    rank_0_row = outline._group_partitions.child(0)

    outline._on_eye_clicked(rank_0_row)
    assert (2, 1) in vis_mgr.hidden
    outline._on_eye_clicked(rank_0_row)
    assert (2, 1) not in vis_mgr.hidden


# =====================================================================
# _refresh_eye_states includes partitions
# =====================================================================


def test_programmatic_hide_refreshes_partition_eye_state(qapp):
    """A vis_mgr.set_hidden write fires on_changed → _refresh_eye_states.
    Partition rows must be updated alongside Groups / Types / Parts."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    partition_by_eid = {10: 0, 11: 0}
    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene_single_entity_rank_0(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        view=_make_view(partition_by_eid=partition_by_eid),
    )
    rank_0_row = outline._group_partitions.child(0)
    assert bool(rank_0_row.data(0, ROLE_VISIBLE)) is True

    # Programmatic hide via vis_mgr (fires on_changed -> _refresh_eye_states).
    vis_mgr.set_hidden({(2, 1)})
    assert bool(rank_0_row.data(0, ROLE_VISIBLE)) is False
