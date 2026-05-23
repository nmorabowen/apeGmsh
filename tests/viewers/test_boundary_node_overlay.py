"""PR3 / ADR 0027 — Boundary-node overlay toggle plumbing.

Two surfaces:

* :class:`OverlayVisibilityModel.boundary_nodes_visible` — 5th field
  added in PR3.  Tested as plain Python (no Qt) for the same reasons
  the existing tests cover the original three fields.
* :class:`MeshOutlineTree` — adds a single "Boundary nodes" row
  underneath the per-rank rows inside the Partitions section,
  conditional on ``view.nodes.has_boundary_nodes``.

The actual glyph render path (``MeshViewer._rebuild_boundary_node_overlay``)
requires a real plotter — no GPU in this CI environment per memory
``feedback_viewer_no_gpu`` — so the test only pins the state plumbing
that drives it.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


# =====================================================================
# OverlayVisibilityModel — 5th field
# =====================================================================


def test_fresh_model_has_boundary_nodes_off() -> None:
    """The new field defaults to False (overlay hidden)."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    m = OverlayVisibilityModel()
    assert m.boundary_nodes_visible is False


def test_set_boundary_nodes_visible_updates_and_fires() -> None:
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    m = OverlayVisibilityModel()
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_boundary_nodes_visible(True)

    assert m.boundary_nodes_visible is True
    assert len(calls) == 1


def test_set_boundary_nodes_visible_is_idempotent() -> None:
    """Same-value write is a no-op — no observer fire.  Matches the
    oscillation-fix property of the other setters."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    m = OverlayVisibilityModel()
    m.set_boundary_nodes_visible(True)
    calls: list[None] = []
    m.subscribe(lambda: calls.append(None))

    m.set_boundary_nodes_visible(True)  # already True

    assert calls == []


def test_set_boundary_nodes_visible_coerces_to_bool() -> None:
    """Truthy / falsy values pass through ``bool()``."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    m = OverlayVisibilityModel()
    m.set_boundary_nodes_visible(1)
    assert m.boundary_nodes_visible is True
    m.set_boundary_nodes_visible(0)
    assert m.boundary_nodes_visible is False


# =====================================================================
# Outline integration — boundary nodes row
# =====================================================================

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_scene_two_ranks():
    return SimpleNamespace(
        group_to_breps={},
        brep_dominant_type={},
        brep_to_elems={
            (2, 1): [10, 11],
            (2, 2): [20, 21],
        },
    )


class _StubSelection:
    active_group = None
    picks: list = []


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


def _make_view(*, partition_by_eid, boundary_node_ids):
    """ViewerData-ish stub carrying ``elements`` + ``nodes`` minimally."""
    from apeGmsh.viewers.data._elements import (
        ElementLoadView,
        SurfaceConstraintView,
        ViewerElements,
    )
    from apeGmsh.viewers.data._nodes import (
        MassView,
        NodalLoadView,
        NodeConstraintView,
        SPView,
        ViewerNodes,
        _NamedNodeSelection,
    )

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    elements = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        partition_by_eid=partition_by_eid,
    )
    nodes = ViewerNodes(
        ids=np.array([], dtype=np.int64),
        coords=np.zeros((0, 3), dtype=np.float64),
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=NodalLoadView([]), sp=SPView([]),
        masses=MassView([]), constraints=NodeConstraintView([]),
        boundary_node_ids=boundary_node_ids,
    )
    return SimpleNamespace(elements=elements, nodes=nodes)


def test_outline_omits_boundary_row_when_no_boundary_nodes(qapp):
    """A view with rank labelling but empty boundary set (single-rank
    extension or non-shared partitions) shows only rank rows."""
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND,
    )
    view = _make_view(
        partition_by_eid={10: 0, 11: 0, 20: 1, 21: 1},
        boundary_node_ids=None,
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
    )
    kinds = [
        outline._group_partitions.child(i).data(0, _ROLE_KIND)
        for i in range(outline._group_partitions.childCount())
    ]
    assert "boundary_nodes" not in kinds
    assert kinds.count("partition") == 2


def test_outline_adds_boundary_row_when_boundary_nodes_present(qapp):
    """View with boundary nodes -> 2 partition rows + 1 boundary row."""
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND,
    )
    view = _make_view(
        partition_by_eid={10: 0, 11: 0, 20: 1, 21: 1},
        boundary_node_ids=frozenset({2, 5}),
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
    )
    assert outline._group_partitions.isHidden() is False
    assert outline._group_partitions.childCount() == 3  # 2 ranks + boundary
    kinds = [
        outline._group_partitions.child(i).data(0, _ROLE_KIND)
        for i in range(outline._group_partitions.childCount())
    ]
    # Boundary row appears AFTER the rank rows.
    assert kinds == ["partition", "partition", "boundary_nodes"]


def test_outline_section_visible_when_only_boundary_nodes_present(qapp):
    """Pathological-but-possible: a view carrying boundary nodes but
    no per-entity rank labelling (e.g. the partition emit happened
    outside any bracket).  Section still appears, with just the
    boundary row."""
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND,
    )
    view = _make_view(
        partition_by_eid=None,  # has_partitions == False
        boundary_node_ids=frozenset({2, 5}),
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
    )
    assert outline._group_partitions.isHidden() is False
    assert outline._group_partitions.childCount() == 1
    row = outline._group_partitions.child(0)
    assert row.data(0, _ROLE_KIND) == "boundary_nodes"


def test_outline_boundary_row_element_count_matches_set_size(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND,
    )
    view = _make_view(
        partition_by_eid={10: 0, 11: 0},
        boundary_node_ids=frozenset({2, 5, 7}),
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
    )
    boundary_row = next(
        outline._group_partitions.child(i)
        for i in range(outline._group_partitions.childCount())
        if outline._group_partitions.child(i).data(0, _ROLE_KIND)
        == "boundary_nodes"
    )
    assert boundary_row.text(0) == "Boundary nodes"
    assert boundary_row.text(1) == "3"


def test_outline_boundary_eye_click_fires_callback(qapp):
    """Clicking the boundary row's eye flips its ROLE_VISIBLE in place
    and invokes ``on_boundary_nodes_changed`` with the new value."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[bool] = []
    view = _make_view(
        partition_by_eid={10: 0, 11: 0},
        boundary_node_ids=frozenset({2}),
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
        on_boundary_nodes_changed=captured.append,
    )
    boundary_row = next(
        outline._group_partitions.child(i)
        for i in range(outline._group_partitions.childCount())
        if outline._group_partitions.child(i).data(0, 0x0210)
        == "boundary_nodes"
    )
    # Initial state: hidden (default).
    assert bool(boundary_row.data(0, ROLE_VISIBLE)) is False

    outline._on_eye_clicked(boundary_row)
    assert bool(boundary_row.data(0, ROLE_VISIBLE)) is True
    assert captured == [True]

    outline._on_eye_clicked(boundary_row)
    assert bool(boundary_row.data(0, ROLE_VISIBLE)) is False
    assert captured == [True, False]


def test_outline_boundary_row_syncs_from_overlay_model(qapp):
    """A tab-side write to ``model.set_boundary_nodes_visible(True)``
    propagates to the outline's eye icon via ``_sync_from_overlay_model``."""
    from apeGmsh.viewers.core.overlay_visibility import OverlayVisibilityModel
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    model = OverlayVisibilityModel()
    view = _make_view(
        partition_by_eid={10: 0, 11: 0},
        boundary_node_ids=frozenset({2}),
    )
    outline = MeshOutlineTree(
        scene=_make_scene_two_ranks(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        view=view,
        overlay_model=model,
    )
    boundary_row = next(
        outline._group_partitions.child(i)
        for i in range(outline._group_partitions.childCount())
        if outline._group_partitions.child(i).data(0, 0x0210)
        == "boundary_nodes"
    )
    assert bool(boundary_row.data(0, ROLE_VISIBLE)) is False

    model.set_boundary_nodes_visible(True)  # tab-side write
    assert bool(boundary_row.data(0, ROLE_VISIBLE)) is True
