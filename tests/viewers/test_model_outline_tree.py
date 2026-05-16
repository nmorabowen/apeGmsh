"""``ModelOutlineTree`` — left-rail outline for model.viewer.

The tree's data path is the part we can exercise headlessly:
* group / part rows aggregate their entity members' visibility
  (union semantics — visible iff any member is visible);
* clicking the eye on a group flips every member;
* clicking the eye on a single entity toggles just that one;
* subscribing to ``vis_mgr.on_changed`` keeps eyes in sync after
  programmatic visibility changes (Hide / Isolate / Reveal-all).

Full Gmsh + plotter wiring is too heavy for tests — we use stubs
that mimic the ``SelectionState`` / ``VisibilityManager`` /
``PartsRegistry`` surfaces the outline reads.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# =====================================================================
# Stubs — selection / vis_mgr / parts
# =====================================================================


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
        keep = {tuple(dt) for dt in dts}
        # Smaller stub: pretend we know about exactly these DimTags.
        self.set_hidden(set())    # no-op for tests using this stub

    def reveal_all(self) -> None:
        self.set_hidden(set())


class _StubPartInstance:
    def __init__(self, entities) -> None:
        self.entities = entities


class _StubPartsRegistry:
    def __init__(self, instances) -> None:
        self.instances = instances


# =====================================================================
# Gmsh stub — return canned physical groups
# =====================================================================


@pytest.fixture
def gmsh_two_groups():
    """Patch ``gmsh.model.getPhysicalGroups`` / etc. with canned data.

    Two groups: ``"Body"`` (surfaces 1, 2) and ``"Pinned"`` (point 5).
    """
    with patch("apeGmsh.viewers.ui._model_outline_tree.gmsh") as m:
        m.model.getPhysicalGroups.return_value = [(2, 1), (0, 2)]
        def _name(dim, tag):
            return {(2, 1): "Body", (0, 2): "Pinned"}[(dim, tag)]
        m.model.getPhysicalName.side_effect = _name
        def _ents(dim, tag):
            if (dim, tag) == (2, 1):
                return [1, 2]
            if (dim, tag) == (0, 2):
                return [5]
            return []
        m.model.getEntitiesForPhysicalGroup.side_effect = _ents
        yield m


# =====================================================================
# Construction + initial population
# =====================================================================


def test_outline_renders_two_groups(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._model_outline_tree import (
        ModelOutlineTree,
        _ROLE_KIND,
        _ROLE_PAYLOAD,
    )
    sel = _StubSelection()
    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(selection=sel, vis_mgr=vis_mgr)

    groups_header = outline._group_groups
    # Names appear in tag order (Body has tag 1, Pinned has tag 2 from gmsh).
    assert groups_header.childCount() == 2
    names = [
        groups_header.child(i).data(0, _ROLE_PAYLOAD)
        for i in range(groups_header.childCount())
    ]
    assert names == ["Body", "Pinned"]


def test_outline_hides_parts_section_when_no_registry(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree
    sel = _StubSelection()
    outline = ModelOutlineTree(
        selection=sel, vis_mgr=_StubVisManager(), parts_registry=None,
    )
    assert outline._group_parts.isHidden() is True


def test_outline_shows_parts_when_registry_has_instances(
    qapp, gmsh_two_groups,
):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree
    parts = _StubPartsRegistry(
        instances={"BeamA": _StubPartInstance({3: [10, 11]})},
    )
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        parts_registry=parts,
    )
    assert outline._group_parts.isHidden() is False
    assert outline._group_parts.childCount() == 1


# =====================================================================
# Eye-icon visibility — group / part / entity branches
# =====================================================================


def test_group_eye_click_hides_all_members(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=vis_mgr,
    )
    body_item = outline._group_groups.child(0)    # "Body"
    assert body_item.data(0, ROLE_VISIBLE) is True

    outline._on_eye_clicked(body_item)
    # Both surface DimTags now in hidden set.
    assert (2, 1) in vis_mgr.hidden
    assert (2, 2) in vis_mgr.hidden
    # Refresh propagates → group eye flips to off.
    assert body_item.data(0, ROLE_VISIBLE) is False


def test_group_eye_click_unhides_all_members(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    vis_mgr = _StubVisManager()
    # Pre-state: both Body members hidden.
    vis_mgr.hidden = {(2, 1), (2, 2)}
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=vis_mgr,
    )
    body_item = outline._group_groups.child(0)
    assert body_item.data(0, ROLE_VISIBLE) is False

    outline._on_eye_clicked(body_item)
    assert (2, 1) not in vis_mgr.hidden
    assert (2, 2) not in vis_mgr.hidden
    assert body_item.data(0, ROLE_VISIBLE) is True


def test_entity_eye_click_toggles_only_that_dimtag(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=vis_mgr,
    )
    body_item = outline._group_groups.child(0)
    entity_item = body_item.child(0)    # surf 1
    assert entity_item.data(0, ROLE_VISIBLE) is True

    outline._on_eye_clicked(entity_item)
    assert (2, 1) in vis_mgr.hidden
    assert (2, 2) not in vis_mgr.hidden    # sibling untouched
    # Group eye flips to "any visible" → still on (surf 2 visible).
    assert body_item.data(0, ROLE_VISIBLE) is True


def test_part_eye_click_hides_all_part_entities(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    parts = _StubPartsRegistry(
        instances={"BeamA": _StubPartInstance({3: [10, 11]})},
    )
    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        parts_registry=parts,
    )
    part_item = outline._group_parts.child(0)
    outline._on_eye_clicked(part_item)
    assert (3, 10) in vis_mgr.hidden
    assert (3, 11) in vis_mgr.hidden
    assert part_item.data(0, ROLE_VISIBLE) is False


# =====================================================================
# Sync via vis_mgr.on_changed (Hide / Isolate / Reveal-all path)
# =====================================================================


def test_outline_refreshes_when_vis_mgr_fires(qapp, gmsh_two_groups):
    """Hide via the context menu (or any external caller) fires
    ``vis_mgr.on_changed`` — the outline's subscription must repaint
    the eyes to match."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=vis_mgr,
    )
    body_item = outline._group_groups.child(0)
    assert body_item.data(0, ROLE_VISIBLE) is True

    # Programmatic hide — bypasses our eye-click path.
    vis_mgr.set_hidden({(2, 1), (2, 2)})
    # vis_mgr.on_changed fired; outline's _refresh_eye_states ran.
    assert body_item.data(0, ROLE_VISIBLE) is False


# =====================================================================
# Click → callback wiring
# =====================================================================


def test_group_click_fires_on_group_activated(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    received: list[str] = []
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_group_activated=received.append,
    )
    body_item = outline._group_groups.child(0)
    outline._on_item_clicked(body_item, 0)
    assert received == ["Body"]


def test_entity_click_fires_on_entity_toggled(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    received = []
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_entity_toggled=received.append,
    )
    body_item = outline._group_groups.child(0)
    entity_item = body_item.child(0)
    outline._on_item_clicked(entity_item, 0)
    assert received == [(2, 1)]
