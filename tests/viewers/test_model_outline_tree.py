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
    # ADR 0045 S3c: the outline reads groups from staging, not gmsh.
    # Default mirrors the ``gmsh_two_groups`` fixture (Body srf 1,2 /
    # Pinned pt 5); pass ``groups={name: [(dim, tag), ...]}`` to override.
    def __init__(self, groups: dict | None = None) -> None:
        from apeGmsh.viewers.scene_ir import SelectionTarget
        if groups is None:
            groups = {"Body": [(2, 1), (2, 2)], "Pinned": [(0, 5)]}
        self.active_group: str | None = None
        self.picks: list = []
        self._staged = {
            name: [SelectionTarget.from_dimtag(dt) for dt in dts]
            for name, dts in groups.items()
        }
        self.group_order: list = list(groups)

    @property
    def staged_groups(self) -> dict:
        return dict(self._staged)


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


@pytest.fixture
def gmsh_group_and_label():
    """One user PG ``"Body"`` (surf 1) + one internal label
    ``"_label:shaft"`` (vol 3) — exercises the Labels section."""
    with patch("apeGmsh.viewers.ui._model_outline_tree.gmsh") as m:
        m.model.getPhysicalGroups.return_value = [(2, 1), (3, 7)]
        def _name(dim, tag):
            return {(2, 1): "Body", (3, 7): "_label:shaft"}[(dim, tag)]
        m.model.getPhysicalName.side_effect = _name
        def _ents(dim, tag):
            if (dim, tag) == (2, 1):
                return [1]
            if (dim, tag) == (3, 7):
                return [3]
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


# =====================================================================
# ParaView-faithful 2-column layout
# =====================================================================


def test_tree_has_two_columns_with_eye_moved_first(qapp, gmsh_two_groups):
    """Column 0 = name, column 1 = eye; the eye section is moved to
    visual position 0 (ParaView pqPipelineModel + moveSection(1, 0))."""
    from apeGmsh.viewers.ui._model_outline_tree import (
        ModelOutlineTree,
        _EYE_COL,
    )
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=_StubVisManager(),
    )
    tree = outline._tree
    assert tree.columnCount() == 2
    # Eye is logical column 1 but rendered first (visual index 0).
    assert tree.header().visualIndex(_EYE_COL) == 0
    # Name lives in column 0; eye decoration in column 1.
    body_item = outline._group_groups.child(0)
    assert body_item.text(0) == "Body"
    assert not body_item.icon(_EYE_COL).isNull()


def test_eye_column_click_toggles_visibility_not_activation(
    qapp, gmsh_two_groups,
):
    """A click in the eye column toggles visibility only — it must
    NOT fire on_group_activated (ParaView handleIndexClicked parity)."""
    from apeGmsh.viewers.ui._model_outline_tree import (
        ModelOutlineTree,
        _EYE_COL,
    )

    activated: list[str] = []
    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        on_group_activated=activated.append,
    )
    body_item = outline._group_groups.child(0)
    outline._on_item_clicked(body_item, _EYE_COL)
    assert activated == []                       # no activation
    assert (2, 1) in vis_mgr.hidden              # visibility toggled
    assert (2, 2) in vis_mgr.hidden


# =====================================================================
# Labels group
# =====================================================================


def test_labels_section_lists_internal_labels_stripped(
    qapp, gmsh_group_and_label,
):
    from apeGmsh.viewers.ui._model_outline_tree import (
        ModelOutlineTree,
        _ROLE_KIND,
        _ROLE_PAYLOAD,
    )
    outline = ModelOutlineTree(
        selection=_StubSelection(groups={"Body": [(2, 1)]}),
        vis_mgr=_StubVisManager(),
    )
    # Physical Groups skips the _label: PG → only "Body".
    assert outline._group_groups.childCount() == 1
    assert outline._group_groups.child(0).data(0, _ROLE_PAYLOAD) == "Body"
    # Labels section is visible and carries the prefix-stripped name.
    assert outline._group_labels.isHidden() is False
    assert outline._group_labels.childCount() == 1
    label_item = outline._group_labels.child(0)
    assert label_item.data(0, _ROLE_KIND) == "label"
    assert label_item.data(0, _ROLE_PAYLOAD) == "shaft"
    assert label_item.text(0) == "shaft"


def test_labels_section_hidden_when_no_labels(qapp, gmsh_two_groups):
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=_StubVisManager(),
    )
    assert outline._group_labels.isHidden() is True


def test_label_eye_click_hides_all_label_entities(
    qapp, gmsh_group_and_label,
):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._model_outline_tree import ModelOutlineTree

    vis_mgr = _StubVisManager()
    outline = ModelOutlineTree(
        selection=_StubSelection(), vis_mgr=vis_mgr,
    )
    label_item = outline._group_labels.child(0)        # "shaft" (vol 3)
    assert label_item.data(0, ROLE_VISIBLE) is True
    outline._on_eye_clicked(label_item)
    assert (3, 3) in vis_mgr.hidden
    assert label_item.data(0, ROLE_VISIBLE) is False
