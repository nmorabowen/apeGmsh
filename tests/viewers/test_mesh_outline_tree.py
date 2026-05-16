"""``MeshOutlineTree`` — left-rail outline for mesh.viewer.

Same pattern as ``test_model_outline_tree``: stub the
``MeshSceneData`` + ``VisibilityManager`` surfaces the outline
reads, exercise group / type / part rows + eye-click toggle paths.

The outline subscribes to ``vis_mgr.on_changed`` for live sync —
covered by a programmatic-hide test.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

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


def _make_scene():
    """A minimal scene with two groups + two element types.

    Groups:
      "Body"   → [(2, 1), (2, 2)]  (12 elems total)
      "Pinned" → [(0, 5)]          ( 1 elem)

    Element types:
      surface BReps (2, 1) and (2, 2) → "Quadrilaterals"
      point BRep    (0, 5)            → "Vertices"
    """
    return SimpleNamespace(
        group_to_breps={
            "Body":   [(2, 1), (2, 2)],
            "Pinned": [(0, 5)],
        },
        brep_dominant_type={
            (2, 1): "Quadrilaterals",
            (2, 2): "Quadrilaterals",
            (0, 5): "Vertices",
        },
        brep_to_elems={
            (2, 1): list(range(5)),     # 5 elems
            (2, 2): list(range(7)),     # 7 elems
            (0, 5): [0],                # 1 elem
        },
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
        pass    # not exercised in these tests

    def reveal_all(self) -> None:
        self.set_hidden(set())


class _StubPartInstance:
    def __init__(self, entities) -> None:
        self.entities = entities


class _StubPartsRegistry:
    def __init__(self, instances) -> None:
        self.instances = instances


# =====================================================================
# Construction + sectioning
# =====================================================================


def test_outline_renders_groups_and_types(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import (
        MeshOutlineTree, _ROLE_KIND, _ROLE_PAYLOAD,
    )
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
    )
    # Groups header has 2 children (Body, Pinned), Types header has 2
    # (Quadrilaterals, Vertices) — sorted alphabetically.
    assert outline._group_groups.childCount() == 2
    g_names = [
        outline._group_groups.child(i).data(0, _ROLE_PAYLOAD)
        for i in range(2)
    ]
    assert sorted(g_names) == ["Body", "Pinned"]

    assert outline._group_types.childCount() == 2
    t_names = [
        outline._group_types.child(i).data(0, _ROLE_PAYLOAD)
        for i in range(2)
    ]
    assert sorted(t_names) == ["Quadrilaterals", "Vertices"]


def test_outline_hides_parts_section_when_no_registry(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        parts_registry=None,
    )
    assert outline._group_parts.isHidden() is True


def test_outline_shows_parts_when_registry_has_instances(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    parts = _StubPartsRegistry(
        instances={"BeamA": _StubPartInstance({2: [1, 2]})},
    )
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        parts_registry=parts,
    )
    assert outline._group_parts.isHidden() is False
    assert outline._group_parts.childCount() == 1


def test_outline_hides_sections_on_empty_scene(qapp):
    """An empty scene shouldn't surface empty placeholder rows."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree
    empty = SimpleNamespace(
        group_to_breps={}, brep_dominant_type={}, brep_to_elems={},
    )
    outline = MeshOutlineTree(
        scene=empty,
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
    )
    assert outline._group_groups.isHidden() is True
    assert outline._group_types.isHidden() is True
    assert outline._group_parts.isHidden() is True


# =====================================================================
# Eye-icon toggle — group, type, part branches
# =====================================================================


def test_group_eye_click_hides_all_brep_members(qapp):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
    )
    # Find the "Body" row by payload.
    body_item = None
    for i in range(outline._group_groups.childCount()):
        c = outline._group_groups.child(i)
        if c.text(0) == "Body":
            body_item = c
            break
    assert body_item is not None

    outline._on_eye_clicked(body_item)
    assert (2, 1) in vis_mgr.hidden
    assert (2, 2) in vis_mgr.hidden
    assert (0, 5) not in vis_mgr.hidden    # Pinned untouched
    assert body_item.data(0, ROLE_VISIBLE) is False


def test_type_eye_click_hides_all_breps_of_that_type(qapp):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
    )
    quad_item = None
    for i in range(outline._group_types.childCount()):
        c = outline._group_types.child(i)
        if c.text(0) == "Quadrilaterals":
            quad_item = c
            break
    assert quad_item is not None

    outline._on_eye_clicked(quad_item)
    assert (2, 1) in vis_mgr.hidden
    assert (2, 2) in vis_mgr.hidden
    assert (0, 5) not in vis_mgr.hidden
    assert quad_item.data(0, ROLE_VISIBLE) is False


def test_part_eye_click_hides_all_part_entities(qapp):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    parts = _StubPartsRegistry(
        instances={"BeamA": _StubPartInstance({3: [10, 11]})},
    )
    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene(),
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
# Cross-section consistency — hiding a "type" updates "group" eyes
# =====================================================================


def test_hiding_quad_type_flips_body_group_eye(qapp):
    """Body group contains both Quad BReps. Hiding Quad type via the
    Element Types section should toggle Body's eye too because all
    its members are now hidden."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
    )
    # Initial: Body visible.
    body_item = None
    for i in range(outline._group_groups.childCount()):
        c = outline._group_groups.child(i)
        if c.text(0) == "Body":
            body_item = c
            break
    assert body_item.data(0, ROLE_VISIBLE) is True

    # Hide Quad type — vis_mgr.set_hidden fires on_changed → outline
    # repaints. Body's eye should now reflect "all members hidden".
    quad_item = None
    for i in range(outline._group_types.childCount()):
        c = outline._group_types.child(i)
        if c.text(0) == "Quadrilaterals":
            quad_item = c
            break
    outline._on_eye_clicked(quad_item)

    assert body_item.data(0, ROLE_VISIBLE) is False


# =====================================================================
# Click → on_group_activated callback
# =====================================================================


def test_group_click_fires_on_group_activated(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    received: list[str] = []
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_group_activated=received.append,
    )
    body_item = None
    for i in range(outline._group_groups.childCount()):
        c = outline._group_groups.child(i)
        if c.text(0) == "Body":
            body_item = c
            break
    outline._on_item_clicked(body_item, 0)
    assert received == ["Body"]


def test_type_click_does_not_fire_group_activated(qapp):
    """Type rows aren't physical groups — click shouldn't activate."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    received: list[str] = []
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        on_group_activated=received.append,
    )
    type_item = outline._group_types.child(0)
    outline._on_item_clicked(type_item, 0)
    assert received == []


# =====================================================================
# FEM-input sections — Loads / Masses / Constraints
# =====================================================================


class _StubLoadsComposite:
    def __init__(self, patterns: list[str]) -> None:
        self._patterns = patterns

    def patterns(self) -> list[str]:
        return list(self._patterns)


class _StubMassComposite:
    """Presence-only sentinel — the outline only checks ``is not None``."""


class _StubConstraintsComposite:
    def __init__(self, kinds: list[str]) -> None:
        # Build a list of dicts that look like constraint defs to
        # ``_def_kind_key`` — only the ``kind`` attribute is checked
        # for non-RigidLink branches.
        from types import SimpleNamespace
        self.constraint_defs = [
            SimpleNamespace(kind=k) for k in kinds
        ]


def test_loads_section_renders_one_row_per_pattern(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        loads_composite=_StubLoadsComposite(["self-weight", "wind-x"]),
    )
    assert outline._group_loads.isHidden() is False
    assert outline._group_loads.childCount() == 2
    names = [
        outline._group_loads.child(i).text(0) for i in range(2)
    ]
    assert names == ["self-weight", "wind-x"]


def test_loads_section_hidden_when_no_composite(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        loads_composite=None,
    )
    assert outline._group_loads.isHidden() is True


def test_loads_section_hidden_when_no_patterns(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        loads_composite=_StubLoadsComposite([]),
    )
    assert outline._group_loads.isHidden() is True


def test_load_pattern_eye_click_fires_active_patterns(qapp):
    """Clicking a pattern row's eye toggles ROLE_VISIBLE and fires
    ``on_load_patterns_changed`` with the currently-visible set."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[set[str]] = []
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        loads_composite=_StubLoadsComposite(["a", "b"]),
        on_load_patterns_changed=captured.append,
    )
    # Hide "a" via its eye.
    a_row = outline._group_loads.child(0)
    assert a_row.data(0, ROLE_VISIBLE) is True
    outline._on_eye_clicked(a_row)
    assert a_row.data(0, ROLE_VISIBLE) is False
    assert captured[-1] == {"b"}

    # Hide "b" too → empty active set.
    b_row = outline._group_loads.child(1)
    outline._on_eye_clicked(b_row)
    assert captured[-1] == set()

    # Re-show "a".
    outline._on_eye_clicked(a_row)
    assert captured[-1] == {"a"}


def test_masses_section_renders_single_row(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        mass_composite=_StubMassComposite(),
    )
    assert outline._group_masses.isHidden() is False
    assert outline._group_masses.childCount() == 1


def test_masses_eye_click_fires_boolean(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[bool] = []
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        mass_composite=_StubMassComposite(),
        on_mass_visibility_changed=captured.append,
    )
    row = outline._group_masses.child(0)
    outline._on_eye_clicked(row)
    assert captured == [False]
    outline._on_eye_clicked(row)
    assert captured == [False, True]


def test_constraints_section_renders_one_row_per_kind(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        constraints_composite=_StubConstraintsComposite(
            ["equal_dof", "equal_dof", "tie"],
        ),
    )
    # Two distinct kinds, sorted alphabetically.
    assert outline._group_constraints.isHidden() is False
    assert outline._group_constraints.childCount() == 2
    names = [
        outline._group_constraints.child(i).text(0) for i in range(2)
    ]
    assert names == ["equal_dof", "tie"]


def test_constraint_kind_eye_click_fires_active_kinds(qapp):
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_VISIBLE
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    captured: list[set[str]] = []
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        constraints_composite=_StubConstraintsComposite(["equal_dof", "tie"]),
        on_constraint_kinds_changed=captured.append,
    )
    eq_row = outline._group_constraints.child(0)
    assert eq_row.data(0, ROLE_VISIBLE) is True
    outline._on_eye_clicked(eq_row)
    assert eq_row.data(0, ROLE_VISIBLE) is False
    assert captured[-1] == {"tie"}


def test_constraints_section_hidden_when_no_defs(qapp):
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=_StubVisManager(),
        constraints_composite=_StubConstraintsComposite([]),
    )
    assert outline._group_constraints.isHidden() is True


def test_load_eye_click_does_not_touch_vis_mgr(qapp):
    """FEM-input rows have their own visibility state — they must not
    add DimTags to the VisibilityManager (which would silently hide
    mesh BReps)."""
    from apeGmsh.viewers.ui._mesh_outline_tree import MeshOutlineTree

    vis_mgr = _StubVisManager()
    outline = MeshOutlineTree(
        scene=_make_scene(),
        selection=_StubSelection(),
        vis_mgr=vis_mgr,
        loads_composite=_StubLoadsComposite(["a"]),
        on_load_patterns_changed=lambda _s: None,
    )
    initial_hidden = set(vis_mgr.hidden)
    outline._on_eye_clicked(outline._group_loads.child(0))
    assert set(vis_mgr.hidden) == initial_hidden
