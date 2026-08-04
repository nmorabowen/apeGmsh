"""Plan 03 v2 — layer rows in the outline tree.

The outline now renders a fourth level (Geometry → Composition →
Layer) with an eye icon on each Layer row. Verifies:

* Layer rows are populated under their composition row, one per
  ``comp.layers`` entry, with the diagram stored on the role data.
* Clicking a Layer row fires ``on_diagram_selected(layer)``.
* The Layer-row eye toggles only that layer (sibling rows untouched)
  and clears the owning composition's saved_visibility.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _FakeLayer:
    """Minimal Diagram stand-in for outline tests."""
    def __init__(self, label: str, visible: bool = True) -> None:
        self._label = label
        self.is_visible = bool(visible)
        self.kind = "fake"

    def display_label(self) -> str:
        return self._label

    def set_visible(self, v: bool) -> None:
        self.is_visible = bool(v)


def _build_outline_with_layers(qapp):
    """Construct an OutlineTree wired to a real GeometryManager that
    already contains one composition with three fake layers.

    Returns ``(tree, geometries, registry, comp, layers)``.
    """
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    geometries = GeometryManager()
    geom = geometries.active

    comp = geom.compositions.add(name="Diagram", make_active=True)
    layers = [_FakeLayer(f"layer-{i}") for i in range(3)]
    for L in layers:
        geom.compositions.add_layer(comp.id, L)

    class _Registry:
        def diagrams(self_inner):
            return list(layers)

        def set_visible(self_inner, layer, v):
            layer.set_visible(v)

        def subscribe(self_inner, _cb):
            return lambda: None

    registry = _Registry()

    class _Director:
        def __init__(self_inner):
            self_inner.geometries = geometries
            self_inner.stage_id = None
            self_inner.registry = registry

        def stages(self_inner):
            return []

        def subscribe_stage(self_inner, _cb):
            return lambda: None

        def subscribe_diagrams(self_inner, _cb):
            return lambda: None

    director = _Director()
    tree = OutlineTree(director)
    return tree, geometries, registry, comp, layers


# =====================================================================
# Layer rows render under compositions
# =====================================================================


def test_layer_rows_appear_under_composition(qapp):
    tree, _, _, _, layers = _build_outline_with_layers(qapp)

    from apeGmsh.viewers.ui._outline_tree import _ROLE_DIAGRAM_OBJ

    geom_group = tree._group_diagrams
    assert geom_group.childCount() == 1
    geom_item = geom_group.child(0)
    assert geom_item.childCount() == 1
    comp_item = geom_item.child(0)
    assert comp_item.childCount() == len(layers)
    for i in range(comp_item.childCount()):
        layer_item = comp_item.child(i)
        layer = layer_item.data(0, _ROLE_DIAGRAM_OBJ)
        assert layer is layers[i]


def test_layer_row_label_uses_display_label(qapp):
    tree, _, _, _, layers = _build_outline_with_layers(qapp)
    comp_item = tree._group_diagrams.child(0).child(0)
    for i, L in enumerate(layers):
        assert comp_item.child(i).text(0) == L.display_label()


# =====================================================================
# Tri-state eye — ROLE_EFFECTIVE on layer rows
# =====================================================================


def test_layer_rows_stamp_effective_role(qapp):
    """Intent-visible but gate-hidden layers carry ROLE_EFFECTIVE=False
    and the explanatory tooltip line; effective layers carry True."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_EFFECTIVE

    tree, _, _, _, layers = _build_outline_with_layers(qapp)
    layers[1].is_effectively_visible = False    # gate-hidden
    layers[0].is_effectively_visible = True
    tree._refresh_diagrams()

    comp_item = tree._group_diagrams.child(0).child(0)
    assert comp_item.child(0).data(0, ROLE_EFFECTIVE) is True
    assert comp_item.child(1).data(0, ROLE_EFFECTIVE) is False
    assert "composition gate" in comp_item.child(1).toolTip(0)
    assert "composition gate" not in comp_item.child(0).toolTip(0)


def test_layer_rows_without_effective_attr_default_true(qapp):
    """Fake layers without ``is_effectively_visible`` (and any diagram
    predating the gate) read as effective — backward compatible."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_EFFECTIVE

    tree, _, _, _, _ = _build_outline_with_layers(qapp)
    comp_item = tree._group_diagrams.child(0).child(0)
    for i in range(comp_item.childCount()):
        assert comp_item.child(i).data(0, ROLE_EFFECTIVE) is True


def test_refresh_layer_visibility_roles_updates_in_place(qapp):
    """The gate-follow refresh re-stamps roles on the EXISTING items —
    no tree rebuild — after ``is_effectively_visible`` flips."""
    from apeGmsh.viewers.ui._eye_icon_delegate import ROLE_EFFECTIVE

    tree, _, _, _, layers = _build_outline_with_layers(qapp)
    comp_item = tree._group_diagrams.child(0).child(0)
    item_before = comp_item.child(2)
    assert item_before.data(0, ROLE_EFFECTIVE) is True

    layers[2].is_effectively_visible = False    # gate ran, no event
    tree._refresh_layer_visibility_roles()

    # Same QTreeWidgetItem instance, updated role + tooltip.
    assert comp_item.child(2) is item_before
    assert item_before.data(0, ROLE_EFFECTIVE) is False
    assert "composition gate" in item_before.toolTip(0)

    layers[2].is_effectively_visible = True     # recovery
    tree._refresh_layer_visibility_roles()
    assert item_before.data(0, ROLE_EFFECTIVE) is True
    assert "composition gate" not in item_before.toolTip(0)


# =====================================================================
# Layer-row click → on_diagram_selected
# =====================================================================


def test_layer_row_selection_fires_diagram_selected_callback(qapp):
    tree, _, _, _, layers = _build_outline_with_layers(qapp)
    received = []
    tree.on_diagram_selected(received.append)

    comp_item = tree._group_diagrams.child(0).child(0)
    layer_item = comp_item.child(1)    # middle layer
    tree._tree.setCurrentItem(layer_item)

    assert received[-1] is layers[1]


# =====================================================================
# Layer-row eye toggle
# =====================================================================


def test_layer_row_eye_toggles_only_that_layer(qapp):
    tree, _, _, _, layers = _build_outline_with_layers(qapp)
    # All start visible.
    assert all(L.is_visible for L in layers)

    comp_item = tree._group_diagrams.child(0).child(0)
    layer_item = comp_item.child(1)
    tree._on_eye_clicked(layer_item)

    assert layers[0].is_visible is True
    assert layers[1].is_visible is False
    assert layers[2].is_visible is True


def test_layer_row_eye_clears_composition_snapshot(qapp):
    tree, _, _, comp, layers = _build_outline_with_layers(qapp)
    # Simulate an active snapshot (as if the user had hidden the comp).
    comp.saved_visibility = {L: True for L in layers}

    comp_item = tree._group_diagrams.child(0).child(0)
    tree._on_eye_clicked(comp_item.child(0))

    assert comp.saved_visibility is None

# =====================================================================
# Adversarial-review fixes: the dispatcher wiring is real, and the
# tri-state extends to composition rows
# =====================================================================


def test_gate_event_restamps_roles_through_the_real_dispatcher(qapp):
    """The tri-state repaint must ride the actual UI-lane subscription
    — a test that calls ``_refresh_layer_visibility_roles()`` directly
    proves nothing about the kind list, lane, or flush. Real
    ``Dispatcher`` with an injected ``defer_fn`` (the
    test_ui_tree_perf idiom), a gate-shaped effective flip, then a
    drain: the roles must be stale before the drain and fresh after.
    """
    from apeGmsh.viewers.diagrams._dispatch import (
        COMP_ACTIVE_CHANGED, Dispatcher,
    )
    from apeGmsh.viewers.ui._outline_tree import (
        _ROLE_EFFECTIVE, _ROLE_VISIBLE,
    )

    tree, _geoms, _reg, _comp, layers = _build_outline_with_layers(qapp)

    deferred: list = []
    dispatcher = Dispatcher(
        director=tree._director,
        pump_step=lambda _l: None,
        pump_deform=lambda _l: None,
        # The real gate rewrites layer effective state; the stub gate
        # models exactly that side effect.
        pump_gate=lambda: [
            setattr(L, "is_effectively_visible", False) for L in layers
        ],
        pump_restack=lambda: None,
        render=lambda: None,
        defer_fn=deferred.append,
    )
    tree.attach_dispatcher(dispatcher)

    comp_item = tree._group_diagrams.child(0).child(0)
    row = comp_item.child(0)
    assert row.data(0, _ROLE_VISIBLE) is True
    assert row.data(0, _ROLE_EFFECTIVE) in (True, None)

    dispatcher.fire(COMP_ACTIVE_CHANGED)   # runs the gate primitive

    # Before the UI-lane drain the rows are stale…
    assert row.data(0, _ROLE_EFFECTIVE) in (True, None)
    for fn in deferred:
        fn()
    # …after it, every layer row wears the gate's verdict — and so
    # does the composition row above them (union over effective).
    assert row.data(0, _ROLE_EFFECTIVE) is False
    assert row.data(0, _ROLE_VISIBLE) is True
    assert comp_item.data(0, _ROLE_EFFECTIVE) is False
    assert comp_item.data(0, _ROLE_VISIBLE) is True


def test_composition_row_dims_with_its_layers(qapp):
    """An inactive composition's row must not keep a solid 'on' dot
    while every child under it is dimmed — the parent is the more
    prominent claim (review finding)."""
    from apeGmsh.viewers.ui._outline_tree import (
        _ROLE_EFFECTIVE, _ROLE_VISIBLE,
    )

    tree, _geoms, _reg, comp, layers = _build_outline_with_layers(qapp)
    for L in layers:
        L.is_effectively_visible = False
    tree._refresh_layer_visibility_roles()

    comp_item = tree._group_diagrams.child(0).child(0)
    assert comp_item.data(0, _ROLE_VISIBLE) is True
    assert comp_item.data(0, _ROLE_EFFECTIVE) is False
    assert "composition gate" in comp_item.toolTip(0)

    # One layer effectively back on → the union reads on again.
    layers[1].is_effectively_visible = True
    tree._refresh_layer_visibility_roles()
    assert comp_item.data(0, _ROLE_EFFECTIVE) is True
