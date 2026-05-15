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
