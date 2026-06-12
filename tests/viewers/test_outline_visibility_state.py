"""Plan 03 v2 — outline eye-icon state preservation.

Verifies:

* Hiding a Composition snapshots each layer's prior ``is_visible``
  onto ``comp.saved_visibility``; un-hiding restores from it.
* Layer add / remove invalidates the snapshot (drops it back to
  ``None``).
* ``OutlineTree._apply_composition_visibility`` orchestrates the
  snapshot + restore through the director's registry — verified with
  a stub director so the test stays headless.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apeGmsh.viewers.diagrams._compositions import (
    Composition,
    CompositionManager,
)


# =====================================================================
# Fake Diagram — just exposes is_visible / set_visible.
# =====================================================================


class _FakeLayer:
    def __init__(self, name: str = "L", visible: bool = True) -> None:
        self.name = name
        self.is_visible = bool(visible)

    def set_visible(self, v: bool) -> None:
        self.is_visible = bool(v)

    def __repr__(self) -> str:    # for nicer test failures
        return f"<L {self.name} vis={self.is_visible}>"


def _make_comp_with_layers(n: int = 3) -> tuple[CompositionManager, Composition,
                                                 list[_FakeLayer]]:
    cm = CompositionManager()
    comp = cm.add(name="Diagram", make_active=True)
    layers = [_FakeLayer(f"L{i}") for i in range(n)]
    for layer in layers:
        cm.add_layer(comp.id, layer)
    return cm, comp, layers


# =====================================================================
# Composition.saved_visibility — data-model contract
# =====================================================================


def test_saved_visibility_defaults_to_none():
    comp = Composition(id="c1", name="Diagram")
    assert comp.saved_visibility is None


def test_add_layer_invalidates_snapshot():
    cm, comp, _ = _make_comp_with_layers(2)
    # Simulate a snapshot already in place (e.g. composition was hidden).
    comp.saved_visibility = {object(): True}
    cm.add_layer(comp.id, _FakeLayer("new"))
    assert comp.saved_visibility is None


def test_remove_layer_invalidates_snapshot():
    cm, comp, layers = _make_comp_with_layers(2)
    comp.saved_visibility = {layers[0]: True}
    cm.remove_layer(layers[0])
    assert comp.saved_visibility is None


# =====================================================================
# _apply_composition_visibility — snapshot on hide, restore on show
# =====================================================================


def _stub_outline_tree_with_registry():
    """Construct an OutlineTree-like object with the bare minimum
    state needed by ``_apply_composition_visibility``.

    We deliberately don't import OutlineTree — its __init__ requires a
    QApplication and a real director. The method we're testing reads
    only ``self._director.registry`` and treats it as a thin
    ``set_visible(layer, bool)`` shim. So we bind the method to a
    namespace object with a stub director and call it directly.
    """
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    class _NS:
        pass

    ns = _NS()
    ns._director = MagicMock()

    # Registry shim: forwards to layer.set_visible.
    def _set_visible(layer, v):
        layer.set_visible(bool(v))

    ns._director.registry.set_visible.side_effect = _set_visible

    # Bind the unbound method to our namespace object.
    ns._apply_composition_visibility = (
        OutlineTree._apply_composition_visibility.__get__(ns)
    )
    return ns


def test_hide_snapshots_prior_visibility():
    _, comp, layers = _make_comp_with_layers(3)
    # Pre-state: middle layer is already hidden, others visible.
    layers[1].is_visible = False
    ns = _stub_outline_tree_with_registry()

    ns._apply_composition_visibility(comp, False)

    assert comp.saved_visibility is not None
    assert comp.saved_visibility[layers[0]] is True
    assert comp.saved_visibility[layers[1]] is False
    assert comp.saved_visibility[layers[2]] is True
    # All layers now hidden.
    assert all(not L.is_visible for L in layers)


def test_show_restores_from_snapshot():
    _, comp, layers = _make_comp_with_layers(3)
    layers[1].is_visible = False    # pre-existing hidden middle layer

    ns = _stub_outline_tree_with_registry()
    ns._apply_composition_visibility(comp, False)
    # Sanity: all hidden, snapshot taken.
    assert all(not L.is_visible for L in layers)

    ns._apply_composition_visibility(comp, True)

    assert layers[0].is_visible is True
    assert layers[1].is_visible is False    # restored to its prior state
    assert layers[2].is_visible is True
    assert comp.saved_visibility is None    # cleared


def test_show_without_snapshot_makes_everything_visible():
    """A composition that's never been hidden has no snapshot —
    showing it via the eye should treat that as "all on"."""
    _, comp, layers = _make_comp_with_layers(3)
    layers[0].is_visible = False    # arbitrary pre-state

    ns = _stub_outline_tree_with_registry()
    ns._apply_composition_visibility(comp, True)

    assert all(L.is_visible for L in layers)
    assert comp.saved_visibility is None


def test_hide_then_hide_does_not_overwrite_snapshot():
    """If a second hide fires before the show, the snapshot of the
    real prior state must survive (otherwise the show would restore
    'all hidden' instead of the user's last meaningful state)."""
    _, comp, layers = _make_comp_with_layers(3)
    layers[1].is_visible = False

    ns = _stub_outline_tree_with_registry()
    ns._apply_composition_visibility(comp, False)
    first_snapshot = dict(comp.saved_visibility)
    ns._apply_composition_visibility(comp, False)

    assert comp.saved_visibility == first_snapshot
    assert all(not L.is_visible for L in layers)


def test_layer_added_between_hide_and_show_defaults_visible():
    """If invalidation somehow misses a layer add between hide and
    show (defensive — the manager *does* clear the snapshot in
    add_layer), the show path should still default the unknown layer
    to visible rather than skip it."""
    _, comp, layers = _make_comp_with_layers(2)
    ns = _stub_outline_tree_with_registry()
    ns._apply_composition_visibility(comp, False)

    # Sneak a new layer in WITHOUT calling cm.add_layer (so the
    # snapshot survives). This is the only way to reach the "missing
    # key" fallback under normal use.
    sneaky = _FakeLayer("sneaky", visible=False)
    comp.layers.append(sneaky)

    ns._apply_composition_visibility(comp, True)
    assert sneaky.is_visible is True


# =====================================================================
# Geometry-level eye — ADR 0058 S2b
# =====================================================================
# The Plan 03 v2 geometry-level cascade (``_apply_geometry_visibility``
# + ``Geometry.saved_visibility``) was retired: the geometry row's eye
# now drives ``GeometryManager.set_visible`` and the composition gate
# composes ``owning_geometry.visible`` on top of layer intent. See
# tests/viewers/test_scene_instances_s2b.py for the S2b coverage.


def test_geometry_cascade_helpers_stay_deleted():
    from apeGmsh.viewers.diagrams._geometries import Geometry
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    assert not hasattr(OutlineTree, "_apply_geometry_visibility")
    assert "saved_visibility" not in {
        f.name for f in Geometry.__dataclass_fields__.values()
    }


def test_outline_geometry_eye_reads_visible_flag():
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    gm = GeometryManager()
    geom = gm.active
    assert OutlineTree._is_geometry_visible(geom) is True
    gm.set_visible(geom.id, False)
    assert OutlineTree._is_geometry_visible(geom) is False
