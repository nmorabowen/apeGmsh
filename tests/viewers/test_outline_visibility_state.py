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
# Geometry-level cascade — restores only previously-visible compositions
# =====================================================================


def _make_geometry_with_two_comps():
    """Build a GeometryManager, return (geom, [c1, c2], [c1_layers, c2_layers]).

    c1 has 2 layers, c2 has 1 layer. Bootstrap geometry is used; we
    add 2 fresh compositions inside it.
    """
    from apeGmsh.viewers.diagrams._geometries import GeometryManager
    gm = GeometryManager()
    geom = gm.active    # bootstrap "Geometry 1"
    c1 = geom.compositions.add(name="C1")
    c2 = geom.compositions.add(name="C2")
    l1a = _FakeLayer("l1a")
    l1b = _FakeLayer("l1b")
    l2a = _FakeLayer("l2a")
    geom.compositions.add_layer(c1.id, l1a)
    geom.compositions.add_layer(c1.id, l1b)
    geom.compositions.add_layer(c2.id, l2a)
    return geom, [c1, c2], [[l1a, l1b], [l2a]]


def test_geometry_hide_snapshots_per_composition_visibility():
    geom, comps, comp_layers = _make_geometry_with_two_comps()
    # Pre-state: c2 has been "manually" hidden (all its layers off).
    for L in comp_layers[1]:
        L.is_visible = False

    ns = _stub_outline_tree_with_registry()
    # _is_composition_visible is a staticmethod on OutlineTree; bind it.
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    ns._is_composition_visible = OutlineTree._is_composition_visible
    ns._apply_geometry_visibility = (
        OutlineTree._apply_geometry_visibility.__get__(ns)
    )

    ns._apply_geometry_visibility(geom, False)

    assert geom.saved_visibility is not None
    assert geom.saved_visibility[comps[0].id] is True
    assert geom.saved_visibility[comps[1].id] is False
    # All layers now hidden.
    assert all(not L.is_visible for c in comp_layers for L in c)
    # Each composition snapshotted its layer states.
    assert comps[0].saved_visibility is not None
    assert comps[1].saved_visibility is not None


def test_geometry_show_restores_only_previously_visible():
    geom, comps, comp_layers = _make_geometry_with_two_comps()
    # Pre-state: c2 hidden via its layers.
    for L in comp_layers[1]:
        L.is_visible = False

    ns = _stub_outline_tree_with_registry()
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    ns._is_composition_visible = OutlineTree._is_composition_visible
    ns._apply_geometry_visibility = (
        OutlineTree._apply_geometry_visibility.__get__(ns)
    )

    ns._apply_geometry_visibility(geom, False)
    ns._apply_geometry_visibility(geom, True)

    # c1 was visible → restored. c2 was hidden → stays hidden.
    assert all(L.is_visible for L in comp_layers[0])
    assert all(not L.is_visible for L in comp_layers[1])
    assert geom.saved_visibility is None


def test_geometry_show_without_snapshot_makes_everything_visible():
    geom, _, comp_layers = _make_geometry_with_two_comps()
    # Manually hide one layer.
    comp_layers[0][0].is_visible = False

    ns = _stub_outline_tree_with_registry()
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    ns._is_composition_visible = OutlineTree._is_composition_visible
    ns._apply_geometry_visibility = (
        OutlineTree._apply_geometry_visibility.__get__(ns)
    )

    ns._apply_geometry_visibility(geom, True)

    assert all(L.is_visible for c in comp_layers for L in c)
    assert geom.saved_visibility is None


def test_geometry_show_treats_unknown_composition_as_visible():
    """A composition added while the geometry was hidden is absent
    from the snapshot; the restore should default it to shown rather
    than silently leave it hidden."""
    geom, _, _ = _make_geometry_with_two_comps()

    ns = _stub_outline_tree_with_registry()
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    ns._is_composition_visible = OutlineTree._is_composition_visible
    ns._apply_geometry_visibility = (
        OutlineTree._apply_geometry_visibility.__get__(ns)
    )

    ns._apply_geometry_visibility(geom, False)

    # Sneak in a new composition with a layer (caller didn't go
    # through compositions.add since that would re-trigger a notify
    # but not the geometry's invalidation — we want to test the
    # show-path resilience explicitly).
    c3 = geom.compositions.add(name="C3")
    l3 = _FakeLayer("l3")
    geom.compositions.add_layer(c3.id, l3)
    # The new layer is visible by default; hide it to match what the
    # geometry-hide should logically have done if it knew.
    l3.is_visible = False

    ns._apply_geometry_visibility(geom, True)

    # c3 wasn't in the snapshot — show path defaults it visible.
    assert l3.is_visible is True


def test_geometry_show_skips_stale_composition_ids():
    """A composition removed while the geometry was hidden leaves a
    stale key in the snapshot; restore should silently skip it."""
    geom, comps, comp_layers = _make_geometry_with_two_comps()

    ns = _stub_outline_tree_with_registry()
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    ns._is_composition_visible = OutlineTree._is_composition_visible
    ns._apply_geometry_visibility = (
        OutlineTree._apply_geometry_visibility.__get__(ns)
    )

    ns._apply_geometry_visibility(geom, False)
    # Drop one composition. (In real usage the caller would also tear
    # down its layers via the registry; here the snapshot just sees a
    # stale id.)
    geom.compositions.remove(comps[1].id)

    # Must not raise — the show path treats missing comps as no-ops.
    ns._apply_geometry_visibility(geom, True)
    # The surviving composition restored to its prior state.
    assert all(L.is_visible for L in comp_layers[0])
