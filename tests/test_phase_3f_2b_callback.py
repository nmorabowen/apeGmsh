"""Phase 3F.2b / ADR 0038 — ColorMode "Module" idle-fn dispatch.

Phase 3F.2a plumbed ``ViewerElements.module_for(eid)`` (FEM eid ->
joined compose-module label, e.g. ``"bayP/frameA"`` for a nested
compose). Phase 3F.2b wires the per-entity dispatch on
:class:`apeGmsh.viewers.core.color_mode_controller.ColorModeController`
so the mesh viewer can color BRep entities by their dominant module.

Slice scope (3F.2b): controller side ONLY.

- ``_module_idle`` callback + ``set_mode("Module")`` dispatch.
- Palette reuses ``_GROUP_PALETTE_RGB`` (same as Partition).

Slice 3F.2c will add ``"Module"`` to ``COLOR_MODES`` in
``viewers/ui/mesh_tabs.py`` so the dropdown surfaces the new mode;
this slice does NOT touch UI.

End-to-end rendering can't be exercised in CI (no GPU) — see
:doc:`feedback_viewer_no_gpu` — so these tests exercise the callback
as a pure function with minimal mocks (mirrors the structure of
``tests/viewers/test_color_mode_partition.py``).
"""
from __future__ import annotations

import numpy as np

from apeGmsh.viewers.core.color_mode_controller import (
    _FALLBACK_RGB,
    _GROUP_PALETTE_RGB,
    ColorModeController,
)
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


# =====================================================================
# Minimal dependency stand-ins (mirrors test_color_mode_partition.py)
# =====================================================================


class _NoopColorMgr:
    def __init__(self) -> None:
        self.idle_fn = None

    def set_idle_fn(self, fn) -> None:
        self.idle_fn = fn

    def reset_idle_fn(self) -> None:
        self.idle_fn = None

    def recolor_all(self, *, picks, hidden) -> None:
        pass


class _NoopRegistry:
    dims: list[int] = []
    dim_meshes: dict = {}
    dim_actors: dict = {}


class _NoopSel:
    picks: set = set()


class _NoopVisMgr:
    hidden: set = set()


class _NoopPlotter:
    def render(self) -> None:
        pass


class _FakeScene:
    """Minimal MeshSceneData stand-in — controller only reads
    ``brep_to_elems`` on the ``_module_idle`` path."""
    def __init__(self, brep_to_elems: "dict") -> None:
        self.brep_to_elems = brep_to_elems
        self.brep_to_group: dict = {}
        self.brep_dominant_type: dict = {}
        self.quality: dict = {}
        self.batch_cell_to_elem: dict = {}


def _make_viewer_data(
    *,
    module_by_eid: "dict[int, str] | None" = None,
) -> "tuple[ViewerNodes, ViewerElements]":
    """Build a minimal pair the controller can wear."""
    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    nodes = ViewerNodes(
        ids=np.array([], dtype=np.int64),
        coords=np.zeros((0, 3), dtype=np.float64),
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=NodalLoadView([]), sp=SPView([]),
        masses=MassView([]), constraints=NodeConstraintView([]),
    )
    elements = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        module_by_eid=module_by_eid,
    )
    return nodes, elements


class _MiniView:
    """Stand-in for ViewerData carrying only ``elements``."""
    def __init__(self, elements: ViewerElements) -> None:
        self.elements = elements


def _make_controller(
    *,
    scene: _FakeScene,
    view: "_MiniView | None",
) -> ColorModeController:
    return ColorModeController(
        color_mgr=_NoopColorMgr(),
        registry=_NoopRegistry(),
        scene=scene,
        sel=_NoopSel(),
        vis_mgr=_NoopVisMgr(),
        plotter=_NoopPlotter(),
        view=view,  # type: ignore[arg-type]
    )


def _palette_color_for(label: str) -> np.ndarray:
    """Reference implementation of the label->palette mapping the
    callback uses. Anchored here so the tests fail if the controller
    drifts from the documented hash policy."""
    return _GROUP_PALETTE_RGB[abs(hash(label)) % len(_GROUP_PALETTE_RGB)]


# =====================================================================
# _module_idle — happy path
# =====================================================================


def test_module_idle_returns_palette_color_for_single_label_entity() -> None:
    """Entity (2, 10) owns elements 100 and 101, both labelled "modA".
    Dominant label = "modA"; color = palette[hash("modA") % len]."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA", 101: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("modA"))


def test_module_idle_distinct_colors_for_distinct_labels() -> None:
    """Two entities labelled differently get different palette
    colors (modulo palette wraparound — labels picked so collision
    is unlikely)."""
    scene = _FakeScene(brep_to_elems={
        (2, 10): [100, 101],
        (2, 20): [200, 201],
    })
    _, elements = _make_viewer_data(module_by_eid={
        100: "modA", 101: "modA", 200: "modB", 201: "modB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle((2, 10))
    rgb_b = ctrl._module_idle((2, 20))
    np.testing.assert_array_equal(rgb_a, _palette_color_for("modA"))
    np.testing.assert_array_equal(rgb_b, _palette_color_for("modB"))
    # Two distinct labels — extremely likely to hit different palette
    # slots given the 19-color palette. If a collision ever shows up
    # here the test still verifies determinism (above), so swap the
    # labels for non-colliding ones.
    assert not np.array_equal(rgb_a, rgb_b)


def test_module_idle_dominant_label_wins_in_mixed_entity() -> None:
    """Entity owns 3 elements: 2 labelled "modA", 1 labelled "modB".
    Dominant label = "modA"; mixed-label case degrades to dominant
    color (mirrors _partition_idle's most-common-rank reduction)."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101, 102]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "modA", 101: "modA", 102: "modB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("modA"))


def test_module_idle_same_label_produces_same_color() -> None:
    """Determinism: two separate calls with the same input label return
    the same color (no per-call randomness)."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA", 200: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_first = ctrl._module_idle((2, 10))
    rgb_second = ctrl._module_idle((2, 20))
    np.testing.assert_array_equal(rgb_first, rgb_second)


# =====================================================================
# _module_idle — joined labels (nested compose, ADR 0038)
# =====================================================================


def test_module_idle_nested_joined_label_uses_full_label() -> None:
    """For nested-compose models the label is the full joined string
    (e.g. ``"bayP/frameA"`` — see ``_compose._join_module_label``).
    The callback must NOT split on ``/`` — each unique joined label
    gets its own color."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "bayP/frameA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("bayP/frameA"))


def test_module_idle_distinct_joined_labels_typically_distinct_colors() -> None:
    """``"bayP/frameA"`` and ``"bayP/frameB"`` share the ``"bayP"``
    prefix but the callback hashes the FULL joined label — distinct
    labels typically hash to distinct palette slots (modulo
    wraparound), confirming we don't accidentally split-and-collapse
    on the separator."""
    scene = _FakeScene(brep_to_elems={
        (2, 10): [100],
        (2, 20): [200],
    })
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayP/frameB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle((2, 10))
    rgb_b = ctrl._module_idle((2, 20))
    # If split-on-"/"-and-take-root were the (buggy) policy, both
    # would resolve to palette[hash("bayP")] and be equal. We assert
    # the full-label policy: at least one of the two pairs differs.
    # (We use two distinct second-segment labels so a hash collision
    # on just one of them still leaves the asymmetry visible.)
    assert not np.array_equal(rgb_a, rgb_b)


def test_module_idle_same_joined_label_produces_same_color() -> None:
    """Determinism for joined labels — ``"bayP/frameA"`` always maps
    to the same palette slot regardless of which entity holds it."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayP/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_first = ctrl._module_idle((2, 10))
    rgb_second = ctrl._module_idle((2, 20))
    np.testing.assert_array_equal(rgb_first, rgb_second)


# =====================================================================
# _module_idle — degraded paths (mirror _partition_idle fallback shape)
# =====================================================================


def test_module_idle_no_view_returns_fallback() -> None:
    """No ViewerData bound (e.g. live from_fem with no h5 source +
    no broker enrichment) — every entity gets the fallback color.
    Matches what _partition_idle does."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    ctrl = _make_controller(scene=scene, view=None)

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_module_idle_view_without_modules_returns_fallback() -> None:
    """Uncomposed FEMData / pre-2.9.0 archive — ``has_modules ==
    False``. All entities uniform fallback (matches the
    ``has_partitions == False`` shape)."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid=None)
    assert elements.has_modules is False
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_module_idle_entity_without_elements_returns_fallback() -> None:
    """A DimTag missing from ``scene.brep_to_elems`` gets fallback."""
    scene = _FakeScene(brep_to_elems={})  # entity not registered
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 99))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_module_idle_entity_with_only_host_owned_elements() -> None:
    """Entity owns elements whose ``module_for(...)`` returns
    ``None`` — host-owned in a composed model, or elements emitted
    outside any compose bracket. Fallback color, not crash. (3F.2a
    excludes empty-string labels from the map, so unlabeled elements
    naturally read as None.)"""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101]})
    # module_by_eid maps a different element (50), not 100/101
    _, elements = _make_viewer_data(module_by_eid={50: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _FALLBACK_RGB)


def test_module_idle_partial_host_owned_uses_only_labelled_elements() -> None:
    """Entity owns 3 elements: 1 host-owned (None), 2 labelled "modA".
    The None element is skipped; dominant label = "modA". This
    mirrors _partition_idle's "skip None ranks" path."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101, 102]})
    # Only 101, 102 labelled; 100 is host-owned -> module_for(100) is None
    _, elements = _make_viewer_data(module_by_eid={101: "modA", 102: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("modA"))


# =====================================================================
# set_mode dispatch — Module
# =====================================================================


def test_set_mode_module_installs_module_idle_fn() -> None:
    """``set_mode("Module")`` swaps the ColorManager's idle-fn to
    ``_module_idle`` — same dispatch shape as Element Type / Physical
    Group / Partition, so hover/pick/visibility layering keep
    working."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    ctrl.set_mode("Module")
    assert ctrl.mode == "Module"
    # Bound-method identity check (matches the Partition test pattern).
    assert ctrl._color_mgr.idle_fn == ctrl._module_idle


def test_set_mode_default_after_module_resets_idle_fn() -> None:
    """Switching from Module to Default goes through reset_idle_fn,
    not a no-op — consistent with the other idle-fn modes."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    ctrl.set_mode("Module")
    ctrl.set_mode("Default")
    assert ctrl.mode == "Default"
    assert ctrl._color_mgr.idle_fn is None


def test_set_mode_module_to_partition_swaps_idle_fn() -> None:
    """Crossing between two compose/partition modes swaps the idle
    fn cleanly — Module -> Partition routes through the partition
    branch with no leftover Module state."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    ctrl.set_mode("Module")
    ctrl.set_mode("Partition")
    assert ctrl.mode == "Partition"
    assert ctrl._color_mgr.idle_fn == ctrl._partition_idle
