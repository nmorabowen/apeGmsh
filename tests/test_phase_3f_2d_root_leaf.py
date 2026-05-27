"""Phase 3F.2d — ColorMode "Module: Root" and "Module: Leaf" projection.

Phase 3F.2b shipped the full-label ``_module_idle`` callback that
hashes the FULL joined module label (e.g. ``"bayP/frameA"``) onto a
palette slot. Phase 3F.2d adds two sibling modes that project each
label onto its root or leaf component (via
``apeGmsh.mesh._compose._split_joined_label``) before hashing.

Slice scope (3F.2d): two new color modes — ``"Module: Root"`` and
``"Module: Leaf"`` — exposed via UI dropdown + controller dispatch.
Shares the fallback / dominant-label / palette logic with
``_module_idle`` through ``_module_idle_projected``.

Tests reuse 3F.2b's fixture builders directly so any drift in the
underlying `ViewerElements` constructor is caught in one place.
No GPU verification per ``feedback_viewer_no_gpu``.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.color_mode_controller import _FALLBACK_RGB
from apeGmsh.viewers.ui.mesh_tabs import COLOR_MODES

# Reuse the 3F.2b fixture infrastructure verbatim — same scene fake,
# same ViewerElements builder, same controller assembler, same
# palette-mirror helper. Importing avoids drift if 3F.2b's underlying
# constructor signatures evolve.
from tests.test_phase_3f_2b_callback import (
    _FakeScene,
    _MiniView,
    _make_controller,
    _make_viewer_data,
    _palette_color_for,
)


# =====================================================================
# UI registry — both new entries present and in expected order
# =====================================================================


class TestColorModesRegistry:
    def test_module_root_present(self) -> None:
        assert "Module: Root" in COLOR_MODES

    def test_module_leaf_present(self) -> None:
        assert "Module: Leaf" in COLOR_MODES

    def test_ordering(self) -> None:
        """Module < Module: Root < Module: Leaf < Quality."""
        i_module = COLOR_MODES.index("Module")
        i_root = COLOR_MODES.index("Module: Root")
        i_leaf = COLOR_MODES.index("Module: Leaf")
        i_quality = COLOR_MODES.index("Quality")
        assert i_module < i_root < i_leaf < i_quality

    def test_no_duplicates(self) -> None:
        assert len(COLOR_MODES) == len(set(COLOR_MODES))


# =====================================================================
# _module_idle_by_root — happy path
# =====================================================================


def test_root_colors_depth_2_label_by_top_component() -> None:
    """``"bayP/frameA"`` projects to root ``"bayP"`` for coloring."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "bayP/frameA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_root((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("bayP"))


def test_root_collapses_siblings_under_same_root() -> None:
    """``"bayP/frameA"`` and ``"bayP/frameB"`` color identically
    in Root mode — both project to ``"bayP"``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayP/frameB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle_by_root((2, 10))
    rgb_b = ctrl._module_idle_by_root((2, 20))
    np.testing.assert_array_equal(rgb_a, rgb_b)
    np.testing.assert_array_equal(rgb_a, _palette_color_for("bayP"))


def test_root_separates_different_roots() -> None:
    """``"bayP/frameA"`` and ``"bayZ/frameA"`` project to different
    roots — colors come from ``_palette_color_for(root)``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayZ/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle_by_root((2, 10))
    rgb_b = ctrl._module_idle_by_root((2, 20))
    np.testing.assert_array_equal(rgb_a, _palette_color_for("bayP"))
    np.testing.assert_array_equal(rgb_b, _palette_color_for("bayZ"))


def test_root_on_depth_1_label_projects_to_itself() -> None:
    """A flat (depth-1) label has root == itself, so Root mode
    matches plain Module mode on flat composes."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_root((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("modA"))


def test_root_on_depth_3_label() -> None:
    """``"top.assemblyM/partA"`` (depth-3 joined label) projects to
    its root ``"top"`` for coloring."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "top.assemblyM/partA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_root((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("top"))


# =====================================================================
# _module_idle_by_leaf — happy path
# =====================================================================


def test_leaf_colors_depth_2_label_by_bottom_component() -> None:
    """``"bayP/frameA"`` projects to leaf ``"frameA"``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "bayP/frameA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_leaf((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("frameA"))


def test_leaf_collapses_same_leaf_across_different_roots() -> None:
    """``"bayP/frameA"`` and ``"bayZ/frameA"`` color identically in
    Leaf mode — both project to ``"frameA"``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayZ/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle_by_leaf((2, 10))
    rgb_b = ctrl._module_idle_by_leaf((2, 20))
    np.testing.assert_array_equal(rgb_a, rgb_b)
    np.testing.assert_array_equal(rgb_a, _palette_color_for("frameA"))


def test_leaf_separates_different_leaves() -> None:
    """``"bayP/frameA"`` and ``"bayP/frameB"`` project to different
    leaves."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayP/frameB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_a = ctrl._module_idle_by_leaf((2, 10))
    rgb_b = ctrl._module_idle_by_leaf((2, 20))
    np.testing.assert_array_equal(rgb_a, _palette_color_for("frameA"))
    np.testing.assert_array_equal(rgb_b, _palette_color_for("frameB"))


def test_leaf_on_depth_1_label_projects_to_itself() -> None:
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_leaf((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("modA"))


def test_leaf_on_depth_3_label() -> None:
    """``"top.assemblyM/partA"`` projects to leaf ``"partA"``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "top.assemblyM/partA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_leaf((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("partA"))


# =====================================================================
# Fallback paths (mirror 3F.2b: no view / no modules / no elements)
# =====================================================================


def test_root_no_view_returns_fallback() -> None:
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    ctrl = _make_controller(scene=scene, view=None)
    np.testing.assert_array_equal(ctrl._module_idle_by_root((2, 10)), _FALLBACK_RGB)


def test_root_has_modules_false_returns_fallback() -> None:
    """Uncomposed model: has_modules is False -> fallback."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid=None)  # None -> has_modules False
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))
    np.testing.assert_array_equal(ctrl._module_idle_by_root((2, 10)), _FALLBACK_RGB)


def test_root_no_brep_to_elems_returns_fallback() -> None:
    scene = _FakeScene(brep_to_elems={})
    _, elements = _make_viewer_data(module_by_eid={100: "modA"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))
    np.testing.assert_array_equal(ctrl._module_idle_by_root((2, 10)), _FALLBACK_RGB)


def test_leaf_no_view_returns_fallback() -> None:
    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    ctrl = _make_controller(scene=scene, view=None)
    np.testing.assert_array_equal(ctrl._module_idle_by_leaf((2, 10)), _FALLBACK_RGB)


# =====================================================================
# Dominant-label resolution under projection
# =====================================================================


def test_root_dominant_after_projection() -> None:
    """Three elements: 2 with root ``"bayP"``, 1 with root ``"bayZ"``.
    Dominant projected root = ``"bayP"`` (Counter.most_common over
    projected labels, mirrors _module_idle)."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101, 102]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA",
        101: "bayP/frameB",
        102: "bayZ/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_root((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("bayP"))


def test_leaf_dominant_after_projection() -> None:
    """Two elements with leaf ``"frameA"``, one with leaf ``"frameB"``.
    Dominant projected leaf = ``"frameA"``."""
    scene = _FakeScene(brep_to_elems={(2, 10): [100, 101, 102]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA",
        101: "bayZ/frameA",
        102: "bayP/frameB",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb = ctrl._module_idle_by_leaf((2, 10))
    np.testing.assert_array_equal(rgb, _palette_color_for("frameA"))


# =====================================================================
# Determinism
# =====================================================================


def test_root_deterministic_across_calls() -> None:
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayP/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_first = ctrl._module_idle_by_root((2, 10))
    rgb_again = ctrl._module_idle_by_root((2, 10))
    rgb_other = ctrl._module_idle_by_root((2, 20))
    np.testing.assert_array_equal(rgb_first, rgb_again)
    np.testing.assert_array_equal(rgb_first, rgb_other)


def test_leaf_deterministic_across_calls() -> None:
    scene = _FakeScene(brep_to_elems={(2, 10): [100], (2, 20): [200]})
    _, elements = _make_viewer_data(module_by_eid={
        100: "bayP/frameA", 200: "bayZ/frameA",
    })
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    rgb_first = ctrl._module_idle_by_leaf((2, 10))
    rgb_again = ctrl._module_idle_by_leaf((2, 10))
    rgb_other = ctrl._module_idle_by_leaf((2, 20))
    np.testing.assert_array_equal(rgb_first, rgb_again)
    np.testing.assert_array_equal(rgb_first, rgb_other)


# =====================================================================
# Malformed-label fail-loud
# =====================================================================


def test_malformed_label_raises_compose_error_root() -> None:
    """If a malformed joined label (violating separator alternation)
    appears in module_by_eid, the projector raises ComposeError —
    fail-loud rather than silently fall back, since this indicates
    upstream corruption.
    """
    from apeGmsh.mesh._compose import ComposeError

    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    # "top/foo/bar" is malformed — depth-3 expects "." at the outermost
    # position, but this has "/" instead. _split_joined_label should
    # raise ComposeError.
    _, elements = _make_viewer_data(module_by_eid={100: "top/foo/bar"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    with pytest.raises(ComposeError):
        ctrl._module_idle_by_root((2, 10))


def test_malformed_label_raises_compose_error_leaf() -> None:
    from apeGmsh.mesh._compose import ComposeError

    scene = _FakeScene(brep_to_elems={(2, 10): [100]})
    _, elements = _make_viewer_data(module_by_eid={100: "top/foo/bar"})
    ctrl = _make_controller(scene=scene, view=_MiniView(elements))

    with pytest.raises(ComposeError):
        ctrl._module_idle_by_leaf((2, 10))
