"""ADR 0056 V5 — PreferencesTab is a projection of its owners.

The V5 projection audit found the Session tab was the one panel whose
widgets could NOT rebuild from owners (INV-1):

* the overlay-scale sliders hardcoded 1.0x instead of reading
  ``OverlayVisibilityModel.scales`` — and the "Load arrows" slider
  fired a ``"load_arrow"`` key that no owner ever had, so it was a
  silent no-op pre-V3 and a ``KeyError`` crash after V3 made
  ``set_scale`` fail loud;
* the pick-color swatch hardcoded ``#E74C3C`` instead of reading the
  ``ColorManager``'s effective pick colour;
* both controls were built even when no callback bound them to an
  owner (a silent no-op surface in the other viewer).

These tests lock the fixes. The vocabulary tests are pure (no Qt) and
run in CI; the widget tests open real Qt widgets and carry the ``qt``
marker (run per-file: ``pytest -m qt tests/viewers/test_preferences_projection.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.overlay_visibility import (
    _SCALE_KEY_TO_OVERLAY,
    OverlayVisibilityModel,
)
from apeGmsh.viewers.ui.preferences import _OVERLAY_ITEMS


# ─────────────────────────────────────────────────────────────────────
# Pure — vocabulary contract (CI-safe; preferences.py imports Qt lazily)
# ─────────────────────────────────────────────────────────────────────

def test_overlay_slider_keys_match_owner_vocabulary():
    """Panel rows and owner scale keys are the same set, both ways.

    Panel ⊆ owner: a slider with an unknown key crashes in
    ``set_scale`` (the pre-V5 "load_arrow" bug). Owner ⊆ panel: the
    Overlay Sizing group is the one UI surface for glyph scales, so a
    new owned scale must get a row (or consciously amend this test).
    """
    panel_keys = {key for key, _label in _OVERLAY_ITEMS}
    owner_keys = set(_SCALE_KEY_TO_OVERLAY)
    assert panel_keys == owner_keys


def test_owner_accepts_every_panel_key():
    """Every key the panel can fire round-trips through the owner
    mutator — the exact call path that raised ``KeyError`` pre-V5."""
    model = OverlayVisibilityModel()
    for key, _label in _OVERLAY_ITEMS:
        model.set_scale(key, 2.5)
        assert model.scale(key) == 2.5


def test_color_manager_public_pick_rgb_reads_override():
    """``ColorManager.pick_rgb`` is the owner read path the swatch
    initializes from (ADR 0056 INV-1)."""
    from apeGmsh.viewers.core.color_manager import ColorManager

    override = np.array([0x12, 0x34, 0x56], dtype=np.uint8)
    mgr = ColorManager(None, pick_color=override)  # type: ignore[arg-type]
    assert np.array_equal(mgr.pick_rgb, override)


# ─────────────────────────────────────────────────────────────────────
# Qt — widget projection behaviour (local only, per-file)
# ─────────────────────────────────────────────────────────────────────

def _qapp():
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.qt
def test_prefs_tab_projects_owner_state():
    _qapp()
    from apeGmsh.viewers.ui.preferences import PreferencesTab

    scale_fires: list[tuple[str, float]] = []
    pick_fires: list[str] = []
    tab = PreferencesTab(
        pick_color="#123456",
        on_pick_color=pick_fires.append,
        overlay_scales={"force_arrow": 2.0},
        on_overlay_scale=lambda k, v: scale_fires.append((k, v)),
    )

    # Initial state is the owner's, and construction never fires.
    assert tab._pick_color_hex == "#123456"
    assert tab._overlay_sliders["force_arrow"].value() == 20
    assert tab._overlay_labels["force_arrow"].text().startswith("2.0")
    assert tab._overlay_sliders["mass_sphere"].value() == 10  # default 1.0x
    assert scale_fires == [] and pick_fires == []

    # A gesture forwards the OWNER's key vocabulary.
    tab._overlay_sliders["force_arrow"].setValue(35)
    assert scale_fires == [("force_arrow", 3.5)]


@pytest.mark.qt
def test_unbound_controls_are_not_built():
    """No callback, no widget — an unbound control would be a silent
    no-op surface (ADR 0056 INV-6)."""
    _qapp()
    from apeGmsh.viewers.ui.preferences import PreferencesTab

    tab = PreferencesTab()  # neither on_pick_color nor on_overlay_scale
    assert not hasattr(tab, "_btn_pick_color")
    assert tab._overlay_sliders == {}
