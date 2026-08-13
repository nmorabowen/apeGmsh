"""Graphics-colors window: Okabe–Ito preset + field coverage (headless)."""
from __future__ import annotations

from apeGmsh.viewers.ui.graphics_colors_dialog import (
    OKABE_ITO_GRAPHICS,
    _FIELDS,
)


def test_okabe_ito_preset_is_blue_yellow():
    hover = OKABE_ITO_GRAPHICS["hover_rgb"]
    pick = OKABE_ITO_GRAPHICS["pick_rgb"]
    assert hover == (240, 228, 66)
    assert pick == (0, 114, 178)
    assert OKABE_ITO_GRAPHICS["origin_marker_color"] == "#56B4E9"
    assert OKABE_ITO_GRAPHICS["measure_color"] == "#56B4E9"


def test_graphics_fields_cover_selection_idle_and_overlays():
    names = {n for n, _label, _kind in _FIELDS}
    assert names >= {
        "hover_rgb", "pick_rgb", "hidden_rgb",
        "dim_pt", "dim_crv", "dim_srf", "dim_vol",
        "origin_marker_color", "measure_color",
    }
