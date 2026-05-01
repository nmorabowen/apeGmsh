"""Palette dataclass, stylesheet factory, and ThemeManager behavior.

Headless: no Qt widgets, only pure-Python data.
"""
from __future__ import annotations

import logging

import pytest

from apeGmsh.viewers.ui import theme


@pytest.fixture
def fresh_manager(tmp_path, monkeypatch):
    """A ThemeManager that doesn't touch the user's real QSettings."""
    monkeypatch.setattr(theme.ThemeManager, "_settings_org", "apeGmsh-test")
    monkeypatch.setattr(theme.ThemeManager, "_settings_app", "viewer-test")
    # Force-return None from _load_saved so tests start from dark
    monkeypatch.setattr(
        theme.ThemeManager, "_load_saved", classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        theme.ThemeManager, "_save", classmethod(lambda cls, palette: None),
    )
    return theme.ThemeManager()


def test_all_builtin_palettes_defined():
    # Built-in theme ids ship with apeGmsh; custom themes may be added
    # later via the theme editor and are not covered by this test.
    builtin = {
        "catppuccin_mocha", "catppuccin_latte", "neutral_studio", "paper",
        "solarized_dark", "solarized_light", "nord", "tokyo_night",
        "gruvbox_dark", "high_contrast",
    }
    assert builtin.issubset(set(theme.PALETTES))
    for key in builtin:
        assert theme.PALETTES[key].name == key


def test_legacy_aliases_resolve(fresh_manager):
    # QSettings from before the rename may carry "dark" / "light".
    fresh_manager.set_theme("light")
    assert fresh_manager.current is theme.PALETTE_PAPER
    fresh_manager.set_theme("dark")
    assert fresh_manager.current is theme.PALETTE_CATPPUCCIN_MOCHA


def test_every_palette_has_aesthetic_fields():
    # Smoke-test that all aesthetic fields were populated for every theme.
    required = (
        "background_mode", "body_palette",
        "outline_color", "outline_silhouette_px", "outline_feature_px",
        "mesh_edge_color",
        "node_accent", "grid_major", "grid_minor",
        "bbox_color", "bbox_line_px",
        "cmap_seq", "cmap_div", "ao_intensity", "corner_triad_default",
    )
    for pal in theme.PALETTES.values():
        for field in required:
            assert getattr(pal, field) is not None, (
                f"{pal.name}.{field} is missing"
            )


def test_palettes_are_frozen():
    with pytest.raises(Exception):
        theme.PALETTE_DARK.base = "#000000"  # type: ignore[misc]


def test_build_stylesheet_includes_dark_chrome():
    qss = theme.build_stylesheet(theme.PALETTE_DARK)
    assert "QMainWindow" in qss
    assert theme.PALETTE_DARK.base in qss
    assert theme.PALETTE_DARK.text in qss


def test_build_stylesheet_includes_light_chrome():
    qss = theme.build_stylesheet(theme.PALETTE_LIGHT)
    assert theme.PALETTE_LIGHT.base in qss
    assert theme.PALETTE_LIGHT.text in qss
    # Dark mantle should not leak into light stylesheet
    assert theme.PALETTE_DARK.mantle not in qss


def test_backcompat_stylesheet_constant_matches_factory():
    assert theme.STYLESHEET == theme.build_stylesheet(theme.PALETTE_DARK)


def test_build_stylesheet_accepts_density_tokens():
    from apeGmsh.viewers.ui.density import (
        DENSITY_COMPACT, DENSITY_COMFORTABLE,
    )
    qss_compact = theme.build_stylesheet(theme.PALETTE_DARK, DENSITY_COMPACT)
    qss_comf = theme.build_stylesheet(
        theme.PALETTE_DARK, DENSITY_COMFORTABLE,
    )
    # Density only changes type / row sizing; the row-height appears
    # in the QTreeWidget::item rule (min-height).
    assert f"min-height: {DENSITY_COMPACT.row_h}px" in qss_compact
    assert f"min-height: {DENSITY_COMFORTABLE.row_h}px" in qss_comf
    # Compact and comfortable must produce different output.
    assert qss_compact != qss_comf


def test_backcompat_module_constants_resolve():
    # Downstream code imports these directly
    assert theme.BG_TOP == theme.PALETTE_DARK.bg_top
    assert theme.BG_BOTTOM == theme.PALETTE_DARK.bg_bottom
    assert theme.BASE == theme.PALETTE_DARK.base
    assert theme.TEXT == theme.PALETTE_DARK.text


def test_set_theme_fires_observers(fresh_manager):
    received: list[theme.Palette] = []
    fresh_manager.subscribe(lambda p: received.append(p))
    fresh_manager.set_theme("light")
    assert received == [theme.PALETTE_LIGHT]


def test_set_theme_idempotent(fresh_manager):
    received: list[theme.Palette] = []
    fresh_manager.subscribe(lambda p: received.append(p))
    fresh_manager.set_theme("catppuccin_mocha")  # already default
    assert received == []


def test_set_theme_rejects_unknown(fresh_manager):
    with pytest.raises(ValueError):
        fresh_manager.set_theme("solarized")


def test_unsubscribe_removes_observer(fresh_manager):
    received: list[theme.Palette] = []
    unsub = fresh_manager.subscribe(lambda p: received.append(p))
    unsub()
    fresh_manager.set_theme("light")
    assert received == []


def test_observer_exception_is_logged_not_raised(fresh_manager, caplog):
    def bad(_p):
        raise RuntimeError("boom")

    fresh_manager.subscribe(bad)
    with caplog.at_level(logging.ERROR, logger="apeGmsh.viewer.theme"):
        fresh_manager.set_theme("light")
    assert any("boom" in str(r.exc_info) for r in caplog.records if r.exc_info)


def test_cad_neutral_palettes_have_black_wire_and_gray_fills():
    # CAD-look: dim=0 (pt) and dim=1 (crv) are pure black across all themes;
    # surface/volume fills are gray, with Paper slightly darker than the dark
    # themes so the fill reads against near-white background.
    dark_srf = theme.PALETTE_DARK.dim_srf
    light_srf = theme.PALETTE_LIGHT.dim_srf
    for pal in (theme.PALETTE_DARK, theme.PALETTE_LIGHT):
        assert pal.dim_pt == (0, 0, 0)
        assert pal.dim_crv == (0, 0, 0)
    assert sum(dark_srf) > sum(light_srf)
