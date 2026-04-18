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


def test_both_palettes_defined():
    assert "dark" in theme.PALETTES
    assert "light" in theme.PALETTES
    assert theme.PALETTES["dark"].name == "dark"
    assert theme.PALETTES["light"].name == "light"


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
    fresh_manager.set_theme("dark")  # already dark
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


def test_light_palette_has_contrast_adjusted_vtk_colors():
    # Dark palette's warm white would be invisible on white bg —
    # light palette must pick dark-ink content colors instead.
    dark_pt = theme.PALETTE_DARK.dim_pt
    light_pt = theme.PALETTE_LIGHT.dim_pt
    assert sum(dark_pt) > sum(light_pt)
