"""Theme choice persists across ThemeManager instances via QSettings."""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtCore")

from qtpy.QtCore import QSettings

from apeGmsh.viewers.ui import theme


@pytest.fixture
def isolated_settings(monkeypatch):
    """Route ThemeManager to a test-only QSettings namespace."""
    org = "apeGmsh-test"
    app = "viewer-theme-persistence"
    monkeypatch.setattr(theme.ThemeManager, "_settings_org", org)
    monkeypatch.setattr(theme.ThemeManager, "_settings_app", app)
    QSettings(org, app).clear()
    yield
    QSettings(org, app).clear()


def test_save_roundtrip_via_qsettings(isolated_settings):
    tm = theme.ThemeManager()
    assert tm.current is theme.PALETTE_DARK  # default / fresh store

    tm.set_theme("light")

    # New instance should pick up the persisted choice
    tm2 = theme.ThemeManager()
    assert tm2.current is theme.PALETTE_LIGHT
