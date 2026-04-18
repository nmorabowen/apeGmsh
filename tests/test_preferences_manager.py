"""Tests for PreferencesManager — JSON-backed persistent viewer prefs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.viewers.ui.preferences_manager import (
    DEFAULT_PREFERENCES,
    Preferences,
    PreferencesManager,
)


@pytest.fixture
def manager(tmp_path: Path) -> PreferencesManager:
    # Construct with an explicit tmp path so real user config is never
    # touched by tests.
    return PreferencesManager(path=tmp_path / "preferences.json")


def test_defaults_applied_when_file_missing(manager: PreferencesManager):
    assert manager.current == DEFAULT_PREFERENCES


def test_update_persists_to_disk(manager: PreferencesManager):
    manager.update({"point_size": 22.0})
    assert manager.current.point_size == 22.0
    assert manager.path.exists()
    data = json.loads(manager.path.read_text())
    assert data["point_size"] == 22.0


def test_update_fires_observers(manager: PreferencesManager):
    captured: list[Preferences] = []
    unsub = manager.subscribe(captured.append)
    manager.update({"line_width": 4.0})
    assert len(captured) == 1
    assert captured[0].line_width == 4.0
    unsub()
    manager.update({"line_width": 5.0})
    assert len(captured) == 1  # unsubscribed; no new event


def test_update_noop_when_values_unchanged(manager: PreferencesManager):
    captured: list[Preferences] = []
    manager.subscribe(captured.append)
    manager.update({"point_size": DEFAULT_PREFERENCES.point_size})
    assert captured == []  # no change → no notification


def test_unknown_keys_are_ignored(manager: PreferencesManager):
    manager.update({"point_size": 15.0, "bogus_field": "xyz"})
    assert manager.current.point_size == 15.0
    data = json.loads(manager.path.read_text())
    assert "bogus_field" not in data


def test_reset_deletes_file(manager: PreferencesManager):
    manager.update({"point_size": 20.0})
    assert manager.path.exists()
    manager.reset()
    assert not manager.path.exists()
    assert manager.current == DEFAULT_PREFERENCES


def test_reload_from_existing_file(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({"point_size": 13.0, "show_surface_edges": True}))
    mgr = PreferencesManager(path=path)
    assert mgr.current.point_size == 13.0
    assert mgr.current.show_surface_edges is True
    # Unspecified fields fall back to defaults
    assert mgr.current.line_width == DEFAULT_PREFERENCES.line_width


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text("{not valid json")
    mgr = PreferencesManager(path=path)
    assert mgr.current == DEFAULT_PREFERENCES


def test_settings_function_exported_at_top_level():
    import apeGmsh
    assert callable(apeGmsh.settings)
    # And from the viewers subpackage
    from apeGmsh import viewers
    assert apeGmsh.settings is viewers.settings


def test_all_new_fields_round_trip(tmp_path: Path):
    """New fields (rendering, font sizes, axis, UI) persist through JSON."""
    from dataclasses import fields
    mgr = PreferencesManager(path=tmp_path / "preferences.json")
    overrides = {
        # Rendering
        "smooth_shading": True,
        "anti_aliasing": "fxaa",
        # Mesh
        "mesh_line_width": 4.0,
        "mesh_surface_opacity": 0.8,
        "mesh_show_surface_edges": False,
        # Outlines
        "feature_angle": 40.0,
        # Labels
        "node_label_font_size": 12,
        "element_label_font_size": 11,
        "entity_label_font_size": 14,
        "origin_marker_font_size": 13,
        "coord_precision": 4,
        # Axis
        "axis_line_width": 3.0,
        "axis_labels_visible": False,
        # Interaction & UI
        "drag_threshold": 12,
        "tab_position": "top",
        "dock_min_width": 400,
        "window_maximized": False,
        "show_console": True,
    }
    mgr.update(overrides)

    reloaded = PreferencesManager(path=mgr.path)
    for k, v in overrides.items():
        assert getattr(reloaded.current, k) == v, f"{k} did not round-trip"

    # Every dataclass field must have a default (no unset attrs)
    for f in fields(Preferences):
        assert hasattr(reloaded.current, f.name)


def test_choice_fields_use_known_values():
    from apeGmsh.viewers.ui.preferences_manager import (
        ANTI_ALIASING_CHOICES, TAB_POSITION_CHOICES,
    )
    assert DEFAULT_PREFERENCES.anti_aliasing in ANTI_ALIASING_CHOICES
    assert DEFAULT_PREFERENCES.tab_position in TAB_POSITION_CHOICES
