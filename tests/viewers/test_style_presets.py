"""Style preset store — serialize, save, list, load, delete, sanitize."""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.viewers.diagrams._style_presets import (
    KIND_TO_STYLE_CLASS,
    StylePresetStore,
    style_from_dict,
    style_to_dict,
)
from apeGmsh.viewers.diagrams._styles import (
    ContourStyle,
    DeformedShapeStyle,
    LineForceStyle,
    SpringForceStyle,
    VectorGlyphStyle,
)


# =====================================================================
# Serialization round-trip
# =====================================================================

def test_contour_round_trip():
    style = ContourStyle(
        cmap="plasma", clim=(-1.0, 2.5),
        opacity=0.75, show_edges=True, topology="nodes",
    )
    data = style_to_dict(style)
    back = style_from_dict("contour", data)
    assert back == style


def test_tuple_field_re_tuples_after_json():
    """JSON has no tuple type; lists must become tuples again."""
    style = ContourStyle(clim=(0.0, 1.0))
    data = style_to_dict(style)
    assert data["clim"] == [0.0, 1.0]   # JSON-safe list
    back = style_from_dict("contour", data)
    assert isinstance(back.clim, tuple)
    assert back.clim == (0.0, 1.0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        style_from_dict("not_a_kind", {})


def test_unknown_field_is_ignored():
    """Forward compatibility: older presets carrying fields no longer
    on the dataclass must not break loads."""
    data = style_to_dict(ContourStyle())
    data["legacy_removed_field"] = "ignored"
    style = style_from_dict("contour", data)
    assert isinstance(style, ContourStyle)


def test_style_to_dict_rejects_non_jsonable_field(tmp_path):
    """Internal contract: every shipped *Style uses primitive fields.
    A made-up subclass with a Path would refuse to serialize."""
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class _BadStyle:
        path: Path = field(default_factory=lambda: Path("/x"))

    with pytest.raises(TypeError):
        style_to_dict(_BadStyle())


def test_kind_map_covers_every_shipped_style():
    """Every Style class in _styles.py should be reachable via a
    kind_id; otherwise saving will succeed but loading will fail."""
    expected = {
        "contour", "deformed_shape", "line_force", "fiber_section",
        "layer_stack", "vector_glyph", "gauss_marker", "spring_force",
    }
    assert set(KIND_TO_STYLE_CLASS.keys()) == expected


# =====================================================================
# Store CRUD
# =====================================================================

def _store(tmp_path: Path) -> StylePresetStore:
    return StylePresetStore(directory=tmp_path / "presets")


def test_save_and_load_round_trip(tmp_path):
    s = _store(tmp_path)
    style = LineForceStyle(scale=12.5, flip_sign=True, opacity=0.5)
    s.save("brace shears", "line_force", style)
    kind, loaded = s.load("brace shears")
    assert kind == "line_force"
    assert loaded == style


def test_list_returns_name_kind_pairs(tmp_path):
    s = _store(tmp_path)
    s.save("a", "contour", ContourStyle(cmap="viridis"))
    s.save("b", "deformed_shape", DeformedShapeStyle())
    pairs = sorted(s.list())
    assert pairs == [("a", "contour"), ("b", "deformed_shape")]


def test_list_for_kind_filters(tmp_path):
    s = _store(tmp_path)
    s.save("c1", "contour", ContourStyle())
    s.save("c2", "contour", ContourStyle(cmap="plasma"))
    s.save("v1", "vector_glyph", VectorGlyphStyle())
    contour = s.list_for_kind("contour")
    assert sorted(contour) == ["c1", "c2"]


def test_delete_removes_file(tmp_path):
    s = _store(tmp_path)
    s.save("temp", "spring_force", SpringForceStyle())
    assert s.delete("temp") is True
    assert s.list_for_kind("spring_force") == []


def test_delete_unknown_returns_false(tmp_path):
    s = _store(tmp_path)
    assert s.delete("never-existed") is False


def test_save_unknown_kind_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.save("oops", "not_a_kind", ContourStyle())


def test_save_invalid_name_raises(tmp_path):
    """Names that sanitize to empty (only path-traversal chars) reject."""
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.save("../", "contour", ContourStyle())


def test_save_sanitizes_filename_characters(tmp_path):
    """Slashes and backslashes are stripped — no path traversal possible."""
    s = _store(tmp_path)
    path = s.save("foo/bar..\\baz", "contour", ContourStyle())
    # Result file should live in the store dir, not escape it.
    assert path.parent == s.directory


def test_corrupt_json_skipped_in_list(tmp_path):
    s = _store(tmp_path)
    s.save("good", "contour", ContourStyle())
    bad = s.directory / "bad.json"
    bad.write_text("{ this is not valid", encoding="utf-8")
    pairs = s.list()
    assert ("good", "contour") in pairs
    assert all(name != "bad" for name, _ in pairs)


# =====================================================================
# Default singleton — reset_store + lazy create
# =====================================================================

def test_default_store_lazy_singleton(tmp_path, monkeypatch):
    from apeGmsh.viewers.diagrams import _style_presets
    _style_presets.reset_store(_style_presets.StylePresetStore(
        directory=tmp_path / "presets",
    ))
    s = _style_presets.default_store()
    s.save("zz", "contour", ContourStyle())
    # Singleton — second call returns same instance.
    assert _style_presets.default_store() is s
    _style_presets.reset_store(None)
