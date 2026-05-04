"""Session persistence — DiagramSpec ↔ JSON ↔ DiagramSpec.

Pure unit tests on the serialize / deserialize / save / load surface.
The viewer-side restore (prompt + apply) is exercised separately
through the headless ResultsViewer fixtures in
``test_results_viewer_smoke.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.viewers.diagrams import (
    ContourStyle,
    DeformedShapeStyle,
    DiagramSpec,
    LineForceStyle,
    SlabSelector,
    SpringForceStyle,
)
from apeGmsh.viewers.diagrams._session import (
    SESSION_SCHEMA_VERSION,
    GeometrySnapshot,
    ViewerSession,
    default_session_path,
    deserialize_session,
    deserialize_spec,
    load_session,
    save_session,
    serialize_session,
    serialize_spec,
)


# =====================================================================
# DiagramSpec round-trip
# =====================================================================

def _make_contour_spec():
    return DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z", pg=("Top",)),
        style=ContourStyle(cmap="viridis", clim=(-1.0, 1.0)),
        stage_id="stage_0",
        visible=True,
        label="Top displacement",
    )


def _make_line_force_spec():
    return DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component="bending_moment_z"),
        style=LineForceStyle(scale=2.5, flip_sign=True),
        stage_id="stage_1",
    )


def _make_spring_spec():
    return DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component="spring_force_0"),
        style=SpringForceStyle(direction=(1.0, 0.0, 0.0)),
    )


def _make_deformed_spec():
    return DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DeformedShapeStyle(
            components=("displacement_x", "displacement_y"),
            scale=10.0,
        ),
    )


def test_serialize_spec_returns_json_friendly_dict():
    spec = _make_contour_spec()
    data = serialize_spec(spec)
    # Must JSON-serialize without error.
    json.dumps(data)
    assert data["kind"] == "contour"
    assert data["selector"]["component"] == "displacement_z"
    assert data["stage_id"] == "stage_0"


def test_contour_spec_round_trip():
    spec = _make_contour_spec()
    restored = deserialize_spec(serialize_spec(spec))
    assert restored == spec


def test_line_force_spec_round_trip_preserves_flip_sign():
    spec = _make_line_force_spec()
    restored = deserialize_spec(serialize_spec(spec))
    assert restored == spec
    assert restored.style.flip_sign is True


def test_deformed_spec_round_trip_keeps_components_tuple():
    """Tuples become lists in JSON; deserialize must coerce them back."""
    spec = _make_deformed_spec()
    restored = deserialize_spec(serialize_spec(spec))
    assert restored == spec
    assert isinstance(restored.style.components, tuple)


def test_spring_spec_round_trip_keeps_direction_tuple():
    spec = _make_spring_spec()
    restored = deserialize_spec(serialize_spec(spec))
    assert restored == spec
    assert isinstance(restored.style.direction, tuple)


def test_deserialize_spec_unknown_kind_raises():
    with pytest.raises(KeyError, match="Unknown diagram kind"):
        deserialize_spec({
            "kind": "totally_made_up",
            "selector": {"component": "x"},
            "style": {},
        })


# =====================================================================
# Session round-trip
# =====================================================================

def test_serialize_session_includes_metadata(tmp_path: Path):
    payload = serialize_session(
        specs=[_make_contour_spec(), _make_line_force_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id="abc123",
        active_stage_id="stage_0",
        active_step=42,
    )
    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    assert payload["fem_snapshot_id"] == "abc123"
    assert payload["active_stage_id"] == "stage_0"
    assert payload["active_step"] == 42
    assert len(payload["diagrams"]) == 2


def test_session_round_trip(tmp_path: Path):
    specs = [
        _make_contour_spec(),
        _make_line_force_spec(),
        _make_deformed_spec(),
        _make_spring_spec(),
    ]
    payload = serialize_session(
        specs=specs,
        results_path=tmp_path / "run.h5",
        fem_snapshot_id="abc123",
    )
    session = deserialize_session(payload)
    assert isinstance(session, ViewerSession)
    assert session.fem_snapshot_id == "abc123"
    assert len(session.diagrams) == 4
    for original, restored in zip(specs, session.diagrams):
        assert restored == original


def test_deserialize_skips_corrupt_specs():
    """A corrupt entry must not abort the whole restore — drop it instead."""
    payload = {
        "schema_version": 1,
        "results_path": "/x/y.h5",
        "fem_snapshot_id": None,
        "saved_at": "",
        "diagrams": [
            serialize_spec(_make_contour_spec()),
            {"kind": "completely_unknown", "selector": {"component": "x"}},
            serialize_spec(_make_line_force_spec()),
        ],
    }
    session = deserialize_session(payload)
    # 2 of 3 survived
    assert len(session.diagrams) == 2
    assert session.diagrams[0].kind == "contour"
    assert session.diagrams[1].kind == "line_force"


# =====================================================================
# Disk I/O
# =====================================================================

def test_default_session_path_appends_suffix(tmp_path: Path):
    target = default_session_path(tmp_path / "run.h5")
    assert target.name == "run.h5.viewer-session.json"
    assert target.parent == tmp_path


def test_default_session_path_handles_mpco(tmp_path: Path):
    target = default_session_path(tmp_path / "frame.mpco")
    assert target.name == "frame.mpco.viewer-session.json"


def test_save_then_load_round_trip(tmp_path: Path):
    results_path = tmp_path / "run.h5"
    saved_path = save_session(
        specs=[_make_contour_spec(), _make_line_force_spec()],
        results_path=results_path,
        fem_snapshot_id="snap123",
        active_stage_id="stage_0",
        active_step=7,
    )
    assert saved_path.exists()
    assert saved_path == default_session_path(results_path)

    session = load_session(saved_path)
    assert session.fem_snapshot_id == "snap123"
    assert session.active_stage_id == "stage_0"
    assert session.active_step == 7
    assert len(session.diagrams) == 2
    assert session.diagrams[0].kind == "contour"
    assert session.diagrams[1].kind == "line_force"


def test_save_session_explicit_target(tmp_path: Path):
    target = tmp_path / "custom.json"
    out = save_session(
        specs=[_make_contour_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        target_path=target,
    )
    assert out == target
    assert target.exists()


def test_save_then_load_preserves_visible_and_label_flags(tmp_path: Path):
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
        visible=False,
        label="Custom label",
    )
    saved = save_session(
        specs=[spec],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
    )
    session = load_session(saved)
    restored = session.diagrams[0]
    assert restored.visible is False
    assert restored.label == "Custom label"


# =====================================================================
# Geometry snapshot — display state (schema v3)
# =====================================================================

def test_v3_geometry_snapshot_persists_display_state(tmp_path: Path):
    """show_mesh / show_nodes / display_opacity round-trip through JSON."""
    geom = GeometrySnapshot(
        id="g0",
        name="Test geom",
        deform_enabled=True,
        deform_field="displacement",
        deform_scale=2.0,
        show_mesh=False,
        show_nodes=True,
        display_opacity=0.4,
    )
    saved = save_session(
        specs=[_make_contour_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=[geom],
    )
    session = load_session(saved)
    assert len(session.geometries) == 1
    restored = session.geometries[0]
    assert restored.show_mesh is False
    assert restored.show_nodes is True
    assert restored.display_opacity == pytest.approx(0.4)


def test_v2_session_loads_with_display_defaults(tmp_path: Path):
    """A v2 session JSON (no display fields) must load with defaults."""
    payload = {
        "schema_version": 2,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "geometries": [
            {
                "id": "g0",
                "name": "Geometry 1",
                "deform_enabled": True,
                "deform_field": "displacement",
                "deform_scale": 5.0,
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_contour_spec())],
    }
    target = tmp_path / "v2.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    session = load_session(target)
    geom = session.geometries[0]
    # Display fields default to the historical "everything visible,
    # full alpha" behavior so old saves render the same as before.
    assert geom.show_mesh is True
    assert geom.show_nodes is True
    assert geom.display_opacity == pytest.approx(1.0)
    # Deformation state still loads.
    assert geom.deform_enabled is True
    assert geom.deform_scale == pytest.approx(5.0)


def test_session_schema_version_is_3():
    """Sanity: the constant tracks the latest schema."""
    assert SESSION_SCHEMA_VERSION == 3
