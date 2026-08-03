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

from apeGmsh.cuts import SectionCutDef
from apeGmsh.viewers.diagrams import (
    ContourStyle,
    DiagramSpec,
    LineForceStyle,
    SectionCutStyle,
    SlabSelector,
    SpringForceStyle,
)
from apeGmsh.viewers.diagrams._session import (
    SESSION_SCHEMA_VERSION,
    CompositionSnapshot,
    GeometrySnapshot,
    ViewerSession,
    default_session_path,
    deserialize_session,
    deserialize_spec,
    load_session,
    save_session,
    serialize_session,
    serialize_spec,
    session_restorable,
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


def _make_section_cut_spec(*, with_bounding: bool = False, side="positive"):
    bounding = (
        ((0.0, 0.0, 2.5), (1.0, 0.0, 2.5), (1.0, 1.0, 2.5), (0.0, 1.0, 2.5))
        if with_bounding else None
    )
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 2.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 12),
        side=side,
        label="Story 3 base shear",
        bounding_polygon=bounding,
    )
    return DiagramSpec(
        kind="section_cut",
        selector=SlabSelector(component="Story 3 base shear"),
        style=SectionCutStyle(cut=cut, show_filter_initially=True),
        label="Story 3 base shear",
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


def test_deserialize_spec_retired_deformed_shape_raises():
    """A retired kind (ADR 0058 S4) raises with a migration hint, not a
    bare 'unknown kind' — so the catch-and-skip drops just that spec."""
    with pytest.raises(KeyError, match="retired"):
        deserialize_spec({
            "kind": "deformed_shape",
            "selector": {"component": "displacement"},
            "style": {},
        })


def test_legacy_session_with_deformed_shape_drops_it_keeps_rest():
    """ADR 0058 S4 — a legacy session carrying a retired
    ``deformed_shape`` diagram loads with the spec dropped and the rest
    of its hierarchy intact (the retirement is fail-soft, no schema
    bump)."""
    payload = {
        "schema_version": 1,
        "results_path": "/x/y.h5",
        "fem_snapshot_id": None,
        "saved_at": "",
        "diagrams": [
            serialize_spec(_make_contour_spec()),
            {"kind": "deformed_shape", "selector": {"component": "displacement"}},
            serialize_spec(_make_line_force_spec()),
        ],
    }
    session = deserialize_session(payload)
    # The retired deformed_shape spec is dropped — but its slot stays
    # open as a None sentinel so layer_indices keep pointing at the
    # right specs (ADR 0084 D6).
    assert len(session.diagrams) == 3
    assert session.diagrams[1] is None
    live = [d for d in session.diagrams if d is not None]
    assert len(live) == 2
    assert {d.kind for d in live} == {"contour", "line_force"}


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
    assert len(session.diagrams) == 3
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
    # 2 of 3 survived, each still at its original index — the corrupt
    # one leaves a None hole rather than shifting the third spec up.
    assert len(session.diagrams) == 3
    assert session.diagrams[0].kind == "contour"
    assert session.diagrams[1] is None
    assert session.diagrams[2].kind == "line_force"


def _composition_membership(session) -> dict[tuple[str, str], tuple[str, ...]]:
    """Resolve every composition's layer_indices the way
    ``ResultsViewer._apply_session`` does: positional lookup, bounds
    check, skip holes. Returns ``{(geometry, composition): kinds}``."""
    return {
        (g.name, c.name): tuple(
            session.diagrams[i].kind
            for i in c.layer_indices
            if 0 <= i < len(session.diagrams)
            and session.diagrams[i] is not None
        )
        for g in session.geometries
        for c in g.compositions
    }


def test_dropped_spec_keeps_later_layers_in_their_composition(tmp_path: Path):
    """ADR 0084 D6 — a spec that fails to deserialize leaves a hole, not
    a gap: ``layer_indices`` are positional, so compacting the list would
    re-home every later layer into the wrong composition."""
    payload = serialize_session(
        specs=[
            _make_contour_spec(),
            _make_line_force_spec(),
            _make_spring_spec(),
        ],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=[GeometrySnapshot(
            id="g0",
            name="Geometry 1",
            active_composition_id="c0",
            compositions=(
                CompositionSnapshot(
                    id="c0", name="Displacements", layer_indices=(0, 1),
                ),
                CompositionSnapshot(
                    id="c1", name="Forces", layer_indices=(2,),
                ),
            ),
        )],
    )
    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    # Retire the MIDDLE spec — what a pre-ADR-0058-S4 session looks like
    # once ``deformed_shape`` left the registry.
    payload["diagrams"][1] = {
        "kind": "deformed_shape",
        "selector": {"component": "displacement"},
        "style": {},
    }

    session = deserialize_session(payload)

    # Positions survive: the hole sits where the retired spec was.
    assert len(session.diagrams) == 3
    assert session.diagrams[0].kind == "contour"
    assert session.diagrams[1] is None
    assert session.diagrams[2].kind == "spring_force"

    # And every surviving spec still resolves to its ORIGINAL
    # composition. Under compaction "Displacements" would have swallowed
    # the spring force (index 1) and "Forces" would have been empty.
    assert _composition_membership(session) == {
        ("Geometry 1", "Displacements"): ("contour",),
        ("Geometry 1", "Forces"): ("spring_force",),
    }


def test_session_of_nothing_but_holes_is_not_restorable():
    """Padding must not make an all-dropped session look restorable."""
    session = deserialize_session({
        "schema_version": SESSION_SCHEMA_VERSION,
        "results_path": "/x/y.h5",
        "fem_snapshot_id": None,
        "saved_at": "",
        "diagrams": [
            {"kind": "deformed_shape", "selector": {"component": "d"}},
        ],
    })
    assert session.diagrams == (None,)
    assert session_restorable(session) is False


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


def test_session_schema_version_is_12():
    """Sanity: the constant tracks the latest schema (bumped to 12 for
    the spatial scope box — added ``GeometrySnapshot.scope``; 11 was
    ADR 0084 D1's ``GeometrySnapshot.threshold``)."""
    assert SESSION_SCHEMA_VERSION == 12


# =====================================================================
# Threshold (ADR 0084 D1, schema v11)
# =====================================================================

def test_v11_threshold_round_trips_inside_its_geometry(tmp_path: Path):
    """The spec rides INSIDE ``GeometrySnapshot``, not in a top-level
    map keyed by geometry id — the restore path re-matches geometries
    by name because the saved ids are stale UUIDs."""
    from apeGmsh.viewers.diagrams._session import ThresholdSnapshot

    saved = save_session(
        specs=[_make_contour_spec()],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        geometries=[
            GeometrySnapshot(id="g0", name="Filtered", threshold=(
                ThresholdSnapshot(
                    component="s_xx", lo=-2.5, hi=7.5, topology="gauss",
                )
            )),
            GeometrySnapshot(id="g1", name="Plain"),
        ],
    )
    # The payload nests it under the geometry, keyed by nothing.
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["geometries"][0]["threshold"] == {
        "component": "s_xx", "lo": -2.5, "hi": 7.5, "topology": "gauss",
    }
    assert payload["geometries"][1]["threshold"] is None
    assert "thresholds" not in payload

    session = load_session(saved)
    restored = session.geometries[0].threshold
    assert restored == ThresholdSnapshot("s_xx", -2.5, 7.5, "gauss")
    assert session.geometries[1].threshold is None


def test_a_v10_session_loads_with_no_threshold(tmp_path: Path):
    """Legacy sessions carry no ``threshold`` key — they must read as
    "no threshold" and must not crash."""
    payload = {
        "schema_version": 10,
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
                "stage_id": None,
                "visible": True,
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_contour_spec())],
    }
    target = tmp_path / "v10.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    session = load_session(target)
    assert session.schema_version == 10
    assert session.geometries[0].threshold is None
    # …and the rest of the geometry still loads.
    assert session.geometries[0].deform_scale == pytest.approx(5.0)


def test_a_malformed_threshold_degrades_to_none_not_a_lost_geometry(
    tmp_path: Path,
):
    """A half-written threshold must not take its geometry down with
    it — ``_deserialize_geometry`` failing would drop the whole row and
    slide every later ``layer_indices`` reference."""
    payload = {
        "schema_version": 11,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "geometries": [
            {
                "id": "g0",
                "name": "Geometry 1",
                "threshold": {"component": "s_xx"},      # no lo / hi
                "active_composition_id": None,
                "compositions": [],
            },
        ],
        "diagrams": [serialize_spec(_make_contour_spec())],
    }
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    session = load_session(target)
    assert len(session.geometries) == 1
    assert session.geometries[0].threshold is None


# =====================================================================
# Colour-scale placement (ADR 0081 L3, schema v8)
# =====================================================================

def test_legend_placement_round_trips(tmp_path):
    """A scale the user placed comes back where they left it."""
    from apeGmsh.viewers.diagrams._session import (
        LegendSnapshot, load_session, save_session,
    )

    placed = LegendSnapshot(
        geometry="", component="stress_vm", vertical=False, visible=False,
        fmt="%.4f", font_scale=1.75, slot=None, anchor=(0.62, 0.71),
    )
    docked = LegendSnapshot(geometry="Geometry 2", component="stress_vm")
    out = save_session(
        specs=[], results_path=tmp_path / "r.h5", fem_snapshot_id=None,
        legends=[placed, docked], target_path=tmp_path / "s.json",
    )

    back = load_session(out).legends
    assert back == (placed, docked)


def test_a_legacy_session_restores_with_no_legend_placement(tmp_path):
    """v7 and earlier carry no legends block; that is not an error."""
    import json

    from apeGmsh.viewers.diagrams._session import load_session

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schema_version": 7, "results_path": str(tmp_path / "r.h5"),
        "fem_snapshot_id": None, "saved_at": "", "diagrams": [],
    }), encoding="utf-8")
    assert load_session(path).legends == ()


# =====================================================================
# Section cut — spec + session round-trip with model_h5
# =====================================================================

def test_section_cut_spec_round_trip_bounded():
    spec = _make_section_cut_spec(with_bounding=True)
    restored = deserialize_spec(serialize_spec(spec))
    assert restored.kind == "section_cut"
    assert isinstance(restored.style, SectionCutStyle)
    cut = restored.style.cut
    assert isinstance(cut, SectionCutDef)
    # Plane fields survive the JSON tuple → list → tuple round-trip
    # because SectionCutDef.__post_init__ coerces.
    assert cut.plane_point == (0.0, 0.0, 2.5)
    assert cut.plane_normal == (0.0, 0.0, 1.0)
    assert cut.element_ids == (10, 11, 12)
    assert cut.side == "positive"
    assert cut.label == "Story 3 base shear"
    # Bounding polygon survives — tuple of (x, y, z) triples.
    assert cut.bounding_polygon == (
        (0.0, 0.0, 2.5), (1.0, 0.0, 2.5),
        (1.0, 1.0, 2.5), (0.0, 1.0, 2.5),
    )
    # Style flag carries through too.
    assert restored.style.show_filter_initially is True


def test_section_cut_spec_round_trip_unbounded():
    spec = _make_section_cut_spec(with_bounding=False, side="negative")
    restored = deserialize_spec(serialize_spec(spec))
    cut = restored.style.cut
    assert cut.bounding_polygon is None
    assert cut.side == "negative"


def test_session_carries_model_h5(tmp_path: Path):
    """``serialize_session(model_h5=...)`` round-trips through JSON."""
    h5 = tmp_path / "model.h5"
    payload = serialize_session(
        specs=[_make_section_cut_spec(with_bounding=True)],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id="snap",
        model_h5=h5,
    )
    # Serialized as a string.
    assert payload["model_h5"] == str(h5)
    session = deserialize_session(payload)
    assert session.model_h5 == str(h5)


def test_save_session_with_section_cut(tmp_path: Path):
    h5 = tmp_path / "model.h5"
    saved = save_session(
        specs=[_make_section_cut_spec(with_bounding=True)],
        results_path=tmp_path / "run.h5",
        fem_snapshot_id=None,
        model_h5=h5,
    )
    session = load_session(saved)
    assert len(session.diagrams) == 1
    assert session.diagrams[0].kind == "section_cut"
    assert session.model_h5 == str(h5)


def test_v3_session_without_model_h5_loads_with_none(tmp_path: Path):
    """Pre-v4 saves predate ``model_h5`` — they load with the field None."""
    payload = {
        "schema_version": 3,
        "results_path": str(tmp_path / "run.h5"),
        "fem_snapshot_id": None,
        "saved_at": "",
        "diagrams": [serialize_spec(_make_contour_spec())],
        # No model_h5 key.
    }
    target = tmp_path / "v3.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    session = load_session(target)
    assert session.model_h5 is None
    assert len(session.diagrams) == 1
