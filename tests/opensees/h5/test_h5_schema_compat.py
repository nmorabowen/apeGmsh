"""Schema-compatibility tests for the bridge model.h5 archive.

This file exercises the reference reader's forward-looking
schema-major handling: accept the current major, refuse mismatched
ones, refuse malformed files, and walk the post-write file via
``validate()`` to confirm structural invariants hold.

It is named for the *role* (schema-version compatibility), not for a
specific major, so it does not need to rename every time the major
bumps.  Phase 8.4 (the namespace reshuffle) renamed this from
``test_h5_schema_v1.py``.
"""
from __future__ import annotations

import inspect
import re
import warnings
from typing import Any

import h5py
import pytest

from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import (
    H5Model,
    MalformedH5Error,
    SchemaVersionError,
)
from apeGmsh.opensees._internal.tag_resolution import set_element_nodes


def _open(path: str) -> H5Model:
    return h5_reader.open(path)


def test_reader_accepts_current_schema(tmp_path: Any) -> None:
    e = H5Emitter()
    e.model(ndm=2, ndf=3)
    out = tmp_path / "ok.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        assert m.schema_version.startswith("2.")
        assert m.meta()["ndm"] == 2


def test_reader_refuses_wrong_major(tmp_path: Any) -> None:
    e = H5Emitter(schema_version="3.0.0")
    out = tmp_path / "wrong_major.h5"
    e.write(str(out))
    with pytest.raises(SchemaVersionError) as exc:
        _open(str(out))
    assert "major 3" in str(exc.value) or "major" in str(exc.value)


def test_reader_refuses_missing_meta(tmp_path: Any) -> None:
    out = tmp_path / "no_meta.h5"
    with h5py.File(out, "w") as f:
        f.create_group("nothing_useful")
    with pytest.raises(MalformedH5Error):
        _open(str(out))


def test_reader_refuses_empty_schema_version(tmp_path: Any) -> None:
    out = tmp_path / "no_version.h5"
    with h5py.File(out, "w") as f:
        f.create_group("meta")
    with pytest.raises(MalformedH5Error):
        _open(str(out))


def test_reader_validate_finds_no_violations_in_complete_model(
    tmp_path: Any,
) -> None:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial(
        "Steel02", 1, 420.0e6, 200.0e9, 0.01, 20.0, 0.925, 0.15,
    )
    e.uniaxialMaterial(
        "Concrete02", 2, -30.0e6, -0.002, -25.0e6, -0.006,
        0.1, 2.5e6, 200.0e6,
    )
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    e.fiber(0.1, 0.0, 0.001, 1)
    e.section_close()
    e.timeSeries("Linear", 1, "-factor", 1.0)
    e.pattern_open("Plain", 1, 1)
    e.load(10, 1.0, 0.0, 0.0)
    e.pattern_close()
    out = tmp_path / "complete.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        violations = m.validate()
        assert violations == [], violations


def test_reader_validate_detects_dangling_material_ref(tmp_path: Any) -> None:
    """Hand-craft a file with a bad material_ref and confirm validate catches it."""
    out = tmp_path / "dangling.h5"
    # First write a valid file ...
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.fiber(0.1, 0.0, 0.001, 1)
    e.section_close()
    e.write(str(out))
    # ... then mutate the fiber's material_ref to a dangling path.
    with h5py.File(out, "a") as f:
        ds = f["opensees/sections/Fiber_1/fibers"]
        rows = ds[:]
        rows[0]["material_ref"] = b"/opensees/materials/uniaxial/Nonexistent_99"
        ds[...] = rows
    with _open(str(out)) as m:
        violations = m.validate()
        assert any("Nonexistent_99" in v for v in violations)


def test_reader_accessors_return_attrs(tmp_path: Any) -> None:
    """Bridge-only file: ``/materials`` / ``/transforms`` populated;
    ``/elements`` is broker territory post-Phase-8.5 and therefore
    empty in standalone H5Emitter output."""
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    e.geomTransf("PDelta", 1, 0.0, 0.0, 1.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 1, 1, 2, 1, 1)
    out = tmp_path / "accessors.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        # Phase 8 / ADR 0019 — typed accessors return records.
        by_family = m.materials_by_family()
        assert "uniaxial" in by_family
        assert any(
            mat.type_token == "Steel02" and mat.tag == 1
            for mat in by_family["uniaxial"]
        )
        tx = m.transforms()
        assert any(t.type_token == "PDelta" and t.tag == 1 for t in tx)
        # `/elements` is broker-owned; bridge no longer writes it.
        assert m.elements() == {}


# ===========================================================================
# Phase 7a — Per-zone schema versioning; the reader gate is a floor per
# zone since ADR 0113 (it retired ADR 0023's two-version window).
# Tests below exercise the central helpers in
# :mod:`apeGmsh.opensees._internal.schema_version` plus the read/write
# wiring across the three zones (neutral, opensees, results).
# ===========================================================================

from pathlib import Path

import numpy as np

from apeGmsh.opensees._internal.schema_version import (
    _ZONE_KEY,
    ASSEMBLY,
    ASSEMBLY_KEY,
    ENVELOPE_KEY,
    GEOMETRY,
    NEUTRAL,
    NEUTRAL_KEY,
    OPENSEES,
    OPENSEES_KEY,
    PROVENANCE,
    RESULTS,
    RESULTS_KEY,
    SchemaVersion,
    SchemaVersionError as _PerZoneSchemaError,
    read_zone_version,
    reader_floor,
    reader_version,
    validate_zone_version,
)
from tests.fixtures.schema import (
    GEOMETRY_FLOOR,
    NEUTRAL_FLOOR,
    OPENSEES_FLOOR,
    OPENSEES_PRIOR_MINOR,
    PROVENANCE_FLOOR,
    RESULTS_FLOOR,
)


def _build_composed_results(tmp_path: Path):
    """Build a Composed-file results.h5 + return its path."""
    from apeGmsh.results.writers import NativeWriter
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5

    model_path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "composed.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(name="s", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()
    return results_path, model_path


def test_per_zone_keys_written_on_compose(tmp_path: Any) -> None:
    """A model.h5 written via the composer carries both per-zone keys
    plus the envelope (ADR 0023 §"Three per-zone version stamps")."""
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5

    model_path, _ = build_simple_frame_h5(tmp_path)
    with h5py.File(model_path, "r") as f:
        keys = set(f["meta"].attrs.keys())
    assert ENVELOPE_KEY in keys
    assert NEUTRAL_KEY in keys
    assert OPENSEES_KEY in keys
    # ADR 0117 D5: only Assembly.h5 writes /assembly and its key.
    assert ASSEMBLY_KEY not in keys


def test_assembly_key_is_its_own_and_never_the_envelope() -> None:
    """ADR 0117 D5 / ADR 0112 D2: ``/assembly`` has its own ``/meta`` key,
    and a file without it has no assembly version even if it carries the
    legacy envelope (the zone postdates the envelope)."""
    assert _ZONE_KEY[ASSEMBLY] == ASSEMBLY_KEY == "assembly_schema_version"
    attrs = {ENVELOPE_KEY: str(reader_version(NEUTRAL))}
    assert read_zone_version(attrs, ASSEMBLY) is None
    stamped = {ASSEMBLY_KEY: str(reader_version(ASSEMBLY))}
    assert read_zone_version(stamped, ASSEMBLY) == reader_version(ASSEMBLY)


def test_per_zone_keys_written_on_native_results(tmp_path: Any) -> None:
    """A composed results.h5 carries all four stamps at the root:
    envelope, results, neutral (forwarded), opensees (forwarded)."""
    results_path, _ = _build_composed_results(tmp_path)
    with h5py.File(results_path, "r") as f:
        keys = set(f.attrs.keys())
    assert ENVELOPE_KEY in keys
    assert RESULTS_KEY in keys
    assert NEUTRAL_KEY in keys
    assert OPENSEES_KEY in keys


def test_legacy_envelope_only_file_reads_via_fallback(tmp_path: Any) -> None:
    """A file with only ``schema_version`` (no per-zone keys) reads via
    the envelope fallback in :func:`read_zone_version`."""
    out = tmp_path / "envelope_only.h5"
    with h5py.File(out, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "2.6.0"
    with h5py.File(out, "r") as f:
        attrs = f["meta"].attrs
        v = read_zone_version(attrs, NEUTRAL)
        assert v == SchemaVersion(2, 6, 0)
        v_no_fallback = read_zone_version(attrs, NEUTRAL, envelope_fallback=False)
        assert v_no_fallback is None


# ---------------------------------------------------------------------------
# The floor (ADR 0113 (#1303)): accept iff same major and
# floor.minor <= file.minor <= reader.minor, patch ignored. Grid-tested at
# both edges for every registered zone, so a zone added to _ZONE_KEY is
# covered (and must have a floor) the day it lands.
# ---------------------------------------------------------------------------

_PATCHES = (0, 1, 99)


def _accepted_minors(zone: str) -> list[int]:
    floor, reader = reader_floor(zone), reader_version(zone)
    edges = {floor.minor, floor.minor + 1, reader.minor - 1, reader.minor}
    return sorted(m for m in edges if floor.minor <= m <= reader.minor)


@pytest.mark.parametrize("zone", tuple(_ZONE_KEY))
def test_floor_accepts_every_edge_minor_and_patch(zone: str) -> None:
    reader = reader_version(zone)
    for minor in _accepted_minors(zone):
        for patch in _PATCHES:
            validate_zone_version(
                SchemaVersion(reader.major, minor, patch), reader, zone=zone,
            )


@pytest.mark.parametrize("zone", tuple(_ZONE_KEY))
def test_floor_refuses_one_minor_below(zone: str) -> None:
    floor, reader = reader_floor(zone), reader_version(zone)
    if floor.minor == 0:
        # Nothing lies below X.0 on this major: the newest file of the
        # previous major is the edge, and it is refused as another major.
        with pytest.raises(_PerZoneSchemaError, match="different major"):
            validate_zone_version(
                SchemaVersion(floor.major - 1, 99, 99), reader, zone=zone,
            )
        return
    for patch in _PATCHES:
        with pytest.raises(_PerZoneSchemaError) as exc:
            validate_zone_version(
                SchemaVersion(floor.major, floor.minor - 1, patch), reader,
                zone=zone,
            )
        assert "too old" in str(exc.value)


@pytest.mark.parametrize("zone", tuple(_ZONE_KEY))
def test_floor_refuses_newer_minor(zone: str) -> None:
    """INV-4: refusing a newer minor is safer than silent tolerance."""
    reader = reader_version(zone)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(reader.major, reader.minor + 1, 0), reader, zone=zone,
        )
    assert "newer than this reader" in str(exc.value)


@pytest.mark.parametrize("zone", tuple(_ZONE_KEY))
def test_floor_refuses_other_majors(zone: str) -> None:
    floor, reader = reader_floor(zone), reader_version(zone)
    others = [
        SchemaVersion(reader.major + 1, floor.minor, 0),
        SchemaVersion(reader.major - 1, reader.minor, 0),
    ]
    for file_version in others:
        with pytest.raises(_PerZoneSchemaError) as exc:
            validate_zone_version(file_version, reader, zone=zone)
        assert "different major" in str(exc.value)


@pytest.mark.parametrize("zone", tuple(_ZONE_KEY))
def test_refusal_names_both_ends_of_the_range(zone: str) -> None:
    floor, reader = reader_floor(zone), reader_version(zone)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(reader.major, reader.minor + 1, 0), reader, zone=zone,
        )
    msg = str(exc.value)
    assert f"{zone}_schema_version={reader.major}.{reader.minor + 1}.0" in msg
    assert (
        f"supports {floor.major}.{floor.minor}.x\u2013"
        f"{reader.major}.{reader.minor}.x" in msg
    )
    assert "Upgrade apeGmsh" in msg


def test_too_old_refusal_says_regenerate() -> None:
    floor, reader = reader_floor(NEUTRAL), reader_version(NEUTRAL)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(floor.major, floor.minor - 1, 0), reader, zone=NEUTRAL,
        )
    assert "too old" in str(exc.value)
    assert "Regenerate" in str(exc.value)


def test_reader_below_its_floor_is_a_caller_error() -> None:
    """A reader below the zone's floor is impossible: fail loud, not refuse."""
    floor = reader_floor(OPENSEES)
    with pytest.raises(ValueError) as exc:
        validate_zone_version(
            floor, SchemaVersion(floor.major, floor.minor - 1, 0), zone=OPENSEES,
        )
    assert not isinstance(exc.value, _PerZoneSchemaError)


def test_floor_unknown_zone_refused() -> None:
    with pytest.raises(ValueError, match="unknown zone 'sequence'"):
        reader_floor("sequence")
    with pytest.raises(ValueError, match="unknown zone 'sequence'"):
        validate_zone_version(
            SchemaVersion(1, 0, 0), SchemaVersion(1, 0, 0), zone="sequence",
        )


def test_validate_per_zone_independently() -> None:
    """INV-3 — the zones' ranges are conjunctive but NOT coupled: each
    zone's floor stamp validates against its own reader."""
    validate_zone_version(
        reader_floor(OPENSEES), reader_version(OPENSEES), zone=OPENSEES,
    )
    validate_zone_version(
        reader_floor(NEUTRAL), reader_version(NEUTRAL), zone=NEUTRAL,
    )
    with pytest.raises(_PerZoneSchemaError):
        # The neutral floor is below the opensees floor.
        validate_zone_version(
            reader_floor(NEUTRAL), reader_version(OPENSEES), zone=OPENSEES,
        )


def test_reader_version_reflects_writer_constants() -> None:
    """``reader_version`` / ``reader_floor`` match the writer constants exactly.

    Single source of truth — the reader's per-zone version and floor are
    sourced from the writer module's constants; they cannot drift. The
    floors are pinned to ``tests/fixtures/schema.py`` too, and sit at or
    below their versions on the same major.
    """
    from apeGmsh.mesh._femdata_h5_io import (
        NEUTRAL_SCHEMA_FLOOR,
        NEUTRAL_SCHEMA_VERSION,
    )
    from apeGmsh.opensees._internal.schema_version import (
        ASSEMBLY_SCHEMA_FLOOR,
        ASSEMBLY_SCHEMA_VERSION,
        GEOMETRY_SCHEMA_FLOOR,
        GEOMETRY_SCHEMA_VERSION,
        PROVENANCE_SCHEMA_FLOOR,
        PROVENANCE_SCHEMA_VERSION,
    )
    from apeGmsh.opensees.emitter.h5 import SCHEMA_FLOOR as OPENSEES_SCHEMA_FLOOR
    from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION as OPENSEES_VERSION
    from apeGmsh.results.schema._versions import (
        RESULTS_SCHEMA_FLOOR,
        RESULTS_SCHEMA_VERSION,
    )

    assert reader_version(NEUTRAL) == SchemaVersion.parse(NEUTRAL_SCHEMA_VERSION)
    assert reader_version(OPENSEES) == SchemaVersion.parse(OPENSEES_VERSION)
    assert reader_version(RESULTS) == SchemaVersion.parse(RESULTS_SCHEMA_VERSION)

    writer_floors = {
        NEUTRAL: (NEUTRAL_SCHEMA_FLOOR, NEUTRAL_FLOOR),
        OPENSEES: (OPENSEES_SCHEMA_FLOOR, OPENSEES_FLOOR),
        RESULTS: (RESULTS_SCHEMA_FLOOR, RESULTS_FLOOR),
        GEOMETRY: (GEOMETRY_SCHEMA_FLOOR, GEOMETRY_FLOOR),
        PROVENANCE: (PROVENANCE_SCHEMA_FLOOR, PROVENANCE_FLOOR),
        # ADR 0117 D5: the zone's floor is its first version, so the
        # writer constant is its own fixture (no tests/fixtures/schema.py
        # row until its first minor bump).
        ASSEMBLY: (ASSEMBLY_SCHEMA_FLOOR, ASSEMBLY_SCHEMA_VERSION),
    }
    assert set(writer_floors) == set(_ZONE_KEY)
    for zone, (writer_floor, fixture_floor) in writer_floors.items():
        floor = reader_floor(zone)
        assert floor == SchemaVersion.parse(writer_floor) == SchemaVersion.parse(
            fixture_floor
        ), zone
        reader = reader_version(zone)
        assert floor.major == reader.major and floor.minor <= reader.minor, zone
    # The new zones' floor is their first version: geometry has only that
    # one; provenance gained the additive 1.1.0 (records/origin, V2d
    # #1378) and its floor stays at 1.0.0.
    assert GEOMETRY_SCHEMA_FLOOR == GEOMETRY_SCHEMA_VERSION
    assert PROVENANCE_SCHEMA_FLOOR == PROVENANCE_FLOOR
    assert SchemaVersion.parse(PROVENANCE_SCHEMA_VERSION).minor >= 1


# ---------------------------------------------------------------------------
# The floor through every reader (ADR 0113 (#1303)). A file stamped anywhere
# from the zone's floor to the current minor opens; one minor below refuses.
# The stamps are restamps of a current file: the gate is under test here,
# the real old files are the corpus's job (#1303 PR-3).
# ---------------------------------------------------------------------------

#: Neutral stamps the retired two-version window refused: the floor itself, a
#: mid-history minor, and the minor #1300 (2.34.0) expired.
_OLD_NEUTRAL_STAMPS = (NEUTRAL_FLOOR, "2.12.0", "2.32.0")


def _below(floor: str) -> str:
    v = SchemaVersion.parse(floor)
    return f"{v.major}.{v.minor - 1}.0"


def _restamp_neutral(meta: Any, stamp: str) -> None:
    meta.attrs[ENVELOPE_KEY] = stamp
    meta.attrs[NEUTRAL_KEY] = stamp


def _neutral_file(tmp_path: Path, stamp: str) -> Path:
    from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

    out = tmp_path / f"neutral_{stamp}.h5"
    build_simple_frame_fem().to_h5(str(out))
    with h5py.File(out, "r+") as f:
        _restamp_neutral(f["meta"], stamp)
    return out


@pytest.mark.parametrize("stamp", _OLD_NEUTRAL_STAMPS)
def test_old_neutral_stamp_opens_through_from_h5(tmp_path: Path, stamp: str) -> None:
    """A neutral file stamped 2.12.0 refused under the window; it opens now."""
    from apeGmsh.mesh.FEMData import FEMData

    fem = FEMData.from_h5(str(_neutral_file(tmp_path, stamp)))
    assert len(fem.nodes.ids) > 0


def test_neutral_stamp_below_floor_refuses_through_from_h5(tmp_path: Path) -> None:
    from apeGmsh.mesh.FEMData import FEMData

    path = _neutral_file(tmp_path, _below(NEUTRAL_FLOOR))
    with pytest.raises(_PerZoneSchemaError, match="too old"):
        FEMData.from_h5(str(path))


@pytest.mark.parametrize("stamp", _OLD_NEUTRAL_STAMPS)
def test_old_neutral_stamp_passes_compose_span(tmp_path: Path, stamp: str) -> None:
    from apeGmsh.mesh._compose import _compute_source_span

    _compute_source_span(_neutral_file(tmp_path, stamp))


def test_neutral_stamp_below_floor_refuses_compose_span(tmp_path: Path) -> None:
    from apeGmsh.mesh._compose import _compute_source_span

    path = _neutral_file(tmp_path, _below(NEUTRAL_FLOOR))
    with pytest.raises(_PerZoneSchemaError, match="too old"):
        _compute_source_span(path)


@pytest.mark.parametrize("stamp", [OPENSEES_FLOOR, OPENSEES_PRIOR_MINOR])
def test_opensees_stamp_at_floor_opens_through_h5_reader(
    tmp_path: Path, stamp: str,
) -> None:
    out = tmp_path / "bridge.h5"
    e = H5Emitter(schema_version=stamp)
    e.model(ndm=3, ndf=6)
    e.write(str(out))
    with h5_reader.open(str(out)) as m:
        assert m.schema_version == stamp


@pytest.mark.parametrize("patch", _PATCHES)
def test_opensees_stamp_below_floor_refuses_through_h5_reader(
    tmp_path: Path, patch: int,
) -> None:
    """One minor below the opensees floor refuses at every patch, and the
    refusal names the floor (2.12.0 since the ADR 0113 D3 evidence gate
    raised it past the 2.11 era, whose files never open: #1329)."""
    floor, reader = reader_floor(OPENSEES), reader_version(OPENSEES)
    stamp = f"{floor.major}.{floor.minor - 1}.{patch}"
    out = tmp_path / "bridge_old.h5"
    e = H5Emitter(schema_version=stamp)
    e.model(ndm=3, ndf=6)
    e.write(str(out))
    with pytest.raises(SchemaVersionError) as exc:
        h5_reader.open(str(out))
    msg = str(exc.value)
    assert f"opensees_schema_version={stamp}: too old" in msg
    assert (
        f"supports {floor.major}.{floor.minor}.x–"
        f"{reader.major}.{reader.minor}.x" in msg
    )


def _restamp_results(path: Path, *, neutral: str, opensees: str, results: str) -> None:
    with h5py.File(path, "r+") as f:
        f.attrs[RESULTS_KEY] = results
        f.attrs[NEUTRAL_KEY] = neutral
        f.attrs[OPENSEES_KEY] = opensees
        _restamp_neutral(f["model/meta"], neutral)
        f["model/meta"].attrs[OPENSEES_KEY] = opensees


def test_results_with_floor_stamped_zones_open_through_native_reader(
    tmp_path: Path,
) -> None:
    """The embedded /model and /opensees at their floors no longer expire a
    results file (RS6)."""
    from apeGmsh.results.readers._native import NativeReader

    results_path, _ = _build_composed_results(tmp_path)
    _restamp_results(
        results_path, neutral=NEUTRAL_FLOOR, opensees=OPENSEES_FLOOR,
        results=RESULTS_FLOOR,
    )
    # Warn-as-contract: a file inside every floor opens silently; the D9
    # warning below fires only on a zone the reader cannot open.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reader = NativeReader(results_path)
    with reader:
        assert reader.unavailable_zones == {}
        assert reader.fem() is not None
        assert reader.opensees_model() is not None


# ADR 0113 D9 / INV-11: a results file outlives its embedded model zones.
# An embedded /model (neutral) or /opensees below its floor no longer
# expires the file: /stages opens read-only and flagged, model access
# refuses. An embedded zone NEWER than the reader still refuses (INV-4, D2).


def _flagged_results(tmp_path: Path, zone: str, stamp: str) -> Path:
    results_path, _ = _build_composed_results(tmp_path)
    stamps = {NEUTRAL: NEUTRAL_FLOOR, OPENSEES: OPENSEES_FLOOR}
    stamps[zone] = stamp
    _restamp_results(
        results_path, neutral=stamps[NEUTRAL], opensees=stamps[OPENSEES],
        results=RESULTS_FLOOR,
    )
    return results_path


@pytest.mark.parametrize("zone", [NEUTRAL, OPENSEES])
def test_results_with_embedded_zone_below_floor_opens_stages_flagged(
    tmp_path: Path, zone: str,
) -> None:
    """D9: the embedded zone below its floor is flagged, /stages reads."""
    from apeGmsh.results.readers._native import NativeReader
    from apeGmsh.results.readers._protocol import ResultLevel

    floor = {NEUTRAL: NEUTRAL_FLOOR, OPENSEES: OPENSEES_FLOOR}[zone]
    results_path = _flagged_results(tmp_path, zone, _below(floor))
    with pytest.warns(UserWarning, match=f"{zone}_schema_version") as rec:
        reader = NativeReader(results_path)
    with reader:
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "too old" in msg and str(results_path) in msg
        # Flagged: the reader names the zone and why.
        assert set(reader.unavailable_zones) == {zone}
        assert "too old" in reader.unavailable_zones[zone]
        # /stages opens read-only.
        (stage,) = reader.stages()
        assert stage.kind == "static" and stage.n_steps == 1
        assert reader.time_vector(stage.id).tolist() == [0.0]
        assert reader.available_components(stage.id, ResultLevel.NODES) == [
            "displacement_z",
        ]
        slab = reader.read_nodes(stage.id, "displacement_z")
        assert slab.values.shape[0] == 1
        # Model access refuses, naming the zone.
        with pytest.raises(_PerZoneSchemaError, match=f"{zone}_schema_version"):
            reader.opensees_model()
        if zone == NEUTRAL:
            with pytest.raises(_PerZoneSchemaError, match="neutral_schema_version"):
                reader.fem()
        else:
            # The neutral zone is inside its floor: the FEM still reads.
            assert reader.fem() is not None


def test_results_with_embedded_neutral_of_an_older_major_is_flagged(
    tmp_path: Path,
) -> None:
    """Below the floor includes the previous major: the reader cannot open
    that /model either, and /stages do not depend on it."""
    from apeGmsh.results.readers._native import NativeReader

    floor = SchemaVersion.parse(NEUTRAL_FLOOR)
    results_path = _flagged_results(
        tmp_path, NEUTRAL, f"{floor.major - 1}.99.0",
    )
    with pytest.warns(UserWarning, match="different major"):
        reader = NativeReader(results_path)
    with reader:
        assert set(reader.unavailable_zones) == {NEUTRAL}
        assert len(reader.stages()) == 1
        with pytest.raises(_PerZoneSchemaError, match="neutral_schema_version"):
            reader.fem()


@pytest.mark.parametrize("zone", [NEUTRAL, OPENSEES])
def test_results_with_embedded_zone_newer_than_reader_refuses(
    tmp_path: Path, zone: str,
) -> None:
    """D2 / INV-4: D9 covers old zones only; a newer embedded zone refuses."""
    from apeGmsh.results.readers._native import NativeReader

    reader = reader_version(zone)
    results_path = _flagged_results(
        tmp_path, zone, f"{reader.major}.{reader.minor + 1}.0",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_PerZoneSchemaError, match=f"{zone}_schema_version"):
            NativeReader(results_path)


def test_results_zone_below_its_floor_still_refuses(tmp_path: Path) -> None:
    """D9 is about the embedded zones; the results zone itself is validated
    as before (its floor is 1.0.0, so the edge is the previous major)."""
    from apeGmsh.results.readers._native import NativeReader

    results_path, _ = _build_composed_results(tmp_path)
    floor = SchemaVersion.parse(RESULTS_FLOOR)
    _restamp_results(
        results_path, neutral=NEUTRAL_FLOOR, opensees=OPENSEES_FLOOR,
        results=f"{floor.major - 1}.0.0",
    )
    with pytest.raises(_PerZoneSchemaError, match="results_schema_version"):
        NativeReader(results_path)


def test_envelope_back_compat_preserves_existing_files(tmp_path: Any) -> None:
    """Files with only the envelope key still read via the fallback.

    Synthesize an envelope-only file at an in-window version; the
    reader accepts it without per-zone keys.
    """
    out = tmp_path / "envelope.h5"
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.write(str(out))
    # Strip the per-zone OpenSees key to simulate a pre-Phase-7a file.
    with h5py.File(out, "a") as f:
        if OPENSEES_KEY in f["meta"].attrs:
            del f["meta"].attrs[OPENSEES_KEY]
    # Re-open: envelope fallback resolves the opensees version from
    # the surviving ``schema_version``; the reader accepts it.
    with h5_reader.open(str(out)) as m:
        assert m.schema_version.startswith("2.")


def test_results_schema_version_independent_of_opensees(tmp_path: Any) -> None:
    """Each zone's version check is independent of the others.

    The results-zone floor check applies to the results version
    only; the opensees-zone check applies to the opensees version
    only — they don't share a major (INV-3).
    """
    reader_neutral = reader_version(NEUTRAL)
    reader_opensees = reader_version(OPENSEES)
    reader_results = reader_version(RESULTS)
    # File at the prior-minor in each zone (still inside the window).
    neutral_prior = SchemaVersion(
        reader_neutral.major, reader_neutral.minor - 1, 0,
    )
    opensees_prior = SchemaVersion(
        reader_opensees.major, reader_opensees.minor - 1, 0,
    )
    validate_zone_version(neutral_prior, reader_neutral, zone=NEUTRAL)
    validate_zone_version(opensees_prior, reader_opensees, zone=OPENSEES)
    validate_zone_version(
        SchemaVersion(1, 1, 0), reader_results, zone=RESULTS,
    )


def test_composed_file_validates_all_three_zones(tmp_path: Any) -> None:
    """Open a Composed results.h5 at the current per-zone writer versions
    (neutral, opensees, results 1.1.0); the NativeReader's __init__
    validation succeeds for all three zones with no warnings raised."""
    from apeGmsh.results.readers._native import NativeReader

    results_path, _ = _build_composed_results(tmp_path)
    reader = NativeReader(results_path)
    try:
        # All three zones validated at __init__; assert the file has
        # the expected keys at root.
        with h5py.File(results_path, "r") as f:
            attrs = dict(f.attrs)
        assert read_zone_version(attrs, RESULTS) is not None
        assert read_zone_version(attrs, NEUTRAL) is not None
        assert read_zone_version(attrs, OPENSEES) is not None
    finally:
        reader.close()


def test_single_stamp_file_fallback_lineage_is_envelope(tmp_path: Any) -> None:
    """A file carrying only the envelope key validates ALL zones via the
    envelope-fallback rule — pre-Phase-7a single-stamp files keep working."""
    out = tmp_path / "single_stamp.h5"
    with h5py.File(out, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "2.6.0"
    with h5py.File(out, "r") as f:
        attrs = f["meta"].attrs
        # All three zones resolve to the same envelope value.
        assert read_zone_version(attrs, NEUTRAL) == SchemaVersion(2, 6, 0)
        assert read_zone_version(attrs, OPENSEES) == SchemaVersion(2, 6, 0)
        assert read_zone_version(attrs, RESULTS) == SchemaVersion(2, 6, 0)


# ---------------------------------------------------------------------------
# Schema 2.12.0 — ASDEmbeddedNodeElement option exposure (ADR 0035)
# (2.7.0 added /opensees/constraints/; 2.8.0 renamed embeddedNode's
#  embedding_ele → cnode; 2.9.0 added /opensees/regions/ for MPCO
#  region-filtered output; 2.10.0 added /opensees/partitions/ +
#  partition_ids column for OpenSeesMP per-rank emission; 2.11.0
#  flipped the runtime-rank seam to 0-based — partition_NN/rank attr
#  and the partition_ids column values are now in [0, N-1] rather
#  than [1, N], matching OpenSeesMP::getPID(); 2.12.0 added five typed
#  columns — stiffness/stiffness_p/has_stiffness_p/rotational/pressure
#  — to /opensees/constraints/embeddedNode so the ASDEmbeddedNodeElement
#  -K/-KP/-rot/-p flags round-trip per ADR 0035.)
# ---------------------------------------------------------------------------


def test_opensees_reader_version_is_2_22_0() -> None:
    """Schema 2.22.0 — ``/opensees/bcs@mass_from_model`` marker (#1304)."""
    assert reader_version(OPENSEES) == SchemaVersion(2, 22, 0)


def test_constraints_group_present_when_emitted(tmp_path: Any) -> None:
    """``H5Emitter.equalDOF`` etc populate ``/opensees/constraints/*``."""
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.equalDOF(1, 2, 1, 2, 3)
    e.rigidLink("beam", 3, 4)
    e.rigidDiaphragm(3, 100, 1, 2, 3, 4)
    e.embeddedNode(1000, 5, 10, 1, 2)
    out = tmp_path / "constraints.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "opensees/constraints/equalDOF" in f
        assert "opensees/constraints/rigidLink" in f
        assert "opensees/constraints/rigidDiaphragm" in f
        assert "opensees/constraints/embeddedNode" in f


def test_phantom_node_tags_present_when_predicate_installed(
    tmp_path: Any,
) -> None:
    """Per S2 (ADR 0033) the phantom discriminator is the stateless
    ``set_phantom_node_tags`` predicate — ``ndf=K`` on
    ``H5Emitter.node`` is now legal for real broker nodes
    (shell-on-solid mixed-ndf models) and no longer implies
    phantom-ness on its own.  The MP-constraint emit pass pre-loads
    the complete phantom-tag set on the emitter before any node
    emission begins; the H5 emitter consults it per call."""
    from apeGmsh.opensees._internal.tag_resolution import (
        set_phantom_node_tags,
    )

    e = H5Emitter()
    e.model(ndm=3, ndf=3)
    # Pre-load the phantom-tag predicate once — order-independent.
    set_phantom_node_tags(e, {200})
    e.node(1, 0.0, 0.0, 0.0)               # regular broker node
    e.node(2, 0.0, 0.0, 1.0, ndf=6)        # real broker node with ndf override
    e.node(200, 0.0, 0.0, 2.0, ndf=6)      # phantom — classified by predicate
    out = tmp_path / "phantoms.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "opensees/constraints/phantom_node_tags" in f
        tags = f["opensees/constraints/phantom_node_tags"][:]
        assert list(int(t) for t in tags) == [200], (
            "phantom_node_tags must contain only tags in the predicate "
            "set installed via set_phantom_node_tags (ADR 0033)."
        )


# ===========================================================================
# Doc-sync ratchet — architecture/h5-schema.md vs the live writer constants
# (2026-08 housekeeping). A 2026-08 audit found the doc's zone-registry
# "Current" column frozen at neutral 2.10.0 / opensees 2.12.0 while the
# writers had moved on to 2.31.0 / 2.20.0, and the "Two zones" bullet list
# silently missing ~10 neutral-zone groups (`/mesh_selections`,
# `/partitions`, `/parts`, `/contacts`, `/interfaces`, the tie groups, …)
# that `write_neutral_zone` actually writes. Nothing failed when either
# drifted. These two tests are the ratchet: they read the live writer
# constants and the live writer source directly (never a hand-copied
# snapshot of either), so a future version bump or a new neutral-zone
# group added without a doc update fails loud here instead of drifting
# silently again.
# ===========================================================================

from pathlib import Path as _Path

_H5_SCHEMA_DOC = (
    _Path(__file__).resolve().parents[3]
    / "architecture" / "h5-schema.md"
)


def _h5_schema_doc_text() -> str:
    assert _H5_SCHEMA_DOC.is_file(), (
        f"h5-schema.md not found at {_H5_SCHEMA_DOC} — update the path "
        "if the doc moved."
    )
    return _H5_SCHEMA_DOC.read_text(encoding="utf-8")


def _zone_registry_current(text: str, zone_label: str) -> str:
    """The bolded ``**X.Y.Z**`` Current-column value for one zone-registry row.

    ``zone_label`` is the row's leading cell, e.g. ``"neutral (broker)"``.
    Matches the whole markdown-table row (one line) so the scan does not
    depend on column count or ordering, only on the row starting with
    ``| {zone_label} |`` and ending with a bolded semver cell.
    """
    row = re.search(
        rf"^\|\s*{re.escape(zone_label)}\s*\|.*\|\s*$", text, re.M,
    )
    assert row is not None, (
        f"h5-schema.md: no zone-registry row found starting with "
        f"`| {zone_label} |` — table reformatted or row renamed, update "
        "the scan (or the doc)."
    )
    version = re.search(r"\*\*(\d+\.\d+\.\d+)\*\*", row.group(0))
    assert version is not None, (
        f"h5-schema.md: zone-registry row for {zone_label!r} has no "
        "bolded **X.Y.Z** Current cell — table format drifted."
    )
    return version.group(1)


def _zone_registry_floor(text: str, zone_label: str) -> str:
    """The Floor-column ``**X.Y.Z**`` value (the row's last cell) for the
    zone-registry row whose leading cell starts with ``zone_label``."""
    row = re.search(
        rf"^\|\s*{re.escape(zone_label)}[^|]*\|.*\|\s*$", text, re.M,
    )
    assert row is not None, (
        f"h5-schema.md: no zone-registry row found starting with "
        f"`| {zone_label}` — table reformatted or row renamed, update "
        "the scan (or the doc)."
    )
    cells = [c.strip() for c in row.group(0).strip().strip("|").split("|")]
    floor = re.fullmatch(r"\*\*(\d+\.\d+\.\d+)\*\*", cells[-1])
    assert floor is not None, (
        f"h5-schema.md: zone-registry row for {zone_label!r} has no "
        "bolded **X.Y.Z** Floor cell in its last column — table format "
        "drifted."
    )
    return floor.group(1)


def test_h5_schema_doc_registry_matches_writer_constants() -> None:
    """The zone-registry's "Current" column must equal the live writer
    constants for the neutral and opensees zones, and its "Floor" column
    must equal every zone's reader floor (ADR 0113 INV-3) — not a
    hand-typed snapshot that can silently go stale."""
    from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
    from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION as OPENSEES_VERSION

    text = _h5_schema_doc_text()
    neutral_doc = _zone_registry_current(text, "neutral (broker)")
    opensees_doc = _zone_registry_current(text, "opensees (bridge)")
    assert neutral_doc == NEUTRAL_SCHEMA_VERSION, (
        f"h5-schema.md zone registry says neutral Current={neutral_doc!r} "
        f"but NEUTRAL_SCHEMA_VERSION={NEUTRAL_SCHEMA_VERSION!r} — the doc "
        "has drifted from mesh/_femdata_h5_io.py; update the table."
    )
    assert opensees_doc == OPENSEES_VERSION, (
        f"h5-schema.md zone registry says opensees Current={opensees_doc!r} "
        f"but SCHEMA_VERSION={OPENSEES_VERSION!r} — the doc has drifted "
        "from opensees/emitter/h5.py; update the table."
    )

    from apeGmsh.opensees._internal.schema_version import (
        ASSEMBLY, GEOMETRY, NEUTRAL, OPENSEES, PROVENANCE, RESULTS,
        reader_floor,
    )

    for zone, label in (
        (NEUTRAL, "neutral (broker)"),
        (OPENSEES, "opensees (bridge)"),
        (RESULTS, "results"),
        (GEOMETRY, "geometry"),
        (PROVENANCE, "provenance"),
        (ASSEMBLY, "assembly"),
    ):
        floor_doc = _zone_registry_floor(text, label)
        floor_live = str(reader_floor(zone))
        assert floor_doc == floor_live, (
            f"h5-schema.md zone registry says {label} Floor={floor_doc!r} "
            f"but reader_floor({zone!r})={floor_live!r} — the doc has "
            "drifted from the zone's *_SCHEMA_FLOOR constant; update the "
            "table."
        )


def _write_neutral_zone_group_names() -> list[str]:
    """Top-level HDF5 group/dataset names ``write_neutral_zone`` writes.

    Derived from the live source rather than hand-maintained, so a new
    ``_write_*`` group added to ``write_neutral_zone`` is picked up
    automatically instead of requiring someone to remember to update a
    parallel list here. For each ``_write_x(fem, f)`` callee in
    ``write_neutral_zone``'s own body, extracts the literal top-level
    name it creates: directly via ``f.create_group("name")`` /
    ``f.create_dataset("name", ...)``, or indirectly via a
    ``group_name="name"`` keyword forwarded to a shared helper (the
    ``/physical_groups`` / ``/labels`` pattern). A callee whose source
    matches none of these patterns fails loud asking the scan to be
    extended — it never silently drops a group from the check.
    """
    from apeGmsh.mesh import _femdata_h5_io as mod

    writer_src = inspect.getsource(mod.write_neutral_zone)
    callees = re.findall(r"\b(_write_\w+)\(fem, f\)", writer_src)
    assert callees, (
        "write_neutral_zone's body no longer matches the `_write_x(fem, "
        "f)` call pattern this scan looks for — update the regex."
    )

    names: list[str] = []
    for callee in callees:
        fn = getattr(mod, callee, None)
        assert fn is not None, (
            f"write_neutral_zone calls `{callee}` but no such attribute "
            "exists on apeGmsh.mesh._femdata_h5_io."
        )
        src = inspect.getsource(fn)
        m = (
            re.search(r'\.create_group\(\s*["\'](\w+)["\']', src)
            or re.search(r'\.create_dataset\(\s*["\'](\w+)["\']', src)
            or re.search(r'group_name\s*=\s*["\'](\w+)["\']', src)
        )
        assert m is not None, (
            f"{callee}: found no `.create_group(\"...\")` / "
            "`.create_dataset(\"...\")` / `group_name=\"...\"` literal in "
            "its source — extend the scan pattern in this test."
        )
        names.append(m.group(1))
    return names


def test_neutral_zone_group_scan_finds_the_full_writer() -> None:
    """Sanity check on the extractor itself: it must find every group
    `write_neutral_zone` is known (as of this housekeeping pass) to
    write, not just a handful — guards against the regex silently
    matching nothing or only the first few calls."""
    names = _write_neutral_zone_group_names()
    expected_minimum = {
        "nodes", "elements", "physical_groups", "labels",
        "mesh_selections", "partitions", "parts", "constraints",
        "reinforce_ties", "embed_ties", "rebar_elements", "contacts",
        "contact_planes", "interfaces", "loads", "masses",
        "composed_from",
    }
    missing = expected_minimum - set(names)
    assert not missing, (
        f"_write_neutral_zone_group_names() no longer finds {missing} — "
        "either the scan regressed or write_neutral_zone stopped writing "
        "them (check mesh/_femdata_h5_io.py)."
    )


def test_h5_schema_doc_names_every_neutral_zone_group() -> None:
    """Every top-level group/dataset ``write_neutral_zone`` writes must be
    named somewhere in ``h5-schema.md`` (as a ``/name`` path mention).
    Catches a new neutral-zone group shipping without any doc update at
    all — the failure mode a 2026-08 audit found for ~10 pre-existing
    groups."""
    text = _h5_schema_doc_text()
    missing = sorted({
        name for name in _write_neutral_zone_group_names()
        if f"/{name}" not in text
    })
    assert not missing, (
        "h5-schema.md never mentions these neutral-zone groups that "
        f"write_neutral_zone writes: {missing} — add them (e.g. to the "
        '"Two zones" bullet list and/or the top-level layout tree).'
    )


def test_control_zone_registry_scan_detects_mismatch() -> None:
    """Control: a synthetic doc with a wrong Current value must be
    flagged by ``_zone_registry_current`` — proves the extractor
    actually reads the value rather than trivially passing."""
    fake_doc = (
        "| Zone | key | paths | writer | Current |\n"
        "|---|---|---|---|---|\n"
        "| neutral (broker) | k | p | w | **9.9.9** |\n"
    )
    assert _zone_registry_current(fake_doc, "neutral (broker)") == "9.9.9"
    assert _zone_registry_current(fake_doc, "neutral (broker)") != "2.31.0"


def test_control_group_name_scan_flags_missing_mention() -> None:
    """Control: a name absent from a synthetic doc must show up as
    missing — proves the membership check isn't vacuously true."""
    fake_names = ["contacts", "interfaces"]
    fake_doc_text = "See /contacts for details."
    missing = [n for n in fake_names if f"/{n}" not in fake_doc_text]
    assert missing == ["interfaces"]
