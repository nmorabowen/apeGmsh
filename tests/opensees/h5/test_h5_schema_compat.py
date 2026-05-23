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
# Phase 7a — Per-zone schema versioning + two-version reader window
# (ADR 0023). Tests below exercise the central helpers in
# :mod:`apeGmsh.opensees._internal.schema_version` plus the read/write
# wiring across the three zones (neutral, opensees, results).
# ===========================================================================

from pathlib import Path

import numpy as np

from apeGmsh.opensees._internal.schema_version import (
    ENVELOPE_KEY,
    NEUTRAL,
    NEUTRAL_KEY,
    OPENSEES,
    OPENSEES_KEY,
    RESULTS,
    RESULTS_KEY,
    SchemaVersion,
    SchemaVersionError as _PerZoneSchemaError,
    read_zone_version,
    reader_version,
    validate_zone_version,
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


def test_two_version_window_accepts_current_minor() -> None:
    """Reader at X.Y.Z accepts X.Y.* — any patch within the current minor."""
    reader = SchemaVersion(2, 6, 0)
    for patch in (0, 1, 99):
        validate_zone_version(
            SchemaVersion(2, 6, patch), reader, zone=NEUTRAL,
        )


def test_two_version_window_accepts_previous_minor() -> None:
    """Reader at X.Y.Z accepts X.(Y-1).* — any patch within the prior minor."""
    reader = SchemaVersion(2, 6, 0)
    for patch in (0, 1, 99):
        validate_zone_version(
            SchemaVersion(2, 5, patch), reader, zone=NEUTRAL,
        )


def test_two_version_window_refuses_too_old_minor() -> None:
    """Reader at 2.7.0 refuses 2.5.0 (outside the two-version window)."""
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(2, 5, 0), SchemaVersion(2, 7, 0), zone=NEUTRAL,
        )
    assert "too old" in str(exc.value)


def test_two_version_window_refuses_newer_minor() -> None:
    """Reader at 2.6.0 refuses 2.7.0 (newer than this reader knows).
    INV-4 — refusing is safer than silent tolerance."""
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(2, 7, 0), SchemaVersion(2, 6, 0), zone=NEUTRAL,
        )
    assert "newer" in str(exc.value)


def test_two_version_window_refuses_different_major() -> None:
    """Reader at 2.6.0 refuses 3.0.0 AND 1.x — any major mismatch."""
    reader = SchemaVersion(2, 6, 0)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(3, 0, 0), reader, zone=NEUTRAL,
        )
    assert "different major" in str(exc.value)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(1, 9, 0), reader, zone=NEUTRAL,
        )
    assert "different major" in str(exc.value)


def test_schema_version_error_message_includes_upgrade_path() -> None:
    """SchemaVersionError text mentions the file's version AND the
    reader's supported range, per ADR 0023 §"Per-zone read validation"."""
    reader = SchemaVersion(2, 6, 0)
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(2, 4, 0), reader, zone=OPENSEES,
        )
    msg = str(exc.value)
    assert "2.4.0" in msg
    assert "2.5.x" in msg or "2.5" in msg
    assert "2.6.x" in msg or "2.6" in msg
    assert "Upgrade" in msg


def test_validate_per_zone_independently() -> None:
    """INV-3 — windows are conjunctive but NOT coupled.

    opensees at 2.6.0 + neutral at 2.5.0 should both validate
    independently when the reader is at 2.6.0 for each zone.
    """
    reader = SchemaVersion(2, 6, 0)
    validate_zone_version(SchemaVersion(2, 6, 0), reader, zone=OPENSEES)
    validate_zone_version(SchemaVersion(2, 5, 0), reader, zone=NEUTRAL)


def test_reader_version_reflects_writer_constants() -> None:
    """``reader_version(NEUTRAL)`` matches ``NEUTRAL_SCHEMA_VERSION`` exactly.

    Single source of truth — the reader's per-zone version is sourced
    from the writer module's constant; they cannot drift.
    """
    from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
    from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION as OPENSEES_VERSION
    from apeGmsh.results.schema._versions import RESULTS_SCHEMA_VERSION

    assert reader_version(NEUTRAL) == SchemaVersion.parse(NEUTRAL_SCHEMA_VERSION)
    assert reader_version(OPENSEES) == SchemaVersion.parse(OPENSEES_VERSION)
    assert reader_version(RESULTS) == SchemaVersion.parse(RESULTS_SCHEMA_VERSION)


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
    """Each zone's version window is independent of the others.

    The results-zone reader window applies to the results version
    only; the opensees-zone window applies to the opensees version
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
# Schema 2.10.0 (ADR 0027) / /opensees/partitions/
# (2.7.0 added /opensees/constraints/; 2.8.0 renamed embeddedNode's
#  embedding_ele → cnode; 2.9.0 added /opensees/regions/ for MPCO
#  region-filtered output; 2.10.0 added /opensees/partitions/ +
#  partition_ids column for OpenSeesMP per-rank emission.)
# ---------------------------------------------------------------------------


def test_opensees_reader_version_is_2_10_0() -> None:
    """Schema 2.10.0 — /opensees/partitions/ zone (ADR 0027)."""
    assert reader_version(OPENSEES) == SchemaVersion(2, 10, 0)


def test_two_version_window_at_2_10_accepts_2_9_and_2_10() -> None:
    """Reader at 2.10.0 accepts 2.9.x and 2.10.x (window: prev minor + current)."""
    reader = SchemaVersion(2, 10, 0)
    for patch in (0, 1, 99):
        validate_zone_version(
            SchemaVersion(2, 9, patch), reader, zone=OPENSEES,
        )
        validate_zone_version(
            SchemaVersion(2, 10, patch), reader, zone=OPENSEES,
        )


def test_two_version_window_at_2_10_refuses_2_8() -> None:
    """Reader at 2.10.0 refuses 2.8.x (outside window)."""
    with pytest.raises(_PerZoneSchemaError) as exc:
        validate_zone_version(
            SchemaVersion(2, 8, 0), SchemaVersion(2, 10, 0), zone=OPENSEES,
        )
    assert "too old" in str(exc.value)


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


def test_phantom_node_tags_present_when_ndf_overrides_emitted(
    tmp_path: Any,
) -> None:
    """``node(tag, *xyz, ndf=6)`` calls populate ``phantom_node_tags``."""
    e = H5Emitter()
    e.model(ndm=3, ndf=3)
    e.node(1, 0.0, 0.0, 0.0)               # regular
    e.node(200, 0.0, 0.0, 1.0, ndf=6)      # phantom
    out = tmp_path / "phantoms.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "opensees/constraints/phantom_node_tags" in f
        tags = f["opensees/constraints/phantom_node_tags"][:]
        assert list(int(t) for t in tags) == [200]
