"""ADR 0019 — :class:`OpenSeesModel` end-to-end read-side broker tests.

Covers the surface listed in ADR 0019 §Decision:

* ``from_h5`` rehydrates into the typed broker.
* ``om.fem`` carries a non-trivial :class:`FEMData`.
* ``to_h5`` round-trip is byte-equivalent (modulo ``created_iso``).
* Typed-record accessors match the dict-style accessors in
  :mod:`apeGmsh.opensees.emitter.h5_reader`.
* Orientation-fan-out files round-trip byte-stably despite the
  one-record-per-call schema deviation.
* The module has no eager edge to :mod:`apeGmsh.mesh` (INV-4).
* ``snapshot_id`` is non-empty / lineage placeholder is populated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.lineage import Lineage
from apeGmsh.opensees._internal.typed_records import (
    BeamIntegrationRecord,
    ElementRecord,
    FixRecord,
    MaterialRecord,
    PatternRecord,
    RecorderRecord,
    SectionComplexRecord,
    SectionSimpleRecord,
    TimeSeriesRecord,
    TransformRecord,
)
from apeGmsh.opensees.emitter import h5_reader

from tests.opensees.h5._opensees_model_fixtures import (
    build_frame_with_orientation_fan_out_h5,
    build_simple_frame_h5,
)


OPENSEES_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "apeGmsh" / "opensees" / "opensees_model.py"
)


# ---------------------------------------------------------------------------
# Basic from_h5
# ---------------------------------------------------------------------------

def test_from_h5_returns_opensees_model_instance(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert isinstance(om, OpenSeesModel)


def test_from_h5_carries_fem(tmp_path: Path) -> None:
    src, fem = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    # The embedded FEM has non-trivial nodes and elements.
    assert len(om.fem.nodes.ids) == 2
    # One element group, one element.
    total_elements = sum(len(g.ids) for g in om.fem.elements)
    assert total_elements == 1


def test_from_h5_reports_model_metadata(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert om.model_name == "simple_frame"
    # Bridge ndm inferred from transforms (broker /meta.ndm is 1 for
    # a line-only FEM; the inference yields 3 from the vecxz vector
    # length).  See ADR 0019 §"On _replay_into" for the rationale.
    assert om.ndm == 3
    assert om.ndf == 6


# ---------------------------------------------------------------------------
# Byte-equivalence
# ---------------------------------------------------------------------------

def test_from_h5_to_h5_byte_equivalent(tmp_path: Path) -> None:
    """Loading a model.h5 and writing it back must be a fixed point on
    the file tree (modulo ``/meta/created_iso``, which the composer
    always re-stamps).

    Compares group / dataset names, attrs, and dataset contents.
    """
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    out = tmp_path / "roundtrip.h5"
    om.to_h5(out)

    with h5py.File(src, "r") as a, h5py.File(out, "r") as b:
        _assert_h5_equal(a, b)


def test_orientation_fanned_out_round_trip(tmp_path: Path) -> None:
    """The schema-deviation transforms zone (one group per
    ``geomTransf`` call) round-trips byte-stably."""
    src, _ = build_frame_with_orientation_fan_out_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    out = tmp_path / "roundtrip.h5"
    om.to_h5(out)

    # Sanity: two transforms (one per emitted call).
    assert len(om.transforms()) == 2

    with h5py.File(src, "r") as a, h5py.File(out, "r") as b:
        _assert_h5_equal(a, b)


# ---------------------------------------------------------------------------
# Typed-record vs dict-style parity
# ---------------------------------------------------------------------------

def test_typed_records_match_h5_reader_view(tmp_path: Path) -> None:
    """The :class:`OpenSeesModel` broker and the bare :mod:`h5_reader`
    surface agree on every record's (type, tag) pair.

    Phase 8 (ADR 0019) pruned the dict-style accessors; both surfaces
    now return typed records, so the parity test compares record sets
    directly.
    """
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    with h5_reader.open(str(src)) as m:
        # Materials.
        reader_materials = m.materials()
        om_materials = om.materials()
        assert (
            {(r.type_token, r.tag) for r in reader_materials}
            == {(r.type_token, r.tag) for r in om_materials}
        )

        # Sections.
        reader_sections = m.sections()
        om_sections = om.sections()
        assert (
            {(r.type_token, r.tag) for r in reader_sections}
            == {(s.type_token, s.tag) for s in om_sections}
        )

        # Transforms.
        reader_transforms = m.transforms()
        om_transforms = om.transforms()
        assert (
            {(r.type_token, r.tag) for r in reader_transforms}
            == {(t.type_token, t.tag) for t in om_transforms}
        )

        # Beam integration.
        reader_bi = m.beam_integration()
        om_bi = om.beam_integration()
        assert (
            {(r.type_token, r.tag) for r in reader_bi}
            == {(r.type_token, r.tag) for r in om_bi}
        )


# ---------------------------------------------------------------------------
# Per-record accessor content
# ---------------------------------------------------------------------------

def test_materials_typed_carry_steel02_params(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    materials = om.materials()
    assert all(isinstance(m, MaterialRecord) for m in materials)
    steel = next(m for m in materials if m.type_token == "Steel02")
    # The bridge fans Steel02(fy, E, b) into the 7-param positional
    # call after defaulting R0/cR1/cR2.
    assert steel.params[0] == pytest.approx(420e6)
    assert steel.params[1] == pytest.approx(200e9)
    assert steel.params[2] == pytest.approx(0.01)


def test_sections_typed_include_fiber_complex(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    sections = om.sections()
    assert len(sections) == 1
    sec = sections[0]
    assert isinstance(sec, SectionComplexRecord)
    assert sec.type_token == "Fiber"
    # The bridge emitted one fiber for this fixture.
    assert len(sec.fibers) == 1
    assert sec.fibers[0].area == pytest.approx(0.01)


def test_transforms_typed_carry_vecxz(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    transforms = om.transforms()
    assert len(transforms) == 1
    t = transforms[0]
    assert isinstance(t, TransformRecord)
    assert t.type_token == "Linear"
    np.testing.assert_allclose(t.vec, (1.0, 0.0, 0.0))


def test_beam_integration_typed(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    bi = om.beam_integration()
    assert len(bi) == 1
    assert isinstance(bi[0], BeamIntegrationRecord)
    assert bi[0].type_token == "Lobatto"


def test_elements_typed_carry_fem_eid(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    elements = om.elements()
    assert len(elements) == 1
    e = elements[0]
    assert isinstance(e, ElementRecord)
    assert e.type_token == "forceBeamColumn"
    # ``fem_eid`` is the broker FEM element id (Phase 8.6 tag_map).
    # The fixture's "Cols" PG contains element id 1.
    assert e.fem_eid == 1


def test_patterns_recorders_time_series_empty_when_unused(
    tmp_path: Path,
) -> None:
    """The simple-frame fixture deliberately omits patterns /
    recorders / time-series; the typed accessors return empty
    tuples without raising."""
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert om.patterns() == ()
    assert om.recorders() == ()
    assert om.time_series() == ()


# ---------------------------------------------------------------------------
# Lineage / snapshot_id
# ---------------------------------------------------------------------------

def test_snapshot_id_present(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert isinstance(om.snapshot_id, str)
    assert om.snapshot_id != ""


def test_lineage_placeholder(tmp_path: Path) -> None:
    src, fem = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert isinstance(om.lineage, Lineage)
    assert om.lineage.fem_hash != ""
    assert om.lineage.fem_hash == fem.snapshot_id
    assert om.lineage.model_hash is not None
    # Phase 3 stub: no results hash, no warnings.
    assert om.lineage.results_hash is None
    assert om.lineage.warnings == ()


# ---------------------------------------------------------------------------
# Lazy FEMData import — ADR 0019 INV-4
# ---------------------------------------------------------------------------

def test_no_module_level_apeGmsh_mesh_import() -> None:
    """AST-scan ``opensees_model.py`` and confirm no module-level
    ``from apeGmsh.mesh ...`` import.

    Every FEMData binding lives inside a method body so the
    import-DAG polarity (ADR 0019 INV-4) is preserved.
    """
    src = OPENSEES_MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(OPENSEES_MODEL_PATH))
    offences: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            if mod.startswith("apeGmsh.mesh"):
                offences.append(
                    f"line {stmt.lineno}: from {mod} import ..."
                )
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name.startswith("apeGmsh.mesh"):
                    offences.append(
                        f"line {stmt.lineno}: import {alias.name}"
                    )
    assert not offences, (
        "OpenSeesModel module must have no module-level "
        "apeGmsh.mesh import (ADR 0019 INV-4). Offences:\n  "
        + "\n  ".join(offences)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_h5_equal(a: h5py.File, b: h5py.File) -> None:
    """Recursively compare two HDF5 files, ignoring ``created_iso``.

    Asserts: same group structure, same datasets (shape, dtype kind,
    values), same attrs (modulo ``created_iso``).
    """
    _assert_attrs_equal(a, b, path="/")
    _assert_group_equal(a, b, path="/")


def _assert_group_equal(a: h5py.Group, b: h5py.Group, *, path: str) -> None:
    a_keys = sorted(a.keys())
    b_keys = sorted(b.keys())
    assert a_keys == b_keys, f"{path}: child keys differ {a_keys} vs {b_keys}"
    for key in a_keys:
        a_sub = a[key]
        b_sub = b[key]
        sub_path = f"{path}{key}/"
        _assert_attrs_equal(a_sub, b_sub, path=sub_path)
        if hasattr(a_sub, "shape"):  # dataset
            assert a_sub.shape == b_sub.shape, (
                f"{sub_path}: shape differs {a_sub.shape} vs {b_sub.shape}"
            )
            assert a_sub.dtype.kind == b_sub.dtype.kind, (
                f"{sub_path}: dtype kind differs {a_sub.dtype.kind} vs "
                f"{b_sub.dtype.kind}"
            )
            # Compare data; skip bytes/object datasets (compound rows)
            # via numpy-friendly tools.
            if a_sub.dtype.kind in ("f", "i", "u"):
                np.testing.assert_array_equal(a_sub[:], b_sub[:])
            elif a_sub.dtype.kind == "O":
                # Variable-length string arrays (e.g. element_loads)
                assert list(a_sub[:].flatten()) == list(b_sub[:].flatten())
            elif a_sub.dtype.kind == "V":  # compound rows
                # Cell-by-cell field comparison.
                rows_a = a_sub[:]
                rows_b = b_sub[:]
                assert rows_a.shape == rows_b.shape
                for ra, rb in zip(rows_a, rows_b):
                    for field_name in a_sub.dtype.names or ():
                        va = ra[field_name]
                        vb = rb[field_name]
                        # Handle nested array fields (the schema's
                        # NaN-padded ``forces`` / ``values`` slots).
                        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                            # NaN-tolerant equality for float arrays.
                            if va.dtype.kind == "f":
                                np.testing.assert_array_equal(
                                    np.nan_to_num(va, nan=0.0),
                                    np.nan_to_num(vb, nan=0.0),
                                )
                            else:
                                np.testing.assert_array_equal(va, vb)
                        else:
                            assert va == vb, (
                                f"{sub_path}: field {field_name} differs "
                                f"{va!r} vs {vb!r}"
                            )
        else:  # group
            _assert_group_equal(a_sub, b_sub, path=sub_path)


def _assert_attrs_equal(
    a: h5py.HLObject, b: h5py.HLObject, *, path: str,
) -> None:
    """Compare attrs, ignoring ``created_iso`` (always re-stamped by
    the composer)."""
    a_keys = {k for k in a.attrs.keys() if k != "created_iso"}
    b_keys = {k for k in b.attrs.keys() if k != "created_iso"}
    assert a_keys == b_keys, (
        f"{path}: attr keys differ; only-in-a={a_keys - b_keys}, "
        f"only-in-b={b_keys - a_keys}"
    )
    for key in a_keys:
        va = a.attrs[key]
        vb = b.attrs[key]
        # Decode bytes to compare across the np.bytes_/str boundary.
        if isinstance(va, bytes):
            va = va.decode("utf-8")
        if isinstance(vb, bytes):
            vb = vb.decode("utf-8")
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
        else:
            assert va == vb, (
                f"{path}: attr {key!r} differs {va!r} vs {vb!r}"
            )


# ---------------------------------------------------------------------------
# Parameterized roots — composed-file rehydrate (ADR 0020 / Phase 4 cleanup)
# ---------------------------------------------------------------------------


def test_from_h5_with_roots(tmp_path: Path) -> None:
    """``OpenSeesModel.from_h5(path, fem_root="/model")`` rehydrates from
    a composed results.h5 whose FEMData rich zone lives under
    ``/model/`` and whose ``/opensees/`` zone lives at root.

    The composed file is produced by :class:`NativeWriter` with
    ``model_h5_src=``.  We assert field-level equivalence to the
    standalone-file rehydrate to lock the parameterisation: same
    materials, sections, transforms, beam_integration, elements,
    fixes, masses, time_series, patterns, recorders.
    """
    from apeGmsh.results.writers import NativeWriter

    src_path, fem = build_simple_frame_h5(tmp_path)
    composed_path = tmp_path / "composed.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(composed_path) as w:
        w.open(fem=fem, model_h5_src=src_path)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()

    # Rehydrate via the composed-file roots.
    om_composed = OpenSeesModel.from_h5(
        composed_path, fem_root="/model", opensees_root="/opensees",
    )
    # Rehydrate via the standalone roots for comparison.
    om_standalone = OpenSeesModel.from_h5(src_path)

    assert om_composed.model_name == om_standalone.model_name
    assert om_composed.ndm == om_standalone.ndm
    assert om_composed.ndf == om_standalone.ndf
    # snapshot_id is opaque carry-through; both rehydrates pick up
    # the same broker-stamped value.
    assert om_composed.snapshot_id == om_standalone.snapshot_id

    # Typed-record collections — same count and (type, tag) identity.
    def _pairs(recs):
        return {(r.type_token, r.tag) for r in recs}

    assert _pairs(om_composed.materials()) == _pairs(om_standalone.materials())
    assert _pairs(om_composed.sections()) == _pairs(om_standalone.sections())
    assert _pairs(om_composed.transforms()) == _pairs(om_standalone.transforms())
    assert _pairs(om_composed.beam_integration()) == _pairs(
        om_standalone.beam_integration()
    )

    # Elements — fem_eid mapping survives through both routes.
    composed_eids = sorted(int(e.fem_eid) for e in om_composed.elements())
    standalone_eids = sorted(int(e.fem_eid) for e in om_standalone.elements())
    assert composed_eids == standalone_eids


def test_from_h5_default_roots_backcompat(tmp_path: Path) -> None:
    """``OpenSeesModel.from_h5(path)`` (no kwargs) reads a standalone
    model.h5 byte-identically to the pre-Phase-4 reader.

    Existing call sites must work unchanged — the parameterisation
    is additive.
    """
    src, _ = build_simple_frame_h5(tmp_path)
    om_default = OpenSeesModel.from_h5(src)
    om_explicit = OpenSeesModel.from_h5(
        src, fem_root="/", opensees_root="/opensees",
    )
    assert om_default.model_name == om_explicit.model_name
    assert om_default.snapshot_id == om_explicit.snapshot_id
    assert len(om_default.materials()) == len(om_explicit.materials())
    assert len(om_default.elements()) == len(om_explicit.elements())


# ---------------------------------------------------------------------------
# Phase 6 / ADR 0021 — lineage chain populated on read; preserved on round-trip
# ---------------------------------------------------------------------------


def test_lineage_on_from_h5(tmp_path: Path) -> None:
    """``om.lineage`` carries both ``fem_hash`` and ``model_hash``.

    Phase 6 (ADR 0021) — the standalone ``model.h5`` write path
    stamps the lineage triple; ``from_h5`` reads it back and
    recomputes for the drift check.  No warnings on a freshly-
    written file.
    """
    src, fem = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    assert om.lineage.fem_hash != ""
    assert om.lineage.fem_hash == fem.snapshot_id  # INV-1
    assert om.lineage.model_hash is not None
    assert om.lineage.results_hash is None  # standalone model.h5
    assert om.lineage.warnings == ()


def test_lineage_recompute_on_to_h5(tmp_path: Path) -> None:
    """Rehydrate, ``to_h5`` round-trip, lineage chain matches.

    The model_hash and fem_hash byte-stable across the round-trip
    (modulo timestamps the lineage stamp does not include).
    """
    src, _ = build_simple_frame_h5(tmp_path)
    om_a = OpenSeesModel.from_h5(src)
    dst = tmp_path / "roundtrip.h5"
    om_a.to_h5(dst)
    om_b = OpenSeesModel.from_h5(dst)
    assert om_a.lineage.fem_hash == om_b.lineage.fem_hash
    assert om_a.lineage.model_hash == om_b.lineage.model_hash
    # No drift warnings — round-trip preserves the chain.
    assert om_b.lineage.warnings == ()
