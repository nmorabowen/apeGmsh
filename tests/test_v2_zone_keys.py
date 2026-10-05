"""V2a (#1304): the ADR 0112 zone keys, ``session_id`` and hash invariance.

Oracles:

* The ``/geometry`` and ``/provenance`` zones are registered with their own
  ``/meta`` keys, and their version comes from ``tests/fixtures/schema.py``.
  The legacy envelope never stands in for them (they postdate it).
* ``FEMData.session_id`` is a canonical uuid4. Every neutral-zone writer
  stamps it as ``/meta/session_id``, ``FEMData.from_h5`` reads it back, and a
  malformed one is refused loudly.
* Hash invariance (V0 decision 2): adding or deleting a ``/geometry`` or
  ``/provenance`` group, and changing ``session_id``, leave ``snapshot_id``,
  ``fem_hash`` and ``model_hash`` unchanged. The expected values are the ones
  the unmodified file carries, so the oracle is the file's own lineage.
"""
from __future__ import annotations

import copy
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.lineage import (
    compute_fem_hash,
    compute_model_hash,
    read_stored_lineage,
)
from apeGmsh.opensees._internal.schema_version import (
    _ZONE_KEY,
    GEOMETRY,
    GEOMETRY_KEY,
    PROVENANCE,
    PROVENANCE_KEY,
    SchemaVersion,
    SchemaVersionError,
    read_zone_version,
    reader_version,
    validate_zone_version,
)
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from tests.fixtures.schema import (
    GEOMETRY_CURRENT,
    NEUTRAL_CURRENT,
    PROVENANCE_CURRENT,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Zone keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "zone, key, current",
    [
        (GEOMETRY, GEOMETRY_KEY, GEOMETRY_CURRENT),
        (PROVENANCE, PROVENANCE_KEY, PROVENANCE_CURRENT),
    ],
)
def test_zone_registered_with_own_key(zone, key, current):
    assert _ZONE_KEY[zone] == key == f"{zone}_schema_version"
    assert reader_version(zone) == SchemaVersion.parse(current)
    # Present key: read and accepted (it sits at the current version).
    got = read_zone_version({key: current}, zone)
    assert got == SchemaVersion.parse(current)
    validate_zone_version(got, reader_version(zone), zone=zone)
    # A different major is refused, as for every other zone.
    with pytest.raises(SchemaVersionError):
        validate_zone_version(
            SchemaVersion.parse(NEUTRAL_CURRENT), reader_version(zone),
            zone=zone,
        )


@pytest.mark.parametrize("zone", [GEOMETRY, PROVENANCE])
def test_new_zones_never_fall_back_to_envelope(zone):
    """A model.h5 has only the envelope: that is not a geometry version."""
    meta = {"schema_version": NEUTRAL_CURRENT}
    assert read_zone_version(meta, zone) is None


def test_unknown_zone_still_refused():
    with pytest.raises(ValueError, match="unknown zone 'sequence'"):
        read_zone_version({}, "sequence")
    with pytest.raises(ValueError, match="unknown zone 'sequence'"):
        reader_version("sequence")


def test_schema_doc_specifies_new_zones_and_session_id():
    doc = (REPO / "architecture" / "h5-schema.md").read_text(encoding="utf-8")
    for needle in (
        "## `/geometry` (sibling `<stem>.geometry.h5`)",
        "## `/provenance`",
        "## `/meta/session_id` and the geometry sibling",
        "## Integer policy for the ADR 0112 zones",
        f"| geometry (ADR 0112 D2) | `{GEOMETRY_KEY}` |",
        f"| provenance (ADR 0112 D3) | `{PROVENANCE_KEY}` |",
    ):
        assert needle in doc, needle


# ---------------------------------------------------------------------------
# session_id
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fem():
    with apeGmsh(model_name="v2a", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.physical.add_volume("b", name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        out = g.mesh.queries.get_fem_data(dim=3)
    return out


def _write_bridge(fem_: FEMData, path: Path) -> None:
    ops = apeSees(fem_)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.h5(str(path))


def _with_session(fem_: FEMData, sid: str) -> FEMData:
    other = copy.copy(fem_)
    other.session_id = sid
    return other


def test_session_id_is_canonical_uuid4(fem):
    sid = fem.session_id
    assert uuid.UUID(sid).version == 4
    assert str(uuid.UUID(sid)) == sid


@pytest.mark.parametrize(
    "bad", ["", "not-a-uuid", str(uuid.uuid1()), uuid.uuid4().hex, 7],
)
def test_constructor_refuses_malformed_session_id(fem, bad):
    with pytest.raises(ValueError, match="session_id"):
        FEMData(fem.nodes, fem.elements, fem.info, session_id=bad)


@pytest.mark.parametrize("writer", ["to_h5", "apesees_h5"])
def test_writers_stamp_and_reader_restores_session_id(fem, tmp_path, writer):
    path = tmp_path / "m.h5"
    if writer == "to_h5":
        fem.to_h5(str(path))
    else:
        _write_bridge(fem, path)
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["session_id"] == fem.session_id
    assert FEMData.from_h5(str(path)).session_id == fem.session_id


def test_derived_snapshots_inherit_session_id(fem):
    """The id belongs to the session: record transforms and copies keep it."""
    from apeGmsh._kernel.records._masses import MassRecord

    nid = int(fem.nodes.ids[0])
    derived = fem.with_mass(MassRecord(node_id=nid, mass=(1.0, 1.0, 1.0)))
    assert derived is not fem
    assert len(derived.nodes.masses) == len(fem.nodes.masses) + 1
    assert derived.session_id == fem.session_id
    assert copy.copy(fem).session_id == fem.session_id
    assert fem._replaced().session_id == fem.session_id


def test_old_pickle_without_session_id_keeps_its_neutral_zone(fem, tmp_path):
    """A FEMData pickled before #1304 has no session_id: it gets a fresh one,
    and the bridge still writes its neutral zone (it is not read as a stub)."""
    import pickle

    old = copy.copy(fem)
    del old.session_id  # the shape of a pre-#1304 pickle
    revived = pickle.loads(pickle.dumps(old))
    assert uuid.UUID(revived.session_id).version == 4
    path = tmp_path / "old.h5"
    _write_bridge(revived, path)
    with h5py.File(path, "r") as f:
        assert "nodes" in f and "elements" in f
        assert f["meta"].attrs["session_id"] == revived.session_id


def test_reader_refuses_malformed_session_id(fem, tmp_path):
    path = tmp_path / "m.h5"
    fem.to_h5(str(path))
    with h5py.File(path, "r+") as f:
        f["meta"].attrs["session_id"] = "garbage"
    with pytest.raises(MalformedH5Error, match="session_id"):
        FEMData.from_h5(str(path))


def test_file_without_session_id_reads_with_fresh_one(fem, tmp_path):
    """A file written before #1304 still reads; it pairs with nothing."""
    path = tmp_path / "m.h5"
    fem.to_h5(str(path))
    with h5py.File(path, "r+") as f:
        del f["meta"].attrs["session_id"]
    back = FEMData.from_h5(str(path))
    assert uuid.UUID(back.session_id).version == 4
    assert back.session_id != fem.session_id


# ---------------------------------------------------------------------------
# Hash invariance (V0 decision 2)
# ---------------------------------------------------------------------------


def _hashes(path: Path) -> tuple[str, str | None, str | None, str, str]:
    """(snapshot_id, stored fem_hash, stored model_hash, recomputed both)."""
    with h5py.File(path, "r") as f:
        fem_hash, model_hash, _ = read_stored_lineage(f["meta"])
        re_fem = compute_fem_hash(f)
        re_model = compute_model_hash(re_fem, f["opensees"])
    return (
        FEMData.from_h5(str(path)).snapshot_id,
        fem_hash, model_hash, re_fem, re_model,
    )


def _add_dummy_zones(path: Path) -> None:
    with h5py.File(path, "r+") as f:
        geo = f.create_group("geometry")
        geo.attrs["status"] = "ok"
        ent = geo.create_group("entities")
        ent.create_dataset("dim", data=np.array([3], dtype=np.int8))
        ent.create_dataset("tag", data=np.array([1], dtype=np.int32))
        # A complete, empty /provenance (h5-schema.md, "/provenance"): the
        # reader refuses a partial zone, so every table and column exists.
        # Since V2d (#1307) the bridge write already carries the real
        # zone (``base`` above hashed with it); the dummy replaces it.
        if "provenance" in f:
            del f["provenance"]
        prov = f.create_group("provenance")
        prov.attrs["base_dir"] = path.parent.as_posix()
        for table, cols in (
            ("files", {"path": str, "sha256": str, "kind": str}),
            ("sites", {"file": int, "line": int, "function": str}),
            ("records", {"path": str, "site": int, "script": int,
                         "seq": int, "origin": str}),  # origin: 1.1.0
        ):
            grp = prov.create_group(table)
            for name, kind in cols.items():
                if kind is str:
                    grp.create_dataset(
                        name, data=np.array([], dtype=object),
                        dtype=h5py.string_dtype("utf-8"))
                else:
                    grp.create_dataset(
                        name, data=np.array([], dtype=np.int32))
        f["meta"].attrs[GEOMETRY_KEY] = GEOMETRY_CURRENT
        f["meta"].attrs[PROVENANCE_KEY] = PROVENANCE_CURRENT


def test_new_zones_and_session_id_leave_hashes_unchanged(fem, tmp_path):
    path = tmp_path / "m.h5"
    _write_bridge(fem, path)
    base = _hashes(path)
    snapshot_id, fem_hash, model_hash, re_fem, re_model = base
    # The file's own lineage is the oracle; it must be self-consistent.
    assert snapshot_id == fem.snapshot_id == fem_hash == re_fem
    assert model_hash is not None and model_hash == re_model

    _add_dummy_zones(path)
    assert _hashes(path) == base

    with h5py.File(path, "r+") as f:
        f["meta"].attrs["session_id"] = str(uuid.uuid4())
    assert _hashes(path) == base

    with h5py.File(path, "r+") as f:
        del f["geometry"]
        del f["provenance"]
        del f["meta"].attrs[GEOMETRY_KEY]
        del f["meta"].attrs[PROVENANCE_KEY]
    assert _hashes(path) == base


def test_two_sessions_write_identical_hashes(fem, tmp_path):
    """One model written under two session ids hashes the same."""
    other = _with_session(fem, str(uuid.uuid4()))
    assert other.session_id != fem.session_id
    assert other.snapshot_id == fem.snapshot_id
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_bridge(fem, a)
    _write_bridge(other, b)
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert fa["meta"].attrs["session_id"] != fb["meta"].attrs["session_id"]
        assert read_stored_lineage(fa["meta"]) == read_stored_lineage(fb["meta"])
    assert _hashes(a) == _hashes(b)
