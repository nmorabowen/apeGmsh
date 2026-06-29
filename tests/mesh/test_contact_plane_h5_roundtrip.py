"""Neutral-zone H5 round-trip for rigid analytical-plane contacts (ADR 0073).

``fem.elements.contact_planes`` (g.constraints.contact_plane → fork
``contactPlane``) persist through ``FEMData.to_h5`` / ``from_h5`` into a
dedicated ``/contact_planes`` group (neutral schema 2.24.0), the analytical-
plane sibling of ``/contacts``. Built on a real one-body mesh — no fork build
needed (records resolve at ``get_fem_data``; only *running* the deck needs the
fork)."""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


def _face_at_z(volume_tag, z, tol=1e-3):
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _plane_fem(**kw):
    with apeGmsh(model_name="cplane_h5", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        bottom = _face_at_z(box, 0.0)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [bottom], name="floor")
        g.constraints.contact_plane(
            "floor", normal=(0, 0, 1), point=(0, 0, 0), **kw)
        return g.mesh.queries.get_fem_data(dim=3)


def _plain_fem():
    with apeGmsh(model_name="cplane_h5_plain", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        return g.mesh.queries.get_fem_data(dim=3)


def _roundtrip(fem, tmp_path):
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    p = str(tmp_path / "m.h5")
    fem.to_h5(p)
    return read_fem_h5(p), p


def _eq(a, b):
    assert list(a.slave_nodes) == list(b.slave_nodes)
    assert tuple(a.normal) == pytest.approx(tuple(b.normal))
    assert tuple(a.point) == pytest.approx(tuple(b.point))
    assert a.kn == pytest.approx(b.kn)
    assert a.name == b.name
    x, y = a.visc, b.visc
    assert (x is None) == (y is None)
    if x is not None:
        assert x == pytest.approx(y)
    assert a.soft == b.soft


def test_contact_plane_roundtrip(tmp_path):
    fem = _plane_fem(kn=1.0e7, visc=2.5, soft=0.1, name="floor")
    src = fem.elements.contact_planes
    assert len(src) == 1 and src[0].kn == 1e7
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contact_planes
    assert len(got) == 1
    _eq(got[0], src[0])
    assert got[0].soft == pytest.approx(0.1)
    assert got[0].visc == pytest.approx(2.5)
    assert got[0].slave_nodes and got[0].normal == pytest.approx((0, 0, 1))


def test_soft_bare_true_roundtrip(tmp_path):
    fem = _plane_fem(kn=1.0e7, soft=True)
    src = fem.elements.contact_planes[0]
    assert src.soft is True
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contact_planes[0]
    _eq(got, src)
    assert got.soft is True


def test_plane_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.contact_planes
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "contact_planes" not in f
    assert back.snapshot_id == fem.snapshot_id


def test_snapshot_id_stable_on_roundtrip(tmp_path):
    fem = _plane_fem(kn=1.0e7)
    back, _ = _roundtrip(fem, tmp_path)
    assert back.snapshot_id == fem.snapshot_id
    assert back.elements.contact_planes


def test_to_h5_no_deferral_warning(tmp_path, recwarn):
    fem = _plane_fem(kn=1.0e7)
    fem.to_h5(str(tmp_path / "m.h5"))
    assert not [w for w in recwarn.list
                if "not persisted" in str(w.message)
                or "deferred" in str(w.message)]


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


def test_reads_prior_minor_file_without_group_within_window(tmp_path):
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _plane_fem(kn=1.0e7)
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        if "contact_planes" in f:
            del f["contact_planes"]
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(p)                          # within window → no raise
    assert back.elements.contact_planes == []      # absent group → no planes


def test_apesees_deck_archive_recovers_contact_plane(tmp_path, recwarn):
    # The apeSees(fem).h5() deck-archive path drives the H5Emitter
    # contact_plane / contact_surface NO-OPs and writes the neutral
    # /contact_planes group; recovery is via the neutral zone (read_fem_h5).
    # Distinct from the broker-direct fem.to_h5 path above — this is the only
    # path that exercises the H5Emitter no-ops the CHANGELOG advertises.
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    from apeGmsh.opensees import apeSees

    fem = _plane_fem(kn=1.0e7, visc=2.0, soft=0.1, name="floor")
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    p = str(tmp_path / "deck.h5")
    ops.h5(p)
    got = read_fem_h5(p).elements.contact_planes
    assert len(got) == 1
    _eq(got[0], fem.elements.contact_planes[0])
    assert not [w for w in recwarn.list
                if "not persisted" in str(w.message)
                or "deferred" in str(w.message)]


def test_encode_rejects_empty_slave():
    from apeGmsh._kernel.records._constraints import ContactPlaneRecord
    from apeGmsh.mesh._femdata_h5_io import _encode_contact_plane
    with pytest.raises(ValueError, match="slave_nodes is empty"):
        _encode_contact_plane(ContactPlaneRecord(
            kind="contact_plane", slave_nodes=[], normal=(0, 0, 1),
            point=(0, 0, 0), kn=1e7))
