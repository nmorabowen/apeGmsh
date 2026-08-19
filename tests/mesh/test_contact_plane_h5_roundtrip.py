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


def _curve_at_y(surface, y, tol=1e-6):
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[1] - y) < tol and abs(bb[4] - y) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at y={y}")


def _plane_fem_2d(slave="base", normal=(0, 1), point=(0, 0), **kw):
    """A unit square seated on a rigid floor — the 2D rigid-plane lane.

    The plane needs neither the chained master surface of the NTS lane nor
    ``ndf == ndm``: ``contactPlane`` has no master mesh and its adapter
    couples the first ``ndm`` DOFs by construction.
    """
    with apeGmsh(model_name="cplane_h5_2d", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        g.model.sync()
        base = _curve_at_y(rect, 0.0)
        g.mesh.structured.set_transfinite([(2, rect)], n=3)
        g.mesh.generation.generate(2)
        g.physical.add(2, [rect], name="body")
        g.physical.add(1, [base], name="base")
        g.constraints.contact_plane(
            slave, normal=normal, point=point, **kw)
        return g.mesh.queries.get_fem_data(dim=2)


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


# =====================================================================
# 2D rigid plane (fork ADR-85 adoption, slice S1)
# =====================================================================

def test_2d_plane_record_is_z_padded_and_round_trips(tmp_path):
    fem = _plane_fem_2d(kn=1.0e7, name="floor")
    src = fem.elements.contact_planes[0]
    assert tuple(src.normal) == (0.0, 1.0, 0.0)      # z-padded, not reshaped
    assert tuple(src.point) == (0.0, 0.0, 0.0)
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contact_planes
    assert len(got) == 1
    _eq(got[0], src)
    assert tuple(got[0].normal) == (0.0, 1.0, 0.0)


def test_2d_deck_emits_the_9_arg_form_with_zero_z_slots(tmp_path):
    """The zero-padded 9-argument form is permanently valid on a 2D slave
    surface (fork T0 back-compat row), so apeGmsh emits ONE grammar."""
    from apeGmsh.opensees import apeSees

    fem = _plane_fem_2d(kn=1.0e7)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeQuad(pg="body", thickness=0.5, material=mat)
    p = tmp_path / "deck.tcl"
    ops.tcl(str(p))
    line = next(ln for ln in p.read_text().splitlines()
                if ln.startswith("contactPlane"))
    toks = line.split()
    assert len(toks) == 10                    # verb + tag + 9 args, no flags
    nx, ny, nz, px, py, pz, kn = (float(t) for t in toks[3:10])
    assert (nx, ny, nz) == (0.0, 1.0, 0.0)
    assert (px, py, pz) == (0.0, 0.0, 0.0)
    assert kn == 1.0e7


def test_2d_dim2_slave_pg_is_refused_by_name():
    """Naming the dim-2 plane — the natural mistake — would collect every
    INTERIOR node of the body as the slave set."""
    with pytest.raises(ValueError) as exc:
        _plane_fem_2d(slave="body", kn=1.0e7)
    msg = str(exc.value)
    assert msg.startswith("contact_plane:")
    assert "the model is 2D" in msg
    assert "INTERIOR node" in msg


def test_2d_out_of_plane_normal_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _plane_fem_2d(normal=(0, 1, 0.5), kn=1.0e7)
    msg = str(exc.value)
    assert msg.startswith("contact_plane:")
    assert "non-zero z" in msg and "normal" in msg


def test_2d_out_of_plane_point_is_refused_by_name():
    with pytest.raises(ValueError, match="non-zero z"):
        _plane_fem_2d(point=(0, 0, 0.5), kn=1.0e7)


def test_3d_slave_stays_ungated(tmp_path):
    """The 3D slave is deliberately NOT dimension-gated (``_collect_node_set``
    is dimension-agnostic there); the gate must not have retrofitted it."""
    with apeGmsh(model_name="cplane_h5_3d_vol", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.constraints.contact_plane(
            "solid", normal=(0, 0, 1), point=(0, 0, 0), kn=1.0e7)
        fem = g.mesh.queries.get_fem_data(dim=3)
    rec = fem.elements.contact_planes[0]
    assert rec.slave_nodes and tuple(rec.normal) == (0.0, 0.0, 1.0)


def test_encode_rejects_empty_slave():
    from apeGmsh._kernel.records._constraints import ContactPlaneRecord
    from apeGmsh.mesh._femdata_h5_io import _encode_contact_plane
    with pytest.raises(ValueError, match="slave_nodes is empty"):
        _encode_contact_plane(ContactPlaneRecord(
            kind="contact_plane", slave_nodes=[], normal=(0, 0, 1),
            point=(0, 0, 0), kn=1e7))
