"""Neutral-zone H5 round-trip for fork contact interactions (ADR 0073).

``fem.elements.contacts`` (g.constraints.contact / g.constraints.mortar
NTS/mortar interactions) now persist through ``FEMData.to_h5`` / ``from_h5``
into a dedicated ``/contacts`` group (neutral schema 2.21.0). Previously
dropped on the OpenSees deck zone with an ``H5FeatureDeferredWarning`` and no
neutral persistence. Built on a real two-body mesh — no fork build needed (the
records are resolved at ``get_fem_data``; only *running* the deck needs the
fork)."""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _build_two_bodies(g):
    box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)  # 0.05 gap
    g.model.sync()
    master = _face_at_z(box1, 1.0)
    slave = _face_at_z(box2, 1.05)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box1, box2], name="solid")
    g.physical.add(2, [master], name="master")
    g.physical.add(2, [slave], name="slave")


def _contact_fem(add):
    with apeGmsh(model_name="contact_h5", verbose=False) as g:
        _build_two_bodies(g)
        add(g)
        return g.mesh.queries.get_fem_data(dim=3)


def _plain_fem():
    with apeGmsh(model_name="contact_h5_plain", verbose=False) as g:
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
    assert a.formulation == b.formulation
    assert a.master_nps == b.master_nps and a.slave_nps == b.slave_nps
    assert a.name == b.name and a.tie == b.tie
    assert a.consistent_tan == b.consistent_tan and a.geom_tan == b.geom_tan
    assert a.edge_edge == b.edge_edge
    assert a.edge_consistent_tan == b.edge_consistent_tan
    assert a.edge_alm == b.edge_alm
    # edge_kn (auto/None/numeric tri-state) + edge_soft (None/bare/numeric)
    for f in ("edge_kn", "edge_soft"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        assert (x == "auto") == (y == "auto"), f
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            assert x == pytest.approx(y), f
        else:
            assert x == y, f
    for f in ("edge_band", "edge_mu", "edge_kt", "edge_cohesion",
              "edge_tau_max", "edge_aug_tol"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x == pytest.approx(y), f
    np.testing.assert_array_equal(
        np.asarray(a.master_faces), np.asarray(b.master_faces))
    # slave node-set (NTS) vs faceted (mortar) — exactly one present
    assert (a.slave_nodes is None) == (b.slave_nodes is None)
    if a.slave_nodes is not None:
        assert list(a.slave_nodes) == list(b.slave_nodes)
    assert (a.slave_faces is None) == (b.slave_faces is None)
    if a.slave_faces is not None:
        np.testing.assert_array_equal(
            np.asarray(a.slave_faces), np.asarray(b.slave_faces))
    assert (a.outward is None) == (b.outward is None)
    if a.outward is not None:
        assert tuple(a.outward) == pytest.approx(tuple(b.outward))
    # tri-state penalties + plain optionals + soft
    for f in ("kn", "eps_n", "eps_t", "soft"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        assert (x == "auto") == (y == "auto"), f
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            assert x == pytest.approx(y), f
        else:
            assert x == y, f
    for f in ("kt", "mu", "cohesion", "tau_max", "aug_tol", "visc", "cell"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x == pytest.approx(y), f
    for f in ("max_aug", "ngp"):
        assert getattr(a, f) == getattr(b, f), f


def test_nts_contact_roundtrip(tmp_path):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=1.0e6, kt=5.0e5, mu=0.3))
    src = fem.elements.contacts
    assert len(src) == 1 and src[0].formulation == "nts"
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts
    assert len(got) == 1
    _eq(got[0], src[0])
    assert got[0].slave_nodes is not None and got[0].slave_faces is None
    assert got[0].kn == pytest.approx(1.0e6)


def test_mortar_contact_roundtrip(tmp_path):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="mortar", eps_n="auto",
        aug_tol=1e-8, max_aug=20, ngp=2))
    src = fem.elements.contacts
    assert len(src) == 1 and src[0].formulation == "mortar"
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts
    assert len(got) == 1
    _eq(got[0], src[0])
    assert got[0].slave_faces is not None and got[0].slave_nodes is None
    assert got[0].eps_n == "auto" and got[0].max_aug == 20 and got[0].ngp == 2


def test_nts_extensions_roundtrip(tmp_path):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=1.0e6, kt=5.0e5, mu=0.3,
        soft=0.1, visc=1.0, consistent_tan=True, geom_tan=True, cell=2.0))
    src = fem.elements.contacts[0]
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    _eq(got, src)
    assert got.soft == pytest.approx(0.1) and got.visc == pytest.approx(1.0)
    assert got.consistent_tan and got.geom_tan
    assert got.cell == pytest.approx(2.0)


def test_soft_bare_true_roundtrip(tmp_path):
    # soft=True (bare -soft, fork default SOFSCL) must round-trip as True, not
    # a float and not None (the soft_mode tri-state).
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn="auto", soft=True))
    src = fem.elements.contacts[0]
    assert src.soft is True and src.kn == "auto"
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    assert got.soft is True
    assert got.kn == "auto"


def test_edge_edge_roundtrip(tmp_path):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="mortar", eps_n="auto",
        edge_edge=True, edge_kn="auto", edge_band=0.01, edge_mu=0.4,
        edge_kt=1.0e6, edge_cohesion=1.0e3, edge_tau_max=5.0e5,
        edge_consistent_tan=True, edge_soft=0.1, edge_alm=True,
        edge_aug_tol=1.0e-6))
    src = fem.elements.contacts[0]
    assert src.edge_edge is True and src.edge_kn == "auto"
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    _eq(got, src)
    assert got.edge_edge is True and got.edge_kn == "auto"
    assert got.edge_band == pytest.approx(0.01)
    assert got.edge_mu == pytest.approx(0.4)
    assert got.edge_consistent_tan and got.edge_alm
    assert got.edge_soft == pytest.approx(0.1)
    assert got.edge_aug_tol == pytest.approx(1.0e-6)


def test_edge_soft_bare_true_roundtrip(tmp_path):
    # edge_soft=True (bare -edgeSoft) must round-trip as True (the soft_mode
    # tri-state), not a float and not None.
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="mortar", eps_n="auto",
        edge_edge=True, edge_soft=True))
    src = fem.elements.contacts[0]
    assert src.edge_soft is True
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    assert got.edge_soft is True


def test_no_edge_edge_defaults_off_roundtrip(tmp_path):
    # a plain mortar contact (no edge knobs) round-trips with the fallback off.
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="mortar", eps_n="auto"))
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    assert got.edge_edge is False
    assert got.edge_kn is None and got.edge_soft is None
    assert got.edge_mu is None and got.edge_aug_tol is None


def test_decode_presence_probes_edge_columns():
    # A row whose payload predates the 2.25.0 edge-edge columns (a genuine
    # 2.24.x file) must still decode → the fallback off. Exercise the
    # presence-probe directly by dropping the edge_* fields from a freshly
    # encoded payload (no on-disk dtype surgery).
    from numpy.lib import recfunctions as rfn

    from apeGmsh._kernel.records._constraints import ContactRecord
    from apeGmsh.mesh._femdata_h5_io import _decode_contact, _encode_contact
    from apeGmsh.mesh._record_h5 import contact_payload_dtype

    rec = ContactRecord(
        kind="contact", formulation="mortar",
        master_faces=np.array([[1, 2, 3]]), master_nps=3,
        slave_faces=np.array([[4, 5, 6]]), slave_nps=3, eps_n="auto")
    full = np.zeros((1,), dtype=contact_payload_dtype())
    full[0] = _encode_contact(rec)
    edge_names = [n for n in full.dtype.names if n.startswith("edge_")]
    trimmed = rfn.drop_fields(full, edge_names)        # repacks without edge_*
    assert not any(n.startswith("edge_") for n in trimmed.dtype.names)
    row = np.zeros((1,), dtype=[("payload", trimmed.dtype)])
    row["payload"] = trimmed
    got = _decode_contact(row[0], ContactRecord)
    assert got.edge_edge is False
    assert got.edge_kn is None and got.edge_soft is None
    assert got.edge_mu is None and got.edge_aug_tol is None
    # the non-edge fields still decode normally
    assert got.formulation == "mortar" and got.eps_n == "auto"


def test_mortar_tie_roundtrip(tmp_path):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="mortar", eps_n=1.0e7, tie=True,
        outward=(0.0, 0.0, 1.0), name="weld"))
    src = fem.elements.contacts[0]
    assert src.tie is True and src.outward == (0.0, 0.0, 1.0)
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    _eq(got, src)
    assert got.tie is True and got.name == "weld"
    assert got.outward == pytest.approx((0.0, 0.0, 1.0))


def test_zero_and_none_penalty_sentinels_roundtrip(tmp_path):
    # kn=0.0 is the fork's inert/zero-force sentinel (distinct from None and
    # from "auto"); kt/mu=None (frictionless) must stay None, not decode to 0.0
    # — the NaN-vs-zero and None-vs-auto-vs-numeric edges the tri-state /
    # NaN-sentinel encoding must preserve.
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=0.0))
    src = fem.elements.contacts[0]
    assert src.kn == 0.0 and src.kt is None and src.mu is None
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.contacts[0]
    assert got.kn == 0.0          # numeric zero, NOT None and NOT "auto"
    assert got.kn != "auto"
    assert got.kt is None and got.mu is None
    _eq(got, src)


def test_contact_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.contacts
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "contacts" not in f                     # group omitted
    assert back.snapshot_id == fem.snapshot_id


def test_contact_snapshot_id_stable_on_roundtrip(tmp_path):
    # snapshot_id excludes the contact overlay (consistent with constraints /
    # ties), so a contact model round-trips with an identical id even though
    # the contacts are read back into the broker.
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=1.0e6))
    back, _ = _roundtrip(fem, tmp_path)
    assert back.snapshot_id == fem.snapshot_id
    assert back.elements.contacts                       # really present


def test_to_h5_no_deferral_warning(tmp_path, recwarn):
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=1.0e6))
    fem.to_h5(str(tmp_path / "m.h5"))
    assert not [w for w in recwarn.list
                if "not persisted" in str(w.message)
                or "deferred" in str(w.message)]


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


def test_reads_prior_minor_file_without_contacts_group_within_window(tmp_path):
    # An in-window prior-minor file with the /contacts group stripped must
    # still read → empty contacts (absence ⇒ no contacts).
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _contact_fem(lambda g: g.constraints.contact(
        "master", "slave", formulation="nts", kn=1.0e6))
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        if "contacts" in f:
            del f["contacts"]
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(p)                               # within window → no raise
    assert back.elements.contacts == []                # absent group → none


# ── encode-side fail-loud hardening ──────────────────────────────────

def _rec(**over):
    from apeGmsh._kernel.records._constraints import ContactRecord
    base = dict(
        kind="contact", formulation="nts",
        master_faces=np.array([[1, 2, 3], [3, 4, 1]]), master_nps=3,
        slave_nodes=[10, 11], kn=1.0e6)
    base.update(over)
    return ContactRecord(**base)


def test_encode_rejects_empty_master_faces():
    from apeGmsh.mesh._femdata_h5_io import _encode_contact
    with pytest.raises(ValueError, match="master_faces is empty|master flat"):
        _encode_contact(_rec(master_faces=np.empty((0, 3), dtype=int)))


def test_encode_rejects_nts_without_slave_nodes():
    from apeGmsh.mesh._femdata_h5_io import _encode_contact
    with pytest.raises(ValueError, match="slave_nodes is empty"):
        _encode_contact(_rec(slave_nodes=[]))


def test_encode_rejects_bad_master_stride():
    from apeGmsh.mesh._femdata_h5_io import _encode_contact
    # 5 flat nodes with nps=3 is not a multiple of the stride
    with pytest.raises(ValueError, match="master flat length"):
        _encode_contact(_rec(
            master_faces=np.array([1, 2, 3, 4, 5]), master_nps=3))
