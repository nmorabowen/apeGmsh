"""Compose carries fork contact interactions (ADR 0073 records under
ADR 0038 compose). A module's ``elements.contacts`` / ``contact_planes``
arrive on the host offset into the module's reservation window with
namespaced names, the host's own contacts survive the merge, and each
record's own geometry (``outward`` / ``normal`` / ``point``) follows the
module's rotate+translate. Previously BOTH sides were silently dropped:
the bundle never read the contact streams and the merge rebuilt
``ElementComposite`` without them — a composed model emitted a deck with
zero ``contact`` verbs and solved as an unbonded free-body problem."""
from __future__ import annotations

import math

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _contact_module_h5(path, **contact_kw):
    """Two stacked boxes with one NTS contact across the gap; saved to H5.

    Returns the source ContactRecord for comparison.
    """
    with apeGmsh(model_name="contact_mod", verbose=False) as g:
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.constraints.contact(
            "master", "slave", formulation="nts",
            kn=1.0e6, kt=5.0e5, mu=0.3, name="weld", **contact_kw)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(str(path))
        assert len(fem.elements.contacts) == 1
        return fem.elements.contacts[0]


def _plane_module_h5(path):
    """One box whose bottom face contacts a rigid plane; saved to H5.

    Returns the source ContactPlaneRecord for comparison.
    """
    with apeGmsh(model_name="plane_mod", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        bottom = _face_at_z(box, 0.0)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [bottom], name="sole")
        g.constraints.contact_plane(
            "sole", normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0),
            kn=2.0e6, name="floor")
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(str(path))
        assert len(fem.elements.contact_planes) == 1
        return fem.elements.contact_planes[0]


def _plain_host_h5(path):
    with apeGmsh(model_name="plain_host", verbose=False) as g:
        box = g.model.geometry.add_box(5, 5, 5, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(path))


def _compose_and_reload(host, mod, tmp_path, **compose_kw):
    g = apeGmsh.from_h5(str(host))
    g.compose(str(mod), **compose_kw)
    out = tmp_path / "out.h5"
    g.save(str(out))
    return FEMData.from_h5(str(out))


def test_compose_carries_module_contact_with_offset_tags(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _contact_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="A", translate=(3.0, 0.0, 0.0))

    assert len(merged.elements.contacts) == 1
    got = merged.elements.contacts[0]
    assert got.name == "A.weld"                       # namespaced
    assert got.formulation == "nts"
    assert got.kn == pytest.approx(1.0e6)
    assert got.mu == pytest.approx(0.3)
    # every node tag shifted by ONE constant positive offset vs the source
    d_master = (np.asarray(got.master_faces, dtype=np.int64)
                - np.asarray(src.master_faces, dtype=np.int64))
    offs = set(np.unique(d_master).tolist())
    offs |= {g - s for g, s in zip(got.slave_nodes, src.slave_nodes)}
    assert len(offs) == 1 and next(iter(offs)) > 0
    # the rewritten tags actually exist on the merged model
    node_ids = set(np.asarray(merged.nodes.ids, dtype=np.int64).tolist())
    assert set(np.asarray(got.master_faces).ravel().tolist()) <= node_ids
    assert set(int(n) for n in got.slave_nodes) <= node_ids


def test_compose_preserves_host_contact_with_plain_module(tmp_path):
    contact_host = tmp_path / "contact_host.h5"
    plain_mod = tmp_path / "plain_mod.h5"
    src = _contact_module_h5(contact_host)
    _plain_host_h5(plain_mod)

    merged = _compose_and_reload(
        contact_host, plain_mod, tmp_path, label="B",
        translate=(3.0, 0.0, 0.0))

    assert len(merged.elements.contacts) == 1
    got = merged.elements.contacts[0]
    assert got.name == "weld"                         # host-owned: unprefixed
    np.testing.assert_array_equal(
        np.asarray(got.master_faces), np.asarray(src.master_faces))
    assert list(got.slave_nodes) == list(src.slave_nodes)


def test_compose_both_sides_carry_contacts(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _contact_module_h5(a)
    _contact_module_h5(b)

    merged = _compose_and_reload(
        a, b, tmp_path, label="M", translate=(4.0, 0.0, 0.0))

    names = sorted(c.name for c in merged.elements.contacts)
    assert names == ["M.weld", "weld"]


def test_compose_rotates_contact_outward(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    _contact_module_h5(mod, outward=(0.0, 0.0, 1.0))
    _plain_host_h5(host)

    # +90 deg about x: (0,0,1) -> (0,-1,0); translate must NOT move it
    merged = _compose_and_reload(
        host, mod, tmp_path, label="A",
        translate=(0.0, 9.0, 0.0), rotate=(1.0, 0.0, 0.0, math.pi / 2.0))

    got = merged.elements.contacts[0]
    assert got.outward == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)


def test_compose_transforms_contact_plane_geometry(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    _plane_module_h5(mod)
    _plain_host_h5(host)

    # rotate +90 deg about x, then translate (0, 9, 0):
    # normal (0,0,1) -> (0,-1,0) (direction: rotate only);
    # point (0,0,0) -> (0,9,0) (position: rotate + translate).
    merged = _compose_and_reload(
        host, mod, tmp_path, label="P",
        translate=(0.0, 9.0, 0.0), rotate=(1.0, 0.0, 0.0, math.pi / 2.0))

    assert len(merged.elements.contact_planes) == 1
    got = merged.elements.contact_planes[0]
    assert got.name == "P.floor"
    assert got.kn == pytest.approx(2.0e6)
    assert got.normal == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert got.point == pytest.approx((0.0, 9.0, 0.0), abs=1e-12)


def test_compose_translate_only_keeps_plane_normal(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _plane_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="P", translate=(0.0, 0.0, 2.0))

    got = merged.elements.contact_planes[0]
    assert got.normal == pytest.approx(tuple(src.normal))
    assert got.point == pytest.approx((0.0, 0.0, 2.0))
    # slave node set shifted by one constant offset
    offs = {g - s for g, s in zip(got.slave_nodes, src.slave_nodes)}
    assert len(offs) == 1 and next(iter(offs)) > 0


def test_live_session_compose_keeps_contact_on_reextraction(tmp_path):
    """The probe-2 hole: a LIVE session with a resolved contact composes a
    module, and the post-compose ``get_fem_data`` (which re-extracts and
    re-applies every bundle through the merge) must keep the contact."""
    plain = tmp_path / "plain.h5"
    _plain_host_h5(plain)

    with apeGmsh(model_name="live", verbose=False) as g:
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.constraints.contact("master", "slave", formulation="nts", kn=1.0e6)
        fem0 = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem0.elements.contacts) == 1

        g.compose(str(plain), label="M", translate=(10.0, 0.0, 0.0))
        fem1 = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem1.elements.contacts) == 1

        out = tmp_path / "live_out.h5"
        g.save(str(out))
    assert len(FEMData.from_h5(str(out)).elements.contacts) == 1
