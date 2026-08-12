"""Compose carries oriented coincident-pair zeroLength interfaces
(ADR 0093 records under ADR 0038 compose, S6).

A module's ``elements.interfaces`` arrive on the host offset into the
module's reservation window with namespaced names, the host's own
interfaces survive the merge, and each record's own geometry follows the
module's rotate+translate per INV-2 — **both** ``orient`` direction
vectors rotated, ``phantom_coords`` rotated and translated. Until S6
compose refused outright rather than merge a model whose springs had
silently vanished.

The rotation cases are load-bearing, not decoration: a translate-only
test greens a broken rotation, and a rotated interface whose local-x no
longer follows the master face is compression-only in the wrong
direction — the silent failure class ADR 0093 exists to kill.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.mesh.FEMData import FEMData

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)
THICKNESS = 0.5

#: The verb's own frame on this fixture: the left square's right edge
#: has outward normal +x, and local-y is the right-handed completion.
BASE_ORIENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _interface_module_h5(path, *, slave_ndf=None, name="RockLiner", n=2):
    """Two abutting squares with one interface across x=1; saved to H5.

    Returns the source InterfaceRecord list for comparison.
    """
    with apeGmsh(model_name="iface_mod", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(2, [right], name="liner")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name=name)
        fem = g.mesh.queries.get_fem_data()
        fem.to_h5(str(path))
        assert len(fem.elements.interfaces) == 2
        return list(fem.elements.interfaces)


def _plain_host_h5(path):
    with apeGmsh(model_name="plain_host", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(9, 9, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, surf)], n=2)
        g.mesh.generation.generate(2)
        g.physical.add(2, [surf], name="host")
        g.mesh.queries.get_fem_data(dim=2).to_h5(str(path))


def _compose_and_reload(host, mod, tmp_path, out="out.h5", **compose_kw):
    g = apeGmsh.from_h5(str(host))
    g.compose(str(mod), **compose_kw)
    p = tmp_path / out
    g.save(str(p))
    return FEMData.from_h5(str(p))


# ======================================================================
# Tag rewrite — nodes AND the backing element
# ======================================================================
def test_compose_carries_module_interface_with_offset_tags(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _interface_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="A", translate=(3.0, 0.0, 0.0))

    got = merged.elements.interfaces
    assert len(got) == len(src)
    offs = set()
    for a, b in zip(got, src):
        assert a.name == "A.RockLiner"                 # namespaced
        assert a.normal_law == b.normal_law
        assert a.tangential_law == b.tangential_law
        assert a.a_trib == pytest.approx(b.a_trib)
        offs.add(a.master_node - b.master_node)
        offs.add(a.slave_node - b.slave_node)
        # the element tag rides the SAME single offset as the node tags
        offs.add(a.backing_element - b.backing_element)
    assert len(offs) == 1 and next(iter(offs)) > 0

    # the rewritten tags really exist on the merged model
    node_ids = set(np.asarray(merged.nodes.ids, dtype=np.int64).tolist())
    elem_ids = {
        int(t) for grp in merged.elements._groups.values() for t in grp.ids
    }
    for a in got:
        assert a.master_node in node_ids and a.slave_node in node_ids
        assert a.backing_element in elem_ids


def test_compose_preserves_host_interface_with_plain_module(tmp_path):
    iface_host = tmp_path / "iface_host.h5"
    plain_mod = tmp_path / "plain_mod.h5"
    src = _interface_module_h5(iface_host)
    _plain_host_h5(plain_mod)

    merged = _compose_and_reload(
        iface_host, plain_mod, tmp_path, label="B",
        translate=(3.0, 0.0, 0.0))

    got = merged.elements.interfaces
    assert len(got) == len(src)
    for a, b in zip(got, src):
        assert a.name == "RockLiner"                   # host-owned
        assert a.master_node == b.master_node
        assert a.slave_node == b.slave_node
        assert a.backing_element == b.backing_element
        assert tuple(a.orient) == pytest.approx(tuple(b.orient))


def test_compose_both_sides_carry_interfaces(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _interface_module_h5(a, name="HostFace")
    _interface_module_h5(b, name="ModFace")

    merged = _compose_and_reload(
        a, b, tmp_path, label="M", translate=(4.0, 0.0, 0.0))

    names = sorted({r.name for r in merged.elements.interfaces})
    assert names == ["HostFace", "M.ModFace"]
    assert len(merged.elements.interfaces) == 4


# ======================================================================
# INV-2 — the per-pair frame survives a ROTATION
# ======================================================================
def test_compose_rotates_both_orient_vectors(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _interface_module_h5(mod)
    _plain_host_h5(host)
    assert tuple(src[0].orient) == pytest.approx(BASE_ORIENT)

    # +90 deg about z: local-x (1,0,0) -> (0,1,0);
    #                  local-y (0,1,0) -> (-1,0,0).
    # The translate must NOT move either — they are directions.
    merged = _compose_and_reload(
        host, mod, tmp_path, label="R",
        translate=(0.0, 7.0, 0.0), rotate=(0.0, 0.0, 1.0, math.pi / 2.0))

    for rec in merged.elements.interfaces:
        assert tuple(rec.orient) == pytest.approx(
            (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), abs=1e-12)


def test_compose_rotated_orient_still_points_out_of_the_master(tmp_path):
    """The check a numeric-only rotation assertion cannot make: after the
    rotation the record's local-x must still point from the master face
    AWAY from the continuum it backs (INV-1 on the rotated frame).

    A rewrite that rotated the node coords but left ``orient`` alone
    would pass every tag test and every "the deck has 2 zeroLengths"
    test, and produce a tension-only interface that converges.
    """
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    _interface_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="R",
        translate=(0.0, 7.0, 0.0), rotate=(0.0, 0.0, 1.0, math.pi / 2.0))

    xyz = {
        int(t): merged.nodes.coords[i]
        for i, t in enumerate(np.asarray(merged.nodes.ids).tolist())
    }
    conn_of = {}
    for grp in merged.elements._groups.values():
        for eid, row in zip(np.asarray(grp.ids).tolist(),
                            np.asarray(grp.connectivity)):
            conn_of[int(eid)] = [int(x) for x in row]

    for rec in merged.elements.interfaces:
        local_x = np.asarray(rec.orient[:3], dtype=float)
        node = np.asarray(xyz[int(rec.master_node)], dtype=float)
        centroid = np.mean(
            [xyz[n] for n in conn_of[int(rec.backing_element)]], axis=0)
        # outward ⇒ the backing element's centroid sits BEHIND local-x
        assert float(np.dot(centroid - node, local_x)) < 0.0


def test_compose_translate_only_keeps_orient(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _interface_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="T", translate=(0.0, 0.0, 0.0))

    for a, b in zip(merged.elements.interfaces, src):
        assert tuple(a.orient) == pytest.approx(tuple(b.orient))


# ======================================================================
# Phantom bridge — coords transform, tags stay collision-free
# ======================================================================
def test_compose_rotates_and_translates_phantom_coords(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _interface_module_h5(mod, slave_ndf=3)
    _plain_host_h5(host)
    assert all(r.phantom_coords is not None for r in src)

    rotate = (0.0, 0.0, 1.0, math.pi / 2.0)
    translate = (0.0, 7.0, 0.0)
    merged = _compose_and_reload(
        host, mod, tmp_path, label="P",
        translate=translate, rotate=rotate)

    for a, b in zip(merged.elements.interfaces, src):
        x, y, z = (float(v) for v in b.phantom_coords)
        # rotate +90 about z: (x, y) -> (-y, x); then translate.
        expected = (-y + translate[0], x + translate[1], z + translate[2])
        assert tuple(float(v) for v in a.phantom_coords) == pytest.approx(
            expected, abs=1e-12)
        # …and the phantom node now sits on its real coordinates: the
        # slave node it bridges to moved the same way.
        assert a.phantom_ndf == 2


def test_two_phantom_carrying_models_do_not_collide(tmp_path):
    """The S6 sharp edge: both models mint phantoms from their OWN
    ``max(node_tag) + 1``, so the raw tags overlap. After the merge the
    module's phantoms must have moved into its reservation window, and
    no phantom may land on a real node of the merged model.
    """
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    src_a = _interface_module_h5(a, slave_ndf=3, name="HostFace")
    src_b = _interface_module_h5(b, slave_ndf=3, name="ModFace")
    # the raw phantom ranges really DO overlap before the rewrite
    assert ({int(r.phantom_node) for r in src_a}
            & {int(r.phantom_node) for r in src_b})

    merged = _compose_and_reload(
        a, b, tmp_path, label="M", translate=(4.0, 0.0, 0.0))

    recs = merged.elements.interfaces
    assert len(recs) == 4
    phantoms = [int(r.phantom_node) for r in recs]
    assert len(set(phantoms)) == len(phantoms)         # all distinct
    # …and distinct from every real node tag, so the emitted ``node``
    # lines cannot redeclare an existing node.
    node_ids = set(np.asarray(merged.nodes.ids, dtype=np.int64).tolist())
    assert not (set(phantoms) & node_ids)
    # the module's phantoms rode the node rewrite into its window
    host_phantoms = {int(r.phantom_node) for r in recs
                     if r.name == "HostFace"}
    mod_phantoms = {int(r.phantom_node) for r in recs
                    if r.name == "M.ModFace"}
    assert min(mod_phantoms) > max(host_phantoms)
    # each nested equalDOF still points at its own phantom
    for r in recs:
        assert r.equal_dof_records[0].slave_node == r.phantom_node
        assert r.equal_dof_records[0].master_node == r.slave_node


def test_post_compose_reresolution_mints_above_the_composed_phantoms(tmp_path):
    """A composed model whose phantoms sit above ``max(node_tag)`` is the
    one shape where a later mint could re-use a live tag. Pin that the
    merged model's phantom high-water is visible to a caller: every
    composed phantom is inside the module's reserved window, which sits
    strictly above the host's tags, so the next reservation (and any
    re-resolve on a re-extracted session) starts above it.
    """
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _interface_module_h5(a, slave_ndf=3, name="HostFace")
    _interface_module_h5(b, slave_ndf=3, name="ModFace")

    merged = _compose_and_reload(
        a, b, tmp_path, label="M", translate=(4.0, 0.0, 0.0))

    rec = next(r for r in merged.elements.interfaces if r.name == "M.ModFace")
    base = min(
        int(t) for t in np.asarray(merged.nodes.ids).tolist()
        if int(t) >= 1_000_000
    )
    # module tags live at/above the 1M reservation base; the phantom
    # rode the same offset and stays inside that window.
    assert rec.phantom_node >= base
    assert rec.phantom_node < base + 1_000_000


# ======================================================================
# End-to-end: compose (one module ROTATED) -> h5 -> reload -> emit
# ======================================================================
def test_composed_rotated_model_emits_both_interface_blocks(tmp_path):
    from apeGmsh.opensees import apeSees

    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    src_host = _interface_module_h5(a, name="HostFace")
    src_mod = _interface_module_h5(b, name="ModFace")

    merged = _compose_and_reload(
        a, b, tmp_path, label="M",
        translate=(0.0, 7.0, 0.0), rotate=(0.0, 0.0, 1.0, math.pi / 2.0))

    recs = merged.elements.interfaces
    assert len(recs) == 4
    host_recs = [r for r in recs if r.name == "HostFace"]
    mod_recs = [r for r in recs if r.name == "M.ModFace"]

    # The rotated module's frame really rotated; the host's did not.
    for r in host_recs:
        assert tuple(r.orient) == pytest.approx(BASE_ORIENT)
    for r in mod_recs:
        assert tuple(r.orient) == pytest.approx(
            (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), abs=1e-12)
    # …and the module's tags really moved.
    for got, src in zip(mod_recs, src_mod):
        assert got.master_node > src.master_node
        assert got.backing_element > src.backing_element

    ops = apeSees(merged)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner", "M.rock", "M.liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    path = tmp_path / "deck.tcl"
    # ``flat=True``: a chain-phase compose leaves the merged model
    # carrying one gmsh partition per module (pre-existing, unrelated to
    # interfaces — a plain-on-plain compose does the same), and every
    # serial-only side list refuses the partitioned path. ``flat`` is the
    # sanctioned escape hatch (see test_interface_emit_e2e.py's
    # ``test_flat_escape_hatch_...``); ADR 0093 S8 lifts the refusal.
    ops.tcl(str(path), flat=True)
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    assert "getPID" not in "\n".join(lines)

    zl = [ln for ln in lines if ln.startswith("element zeroLength ")]
    assert len(zl) == 4                                # BOTH sides emitted

    # Every deck zeroLength matches a record: same (iNode, jNode) pair
    # and the same -orient the record carries after the transform.
    want = {}
    for r in recs:
        want[(int(r.master_node), int(r.slave_node))] = [
            float(v) for v in r.orient]
    seen = {}
    for ln in zl:
        tok = ln.split()
        i = tok.index("-orient")
        seen[(int(tok[3]), int(tok[4]))] = [
            float(v) for v in tok[i + 1:i + 7]]
    assert set(seen) == set(want)
    for key, orient in seen.items():
        assert orient == pytest.approx(want[key], abs=1e-12)

    # Sanity on the host side: unchanged tags, unchanged frame.
    for r in src_host:
        assert (int(r.master_node), int(r.slave_node)) in seen
