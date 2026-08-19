"""2D contact — the wound chain, end to end (fork ADR-85 adoption, slice A).

The fork's 2D contact surface is declared as a **flat stride-2 pair list
chained head-to-tail**::

    contactSurface 10 -master 2  101 102  102 103  103 104   ;# 3 segments

The natural-looking shorthand ``-master 2 101 102 103 104`` is *silently
legal* fork-side and declares two DISJOINT segments with a hole where the
middle one should be — the deck converges, balances its reactions, and
transmits the load through the wrong distribution. apeGmsh generates this
connectivity from a meshed PG, so this file pins that the holed form is
unreachable by construction.

It also pins the dimension gate. Before it, a 2D model that named its
dim-2 plane PG (the natural user mistake) collected the continuum's own
tri3/quad4 elements as "facets" and built a structurally valid
``ContactRecord`` out of SOLID elements, uncaught all the way to emit.

The chain / winding math itself is pinned on bare arrays in
``tests/_kernel/geometry/test_boundary_chain.py``; the fixture here is the
two-squares topology of ``tests/mesh/test_interface_verb.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

THICKNESS = 0.5


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _build_two_squares(g, n: int = 4, order: int = 1):
    """Left square [0,1]^2 + right square [1,2]^2, un-fragmented, so the
    two curves at x=1 carry coincident but distinct node sets — exactly
    the 2D contact topology."""
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    if order != 1:
        gmsh.model.mesh.setOrder(order)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    return left, right


def _contact_fem(master="face", slave="wire", *, n=4, order=1,
                 partition=0, **contact_kw):
    # ``outward=(1, 0)`` is not decoration: the two squares meet flush at
    # x=1, so their surface centroids coincide and the fork's 2D
    # orientation vote is genuinely ambiguous — slice B refuses that deck
    # by name rather than emitting one that aborts at handle(). +x is the
    # direction from the master (the LEFT square's right edge) toward the
    # slave. See test_flush_without_orientation_is_refused_by_name.
    kw = dict(formulation="nts", kn=1.0e6, kt=5.0e5, mu=0.3, name="joint",
              outward=(1.0, 0.0))
    kw.update(contact_kw)
    with apeGmsh(model_name="contact2d", verbose=False) as g:
        _build_two_squares(g, n=n, order=order)
        g.constraints.contact(master, slave, **kw)
        if partition:
            g.mesh.partitioning.partition(partition)
        return g.mesh.queries.get_fem_data(dim=2)


def _quad_ops(fem, ndm=2):
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndm)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(pg=pg, thickness=THICKNESS, material=mat)
    return ops


# =====================================================================
# The record: nps=2 and chained
# =====================================================================

def test_two_squares_yield_a_chained_nps2_record():
    fem = _contact_fem()
    assert len(fem.elements.contacts) == 1
    rec = fem.elements.contacts[0]
    assert rec.master_nps == 2
    assert rec.ndm == 2                       # derived, never stored
    faces = np.asarray(rec.master_faces)
    assert faces.shape == (3, 2)              # n=4 divisions ⇒ 3 segments
    for k in range(faces.shape[0] - 1):
        assert int(faces[k, 1]) == int(faces[k + 1, 0])


def test_chain_follows_the_geometry_monotonically():
    """The chain is a real traversal of the curve, not just a valid graph."""
    fem = _contact_fem()
    coords = {int(t): c for t, c in zip(fem.nodes.ids, fem.nodes.coords)}
    faces = np.asarray(fem.elements.contacts[0].master_faces)
    ys = [coords[int(faces[0, 0])][1]] + [
        coords[int(b)][1] for b in faces[:, 1]]
    assert ys == sorted(ys) or ys == sorted(ys, reverse=True)
    assert all(abs(coords[int(t)][0] - 1.0) < 1e-9
               for t in faces.reshape(-1))


def test_get_fem_data_is_deterministic():
    a = np.asarray(_contact_fem().elements.contacts[0].master_faces)
    b = np.asarray(_contact_fem().elements.contacts[0].master_faces)
    np.testing.assert_array_equal(a, b)


# =====================================================================
# The emitted deck: SIX tags for three segments, chained
# =====================================================================

def test_deck_emits_six_tags_for_three_segments(tmp_path):
    fem = _contact_fem()
    path = tmp_path / "deck.tcl"
    _quad_ops(fem).tcl(str(path))
    lines = [ln for ln in path.read_text().splitlines()
             if "contactSurface" in ln and "-master" in ln]
    assert len(lines) == 1
    toks = lines[0].split()
    assert toks[0] == "contactSurface"
    assert toks[2] == "-master" and toks[3] == "2"
    tags = [int(t) for t in toks[4:]]
    assert len(tags) == 6                     # NOT 4 — the holed shorthand
    pairs = list(zip(tags[0::2], tags[1::2]))
    for k in range(len(pairs) - 1):
        assert pairs[k][1] == pairs[k + 1][0]


def test_deck_never_emits_the_holed_four_tag_form(tmp_path):
    """The single highest-value guarantee of this lane.

    ``-master 2 101 102 103 104`` is silently legal fork-side and declares
    a HOLED surface; the fork's own chain scan structurally cannot refuse
    it (an even tag count with no repeat is indistinguishable from a
    genuinely disjoint surface, which is legitimate). apeGmsh owns this
    connectivity, so it must never be able to write that shape.
    """
    fem = _contact_fem()
    path = tmp_path / "deck.tcl"
    _quad_ops(fem).tcl(str(path))
    line = next(ln for ln in path.read_text().splitlines()
                if "-master 2" in ln)
    tags = [int(t) for t in line.split()[4:]]
    chain_nodes = [int(x) for x in
                   np.asarray(fem.elements.contacts[0].master_faces)[:, 0]]
    chain_nodes.append(
        int(np.asarray(fem.elements.contacts[0].master_faces)[-1, 1]))
    # the holed shorthand would be the bare node chain, four tags
    assert tags != chain_nodes
    assert len(tags) == 2 * (len(chain_nodes) - 1)
    # every interior node appears exactly twice, endpoints exactly once
    _, counts = np.unique(tags, return_counts=True)
    assert sorted(counts.tolist()) == [1, 1, 2, 2]


# =====================================================================
# Persistence
# =====================================================================

def test_nps2_record_round_trips_h5(tmp_path):
    fem = _contact_fem()
    path = tmp_path / "model.h5"
    fem.to_h5(str(path))
    from apeGmsh.mesh.FEMData import FEMData
    back = FEMData.from_h5(str(path))
    got = back.elements.contacts[0]
    src = fem.elements.contacts[0]
    assert got.master_nps == 2 and got.ndm == 2
    np.testing.assert_array_equal(
        np.asarray(got.master_faces), np.asarray(src.master_faces))
    assert got.slave_nodes == src.slave_nodes


# =====================================================================
# The dimension gate
# =====================================================================

def test_naming_the_dim2_plane_pg_as_master_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _contact_fem(master="rock")
    msg = str(exc.value)
    assert "the model is 2D" in msg
    assert "must be the meshed interface CURVE" in msg
    assert "SOLID elements" in msg
    # NOT the old, misleading fall-through
    assert "carries no surface mesh faces" not in msg


def test_dim2_slave_pg_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _contact_fem(slave="liner")
    msg = str(exc.value)
    assert "slave label 'liner'" in msg
    assert "INTERIOR node" in msg


def test_2d_mortar_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="2D mortar"):
        _contact_fem(formulation="mortar", kn=None, kt=None, mu=None,
                     eps_n="auto")


def test_dim1_master_in_a_3d_model_is_refused_at_the_gate():
    """A dim-1 master in 3D must refuse by name, not fall through to
    'carries no surface mesh faces (is it meshed?)'."""
    with apeGmsh(model_name="contact2d_in3d", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        edge = gmsh.model.getBoundary(
            gmsh.model.getBoundary([(3, box)], oriented=False)[:1],
            oriented=False)[0][1]
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(1, [abs(int(edge))], name="wire")
        g.physical.add(2, [1], name="face")
        g.constraints.contact("wire", "face", formulation="nts", kn=1e6)
        with pytest.raises(ValueError) as exc:
            g.mesh.queries.get_fem_data(dim=3)
    msg = str(exc.value)
    assert "the model is 3D" in msg
    assert "dim-2 face physical group" in msg
    assert "carries no surface mesh faces" not in msg


# =====================================================================
# line3 → line2 corner drop
# =====================================================================

def test_line3_master_drops_to_corner_segments():
    fem = _contact_fem(order=2)
    rec = fem.elements.contacts[0]
    assert rec.master_nps == 2
    faces = np.asarray(rec.master_faces)
    assert faces.shape == (3, 2)
    for k in range(faces.shape[0] - 1):
        assert int(faces[k, 1]) == int(faces[k + 1, 0])


def test_gmsh_orders_line3_corner_nodes_first():
    """The corner drop takes the LEADING columns, which assumes gmsh lists
    a line3's two corners before its mid-side node.

    ``ConstraintsComposite`` already claims this for surfaces; it was never
    verified for ``line3`` specifically, and the whole 2D corner drop rests
    on it.
    """
    with apeGmsh(model_name="line3_order", verbose=False) as g:
        _build_two_squares(g, n=3, order=2)
        curve = _curve_at_x(1, 1.0)
        etypes, _, enodes = gmsh.model.mesh.getElements(1, curve)
        rows = None
        for etype, conn in zip(etypes, enodes):
            _, _, _, npe, *_ = gmsh.model.mesh.getElementProperties(int(etype))
            assert int(npe) == 3, "expected a line3 mesh at order 2"
            rows = np.asarray(conn, dtype=int).reshape(-1, 3)
        coord = {}
        for row in rows:
            for t in row:
                c, *_ = gmsh.model.mesh.getNode(int(t))
                coord[int(t)] = np.asarray(c, dtype=float)
        for a, b, mid in rows:
            expect = 0.5 * (coord[int(a)] + coord[int(b)])
            np.testing.assert_allclose(coord[int(mid)], expect, atol=1e-9)


# =====================================================================
# Guards outside the resolver
# =====================================================================

def test_2d_contact_under_partitioning_is_refused_by_name(tmp_path):
    fem = _contact_fem(partition=2)
    assert len(fem.partitions) == 2
    with pytest.raises(BridgeError) as exc:
        _quad_ops(fem).tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "2D line-segment contact (master_nps=2)" in msg
    assert "out of scope" in msg


# =====================================================================
# Orientation (slice B)
# =====================================================================
#
# The fork's 2D lanes take their normal sign from ONE interface-level
# centroid vote. That vote is ambiguous on a flush interface — which in 2D
# is the workhorse case, not an edge case — and it is bypassed entirely
# under declared winding. Both holes are apeGmsh's to close.


def test_flush_without_orientation_is_refused_by_name():
    """The two squares meet flush at x=1, so the fork's centroid datum is
    exactly zero and handle() aborts. Refuse here instead, naming both
    surfaces and the call to add."""
    with pytest.raises(ValueError) as exc:
        _contact_fem(outward=None)
    msg = str(exc.value)
    assert "FLUSH" in msg
    assert "master 'face'" in msg and "slave 'wire'" in msg
    assert "outward=(ox, oy)" in msg
    assert "outward='winding'" in msg


def test_flush_refusal_names_the_contact_when_it_has_one():
    with pytest.raises(ValueError, match=r"^contact 'joint':"):
        _contact_fem(outward=None)


def test_wrong_side_master_is_refused_by_name():
    """The master named on the FAR side of the block (x=0), with the slave
    a whole square away at x=1. Under declared winding the fork's centroid
    vote is bypassed, so nothing downstream would catch this — the deck
    would converge against a boundary that never meets the slave."""
    with apeGmsh(model_name="contact2d_wrong_side", verbose=False) as g:
        left, _ = _build_two_squares(g)
        g.physical.add(1, [_curve_at_x(left, 0.0)], name="far")
        g.constraints.contact("far", "wire", formulation="nts", kn=1.0e6,
                              outward="winding", name="backwards")
        with pytest.raises(ValueError) as exc:
            g.mesh.queries.get_fem_data(dim=2)
    msg = str(exc.value)
    assert "FACE AWAY from the slave" in msg
    assert "BYPASSES the fork's own centroid vote" in msg


def test_flush_interface_does_not_trip_the_wrong_side_guard():
    """The guard must be silent on the case winding exists to unlock — the
    default fixture IS a flush interface."""
    rec = _contact_fem().elements.contacts[0]
    assert rec.master_nps == 2                    # resolved, not refused


# ── the 2-component -outward the fork's 2D lane requires ────────────

def test_2d_outward_emits_two_components(tmp_path):
    """The fork picks the arity from the nodes' getCrds().Size() and then
    checks the trailing token, so a stray oz kills the deck at parse."""
    fem = _contact_fem(outward=(1.0, 0.0))
    assert fem.elements.contacts[0].outward == (1.0, 0.0, 0.0)   # z-padded
    path = tmp_path / "deck.tcl"
    _quad_ops(fem).tcl(str(path))
    line = next(ln for ln in path.read_text().splitlines()
                if ln.startswith("contact ") and "-outward" in ln)
    toks = line.split()
    assert toks[toks.index("-outward"):] == ["-outward", "1.0", "0.0"]


def test_2d_outward_with_a_nonzero_oz_is_refused():
    with pytest.raises(ValueError, match="non-zero"):
        _contact_fem(outward=(1.0, 0.0, 0.5))


# ── outward="winding" (fork F1) ─────────────────────────────────────

def test_winding_record_and_deck():
    fem = _contact_fem(outward="winding")
    assert fem.elements.contacts[0].outward == "winding"


def test_winding_emits_the_keyword(tmp_path):
    fem = _contact_fem(outward="winding")
    path = tmp_path / "deck.tcl"
    _quad_ops(fem).tcl(str(path))
    line = next(ln for ln in path.read_text().splitlines()
                if ln.startswith("contact ") and "-outward" in ln)
    assert line.split()[-2:] == ["-outward", "winding"]


def test_winding_round_trips_h5(tmp_path):
    fem = _contact_fem(outward="winding")
    path = tmp_path / "model.h5"
    fem.to_h5(str(path))
    from apeGmsh.mesh.FEMData import FEMData
    assert FEMData.from_h5(str(path)).elements.contacts[0].outward == "winding"


def test_winding_is_refused_in_a_3d_model():
    """A 3D master is a facet SET with no head-to-tail chain to wind, and
    the 3D kernel already derives a correct per-facet normal."""
    with apeGmsh(model_name="winding_in_3d", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        faces = [abs(t) for _, t in gmsh.model.getBoundary(
            [(3, box)], oriented=False)]
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [faces[0]], name="m3d")
        g.physical.add(2, [faces[1]], name="s3d")
        g.constraints.contact("m3d", "s3d", formulation="nts", kn=1.0e6,
                              outward="winding")
        with pytest.raises(ValueError) as exc:
            g.mesh.queries.get_fem_data(dim=3)
    msg = str(exc.value)
    assert "outward='winding'" in msg and "the model is 3D" in msg
    assert "2D NTS lane only" in msg


def test_winding_is_refused_on_the_mortar_lane_at_declaration():
    """F1 shipped winding on the NTS lane only — the fork's 2D mortar lane
    has no chain scan to rest it on, so a flush mortar interface still
    needs an explicit outward=(ox, oy)."""
    from apeGmsh._kernel.defs.constraints import ContactDef
    with pytest.raises(ValueError, match="NTS-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n="auto", outward="winding")
