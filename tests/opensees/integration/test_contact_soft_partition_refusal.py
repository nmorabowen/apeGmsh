"""ADR 0092 S3 — the `soft=` refusal (INV-3) + the `-kn auto` token pin.

Under partitioning the explicit SOFT penalty is structurally unavailable:
k_soft = SOFSCL*4*m_eff/dt^2 needs the ASSEMBLED mass of BOTH contact
surfaces, and the ghosted side of the interface contributes zero assembled
mass on the owner rank — no owner rule can fix that (fork ADR-78 D4; the
fork engine refuses -soft/-edgeSoft under MPI at handle() time, ADR-78 P2).
These tests lock the emit-side half: a NAMED refusal in
``BuiltModel._emit_partitioned`` that fires BEFORE the blanket serial-only
contact refusal (so it survives S4 relaxing the blanket), stays silent on
every serial path, and covers every SOFT-family knob the bridge exposes
(``soft=`` on contact + contact_plane, ``edge_soft=`` on the mortar
edge-edge fallback).

INV-3's preserved half is pinned too: ``kn="auto"`` is NEVER rewritten to a
literal — a partition-carrying model emits the exact same ``auto`` token as
its serial twin. Today the only emit lane a partitioned contact model can
take is the sanctioned serial escape hatch (``flat=True``); at S4 the
per-rank owner block must carry the identical token, so the twin assertion
here extends naturally.

Emit-time only — no fork build needed (every guard fires before any ops
command).
"""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _two_box_fem(model_name: str, *, partition: bool, **contact_kw):
    """Two stacked boxes + one contact interaction (kwargs pass through)."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
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
        g.constraints.contact("master", "slave", **contact_kw)
        if partition:
            g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _plane_fem(model_name: str, *, partition: bool, **plane_kw):
    """One box + a rigid-plane contact on its bottom face."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        bottom = _face_at_z(box, 0.0)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [bottom], name="floor")
        g.constraints.contact_plane(
            "floor", normal=(0, 0, 1), point=(0, 0, 0), kn=1.0e7, **plane_kw)
        if partition:
            g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="solid", material=mat)
    return ops


# ── the refusal fires under partitioning, with the NAMED message ─────


def _assert_s3_refusal(excinfo) -> None:
    msg = str(excinfo.value)
    assert "ADR 0092" in msg                     # the named error, not the
    assert "serial-only" not in msg              # blanket serial-only one
    assert "ASSEMBLED mass of BOTH contact surfaces" in msg
    assert "ghosted side" in msg
    assert "kn='auto'" in msg                    # the suggested workaround


def test_soft_contact_under_partitioning_refused_named(tmp_path):
    fem = _two_box_fem("s3_soft_nts", partition=True,
                       formulation="nts", kn=1.0e6, soft=True,
                       name="pound")
    assert len(fem.partitions) == 2
    assert fem.elements.contacts and fem.elements.contacts[0].soft is True
    with pytest.raises(BridgeError) as excinfo:
        _ops(fem).tcl(str(tmp_path / "deck.tcl"))
    _assert_s3_refusal(excinfo)
    msg = str(excinfo.value)
    assert "g.constraints.contact interaction #1 ('pound')" in msg
    assert "soft=" in msg


def test_edge_soft_mortar_under_partitioning_refused(tmp_path):
    fem = _two_box_fem("s3_edge_soft", partition=True,
                       formulation="mortar", eps_n="auto",
                       edge_edge=True, edge_soft=0.2)
    assert fem.elements.contacts[0].edge_soft == 0.2
    with pytest.raises(BridgeError) as excinfo:
        _ops(fem).tcl(str(tmp_path / "deck.tcl"))
    _assert_s3_refusal(excinfo)
    assert "edge_soft=" in str(excinfo.value)


def test_soft_contact_plane_under_partitioning_refused(tmp_path):
    fem = _plane_fem("s3_soft_plane", partition=True, soft=0.1)
    assert fem.elements.contact_planes[0].soft == 0.1
    assert not fem.elements.contacts
    with pytest.raises(BridgeError) as excinfo:
        _ops(fem).tcl(str(tmp_path / "deck.tcl"))
    _assert_s3_refusal(excinfo)
    assert "g.constraints.contact_plane interaction #1" in str(excinfo.value)


# ── negative controls: what S3 must NOT refuse ───────────────────────


def test_kn_auto_without_soft_is_not_refused_by_s3(tmp_path):
    # A partitioned kn="auto" deck (no SOFT knob) EMITS since S4 relaxed
    # the blanket serial-only gate to the locality contract — the S3 soft
    # refusal must stay silent on it.
    fem = _two_box_fem("s3_auto_no_soft", partition=True,
                       formulation="nts", kn="auto")
    deck = tmp_path / "deck.tcl"
    _ops(fem).tcl(str(deck))                     # no BridgeError (S4)
    contact_lines = [ln for ln in deck.read_text().splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1


def test_visc_without_soft_is_not_refused_by_s3(tmp_path):
    # visc is a damper on the owner-local active set — not SOFT-family.
    # Emits under partitioning since S4 (the blanket is gone).
    fem = _two_box_fem("s3_visc_no_soft", partition=True,
                       formulation="nts", kn=1.0e6, visc=2.5)
    deck = tmp_path / "deck.tcl"
    _ops(fem).tcl(str(deck))                     # no BridgeError (S4)
    contact_lines = [ln for ln in deck.read_text().splitlines()
                     if ln.strip().startswith("contact ")]
    assert len(contact_lines) == 1
    assert "-visc" in contact_lines[0]


# ── serial emit is untouched (byte-identical serial emit is a hard rule) ──


def test_soft_contact_serial_emit_unchanged(tmp_path):
    fem = _two_box_fem("s3_soft_serial", partition=False,
                       formulation="nts", kn=1.0e6, soft=True)
    assert not fem.partitions
    deck = tmp_path / "deck.tcl"
    _ops(fem).tcl(str(deck))                     # no BridgeError
    contact_lines = [ln for ln in deck.read_text().splitlines()
                     if ln.startswith("contact ")]
    assert len(contact_lines) == 1
    assert "-soft" in contact_lines[0]


def test_soft_plane_serial_emit_unchanged(tmp_path):
    fem = _plane_fem("s3_plane_serial", partition=False, soft=True)
    deck = tmp_path / "deck.tcl"
    _ops(fem).tcl(str(deck))                     # no BridgeError
    plane_lines = [ln for ln in deck.read_text().splitlines()
                   if ln.startswith("contactPlane ")]
    assert len(plane_lines) == 1
    assert "-soft" in plane_lines[0]


# ── INV-3 preserved half: the `-kn auto` token pin ───────────────────


def _contact_deck_shape(deck) -> tuple[list[str], list[tuple[str, frozenset]]]:
    """(exact `contact` verb lines, order-insensitive `contactSurface` sets).

    Partitioning legitimately reorders node EXTRACTION (the slave node-set
    listing order differs between twins), so surface lines compare as
    (prefix, node-set); the `contact` verb line — which carries the kn
    token INV-3 pins — compares byte-exact.
    """
    verbs, surfaces = [], []
    for ln in deck.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("contact "):
            verbs.append(ln)
        elif ln.startswith("contactSurface"):
            head, _, nodes = ln.partition("-slave" if "-slave" in ln
                                          else "-master")
            surfaces.append((head, frozenset(nodes.split())))
    return sorted(verbs), sorted(surfaces, key=repr)


def test_kn_auto_token_survives_partitioning_and_matches_serial_twin(tmp_path):
    # The record → token mapping is partition-independent: partitioning the
    # mesh must not rewrite the record's kn="auto" into a literal, and the
    # deck a partition-carrying model emits — BOTH the sanctioned flat=True
    # serial escape hatch AND (since S4) the owner rank's block of the real
    # partitioned fan-out — must carry the IDENTICAL `auto` token as its
    # serial twin's deck.
    fem_serial = _two_box_fem("s3_auto_twin_serial", partition=False,
                              formulation="nts", kn="auto")
    fem_part = _two_box_fem("s3_auto_twin_part", partition=True,
                            formulation="nts", kn="auto")

    # Partitioning never rewrites the record's sentinel.
    assert fem_part.elements.contacts[0].kn == "auto"

    serial_deck = tmp_path / "serial.tcl"
    flat_deck = tmp_path / "part_flat.tcl"
    part_deck = tmp_path / "part_mp.tcl"
    _ops(fem_serial).tcl(str(serial_deck))
    _ops(fem_part).tcl(str(flat_deck), flat=True)
    _ops(fem_part).tcl(str(part_deck))           # S4: the per-rank fan-out

    serial_verbs, serial_surfaces = _contact_deck_shape(serial_deck)
    flat_verbs, flat_surfaces = _contact_deck_shape(flat_deck)
    part_verbs, part_surfaces = _contact_deck_shape(part_deck)
    assert serial_verbs == flat_verbs == part_verbs   # byte-identical verb
    assert serial_surfaces == flat_surfaces == part_surfaces  # same node sets
    contact_verbs = serial_verbs
    assert len(contact_verbs) == 1
    # The literal token, un-rewritten: `contact <tag> <m> <s> auto` ends on
    # the sentinel itself — no numeric penalty was substituted for it.
    assert contact_verbs[0].split()[-1] == "auto"
