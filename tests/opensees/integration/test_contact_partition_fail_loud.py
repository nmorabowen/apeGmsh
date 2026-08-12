"""Partitioned-emit guards for the fork contact + g.embed stacks.

Born as the fail-loud battery for the blanket "serial-only" contact
refusal (ADR 0073 era). ADR 0092 S4 relaxed that blanket to the locality
contract — partitioned contact now EMITS, inside exactly one owner rank's
block — so the contact halves here pin the new behaviour (emission, not
refusal; the deep shape assertions live in
``test_emit_partitioned_contact.py``). What still refuses, and is pinned
here: STAGED partitioned contact (a named ADR 0092 S4 refusal — the staged
pipeline skips the analysis-chain auto-emit, so ``LadrunoContact`` would
never be forced) and g.embed ties (per-rank node-ownership routing still
deferred — the guard mirrors the reinforce-ties / rebar-elements guards).
Emit-time only — no fork build needed.
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


def _contact_fem_partitioned():
    """Two stacked boxes with a contact interaction, partitioned across 2."""
    with apeGmsh(model_name="contact_part_guard", verbose=False) as g:
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
        g.constraints.contact("master", "slave",
                              formulation="nts", kn=1.0e6, mu=0.3, kt=5.0e5)
        g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _contact_plane_fem_partitioned():
    """One box with a rigid-plane contact on its bottom face, partitioned
    across 2 — and NO face-to-face contact (so only the contact_plane guard
    can catch it; otherwise it is silently dropped AND a spurious
    LadrunoContact handler is auto-emitted)."""
    with apeGmsh(model_name="cplane_part_guard", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        bottom = _face_at_z(box, 0.0)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [bottom], name="floor")
        g.constraints.contact_plane(
            "floor", normal=(0, 0, 1), point=(0, 0, 0), kn=1.0e7)
        g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _embed_fem_partitioned():
    """A rebar line embedded in a box host, partitioned across 2."""
    with apeGmsh(model_name="embed_part_guard", verbose=False) as g:
        box = g.model.geometry.add_box(0.0, 0.0, 0.0, 4.0, 0.4, 0.4)
        p0 = g.model.geometry.add_point(0.0, 0.2, 0.2, lc=0.4)
        p1 = g.model.geometry.add_point(4.0, 0.2, 0.2, lc=0.4)
        rebar = g.model.geometry.add_line(p0, p1)
        g.model.sync()
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [rebar], name="rebar")
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.embed(host="concrete", nodes="rebar")
        g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _one_owner_block_only(deck_path, verb: str) -> None:
    """The verb appears exactly once, inside a getPID rank bracket."""
    text = deck_path.read_text()
    lines = [ln.strip() for ln in text.splitlines()]
    assert sum(1 for ln in lines if ln.startswith(verb)) == 1
    assert "getPID" in text                           # really partitioned


def test_contact_under_partitioned_emit_now_emits(tmp_path):
    # Pre-S4 this refused ("serial-only"). ADR 0092 S4: it emits, on
    # exactly one owner rank (INV-1).
    fem = _contact_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.contacts                      # really present
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))                                # no BridgeError
    _one_owner_block_only(deck, "contact ")


def test_contact_plane_under_partitioned_emit_now_emits(tmp_path):
    # Pre-S4 regression case (adversarial review): a plane-only partitioned
    # model used to be the silent-drop trap. ADR 0092 S4: the contactPlane
    # emits on its owner rank, and the LadrunoContact handler auto-emit is
    # no longer spurious — it has a contactPlane to enforce.
    fem = _contact_plane_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.contact_planes                # really present
    assert not fem.elements.contacts                  # plane-only
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))                                # no BridgeError
    _one_owner_block_only(deck, "contactPlane ")
    assert "constraints LadrunoContact" in deck.read_text()


def test_contact_plane_under_partitioned_staged_emit_fails_loud(tmp_path):
    # Was defense-in-depth for the #761 blanket guard; since ADR 0092 S4 the
    # refusal that fires here is the NAMED staged one (the partitioned staged
    # pipeline skips the analysis-chain auto-emit, so the LadrunoContact
    # handler would never be forced and the plane would be silently
    # unenforced). Same fail-loud outcome, sharper reason.
    fem = _contact_plane_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.contact_planes                # really present
    assert not fem.elements.contacts                  # plane-only (the trap)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    # Make the model STAGED (stage_records non-empty ⇒ the staged dispatch).
    with ops.stage(name="hold") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    assert ops._stage_records                         # really staged
    with pytest.raises(BridgeError,
                       match="contact.*partitioned|partitioned.*contact"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_embed_under_partitioned_emit_fails_loud(tmp_path):
    fem = _embed_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.embed_ties                    # really present
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    with pytest.raises(BridgeError, match="embed.*partitioned|partitioned.*embed"):
        ops.tcl(str(tmp_path / "deck.tcl"))


# ── ops.tcl(flat=True): the sanctioned serial escape hatch ───────────
#
# Since ADR 0092 S4 a partition-carrying contact model takes the real
# per-rank fan-out; ``flat=True`` remains the serial escape hatch (for
# g.embed ties and the contact cases S4 refuses — SOFT, staged,
# undecidable owner) and must keep emitting the single-domain contact
# deck unchanged.


def _tet_ops(fem, *pgs):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in pgs:
        ops.element.FourNodeTetrahedron(pg=pg, material=mat)
    return ops


def _assert_serial_contact_deck(deck_path):
    text = deck_path.read_text()
    lines = [ln.strip() for ln in text.splitlines()]
    assert sum(1 for ln in lines if ln.startswith("contactSurface")) == 2
    assert sum(1 for ln in lines if ln.startswith("contact ")) == 1
    assert "getPID" not in text                       # truly single-domain


def test_flat_emits_contact_deck_from_partitioned_model(tmp_path):
    fem = _contact_fem_partitioned()
    assert len(fem.partitions) == 2
    deck = tmp_path / "deck.tcl"
    _tet_ops(fem, "solid").tcl(str(deck), flat=True)  # no BridgeError
    _assert_serial_contact_deck(deck)


def test_flat_emits_composed_contact_model(tmp_path):
    # The motivating case: plain host composes a contact-carrying module.
    from apeGmsh.mesh.FEMData import FEMData

    mod = tmp_path / "mod.h5"
    plain = tmp_path / "plain.h5"
    with apeGmsh(model_name="flat_mod", verbose=False) as g:
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
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(mod))
    with apeGmsh(model_name="flat_host", verbose=False) as g:
        box = g.model.geometry.add_box(5, 5, 5, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="hostvol")
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(plain))
    fem = FEMData.from_h5(str(plain)).compose(
        str(mod), label="C", translate=(0.0, 10.0, 0.0))
    assert len(fem.partitions) == 2                   # auto one-rank-per-module
    assert len(fem.elements.contacts) == 1

    # default (partitioned) path now EMITS (ADR 0092 S4): the module's
    # interaction lands whole inside its owner rank's block …
    default_deck = tmp_path / "deck_default.tcl"
    _tet_ops(fem, "hostvol", "C.solid").tcl(str(default_deck))
    _one_owner_block_only(default_deck, "contact ")
    # … and flat=True stays the serial escape hatch, unchanged.
    deck = tmp_path / "deck_flat.tcl"
    _tet_ops(fem, "hostvol", "C.solid").tcl(str(deck), flat=True)
    _assert_serial_contact_deck(deck)


def test_flat_conflicts_with_per_rank_and_split(tmp_path):
    fem = _contact_fem_partitioned()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    with pytest.raises(ValueError, match="flat=True and per_rank=True"):
        ops.tcl(str(tmp_path / "x.tcl"), flat=True, per_rank=True)
    with pytest.raises(ValueError, match="flat=True and split=True"):
        ops.tcl(str(tmp_path / "x.tcl"), flat=True, split=True)
