"""Partitioned-emit fail-loud guards for the fork contact + g.embed stacks.

The fork contact subsystem is serial-only and g.embed ties need per-rank
node-ownership routing that is deferred — neither is wired into the
partitioned (OpenSeesMP) emit path. Without a guard the contact / embed
records would be silently DROPPED from every rank's deck (the partitioned
emitter never calls ``emit_contacts`` / ``emit_embed_ties``). These tests
lock the fail-loud guard in ``BuiltModel._emit_partitioned`` (mirrors the
reinforce-ties / rebar-elements guards). Emit-time only — no fork build
needed (the guard fires before any ops command).
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


def test_contact_under_partitioned_emit_fails_loud(tmp_path):
    fem = _contact_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.contacts                      # really present
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    with pytest.raises(BridgeError, match="contact.*partitioned|partitioned.*contact"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_contact_plane_under_partitioned_emit_fails_loud(tmp_path):
    # Regression (adversarial review): a plane-only partitioned model must fail
    # loud, NOT silently drop the contactPlane and auto-emit a spurious
    # LadrunoContact handler (which would unenforce the cross-partition MP
    # constraints, ADR 0027).
    fem = _contact_plane_fem_partitioned()
    assert len(fem.partitions) == 2
    assert fem.elements.contact_planes                # really present
    assert not fem.elements.contacts                  # plane-only (the trap)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    with pytest.raises(BridgeError, match="contact.*partitioned|partitioned.*contact"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_contact_plane_under_partitioned_staged_emit_fails_loud(tmp_path):
    # Defense-in-depth for the #761 guard: the contact serial-only guard sits in
    # BuiltModel._emit_partitioned BEFORE the staged dispatch
    # (_emit_stages_partitioned), so a STAGED + partitioned contact_plane model
    # must ALSO fail loud — the staged path must not let a plane slip through
    # (which would drop the contactPlane silently AND auto-emit a spurious
    # LadrunoContact handler, unenforcing the cross-partition MP constraints).
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
