"""Fork-only end-to-end — `g.constraints.contact` → contactSurface + contact.

Drives the apeGmsh contact generator (two named meshed faces) into the live
*fork* domain and asserts the fork accepts the emitted `contactSurface` +
`contact` + `constraints('LadrunoContact')` deck (it parses/registers without
error — a stock build has no such commands). NTS and mortar variants.

Two unit cubes stacked with a small gap so they are distinct bodies with
distinct interface nodes; the contact is defined on the facing faces. No solid
element/material is assigned (like the reinforce/embed live tests) — the face
nodes alone are what `contactSurface` references. Gated on the backend
resolver via the `ladruno_fork` marker.
"""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

pytestmark = pytest.mark.ladruno_fork


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    """The boundary surface of *volume_tag* whose centroid sits at ``z``."""
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _build(g):
    box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)  # 0.05 gap
    g.model.sync()
    master = _face_at_z(box1, 1.0)     # box1 top
    slave = _face_at_z(box2, 1.05)     # box2 bottom
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box1, box2], name="solid")
    g.physical.add(2, [master], name="master")
    g.physical.add(2, [slave], name="slave")


def test_nts_contact_runs_on_fork() -> None:
    with apeGmsh(model_name="contact_nts", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="nts", kn=1.0e6, mu=0.3, kt=5.0e5)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_contacts = len(fem.elements.contacts)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # executes contactSurface + contact + LadrunoContact

    assert n_contacts == 1
    # auto-outward computed from the master CAD normal, oriented toward slave
    # (the slave sits at +z above the master) ⇒ +z dominant.
    assert rec.outward is not None and rec.outward[2] > 0.5


def test_mortar_contact_runs_on_fork() -> None:
    with apeGmsh(model_name="contact_mortar", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="mortar", eps_n="auto",
                              aug_tol=1e-8, max_aug=20, ngp=2)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_contacts = len(fem.elements.contacts)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)

    assert n_contacts == 1
    # mortar slave is faceted (slave_faces set, slave_nodes None).
    assert rec.slave_faces is not None and rec.slave_nodes is None


def test_nts_soft_explicit_runs_on_fork() -> None:
    # The explicit-only SOFT=1 Courant-stable penalty: the fork must accept the
    # `-soft SOFSCL` grammar emitted by g.constraints.contact(soft=...).
    with apeGmsh(model_name="contact_soft", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="nts", kn=1.0e6, soft=0.2, visc=5.0)
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)

    assert rec.soft == 0.2 and rec.visc == 5.0
