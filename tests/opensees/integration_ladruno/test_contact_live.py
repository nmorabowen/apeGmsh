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


@pytest.fixture(autouse=True)
def _require_contact_command() -> None:
    """Skip when the fork build lacks the (newer) contact subsystem.

    The ``ladruno_fork`` marker only confirms the backend IS the fork — an
    older fork build can carry every other Ladruno feature yet not the
    contact commands (``contactSurface`` / ``contact`` land later in the
    fork's history). Probe the SAME ops module the live emitter resolves
    and skip cleanly (the conftest's "green-or-skipped everywhere" policy)
    rather than failing on a partial-fork box.
    """
    from apeGmsh.opensees.emitter.live import _get_ops

    if not hasattr(_get_ops(), "contactSurface"):
        pytest.skip(
            "fork build lacks the contact subsystem (no ops.contactSurface) "
            "— rebuild the Ladruno fork with contact to run these live tests"
        )


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
    # No outward is auto-derived: the fork computes a correct per-facet normal
    # from each facet's connectivity, and a single global -outward would skip
    # facets perpendicular to it / invert opposed ones on a non-flat master.
    # outward is carried only when the user sets it explicitly.
    assert rec.outward is None


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


def test_nts_contact_extensions_run_on_fork() -> None:
    # The fork parser accepts the NTS extension modifiers
    # (-soft/-visc/-consistanttan/-geomtan) emitted after the kn kt mu triple.
    # consistent_tan needs an unsymmetric solver; run() defaults are fine for a
    # parse/register-only smoke (no analyze step here).
    with apeGmsh(model_name="contact_nts_ext", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="nts", kn=1.0e6, kt=5.0e5, mu=0.3,
                              soft=0.1, visc=1.0,
                              consistent_tan=True, geom_tan=True)
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # fork parses contact … -soft -visc -consistanttan -geomtan

    assert rec.soft == 0.1 and rec.visc == 1.0
    assert rec.consistent_tan and rec.geom_tan


def test_mortar_tie_via_deprecated_mortar_alias_runs_on_fork() -> None:
    # g.constraints.mortar() is a deprecated alias delegating to the fork
    # ALM-penalty mortar mesh-tie (contact formulation='mortar', tie=True). It
    # emits `contact … -mortar -epsN auto -tie -outward …`; the fork must parse
    # and register it. tie=True mandates an explicit outward.
    import warnings

    with apeGmsh(model_name="mortar_tie_alias", verbose=False) as g:
        _build(g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rec_def = g.constraints.mortar("master", "slave",
                                          outward=(0.0, 0.0, 1.0))
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # fork parses contact … -mortar -tie -outward …

    assert rec_def.formulation == "mortar" and rec_def.tie is True
    assert rec.tie is True and rec.eps_n == "auto"
    assert rec.outward == (0.0, 0.0, 1.0)


def test_nts_numeric_kn_soft_runs_on_fork() -> None:
    # Regression (review #1/#2/#5): a numeric kn + an extension flag with NO
    # friction and NO outward must still parse on the fork — the emitted
    # `kn kt mu` triple keeps the fork's m=(remaining>=3)?3:1 reader from
    # consuming `-soft` as a double and aborting the `contact` command. The
    # other live tests pass friction, which masked this path.
    with apeGmsh(model_name="contact_nts_kn_soft", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="nts", kn=1.0e6, soft=0.1, visc=1.0)
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # fork parses `contact … 1e6 0.0 0.0 -soft 0.1 -visc 1.0`

    assert rec.kn == 1.0e6 and rec.soft == 0.1 and rec.mu is None


def test_mortar_contact_extensions_run_on_fork() -> None:
    # The fork parser accepts the mortar extension modifiers (-soft SOFT=2 +
    # -visc); geom_tan is NTS-only so it is not exercised here.
    with apeGmsh(model_name="contact_mortar_ext", verbose=False) as g:
        _build(g)
        g.constraints.contact("master", "slave",
                              formulation="mortar", eps_n="auto",
                              soft=0.1, visc=0.5)
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = fem.elements.contacts[0]
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)

    assert rec.soft == 0.1 and rec.visc == 0.5
