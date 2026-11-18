"""Masses on a mixed solid + frame model run in openseespy.

A tet block (3-dof nodes, inferred per ADR 0048) and a separate
cantilever beam (6-dof nodes) under the frame envelope
``ops.model(ndm=3, ndf=6)``, with masses on both kinds of node.
openseespy's ``mass`` sizes the matrix by the builder ndf, so before the
bracket fix the first solid-node mass died in ``Node::setMass``.  Both
the emitted py deck (run in-process) and the live emitter must now run
an eigen analysis.  Unit coverage: ``tests/opensees/integration/
test_mixed_ndf_mass.py``.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter

openseespy = pytest.importorskip("openseespy.opensees")


def _mixed_model(g) -> apeSees:
    import gmsh

    G = g.model.geometry
    box = G.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    p1 = G.add_point(3.0, 0.0, 0.0)
    p2 = G.add_point(3.0, 0.0, 2.0)
    beam = G.add_line(p1, p2)
    g.model.sync()
    g.physical.add(3, [box], name="Block")
    g.physical.add(1, [beam], name="Beam")
    g.physical.add(0, [p1], name="BeamBase")
    g.physical.add(0, [p2], name="BeamTip")
    g.model.select(dim=2).on_plane(
        (0, 0, 0), (0, 0, 1), tol=1e-6).to_physical("BlockBase")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    assert gmsh.model.getEntities(3)
    fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Block",
        material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.elasticBeamColumn(
        pg="Beam", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="BlockBase", dofs=(1, 1, 1))
    ops.fix(pg="BeamBase", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Block", values=(10.0, 10.0, 10.0))
    ops.mass(pg="Beam", values=(5.0, 5.0, 5.0, 0.1, 0.1, 0.1))
    return ops


@pytest.mark.live
def test_mixed_ndf_masses_run_from_py_deck(g) -> None:
    ops = _mixed_model(g)
    em = PyEmitter()
    ops.build().emit(em)
    deck = "\n".join(em.lines())
    assert "'-ndf', 3)" in deck
    ns: dict[str, object] = {}
    try:
        exec(compile(deck, "<mixed_ndf_mass_deck>", "exec"), ns)
        lam = openseespy.eigen(2)
    finally:
        openseespy.wipe()
    assert len(lam) == 2 and all(v > 0.0 for v in lam)


@pytest.mark.live
def test_mixed_ndf_masses_run_live(g) -> None:
    ops = _mixed_model(g)
    try:
        result = ops.eigen(num_modes=2)
    finally:
        openseespy.wipe()
    assert result.eigenvalues.shape == (2,)
    assert float(result.eigenvalues.min()) > 0.0
