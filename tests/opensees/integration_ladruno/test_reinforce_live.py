"""Fork-only element end-to-end — ``g.reinforce`` -> LadrunoEmbeddedRebar.

Drives the apeGmsh composite reinforcement path (mesh a concrete volume,
thread a rebar line through it, ``g.reinforce``) into the live *fork*
domain and asserts the fork-only ``LadrunoEmbeddedRebar`` couplings
actually load (appear in ``getEleTags``). This is the embedded-rebar half
of the "online" proof: a fork-only *element* emitted by an apeGmsh
generator and run on the fork build.

Mirrors tests/test_reinforce_explicit.py::test_perfect_bond_live_roundtrip,
but gated on the new backend resolver (the ``ladruno_fork`` marker) so it
runs in the fork-backend integration env (no stock ``openseespy`` present).
"""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import _get_ops

pytestmark = pytest.mark.ladruno_fork


def _build(g, size: float = 0.5) -> None:
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    p0 = gmsh.model.occ.addPoint(0.5, 0.5, 0.2)
    p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.8)
    ln = gmsh.model.occ.addLine(p0, p1)
    g.model.sync()
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box], name="concrete")
    g.physical.add(1, [ln], name="rebar")


def test_perfect_bond_rebar_loads_on_fork() -> None:
    with apeGmsh(model_name="rbar_live", verbose=False) as g:
        _build(g, size=0.5)
        g.reinforce(host="concrete", bars="rebar", perfect=1.0e12)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_ties = len(fem.elements.reinforce_ties)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # drives a LiveOpsEmitter through the full deck

    # The fork backend is a process-global singleton; query it directly.
    ele_tags = _get_ops().getEleTags() or []
    assert n_ties > 0
    assert len(ele_tags) >= n_ties
