"""Live: truss axial force through ``DomainCaptureSpec.gauss``.

``axial_force`` used to be accepted only by ``DomainCaptureSpec.line_stations``,
whose capturer knows beam-column sections alone and drops a truss with "no
line-stations capture path". The truss scalar is a Gauss read — the Truss
family answers ``ops.eleResponse(eid, "axialForce")`` with one value at its
single ``Line_GL_1`` point (``RESPONSE_CATALOG``) — so ``gauss`` now accepts
it.

The model is one gmsh line element between two tagged points, so the
element's ``fem_eid`` (3: the two point elements come first) differs from
its ops tag (1). Reading the value back under ``element_index == 3`` checks
the fem <-> ops translation on this route too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

P = 1000.0


def _one_truss(g):
    G = g.model.geometry
    a = G.add_point(0.0, 0.0, 0.0)
    b = G.add_point(2.0, 0.0, 0.0)
    bar = G.add_line(a, b)
    g.model.sync()
    g.physical.add(1, [bar], name="Bar")
    g.physical.add(0, [a], name="A")
    g.physical.add(0, [b], name="B")
    g.mesh.structured.set_transfinite_curve(bar, 2)
    g.mesh.generation.generate(1)
    return g.mesh.queries.get_fem_data(dim=1)


@pytest.mark.live
@pytest.mark.parametrize("cls", ["Truss", "CorotTruss"])
def test_gauss_axial_force_capture_reads_the_truss_force(
    g, tmp_path: Path, cls: str,
) -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    fem = _one_truss(g)
    (bar_eid,) = [int(e) for e in fem.elements.physical.element_ids("Bar")]

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    getattr(ops.element, cls)(pg="Bar", A=1e-3, material=steel)
    ops.fix(pg="A", dofs=(1, 1, 1))
    ops.fix(pg="B", dofs=(0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="B", forces=(P, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    # Build once and step the live domain: ``ops.analyze`` rebuilds from
    # scratch on every call, so two calls would both land at lambda = 0.5.
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    spec = DomainCaptureSpec(opensees=ops)
    spec.gauss(pg="Bar", components="axial_force", name="axial")
    path = str(tmp_path / "truss.h5")
    with ops.domain_capture(spec, path=path, ops=emitter.ops) as cap:
        cap.begin_stage("pull", kind="static")
        for _ in range(2):
            assert emitter.analyze(steps=1) == 0
            cap.step(t=emitter.ops.getTime())
        cap.end_stage()

    om = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=om) as r:
        slab = r.elements.gauss.get(component="axial_force")
    assert slab.values.shape == (2, 1)
    assert slab.values[:, 0] == pytest.approx([P / 2, P], rel=1e-6)
    assert [int(e) for e in slab.element_index] == [bar_eid]
