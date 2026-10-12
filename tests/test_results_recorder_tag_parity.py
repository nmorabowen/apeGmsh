"""#1711 / #1727 parity: recorder-transcoded element rows vs the live domain.

A 2-D portal frame on a default (``element_tags="sequential"``) bridge,
the beam declared before the columns, so each beam element's OpenSees tag
is not its FEM id (FEM ids [8, 9, 10] -> ops tags [3, 4, 5]). A
:class:`ResolvedRecorderSpec` names elements by FEM id (its contract,
the same one ``DomainCapture`` honours), is emitted live through
``spec.emit_recorders``, and is read back through
``Results.from_recorders``. Every transcoded row must equal
``eleResponse`` on the OpenSees tag of the FEM id that row is labelled
with: the oracle is the live domain itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees

pytestmark = pytest.mark.live

H, L = 3.0, 5.0


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


def _portal_fem() -> Any:
    with apeGmsh(model_name="b5c_portal", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, H, 0.0)
        p2 = geo.add_point(L, H, 0.0)
        p3 = geo.add_point(L, 0.0, 0.0)
        c1 = geo.add_line(p0, p1)
        bm = geo.add_line(p1, p2)
        c2 = geo.add_line(p3, p2)
        for c in (c1, bm, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, 4)
        g.physical.add(1, [c1, c2], name="Cols")
        g.physical.add(1, [bm], name="Beam")
        g.labels.add(0, [p0, p3], "base")
        g.labels.add(0, [p1], "tl")
        g.labels.add(0, [p2], "tr")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _portal_bridge(fem: Any) -> apeSees:
    ops = apeSees(fem, default_orientation=None)   # sequential tags
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Beam first: its sequential tags are taken before the columns'.
    ops.element.elasticBeamColumn(pg="Beam", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    sec = ops.section.Elastic(E=2.0e11, A=0.03, Iz=5.0e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=4)
    ops.element.forceBeamColumn(pg="Cols", transf=tr, integration=integ)
    ops.fix(label="base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(label="tl"), forces=(10e3, -30e3, 0.0))
        p.load(node=ops.nodes.one(label="tr"), forces=(0.0, -50e3, 4e3))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def test_transcoded_global_force_rows_are_the_labelled_elements(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results import Results
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    fem = _portal_fem()
    ops = _portal_bridge(fem)
    assert ops.analyze(steps=1) == 0
    plan = ops._read_state.element_plan  # type: ignore[union-attr]
    fem_to_ops = {int(e): int(t) for _s, sub in plan for e, _c, t in sub}
    beam = [int(e) for e in ops.elements.get(pg="Beam").ids]
    # The case is not vacuous: sequential tags differ from FEM ids.
    assert any(fem_to_ops[e] != e for e in beam)

    ops.h5(str(tmp_path / "model.h5"))
    model = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="elements", name="beam",
            components=("nodal_resisting_force_x", "nodal_resisting_force_y",
                        "nodal_resisting_moment_z"),
            dt=None, n_steps=None, element_ids=np.asarray(beam),
            element_class_name="ElasticBeam2d",
        ),),
    )
    osp = get_ops()
    with spec.emit_recorders(str(tmp_path / "out"), ops=osp) as live:
        live.begin_stage("s", kind="static")
        assert osp.analyze(1) == 0
        live.end_stage()

    truth = {e: np.asarray(osp.eleResponse(fem_to_ops[e], "globalForce"))
             for e in beam}
    res = Results.from_recorders(
        spec, tmp_path / "out", fem=fem, stage_id="s", model=model,
        cache_root=tmp_path / "cache",
    )
    for k, comp in enumerate(("nodal_resisting_force_x",
                              "nodal_resisting_force_y",
                              "nodal_resisting_moment_z")):
        slab = res.elements.get(component=comp, ids=beam)
        got_ids = [int(i) for i in np.asarray(slab.element_ids)]
        assert sorted(got_ids) == sorted(beam)
        vals = np.asarray(slab.values)[-1]          # (E, n_nodes)
        for row, eid in enumerate(got_ids):
            expected = truth[eid][[k, k + 3]]       # node 1, node 2
            np.testing.assert_allclose(
                vals[row], expected, rtol=1e-9, atol=1e-6,
                err_msg=f"{comp} of element {eid} (ops tag {fem_to_ops[eid]})",
            )
