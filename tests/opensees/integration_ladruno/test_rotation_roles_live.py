"""Fork lane: a 2-D u-p node's pore pressure never reads as ``rotation_z``
(#2029), on ``.ladruno`` and MPCO.

A 2-D ``LadrunoUP`` vertex node and a 2-D frame node both have ndf 3, and
the fork's recorder writes both third DOFs in the ``rotation_z`` column:
the frame's rotation, and the u-p node's pore pressure. A drained-top soil
column under a sudden load (the Terzaghi set of
``tests/opensees/integration/test_ladruno_up_emission.py``) builds a
non-zero pore pressure; a frame column beside it, under a lateral tip
load, a non-zero rotation. Both are solved through the bridge and
recorded by the fork's own recorder.

The oracle is the live domain: ``nodeDisp(n, 3)``. Read raw (no model),
the file's ``rotation_z`` at a u-p node equals the pore pressure, which
is the premise. Read through the public factory with ``model_h5=``, the
u-p nodes give NaN and the frame nodes give their live rotation at
``rtol=1e-10``.

Gated by the ``ladruno_fork`` marker, so the nightly ``live-fork`` lane runs
it against the pinned build (``FORK_PIN``).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._mpco import MPCOReader

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10


def _fem(g: Any) -> Any:
    """A 1 x 2 quad soil column and a frame column 3 m to its right."""
    G = g.model.geometry
    soil = G.add_rectangle(0.0, 0.0, 0.0, 1.0, 2.0)
    base = G.add_point(3.0, 0.0, 0.0)
    tip = G.add_point(3.0, 2.0, 0.0)
    col = G.add_line(base, tip)
    g.model.sync()
    g.physical.add(2, [soil], name="Soil")
    g.physical.add(1, [col], name="Col")
    g.physical.add(0, [base], name="ColBase")
    g.physical.add(0, [tip], name="ColTip")
    g.mesh.structured.set_recombine("Soil", dim=2)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data()


def _run(g: Any, tmp_path: Path, kind: str) -> tuple[str, str, list[int],
                                                     list[int], dict]:
    fem = _fem(g)
    xyz = {int(n): p for n, p in zip(fem.nodes.ids, np.asarray(fem.nodes.coords))}
    up = sorted({int(n) for n in fem.nodes.select(pg="Soil").ids})
    frame = sorted({int(n) for n in fem.nodes.select(pg="Col").ids})
    soil_base = [n for n in up if abs(xyz[n][1]) < 1e-9]
    soil_top = [n for n in up if abs(xyz[n][1] - 2.0) < 1e-9]

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    ops.element.LadrunoUP(
        pg="Soil", material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.0, rho=0.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4), dynSeepage="off",
    )
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4)
    ops.fix(nodes=soil_base, dofs=(1, 1, 0))
    ops.fix(nodes=soil_top, dofs=(0, 0, 1))   # drained top: p = 0
    ops.fix(pg="ColBase", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Constant()) as pat:
        for n in soil_top:
            pat.load(node=n, forces=(0.0, -1.0 / len(soil_top), 0.0))
        for n in fem.nodes.select(pg="ColTip").ids:
            pat.load(node=int(n), forces=(1e3, 0.0, 0.0))
    ext = "ladruno" if kind == "Ladruno" else "mpco"
    path = str(tmp_path / f"up_frame.{ext}")
    getattr(ops.recorder, kind)(
        file=path, nodal_responses=("displacement", "rotation"))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.6, beta=0.3025)
    ops.analysis.Transient()

    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for _ in range(3):
        assert emitter.analyze(steps=1, dt=0.05) == 0
    live = {n: float(emitter.ops.nodeDisp(n, 3)) for n in up + frame}
    emitter.ops.remove("recorders")
    emitter.ops.wipe()
    return path, model_h5, up, frame, live


@pytest.mark.parametrize("kind", ["Ladruno", "MPCO"])
def test_up_pressure_never_reads_as_rotation_z(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    path, model_h5, up, frame, live = _run(g, tmp_path, kind)
    pressured = [n for n in up if abs(live[n]) > 0.0]
    assert pressured, "the soil column built no pore pressure"
    assert any(abs(live[n]) > 0.0 for n in frame), "the frame did not rotate"

    # Premise: read raw, the file's rotation_z column holds the pressure.
    reader_cls = LadrunoReader if kind == "Ladruno" else MPCOReader
    with reader_cls(Path(path)) as raw, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage = raw.stages()[0].id
        slab = raw.read_nodes(stage, "rotation_z")
    raw_rz = {int(i): float(v) for i, v in zip(slab.node_ids, slab.values[-1])}
    np.testing.assert_allclose(
        [raw_rz[n] for n in pressured], [live[n] for n in pressured],
        rtol=RTOL, err_msg="premise: the rotation_z column holds the pressure",
    )

    factory = Results.from_ladruno if kind == "Ladruno" else Results.from_mpco
    r = factory(path, model_h5=model_h5)
    slab = r.nodes.get(component="rotation_z")
    got = {int(i): float(v) for i, v in zip(slab.node_ids,
                                            np.asarray(slab.values)[-1])}
    assert all(np.isnan(got[n]) for n in up), {n: got[n] for n in pressured}
    want = np.array([live[n] for n in frame])
    np.testing.assert_allclose(
        [got[n] for n in frame], want, rtol=RTOL,
        atol=RTOL * np.abs(want).max(), err_msg=f"frame rotation_z on {kind}",
    )
