"""Fork-only — dual concrete damage read back live and from ``.ladruno``.

One ``LadrunoBrick`` (unit cube) of ``LadrunoConcrete3D`` pulled in uniaxial
tension past its peak by a prescribed top-face displacement. The material's
``damage`` response is ``[omega_t, omega_c]``; it has no element-level
branch, so the live capture reads it per Gauss point through
``eleResponse(eid, "material", gp, "damage")`` and lands it on the
``damage_tension`` / ``damage_compression`` canonicals. The same run writes
a ``.ladruno`` file with ``material.damage``, whose ``C1, C2`` columns the
reader maps by position onto the same two names.

Authority for the values: ``eleResponse`` on the live domain after the last
step, independent of both readers under test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

pytestmark = [pytest.mark.ladruno_fork, pytest.mark.live]

E, FT = 30e9, 2.0e6
U_PEAK = FT / E              # 1 m cube: the top displacement at the peak
N_STEPS = 30
U_MAX = 6.0 * U_PEAK


def _cube(g):
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="cube")
    g.model.sync()
    g.physical.add_volume("cube", name="Body")
    sel = g.model.select(dim=2)
    sel.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6).to_physical("Bottom")
    g.model.select(dim=2).on_plane((0, 0, 1), (0, 0, 1), tol=1e-6).to_physical("Top")
    g.model.select(dim=2).on_plane((0, 0, 0), (1, 0, 0), tol=1e-6).to_physical("X0")
    g.model.select(dim=2).on_plane((0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Y0")
    g.mesh.structured.set_transfinite_box("cube", n=2)
    g.mesh.generation.generate(3)
    return g.mesh.queries.get_fem_data(dim=3)


def test_ladruno_brick_tension_damage_capture_and_ladruno_read(
    g, tmp_path: Path,
) -> None:
    fem = _cube(g)
    (eid,) = [int(e) for e in fem.elements.physical.element_ids("Body")]

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.LadrunoConcrete3D(
        E=E, nu=0.2, fc=30e6, ft=FT, Gf=200.0, Gc=20000.0,
    )
    ops.element.LadrunoBrick(pg="Body", material=conc)
    ops.fix(pg="Bottom", dofs=(0, 0, 1))
    ops.fix(pg="X0", dofs=(1, 0, 0))
    ops.fix(pg="Y0", dofs=(0, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="Top", dof=3, value=U_MAX)
    ladruno = str(tmp_path / "pull.ladruno")
    ops.recorder.Ladruno(file=ladruno, elem_responses=("material.damage",))
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / N_STEPS)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    live = emitter.ops

    spec = DomainCaptureSpec(opensees=ops)
    spec.gauss(pg="Body", components=("damage_tension", "damage_compression"),
               name="dmg")
    path = str(tmp_path / "pull.h5")
    with ops.domain_capture(spec, path=path, ops=live) as cap:
        cap.begin_stage("pull", kind="static")
        for _ in range(N_STEPS):
            assert emitter.analyze(steps=1) == 0
            cap.step(t=live.getTime())
        cap.end_stage()
    (ops_tag,) = live.getEleTags()          # not the fem_eid
    truth = np.array([
        list(live.eleResponse(ops_tag, "material", str(gp), "damage"))
        for gp in range(1, 9)
    ])                                                      # (8 GP, 2)
    live.remove("recorders")                                # flush .ladruno

    assert truth[:, 0].min() > 0.1          # well past the peak: cracked
    assert np.allclose(truth[:, 1], 0.0)    # no compression damage

    om = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=om) as r:
        dt = r.elements.gauss.get(component="damage_tension")
        dc = r.elements.gauss.get(component="damage_compression")
    assert dt.values.shape == (N_STEPS, 8)
    assert set(int(e) for e in dt.element_index) == {eid}
    np.testing.assert_allclose(dt.values[-1], truth[:, 0], rtol=1e-12)
    np.testing.assert_allclose(dc.values[-1], truth[:, 1], atol=1e-14)
    # Zero before the peak, then monotone (no healing).
    first_step = 0                      # u = U_MAX / N_STEPS = 0.2 U_PEAK
    assert np.all(dt.values[first_step] == 0.0)
    assert np.all(np.diff(dt.values, axis=0) >= -1e-14)

    with Results.from_ladruno(ladruno) as rl:
        comps = rl.elements.gauss.available_components()
        assert {"damage_tension", "damage_compression"} <= set(comps)
        lt = rl.elements.gauss.get(component="damage_tension")
    np.testing.assert_allclose(lt.values[-1], truth[:, 0], rtol=1e-12)
