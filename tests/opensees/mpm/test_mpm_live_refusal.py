"""The stock side of the ``mpm`` warrant rows (ADR 0121 D3, ADR 0134 D7).

MPM is the fork's engine and deck-only (ADR 0134 D1): the live route refuses
a model that declares any ``mpm`` member, on every build, before the domain
is touched, with an error that names the fork. One test per member, each a
row of ``tests/opensees/warrant_rows.py`` (``refused``); they run in the
``live-stock`` lane on stock openseespy. The fork side is
``tests/opensees/integration_ladruno/test_mpm_deck_fork.py``.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from apeGmsh.opensees import apeSees

from ._meshes import grid_nodes, make_fem, quads

pytestmark = pytest.mark.live


def _with(member: Callable[[apeSees], object]) -> apeSees:
    """A model of an ``mpm`` grid and body, plus ``member``."""
    nodes, ids = grid_nodes((4, 2), (0.5, 0.0), (1.5, 0.4))
    ops = apeSees(make_fem(nodes, {"Soil": quads(ids)}), units="N_m_s",
                  _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.ElasticIsotropic(E=1.0e7, nu=0.3, name="soil")
    ops.timeSeries.Linear(name="ramp")
    ops.mpm.grid(origin=(0.0, 0.0), h=0.1, cells=(20, 10))
    ops.mpm.body("Soil", material="soil", density=1000.0, ppc=2)
    member(ops)
    ops.fix(pg="Soil", dofs=[1, 1])
    return ops


def test_live_refuses_mpm_grid_naming_the_fork() -> None:
    ops = _with(lambda ops: None)
    with pytest.raises(RuntimeError, match="Grid.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_body_naming_the_fork() -> None:
    ops = _with(lambda ops: None)
    with pytest.raises(RuntimeError, match="Body.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_fix_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.fix("xmin", "x"))
    with pytest.raises(RuntimeError, match="Fix.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_wall_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.wall("ymin", 0.3))
    with pytest.raises(RuntimeError, match="Wall.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_gridvel_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.gridvel(
        "push", ((0.4, -0.1), (1.6, 0.5)), (0.1, 0.0)))
    with pytest.raises(RuntimeError, match="Gridvel.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_gridvel_affine_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.gridvelAffine(
        "shear", ((0.4, -0.1), (1.6, 0.5)), ((0.0, 0.5), (0.0, 0.0))))
    with pytest.raises(RuntimeError,
                       match="GridvelAffine.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_gravity_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.gravity(0.0, -9.81))
    with pytest.raises(RuntimeError, match="Gravity.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_time_series_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.timeSeries("ramp"))
    with pytest.raises(RuntimeError, match="TimeSeries.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_damping_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.damping(0.05))
    with pytest.raises(RuntimeError, match="Damping.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_flip_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.flip(0.99))
    with pytest.raises(RuntimeError, match="Flip.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_mass_cutoff_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.massCutoff(1.0e-9))
    with pytest.raises(RuntimeError, match="MassCutoff.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_safety_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.safety(0.9))
    with pytest.raises(RuntimeError, match="Safety.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_units_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.units("m", "N", "s"))
    with pytest.raises(RuntimeError, match="Units.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_output_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.output("run.ladruno", every=10))
    with pytest.raises(RuntimeError, match="Output.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_stage_naming_the_fork() -> None:
    def stage(ops: apeSees) -> None:
        ops.nDMaterial.DruckerPrager(
            K=1.0e7, G=5.0e6, sigmaY=1.0e3, rho=0.1, rhoBar=0.1, Kinf=0.0,
            Ko=0.0, delta1=0.0, delta2=0.0, H=0.0, theta=1.0, name="weak")
        ops.mpm.body("Soil", material="weak", density=1000.0, ppc=2,
                     name="weak_soil")
        ops.mpm.stage("weak", 0)
        ops.mpm.analyze(10, 1.0e-4)

    ops = _with(stage)
    with pytest.raises(RuntimeError, match="Stage.*the Ladruno fork's mpm engine"):
        ops.run()


def test_live_refuses_mpm_analyze_naming_the_fork() -> None:
    ops = _with(lambda ops: ops.mpm.analyze(10, 1.0e-4))
    with pytest.raises(RuntimeError, match="Analyze.*the Ladruno fork's mpm engine"):
        ops.run()
