"""Live smoke test of an ``RCLayeredShell`` section on one ``ASDShellQ4``.

A 1 x 1 shell in the XY plane with a helper-built section: elastic concrete
(``nu = 0``) and two ``Steel01`` curtains, bars along x (``angle = 0``) and
along y (``angle = 90``) at different ``A_s / s``. ``local_cs=(1, 0, 0)``
pins the section x axis on every build (see ``PlateRebar``). Pulled in
membrane tension along x, then along y, with every rotation and the
out-of-plane translation held, the membrane stiffness is closed-form::

    N / eps = E_c * (h - sum(t_s)) + E_s * sum(t_s along the pull)

The concrete term uses the *reduced* concrete thickness, so a builder that
overlaid the bars on the concrete (``h`` instead of ``h - sum(t_s)``) fails
by ``E_c * sum(t_s)``; a bar counted in the wrong direction fails by
``E_s * (t_x - t_y)``. The loads keep the steel elastic.

Gated by the ``live`` marker; needs ``openseespy`` (stock or the fork — every
material here is stock OpenSees). Run with ``-s`` to see which binary ran.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.section.plate import RebarMesh

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import (  # noqa: E402
    LiveOpsEmitter,
    get_backend_name,
)

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

H = 0.20
E_C = 30e9
E_S = 200e9
T_X = 113.1e-6 / 0.15      # 12 mm bars @ 150 mm, along x
T_Y = 78.5e-6 / 0.20       # 10 mm bars @ 200 mm, along y
SUM_T = 2 * T_X + 2 * T_Y
P = 1.0e5                  # total pull, N (steel strain ~2e-5: elastic)


def _one_quad() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        node_pgs={"All": [1, 2, 3, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={"Wall": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),))},
    )
    return FEMStub(nodes=nodes, elements=elements)


def _pull(direction: int) -> float:
    """Membrane stiffness N / eps (per unit width) for a pull along x (1) or y (2)."""
    ops = apeSees(cast("object", _one_quad()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    conc = ops.nDMaterial.ElasticIsotropic(E=E_C, nu=0.0)
    steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=E_S, b=0.01)
    sec = ops.section.RCLayeredShell(h=H, concrete=conc, meshes=[
        RebarMesh(material=steel, angle=0.0, area_per_width=T_X,
                  cover=0.031, face="bottom"),
        RebarMesh(material=steel, angle=90.0, area_per_width=T_Y,
                  cover=0.042, face="bottom"),
        RebarMesh(material=steel, angle=0.0, area_per_width=T_X,
                  cover=0.031, face="top"),
        RebarMesh(material=steel, angle=90.0, area_per_width=T_Y,
                  cover=0.042, face="top"),
    ])
    ops.element.ASDShellQ4(pg="Wall", section=sec, local_cs=(1.0, 0.0, 0.0))

    # Membrane only: out-of-plane translation and all rotations held.
    ops.fix(pg="All", dofs=(0, 0, 1, 1, 1, 1))
    if direction == 1:
        ops.fix(nodes=[1], dofs=(1, 1, 0, 0, 0, 0))
        ops.fix(nodes=[4], dofs=(1, 0, 0, 0, 0, 0))
        loaded, probe = (2, 3), 2
        force = (P / 2, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        ops.fix(nodes=[1], dofs=(1, 1, 0, 0, 0, 0))
        ops.fix(nodes=[2], dofs=(0, 1, 0, 0, 0, 0))
        loaded, probe = (3, 4), 4
        force = (0.0, P / 2, 0.0, 0.0, 0.0, 0.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in loaded:
            p.load(node=n, forces=force)

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    print(
        f"\n[rc-layered-shell live] backend={get_backend_name()} "
        f"module={getattr(emitter.ops, '__file__', '?')}"
    )
    assert emitter.analyze(steps=1) == 0
    eps = float(emitter.ops.nodeDisp(probe, direction))  # L = 1
    return P / eps                                       # b = 1


def _expected(t_along: float) -> float:
    return E_C * (H - SUM_T) + E_S * 2 * t_along


@pytest.mark.live
def test_rc_layered_shell_membrane_stiffness_along_x() -> None:
    assert _pull(1) == pytest.approx(_expected(T_X), rel=1e-6)


@pytest.mark.live
def test_rc_layered_shell_membrane_stiffness_along_y() -> None:
    assert _pull(2) == pytest.approx(_expected(T_Y), rel=1e-6)


@pytest.mark.live
def test_rc_layered_shell_concrete_is_reduced_not_overlaid() -> None:
    # The two errors the closed form guards against are both far above
    # the 1e-6 tolerance: an overlay adds E_c * sum(t_s), a swapped bar
    # direction E_s * 2 * (T_X - T_Y).
    k = _pull(1)
    assert abs(k - (E_C * H + E_S * 2 * T_X)) > 1e-4 * k
    assert abs(k - _expected(T_Y)) > 1e-4 * k
