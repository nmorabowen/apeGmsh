"""Live transient smoke: a ``Viscous`` dashpot produces a velocity-
proportional force through an apeGmsh-emitted deck.

Gated by the ``live`` marker — only runs when ``openseespy`` is
installed.

Physics: a single free mass ``m`` on DOF 1, grounded through a
``zeroLength`` carrying a **pure** linear ``Viscous`` dashpot (C,
alpha=1), driven by a constant force ``F``. The equation of motion is
``m·v' = F − C·v`` → ``v(t) = (F/C)·(1 − e^(−C·t/m))``, so the terminal
velocity is exactly ``F/C``. Reading that back proves the element fed
the material a strain *rate* and the dashpot force balanced the load.

(A pure dashpot has zero static stiffness — §5.2 of the ZeroLength
guide — but in a transient analysis the Newmark mass term regularizes
the tangent, so no parallel elastic spring is needed here.)
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.element.zero_length import (  # noqa: E402
    ZeroLengthMatDir,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


@pytest.mark.live
def test_viscous_dashpot_reaches_terminal_velocity_F_over_C() -> None:
    fem = make_two_node_beam()  # node 1 @origin (Base), node 2; line "Cols"
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)

    C = 1.0e4
    F = 1.0e3
    m = 1.0
    v_terminal = F / C  # 0.1

    dashpot = ops.uniaxialMaterial.Viscous(C=C, alpha=1.0)
    ops.element.ZeroLength(
        pg="Cols",
        mat_dirs=(ZeroLengthMatDir(material=dashpot, dof=1),),
    )

    # Base fully fixed; free node moves on DOF 1 only.
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(nodes=(2,), dofs=(0, 1, 1))
    ops.mass(nodes=(2,), values=(m, m, m))

    ts = ops.timeSeries.Constant()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(F, 0.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=20)
    ops.algorithm.Linear()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    # tau = m/C = 1e-4 s; 300 steps @ 1e-5 s = 30·tau → fully plateaued.
    assert emitter.analyze(steps=300, dt=1.0e-5) == 0

    v = emitter.ops.nodeVel(2, 1)
    assert v == pytest.approx(v_terminal, rel=1e-2)
