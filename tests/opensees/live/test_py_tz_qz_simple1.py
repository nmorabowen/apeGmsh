"""Live acceptance: ``PySimple1`` / ``TzSimple1`` / ``QzSimple1`` load
into OpenSees through the bridge and trace their backbones.

Gated by the ``live`` marker. All three are stock OpenSees materials
(``SRC/material/uniaxial/PY``), so this runs on stock openseespy and on
the Ladruno fork alike.

Each spring sits on a ``zeroLength`` in global dir 2 (ndm 2 / ndf 3),
node 1 fixed and node 2 driven by DisplacementControl, so every
increment is kinematically determined and the measured force is the
material response at the prescribed deformation. The spring force is
read as minus the reaction at the grounded node (equilibrium), so it
does not depend on a bridge-allocated element tag.

What is gated is what a wrong emit would break: the sign convention
(force follows deformation), the capacity the backbone approaches
without exceeding, the ``*50`` reference point, and the q-z
compression/uplift asymmetry (full ``qult`` in bearing, capped at
``suction * qult`` in uplift). The ``*50`` bands are loose on purpose:
the C++ backbones only approximate 50 % at ``y50`` (PySimple1 soft clay
gives 0.49, API sand 0.47; TzSimple1 hits 0.50 exactly).
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.types import UniaxialMaterial
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

from tests.opensees.live._springs import coincident_spring

_N_STEPS = 200


def _drive(
    make_mat: str, kwargs: dict[str, float], u_end: float,
) -> tuple[list[float], list[float]]:
    """Push the spring monotonically to ``u_end``; return (u, F) samples."""
    fem = coincident_spring()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)

    mat = cast(
        UniaxialMaterial, getattr(ops.uniaxialMaterial, make_mat)(**kwargs),
    )
    ops.element.ZeroLength(
        pg="Cols",
        mat_dirs=(ZeroLengthMatDir(material=mat, dof=2),),
    )
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(nodes=(2,), dofs=(1, 0, 1))

    # DisplacementControl needs a nonzero reference load on the
    # controlled DOF; it solves for the load factor, so the magnitude
    # does not contaminate the measured force.
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(0.0, 1.0 if u_end > 0 else -1.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.DisplacementControl(node=2, dof=2, dU=u_end / _N_STEPS)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    us: list[float] = []
    forces: list[float] = []
    for _ in range(_N_STEPS):
        assert emitter.analyze(steps=1) == 0
        emitter.ops.reactions()
        us.append(float(emitter.ops.nodeDisp(2, 2)))
        forces.append(-float(emitter.ops.nodeReaction(1, 2)))
    return us, forces


def _force_at(us: list[float], forces: list[float], u: float) -> float:
    """Force at the sample nearest to deformation ``u``."""
    i = min(range(len(us)), key=lambda k: abs(us[k] - u))
    return forces[i]


# (material, kwargs, capacity key, reference key, band for F(ref)/ult)
_SYMMETRIC = [
    pytest.param(
        "PySimple1",
        {"soil_type": 1, "pult": 150.0, "y50": 0.01, "Cd": 1.0},
        "pult", "y50", (0.45, 0.55), id="py-matlock-clay",
    ),
    pytest.param(
        "PySimple1",
        {"soil_type": 2, "pult": 150.0, "y50": 0.01, "Cd": 0.3},
        "pult", "y50", (0.42, 0.55), id="py-api-sand",
    ),
    pytest.param(
        "TzSimple1",
        {"tz_type": 1, "tult": 40.0, "z50": 0.002},
        "tult", "z50", (0.49, 0.51), id="tz-reese-oneill-clay",
    ),
    pytest.param(
        "TzSimple1",
        {"tz_type": 2, "tult": 40.0, "z50": 0.002},
        "tult", "z50", (0.49, 0.51), id="tz-mosher-sand",
    ),
]


@pytest.mark.live
@pytest.mark.parametrize("direction", [-1.0, 1.0], ids=["neg", "pos"])
@pytest.mark.parametrize(
    ("mat_name", "kwargs", "ult_key", "ref_key", "band"), _SYMMETRIC,
)
def test_symmetric_spring_approaches_capacity(
    mat_name: str,
    kwargs: dict[str, float],
    ult_key: str,
    ref_key: str,
    band: tuple[float, float],
    direction: float,
) -> None:
    ult, ref = kwargs[ult_key], kwargs[ref_key]
    us, forces = _drive(mat_name, kwargs, direction * 50.0 * ref)

    # Force follows deformation and grows monotonically, never past ult.
    assert all(f * direction > 0.0 for f in forces)
    mags = [abs(f) for f in forces]
    assert all(b >= a - 1e-9 * ult for a, b in zip(mags, mags[1:]))
    assert max(mags) < ult

    ratio_ref = abs(_force_at(us, forces, direction * ref)) / ult
    assert band[0] <= ratio_ref <= band[1]
    assert mags[-1] / ult > 0.97


@pytest.mark.live
@pytest.mark.parametrize("qz_type", [1, 2], ids=["reese-oneill", "vijayvergiya"])
def test_qz_bearing_mobilizes_qult_in_compression(qz_type: int) -> None:
    qult, z50 = 900.0, 0.005
    us, forces = _drive(
        "QzSimple1", {"qz_type": qz_type, "qult": qult, "z50": z50},
        -50.0 * z50,
    )
    assert all(f < 0.0 for f in forces), "bearing is the NEGATIVE direction"
    assert max(abs(f) for f in forces) < qult
    assert abs(forces[-1]) / qult > 0.97


@pytest.mark.live
def test_qz_uplift_is_capped_at_suction_times_qult() -> None:
    qult, z50, suction = 900.0, 0.005, 0.1
    _us, forces = _drive(
        "QzSimple1",
        {"qz_type": 2, "qult": qult, "z50": z50, "suction": suction},
        50.0 * z50,
    )
    assert all(f > 0.0 for f in forces)
    cap = suction * qult
    assert max(forces) <= cap
    assert forces[-1] > 0.95 * cap
