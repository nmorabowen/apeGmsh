"""B5-b (#1726, fixes #1723) live oracle: ``recorder.declare`` maps node
components through the per-node DOF layout.

The 2-D cantilever at ``ndf=3`` (the R2-b fixture), tip load ``P``, run as
an openseespy deck in a subprocess of this interpreter. The declared tip
``rotation_z`` is DOF 3 of a planar beam node; the fixed 6-DOF table sent
it to ``-dof 6``, which OpenSees answers with 0 and no warning. The
recorder file must read ``PL^2/2EI`` (and ``PL^3/3EI`` for the
deflection beside it).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
    cantilever_bridge,
)

pytestmark = pytest.mark.live

UY = -P_TIP * L_BEAM ** 3 / (3.0 * E * I_BEAM)
RZ = -P_TIP * L_BEAM ** 2 / (2.0 * E * I_BEAM)


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def test_declared_tip_rotation_matches_pl2_over_2ei(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(
        nodes=("displacement_y", "rotation_z"), label="tip",
        file_root=str(tmp_path), name="tip",
    )
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)

    out = tmp_path / "tip__default__disp.out"
    assert out.is_file(), sorted(p.name for p in tmp_path.iterdir())
    rows = np.atleast_2d(np.loadtxt(out))
    # [time, uy, rz] for the one tip node, -dof 2 3.
    assert rows.shape[1] == 3, rows
    _t, uy, rz = rows[-1]
    assert uy == pytest.approx(UY, rel=1e-10)
    assert rz == pytest.approx(RZ, rel=1e-10)
