"""ADR 0115 D3 (R2-c, #1704) on the Tcl route: ``ops.results(path)``
after ``ops.tcl(path, run=True)``, where an OpenSees binary is available.

The 2-D cantilever at ``ndf=3``: tip ``displacement_y`` against
``PL^3/3EI`` and ``rotation_z`` against ``PL^2/2EI``, through a typed
``recorder.Node`` and through ``recorder.declare``. The openseespy-deck
oracles are ``tests/opensees/live/test_results_route_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees._target import resolve_opensees_binary

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
    cantilever_bridge,
)

pytestmark = pytest.mark.subprocess

UY = -P_TIP * L_BEAM ** 3 / (3.0 * E * I_BEAM)
RZ = -P_TIP * L_BEAM ** 2 / (2.0 * E * I_BEAM)


@pytest.fixture(autouse=True)
def _binary() -> None:
    try:
        resolve_opensees_binary(None, None)
    except Exception as e:  # no binary on this machine
        pytest.skip(f"no OpenSees binary: {e}")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def test_tcl_run_reads_back_typed_and_declared_records(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ops = cantilever_bridge(cantilever)
    ops.recorder.Node(file="tip.out", label="tip", dofs=(2, 3),
                      response="disp", time_format="dt")
    ops.recorder.declare(nodes=("displacement_x",), label="tip",
                         file_root=".", name="tipx")
    ops.tcl("deck.tcl", run=True, analyze_steps=1)
    r = ops.results("deck.tcl")
    get = r.nodes.get
    assert get(label="tip", component="displacement_y").item() == (
        pytest.approx(UY, rel=1e-9))
    assert get(label="tip", component="rotation_z").item() == (
        pytest.approx(RZ, rel=1e-9))
    assert get(label="tip", component="displacement_x").item() == (
        pytest.approx(0.0, abs=1e-15))


def test_an_aborted_tcl_deck_exits_0_and_the_read_refuses(
    cantilever: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deck whose analyze fails stops at the deck's own Tcl ``error``,
    and OpenSees.exe still exits 0: only the missing completion line
    says the run did not finish. ``ops.tcl`` raises on it (R3-S6, ADR
    0106 Amendment 1), and the read refuses."""
    from apeGmsh.opensees._internal.build.errors import BridgeError
    from apeGmsh.opensees.apesees import apeSees

    monkeypatch.chdir(tmp_path)
    ops = apeSees(cantilever, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=1e-2, E=E,
                                  Iz=I_BEAM)
    # No support: the stiffness is singular, so analyze fails.
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -P_TIP, 0.0))
    ops.recorder.Node(file="tip.out", label="tip", dofs=(2,),
                      response="disp", time_format="dt")
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    with pytest.raises(RuntimeError, match=r"exited 0 but the deck did not "
                                           r"run to its end"):
        ops.tcl("deck.tcl", run=True, analyze_steps=2)   # rc 0, no marker
    log = (tmp_path / "deck.log").read_text(encoding="utf-8")
    assert "analyze FAILED" in log
    with pytest.raises(BridgeError, match=r"did not finish"):
        ops.results("deck.tcl")
