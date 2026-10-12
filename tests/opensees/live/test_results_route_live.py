"""ADR 0115 D3 live oracles (R2-c, #1704): ``ops.results(path)`` after a
real ``ops.py(path, run=True)``.

The 2-D cantilever at ``ndf=3`` (the R2-b fixture), tip load ``P``, run
as an openseespy deck in a subprocess of this interpreter:

- a typed ``recorder.Node`` (DOFs 1, 2, 3, ``time_format="dt"``) read
  through the synthesised spec: tip ``displacement_y`` against
  ``PL^3/3EI``, tip ``rotation_z`` against ``PL^2/2EI`` (DOF 3 at
  ``ndf=3``, which the results capture table would send to DOF 6);
- the same deflection through ``recorder.declare(nodes=...)``;
- the same deflection through ``recorder.MPCO``, where the backend
  writes MPCO (stock pip openseespy has no MPCO recorder: a named skip).

The refusals need no solver: ``tests/opensees/unit/test_results_route.py``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees.apesees import apeSees

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


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_deck(ops: apeSees, deck: Path) -> None:
    # The deck runs in this interpreter, so it binds the same openseespy.
    ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)


def test_typed_node_tip_matches_the_closed_form(
    cantilever: Any, run_dir: Path,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.Node(file=str(run_dir / "tip.out"), label="tip",
                      dofs=(1, 2, 3), response="disp", time_format="dt")
    deck = run_dir / "deck.py"
    run_deck(ops, deck)
    r = ops.results(str(deck))
    tip = {c: r.nodes.get(label="tip", component=c).item()
           for c in ("displacement_x", "displacement_y", "rotation_z")}
    assert tip["displacement_y"] == pytest.approx(UY, rel=1e-9)
    assert tip["rotation_z"] == pytest.approx(RZ, rel=1e-9)
    assert tip["displacement_x"] == pytest.approx(0.0, abs=1e-15)


def test_declared_tip_deflection_matches_the_closed_form(
    cantilever: Any, run_dir: Path,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(nodes=("displacement_y",), label="tip",
                         file_root=str(run_dir), name="tip")
    deck = run_dir / "deck.py"
    run_deck(ops, deck)
    uy = ops.results(str(deck)).nodes.get(
        label="tip", component="displacement_y").item()
    assert uy == pytest.approx(UY, rel=1e-9)


def _subprocess_has_mpco(where: Path) -> bool:
    """The deck's interpreter accepts ``recorder mpco``."""
    probe = (
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 1, '-ndf', 1)\n"
        "ops.node(1, 0.0)\n"
        "ops.recorder('mpco', 'probe.mpco', '-N', 'displacement')\n"
        "ops.wipe()\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=where,
                          capture_output=True, check=False)
    return proc.returncode == 0


def test_mpco_tip_deflection_matches_the_closed_form(
    cantilever: Any, run_dir: Path, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if not _subprocess_has_mpco(tmp_path_factory.mktemp("mpco_probe")):
        pytest.skip(
            "the openseespy build the deck runs on has no MPCO recorder "
            "(stock pip openseespy ships none); the MPCO route is read "
            "only where the backend writes MPCO")
    ops = cantilever_bridge(cantilever)
    ops.recorder.MPCO(file=str(run_dir / "r.mpco"),
                      nodal_responses=("displacement",))
    deck = run_dir / "deck.py"
    run_deck(ops, deck)
    uy = ops.results(str(deck)).nodes.get(
        label="tip", component="displacement_y").item()
    assert uy == pytest.approx(UY, rel=1e-9)
