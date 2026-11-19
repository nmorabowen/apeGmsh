"""The fork side of ``mpm.Gridvel`` / ``mpm.GridvelAffine`` (ADR 0134
Amendment 3, slice #3064).

Since ``ladruno-v20261118b`` (fork #1109, D18) the explicit engine (mode F)
applies ``mpm gridvel`` in both projections; earlier releases accepted the
command and ignored it. Each deck here is emitted by ``apeSees.py`` and run
on the pinned fork in a fresh process, like ``test_mpm_deck_fork.py``.

Two elastic blocks rest, with no gravity, on a 20 x 10 grid at h = 0.1:
``Pushed`` on [0.3, 0.7]^2 and ``Still`` on [1.3, 1.7] x [0.3, 0.7]. A
gridvel box over ``Pushed`` reaches every grid node a ``Pushed`` particle
touches (a uGIMP particle reaches h + h/4 from its centre) and none a
``Still`` particle touches. The runs use pure PIC (``mpm flip 0``), so a
particle's velocity is the grid velocity interpolated at its centre. uGIMP
interpolates constant and linear fields exactly when every supporting node is
inside the grid, so the oracles are closed forms:

* ``Gridvel`` with ``v``: every ``Pushed`` particle moves at ``v`` and is
  displaced by ``v T``; every ``Still`` particle stays at rest. The control
  deck, the same model without the gridvel, leaves both at rest.
* ``Gridvel`` with ``translate`` and ``profile (t0, Tr, t1)``: the box rides
  with the body, and each step prescribes the step's mean of ``v phi(t)``,
  so the displacement is ``v Phi(T)``, ``Phi`` the integral of the trapezoid.
* ``GridvelAffine`` with ``L`` and ``ref``: every ``Pushed`` particle moves at
  ``L (x - ref)``; at the last output that holds to ``dt |L| |v|`` (the
  centre moved by ``dt v`` after the interpolation).

The two warrant-row tests (``tests/opensees/warrant_rows.py``) are
``test_mpm_gridvel_moves_the_body_at_v_on_fork`` and
``test_mpm_gridvel_affine_prescribes_l_times_x_minus_ref_on_fork``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.results import Results
from tests.opensees.mpm._meshes import grid_nodes, make_fem, quads

pytestmark = pytest.mark.ladruno_fork

_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n")

H, PPC, RHO, E, NU = 0.1, 2, 1000.0, 1.0e7, 0.3
STEPS, DT, EVERY = 100, 2.0e-4, 25
T = STEPS * DT
#: Over ``Pushed`` ([0.3, 0.7]^2) with a margin of two cells; ``Still``'s
#: particles reach no node with x < 1.2.
BOX = ((0.1, 0.1), (0.9, 0.9))
V = (0.2, -0.1)
L = ((0.0, 0.5), (-0.25, 0.0))
REF = (0.5, 0.5)
PROFILE = (0.0, 0.25 * T, T)


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _bridge() -> apeSees:
    a_nodes, a_ids = grid_nodes((4, 4), (0.3, 0.3), (0.7, 0.7))
    b_nodes, b_ids = grid_nodes((4, 4), (1.3, 0.3), (1.7, 0.7), first=100)
    ops = apeSees(make_fem({**a_nodes, **b_nodes},
                           {"Pushed": quads(a_ids), "Still": quads(b_ids)}),
                  units="N_m_s", _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, name="soil")
    ops.fix(pg="Pushed", dofs=[1, 1])          # no free FE DOF (fork v1)
    ops.fix(pg="Still", dofs=[1, 1])
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.body("Pushed", material="soil", density=RHO, ppc=PPC)
    ops.mpm.body("Still", material="soil", density=RHO, ppc=PPC)
    ops.mpm.flip(0.0)
    return ops


def _run(root: Path, name: str, declare: Callable[[apeSees], object]) -> "dict[str, Any]":
    """Emit ``_bridge()`` plus ``declare``, run it on the fork, and return
    the first and last outputs of each body."""
    ops = _bridge()
    declare(ops)
    deck_dir, cwd = root / f"{name}_deck", root / f"{name}_cwd"
    deck_dir.mkdir()
    cwd.mkdir()
    out = deck_dir / f"{name}.ladruno"
    ops.mpm.output(str(out).replace("\\", "/"), every=EVERY,
                   fields=("VELOCITY",))
    ops.mpm.analyze(STEPS, DT)
    deck = deck_dir / f"{name}.py"
    ops.py(str(deck))
    text = deck.read_text(encoding="utf-8")
    stock = "import openseespy.opensees as ops"
    assert stock in text
    deck.write_text(text.replace(stock, _FORK_IMPORT), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=cwd, env=_env(),
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = Results.from_ladruno(str(out))
    try:
        assert r.particles.info().mode == "explicit"       # mode F
        res: dict[str, Any] = {"deck": text}
        for body in ("Pushed", "Still"):
            pos = r.particles.get(component="POSITION", body=body)
            vel = r.particles.get(component="VELOCITY", body=body)
            assert pos.n_outputs == STEPS // EVERY + 1
            np.testing.assert_allclose(pos.time[-1], T, rtol=1e-9)
            i0, x0 = pos.output(0)
            i1, x1 = pos.output(-1)
            iv, v1 = vel.output(-1)
            assert np.array_equal(i0, i1) and np.array_equal(i0, iv)
            res[body] = {"x0": np.asarray(x0)[:, :2], "x1": np.asarray(x1)[:, :2],
                         "v1": np.asarray(v1)[:, :2]}
        return res
    finally:
        r.close()


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    root = tmp_path_factory.mktemp("mpm_gridvel")
    return {
        "control": _run(root, "control", lambda ops: None),
        "v": _run(root, "v", lambda ops: ops.mpm.gridvel("push", BOX, V)),
        "profile": _run(root, "profile", lambda ops: ops.mpm.gridvel(
            "ride", BOX, V, translate="x", profile=PROFILE)),
        "affine": _run(root, "affine", lambda ops: ops.mpm.gridvelAffine(
            "shear", BOX, L, ref=REF)),
    }


def _at_rest(body: "dict[str, Any]") -> None:
    np.testing.assert_array_equal(body["v1"], 0.0)
    np.testing.assert_array_equal(body["x1"], body["x0"])


def test_mpm_control_without_gridvel_stays_at_rest_on_fork(
    runs: "dict[str, Any]",
) -> None:
    assert "'gridvel'" not in runs["control"]["deck"]
    _at_rest(runs["control"]["Pushed"])
    _at_rest(runs["control"]["Still"])


def test_mpm_gridvel_moves_the_body_at_v_on_fork(runs: "dict[str, Any]") -> None:
    run = runs["v"]
    assert ("ops.mpm('gridvel', 'push', 'box', 0.1, 0.1, 0.9, 0.9, 'v', "
            "0.2, -0.1)") in run["deck"]
    pushed = run["Pushed"]
    # VELOCITY may be stored as float32 (fork Contract 2); POSITION is float64.
    np.testing.assert_allclose(pushed["v1"], np.broadcast_to(V, pushed["v1"].shape),
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(pushed["x1"] - pushed["x0"],
                               np.broadcast_to(np.multiply(V, T), pushed["x0"].shape),
                               rtol=0, atol=1e-12)
    _at_rest(run["Still"])


def test_mpm_gridvel_translate_profile_moves_the_body_by_v_phi_on_fork(
    runs: "dict[str, Any]",
) -> None:
    run = runs["profile"]
    assert "'translate', 'x', 'profile', 0.0, " in run["deck"]
    t0, tr, t1 = PROFILE
    phi_t = (t1 - t0) - tr                    # the trapezoid's full integral, T <= t1
    pushed = run["Pushed"]
    np.testing.assert_allclose(pushed["x1"] - pushed["x0"],
                               np.broadcast_to(np.multiply(V, phi_t), pushed["x0"].shape),
                               rtol=0, atol=1e-12)
    _at_rest(run["Still"])


def test_mpm_gridvel_affine_prescribes_l_times_x_minus_ref_on_fork(
    runs: "dict[str, Any]",
) -> None:
    run = runs["affine"]
    assert ("ops.mpm('gridvel', 'shear', 'box', 0.1, 0.1, 0.9, 0.9, 'affine', "
            "0.0, 0.5, -0.25, 0.0, 'ref', 0.5, 0.5)") in run["deck"]
    pushed = run["Pushed"]
    want = (pushed["x1"] - np.asarray(REF)) @ np.asarray(L).T
    vmax = float(np.abs(want).max())
    assert vmax > 0.05                        # a field, not a zero
    np.testing.assert_allclose(pushed["v1"], want, rtol=0,
                               atol=2.0 * DT * float(np.abs(L).max()) * vmax + 1e-6)
    _at_rest(run["Still"])
