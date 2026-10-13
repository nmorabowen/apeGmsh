"""R3-S5 (#1789) live checks: ``ops.wipe()`` (D6), recorder
``precision=`` (D7), and the unstaged ``Linear`` without a test.

A cantilever (``test_readback_live``'s mesh) under a tip load ramped over
``N`` load steps. Elastic and small-displacement, so step ``k`` deflects
the tip by ``k / N`` of the closed form ``-P L^3 / (3 E I)``.

* **wipe flushes and closes every recorder.** After ``analyze`` +
  ``ops.wipe()`` the text ``.out`` file holds ``N`` rows, each the closed
  form at its step. (OpenSees writes a text row through at each step, so
  the row count also holds before the wipe; the ``.mpco`` is what the
  wipe completes.) On the fork, an ``.mpco`` the run wrote does not open
  before the wipe (HDF5: "file is already open for write") and opens with
  ``Results.from_mpco`` after it, holding the ``N`` closed-form steps.
* **wipe closes the session, names why, and moves the epoch.** A
  ``.read`` after it raises with the ``ops.wipe()`` cause; the domain has
  no node; another bridge's read session over the same domain raises
  "domain was rebuilt".
* **wipe never reaches a deck.** The Tcl and Python decks emitted after
  a wipe are byte-identical to those emitted before it.
* **precision=10 gives a 10-significant-digit column.** The tip load is
  ``P / 7``, so the deflection has no short decimal form. The
  ``-precision 10`` column has 10 significant digits and matches the
  live value to 1e-9; the default column (6 digits) does not.
* **The frame without its test.** The two-storey portal frame of the
  R3-S1 stage test, unstaged: ``Linear`` with no test gives the same
  storey drifts and the same eigen periods as with one.

Stock openseespy except the ``.mpco`` case (``ladruno_fork``).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
)

pytestmark = pytest.mark.live

N = 5
A_BEAM = 1.0e-2


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def _bridge(fem: Any, *, p_tip: float = P_TIP, with_test: bool = True,
            ) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(pg="Beam", transf=ops.geomTransf.Linear(),
                                  A=A_BEAM, E=E, Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -p_tip, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    if with_test:
        ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0 / N)
    ops.analysis.Static()
    return ops


def _tip_closed_form(p_tip: float) -> float:
    return -p_tip * L_BEAM ** 3 / (3.0 * E * I_BEAM)


def _ramp(p_tip: float) -> np.ndarray:
    return _tip_closed_form(p_tip) * np.arange(1, N + 1) / N


# ---------------------------------------------------------------------------
# wipe (D6)
# ---------------------------------------------------------------------------


def test_wipe_leaves_every_row_and_closes_the_session(
    cantilever: Any, tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = _bridge(cantilever)
    tip = ops.nodes.one(label="tip")
    out = tmp_path / "tip.out"
    ops.recorder.Node(file=str(out), response="disp", nodes=(tip.tag,),
                      dofs=(2,))
    assert ops.analyze(steps=N) == 0
    assert tip.read("displacement_y") == pytest.approx(
        _tip_closed_form(P_TIP), rel=1e-9)

    ops.wipe()

    rows = np.loadtxt(out, ndmin=1)
    assert rows.shape == (N,)
    assert rows == pytest.approx(_ramp(P_TIP), rel=1e-5)  # 6 digits
    with pytest.raises(BridgeError, match=r"closed: ops\.wipe\(\) closed "
                       r"the live domain"):
        tip.read("displacement_y")
    assert list(get_ops().getNodeTags()) == []
    # The declarations stay: a later analyze builds the model again.
    assert ops.analyze(steps=N) == 0
    assert tip.read("displacement_y") == pytest.approx(
        _tip_closed_form(P_TIP), rel=1e-9)
    ops.wipe()


def test_wipe_moves_the_epoch_for_every_bridge(cantilever: Any) -> None:
    first = _bridge(cantilever)
    first.analyze(steps=N)
    tip = first.nodes.one(label="tip")
    tip.read("displacement_y")
    # A bridge that never analyzed wipes the shared domain.
    _bridge(cantilever).wipe()
    with pytest.raises(BridgeError, match="domain was rebuilt"):
        tip.read("displacement_y")


def test_wipe_never_reaches_a_deck(cantilever: Any, tmp_path: Path) -> None:
    ops = _bridge(cantilever)
    ops.recorder.Node(file=str(tmp_path / "tip.out"), response="disp",
                      nodes=(ops.nodes.one(label="tip").tag,), dofs=(2,))
    ops.tcl(str(tmp_path / "before.tcl"))
    ops.py(str(tmp_path / "before.py"))
    ops.analyze(steps=N)
    ops.wipe()
    ops.tcl(str(tmp_path / "after.tcl"))
    ops.py(str(tmp_path / "after.py"))
    for ext in ("tcl", "py"):
        before = (tmp_path / f"before.{ext}").read_bytes()
        assert (tmp_path / f"after.{ext}").read_bytes() == before, ext


@pytest.mark.ladruno_fork
def test_wipe_completes_the_mpco(cantilever: Any, tmp_path: Path) -> None:
    from apeGmsh.results import Results

    ops = _bridge(cantilever)
    mpco, model_h5 = tmp_path / "run.mpco", tmp_path / "model.h5"
    ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))
    ops.h5(str(model_h5))
    assert ops.analyze(steps=N) == 0
    # The recorder still holds the file: it is not readable yet.
    with pytest.raises(OSError, match="already open for write"):
        Results.from_mpco(str(mpco), model_h5=str(model_h5))

    ops.wipe()

    res = Results.from_mpco(str(mpco), model_h5=str(model_h5))
    tip = ops.nodes.one(label="tip").tag
    slab = res.nodes.get(component="displacement_y", ids=[tip])
    assert np.asarray(slab.values)[:, 0] == pytest.approx(
        _ramp(P_TIP), rel=1e-9)


# ---------------------------------------------------------------------------
# precision= (D7)
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _significant_digits(token: str) -> int:
    mantissa = re.split(r"[eE]", token.lstrip("+-"))[0].replace(".", "")
    return len(mantissa.lstrip("0"))


def test_precision_ten_writes_ten_significant_digits(
    cantilever: Any, tmp_path: Path,
) -> None:
    p_tip = P_TIP / 7.0   # a deflection with no short decimal form
    ops = _bridge(cantilever, p_tip=p_tip)
    tip = ops.nodes.one(label="tip")
    files = {n: tmp_path / f"tip_{n}.out" for n in (10, None)}
    for prec, path in files.items():
        ops.recorder.Node(file=str(path), response="disp",
                          nodes=(tip.tag,), dofs=(2,), precision=prec)
    elems = tmp_path / "beam_10.out"
    ops.recorder.Element(file=str(elems), response=("force",), pg="Beam",
                         precision=10)
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    rec_lines = [ln for ln in deck.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("recorder ")]
    assert sum(" -precision 10 " in ln for ln in rec_lines) == 2

    assert ops.analyze(steps=N) == 0
    live = tip.read("displacement_y")
    ops.wipe()

    tok10 = files[10].read_text(encoding="utf-8").split()
    tok6 = files[None].read_text(encoding="utf-8").split()
    assert len(tok10) == len(tok6) == N
    assert all(_NUMBER.fullmatch(t) for t in tok10 + tok6)
    assert [_significant_digits(t) for t in tok10] == [10] * N
    assert max(_significant_digits(t) for t in tok6) == 6
    assert float(tok10[-1]) == pytest.approx(live, rel=1e-9)
    assert float(tok6[-1]) != pytest.approx(live, rel=1e-9)
    assert np.array([float(t) for t in tok10]) == pytest.approx(
        _ramp(p_tip), rel=1e-9)
    # The element column: the base element's end shear is the tip load.
    beam = elems.read_text(encoding="utf-8").split()
    assert max(_significant_digits(t) for t in beam) == 10
    assert any(float(t) == pytest.approx(p_tip, rel=1e-9) for t in beam)


# ---------------------------------------------------------------------------
# The unstaged Linear without a test (brief item 4)
# ---------------------------------------------------------------------------


def _frame(fem: Any, *, with_test: bool) -> tuple[list[float], list[float]]:
    from tests.opensees.live.test_stage_linear_without_test_live import (
        A, E as E_FRAME, F_LATERAL, INERTIA, MASS, STOREY,
    )

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()
    for pg in ("columns", "beams"):
        ops.element.elasticBeamColumn(pg=pg, transf=transf, A=A, E=E_FRAME,
                                      Iz=INERTIA)
    ops.fix(pg="bases", dofs=(1, 1, 1))
    for k in (1, 2):
        ops.mass(pg=f"floor_{k}", values=(MASS, MASS, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for k in (1, 2):
            p.load(pg=f"floor_{k}", forces=(0.0, -MASS * 9.81, 0.0))
            p.load(pg=f"left_{k}", forces=(F_LATERAL * k, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    if with_test:
        ops.test.NormDispIncr(tol=1e-10, max_iter=1)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    assert ops.analyze(steps=1) == 0
    ux = [ops.nodes.one(pg=f"left_{k}").read("displacement_x")
          for k in (1, 2)]
    drifts = [ux[0] / STOREY, (ux[1] - ux[0]) / STOREY]
    periods = [2.0 * math.pi / math.sqrt(lam)
               for lam in ops.eigen(2).eigenvalues]
    ops.wipe()
    return drifts, periods


def test_unstaged_frame_without_a_test_matches_with_one() -> None:
    from tests.opensees.live.test_stage_linear_without_test_live import (
        _frame_fem,
    )

    fem = _frame_fem()
    d_test, t_test = _frame(fem, with_test=True)
    d_none, t_none = _frame(fem, with_test=False)
    assert d_none == pytest.approx(d_test, rel=1e-12)
    assert t_none == pytest.approx(t_test, rel=1e-12)
    assert d_test[0] > 0.0 and d_test[1] > 0.0
    assert all(t > 0.0 for t in t_test)
