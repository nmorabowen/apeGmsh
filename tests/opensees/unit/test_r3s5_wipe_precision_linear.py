"""R3-S5 (#1789): recorder ``precision=`` (D7), the unstaged ``Linear``
without a test, and ``ops.wipe()`` kept off every deck (D6).

Oracles, each naming the right answer:

* **``precision=`` is a positive int, emitted only when set.** ``0``, a
  negative, a ``bool``, a float and a string raise; ``None`` emits the
  argv of a recorder without the field, token for token, so existing
  decks are byte-identical. ``precision=10`` emits ``-precision 10``
  right after the file, on ``Node`` and ``Element``, in the Tcl and the
  Python deck.
* **``decl_params`` carries the field.** The archive's recorder row has
  ``precision`` (``10``, or ``None`` when unset): the additive
  ``/opensees`` minor the maintainer accepted as D7.
* **The unstaged test is optional only for a non-iterating algorithm.**
  The chain check skips the test when the algorithm the domain ends up
  with (the last registered) has ``needs_test`` False (``Linear``), and
  keeps it required for an iterating one, and for ``Linear`` under a
  strategy ladder (its rungs iterate). ``analyze(strategy=...)`` passes
  the ladder to the check.
* **``wipe`` is a live session action.** It is no ``VERBS`` row and no
  emitter Protocol method. (Its live behaviour, and that a deck emitted
  after it is byte-identical, are in
  ``tests/opensees/live/test_wipe_precision_live.py``.)

The live halves (the 10-digit column, the flushed ``.mpco``, the frame
without its test) are in that live module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.algorithm import Linear, Newton
from apeGmsh.opensees.analysis.strategy import Ladder
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.verbs import VERBS
from apeGmsh.opensees.recorder import Element as ElementRec
from apeGmsh.opensees.recorder import Node as NodeRec

from tests.opensees.unit.test_unconsumed_model_definition import _model, _ops


# ---------------------------------------------------------------------------
# precision= (D7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3, True, False, 2.5, 10.0, "10"])
def test_precision_refuses_anything_but_a_positive_int(bad: Any) -> None:
    with pytest.raises(ValueError, match="precision must be a positive int"):
        NodeRec(file="n.out", response="disp", nodes=(1,), dofs=(1,),
                precision=bad)
    with pytest.raises(ValueError, match="precision must be a positive int"):
        ElementRec(file="e.out", response=("force",), elements=(1,),
                   precision=bad)


def _argv(rec: Any) -> tuple[Any, ...]:
    em = RecordingEmitter()
    rec._emit(em, 1)
    ((method, args, _kw),) = em.calls
    assert method == "recorder"
    return tuple(args)


def test_precision_unset_emits_the_argv_without_the_flag() -> None:
    assert _argv(NodeRec(file="n.out", response="disp", nodes=(1, 2),
                         dofs=(1, 2), dT=0.5, time_format="dt")) == (
        "Node", "-file", "n.out", "-dT", 0.5, "-time", "-node", 1, 2,
        "-dof", 1, 2, "disp")
    assert _argv(ElementRec(file="e.out", response=("force",),
                            elements=(3,))) == (
        "Element", "-file", "e.out", "-ele", 3, "force")


def test_precision_emits_the_flag_after_the_file() -> None:
    assert _argv(NodeRec(file="n.out", response="disp", nodes=(1,),
                         dofs=(2,), precision=10)) == (
        "Node", "-file", "n.out", "-precision", 10, "-node", 1,
        "-dof", 2, "disp")
    assert _argv(ElementRec(file="e.out", response=("force",),
                            elements=(3,), precision=12)) == (
        "Element", "-file", "e.out", "-precision", 12, "-ele", 3, "force")


def _bridge_with_recorders(*, precision: int | None) -> apeSees:
    ops = _ops(_model(sp=[], masses=False))
    ops.recorder.Node(file="n.out", response="disp", nodes=(5, 6),
                      dofs=(1,), precision=precision, name="rn")
    ops.recorder.Element(file="e.out", response=("stresses",), pg="Block",
                         precision=precision, name="re")
    return ops


def test_precision_reaches_the_tcl_and_the_py_deck(tmp_path: Path) -> None:
    ops = _bridge_with_recorders(precision=10)
    ops.tcl(str(tmp_path / "d.tcl"))
    ops.py(str(tmp_path / "d.py"))
    tcl = (tmp_path / "d.tcl").read_text(encoding="utf-8").splitlines()
    py = (tmp_path / "d.py").read_text(encoding="utf-8").splitlines()
    rec_tcl = [ln for ln in tcl if ln.startswith("recorder ")]
    rec_py = [ln for ln in py if "ops.recorder(" in ln]
    assert len(rec_tcl) == 2 and len(rec_py) == 2
    assert all(" -precision 10 " in ln for ln in rec_tcl), rec_tcl
    assert all("'-precision', 10," in ln for ln in rec_py), rec_py


def test_precision_unset_writes_no_flag_to_either_deck(tmp_path: Path) -> None:
    ops = _bridge_with_recorders(precision=None)
    ops.tcl(str(tmp_path / "d.tcl"))
    ops.py(str(tmp_path / "d.py"))
    for name in ("d.tcl", "d.py"):
        assert "precision" not in (tmp_path / name).read_text(
            encoding="utf-8")


@pytest.mark.parametrize("precision", [10, None])
def test_decl_params_carries_precision(
    tmp_path: Path, precision: int | None,
) -> None:
    path = tmp_path / "m.h5"
    _bridge_with_recorders(precision=precision).h5(str(path))
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    for key in ("opensees/recorder/rn", "opensees/recorder/re"):
        (row,) = table.params_for(key)
        assert "precision" in row.params, key
        assert row.params["precision"] == precision, key


# ---------------------------------------------------------------------------
# Unstaged Linear without a test (brief item 4, unstaged half)
# ---------------------------------------------------------------------------


def _chain(ops: apeSees, *algorithms: Any, test: bool) -> apeSees:
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    if test:
        ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    for algo in algorithms:
        ops.register(algo)
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _bare() -> apeSees:
    return _ops(_model(sp=[], masses=False))


def test_linear_without_a_test_passes_the_chain_check() -> None:
    _chain(_bare(), Linear(), test=False)._check_analysis_chain_for_analyze()


def test_an_iterating_algorithm_without_a_test_is_refused() -> None:
    ops = _chain(_bare(), Newton(), test=False)
    with pytest.raises(BridgeError, match=r"missing: test\."):
        ops._check_analysis_chain_for_analyze()


def test_the_last_registered_algorithm_decides() -> None:
    # The deck emits both; the domain keeps the last one.
    _chain(_bare(), Newton(), Linear(),
           test=False)._check_analysis_chain_for_analyze()
    ops = _chain(_bare(), Linear(), Newton(), test=False)
    with pytest.raises(BridgeError, match=r"missing: test\."):
        ops._check_analysis_chain_for_analyze()


def test_a_strategy_ladder_keeps_the_test_required() -> None:
    ops = _chain(_bare(), Linear(), test=False)
    with pytest.raises(BridgeError, match=r"missing: test\."):
        ops._check_analysis_chain_for_analyze(strategy_attached=True)
    # analyze() hands its ladder to the check, which refuses before any
    # live module is touched.
    with pytest.raises(BridgeError, match=r"missing: test\."):
        ops.analyze(steps=1, strategy=Ladder(rungs=(Newton(),)))


def test_the_other_chain_slots_stay_required_with_linear() -> None:
    ops = _bare()
    ops.algorithm.Linear()
    with pytest.raises(BridgeError) as err:
        ops._check_analysis_chain_for_analyze()
    msg = str(err.value)
    assert "missing: constraints, numberer, system, integrator, analysis." in msg


def test_a_model_without_a_test_archives_and_replays(tmp_path: Path) -> None:
    from apeGmsh.opensees import OpenSeesModel

    from tests.opensees.live.test_readback_live import build_cantilever_fem

    # A real snapshot: the replay reads the archive's neutral zone.
    ops = apeSees(build_cantilever_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(pg="Beam", transf=ops.geomTransf.Linear(),
                                  A=1.0e-2, E=2.0e11, Iz=8.0e-6)
    ops.fix(label="base", dofs=(1, 1, 1))
    _chain(ops, Linear(), test=False)
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    ops.tcl(str(tmp_path / "direct.tcl"))
    decks = {
        "direct": (tmp_path / "direct.tcl").read_text(encoding="utf-8"),
        "replay": OpenSeesModel.from_h5(str(path)).build("tcl"),
    }
    for name, deck in decks.items():
        lines = [ln.strip() for ln in deck.splitlines()]
        assert not any(ln.startswith("test ") for ln in lines), name
        assert "algorithm Linear" in lines, name


# ---------------------------------------------------------------------------
# wipe() (D6) is no verb
# ---------------------------------------------------------------------------


def test_wipe_is_no_verb_and_no_protocol_method() -> None:
    assert "wipe" not in VERBS
    assert not hasattr(Emitter, "wipe")
    assert callable(apeSees.wipe)
