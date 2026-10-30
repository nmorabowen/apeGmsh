"""X #2613: ``recorder.Ladruno(precision="f64" | "f32")``.

* **The default is ``"f64"`` and emits nothing.** The argv of a default
  recorder is exactly the pre-field argv (``ladruno <file> -N ...``), so a
  default deck is byte-identical to one written before the field existed;
  ``precision="f64"`` spelled out is the same recorder.
* **``"f32"`` emits ``-precision f32`` right after the file**, before the
  ``-N``/``-E``/``-T``/``-R`` tail and the trailing ``-G energy`` (the fork's
  ``-G`` parser eats trailing tokens, so the flag never follows it).
* **Anything else fails closed.** The fork warns and keeps f64 on an unknown
  token, so the bridge refuses every spelling but the two canonical ones at
  construction (``"float32"``, ``"single"``, ``32``, ``None``, ``True``...).
* **The value round-trips through ``ops.h5``**: the ``/opensees/decl_params``
  row carries ``precision`` (``"f64"`` or ``"f32"``, never null) and the
  ``/opensees/recorders/ladruno_N`` argv carries the flag iff f32.

Oracle: the argv / deck text is the fork's documented command grammar
(``LadrunoRecorder.cpp``, ``-precision f32|f64``); the H5 round-trip is read
back through ``h5_reader``. The fork run itself is
``tests/opensees/integration_ladruno/test_ladruno_recorder_precision_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.recorder import LADRUNO_PRECISIONS, Ladruno

from tests.opensees.unit.test_unconsumed_model_definition import _model, _ops


def _argv(rec: Any) -> tuple[Any, ...]:
    em = RecordingEmitter()
    rec._emit(em, 1)
    ((method, args, _kw),) = em.calls
    assert method == "recorder"
    return tuple(args)


# ---------------------------------------------------------------------------
# Construction + argv
# ---------------------------------------------------------------------------


def test_the_vocabulary_is_exactly_the_two_fork_tokens() -> None:
    assert LADRUNO_PRECISIONS == {"f64", "f32"}


def test_default_is_f64_and_emits_the_pre_field_argv() -> None:
    rec = Ladruno(file="r.ladruno", nodal_responses=("displacement",))
    assert rec.precision == "f64"
    assert _argv(rec) == ("ladruno", "r.ladruno", "-N", "displacement")
    # Spelled out, f64 is the same recorder and the same line.
    explicit = Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                       precision="f64")
    assert explicit == rec
    assert _argv(explicit) == _argv(rec)


def test_f32_emits_the_flag_right_after_the_file() -> None:
    rec = Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                  elem_responses=("stresses",), nsteps=2, energy=True,
                  precision="f32")
    assert _argv(rec) == (
        "ladruno", "r.ladruno", "-precision", "f32",
        "-N", "displacement", "-E", "stresses", "-T", "nsteps", 2,
        "-G", "energy",
    )


def test_f32_with_a_materialised_filter_keeps_the_flag_first() -> None:
    # The build pipeline replays a filtered spec with ``_region_tag`` set;
    # the flag stays after the file, ``-R`` stays before ``-G``.
    rec = Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                  nodes=(1, 2), precision="f32", _region_tag=7, energy=True)
    assert _argv(rec) == (
        "ladruno", "r.ladruno", "-precision", "f32", "-N", "displacement",
        "-R", 7, "-G", "energy", 7,
    )


@pytest.mark.parametrize(
    "bad", ["float32", "single", "float64", "double", "F32", "f16", "",
            32, 64, None, True, ("f32",)],
)
def test_precision_refuses_anything_but_the_two_tokens(bad: Any) -> None:
    with pytest.raises(ValueError, match="precision must be 'f64' or 'f32'"):
        Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                precision=bad)


# ---------------------------------------------------------------------------
# Through the bridge: the ns wrapper, the two decks, ops.h5
# ---------------------------------------------------------------------------


def _bridge(precision: "str | None") -> "tuple[apeSees, Ladruno]":
    ops = _ops(_model(sp=[], masses=False))
    kw = {} if precision is None else {"precision": precision}
    rec = ops.recorder.Ladruno(file="r.ladruno",
                               nodal_responses=("displacement",),
                               name="rl", **kw)
    return ops, rec


def test_the_ns_wrapper_defaults_to_f64_and_passes_f32_through() -> None:
    assert _bridge(None)[1].precision == "f64"
    assert _bridge("f32")[1].precision == "f32"
    with pytest.raises(ValueError, match="precision must be 'f64' or 'f32'"):
        _bridge("float32")


def _recorder_lines(ops: apeSees, tmp_path: Path) -> tuple[str, str]:
    ops.tcl(str(tmp_path / "d.tcl"))
    ops.py(str(tmp_path / "d.py"))
    tcl = (tmp_path / "d.tcl").read_text(encoding="utf-8").splitlines()
    py = (tmp_path / "d.py").read_text(encoding="utf-8").splitlines()
    (rec_tcl,) = [ln for ln in tcl if ln.startswith("recorder ")]
    (rec_py,) = [ln for ln in py if "ops.recorder(" in ln]
    return rec_tcl, rec_py


def test_f32_reaches_the_tcl_and_the_py_deck(tmp_path: Path) -> None:
    rec_tcl, rec_py = _recorder_lines(_bridge("f32")[0], tmp_path)
    assert rec_tcl == "recorder ladruno r.ladruno -precision f32 -N displacement"
    assert rec_py == ("ops.recorder('ladruno', 'r.ladruno', '-precision', "
                      "'f32', '-N', 'displacement')")


def test_default_writes_the_pre_field_line_to_both_decks(
    tmp_path: Path,
) -> None:
    for ops, _rec in (_bridge(None), _bridge("f64")):
        rec_tcl, rec_py = _recorder_lines(ops, tmp_path)
        # The exact pre-field lines: no flag, nothing else moved.
        assert rec_tcl == "recorder ladruno r.ladruno -N displacement"
        assert rec_py == ("ops.recorder('ladruno', 'r.ladruno', '-N', "
                          "'displacement')")


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_ops_h5_round_trips_the_precision(
    tmp_path: Path, precision: str,
) -> None:
    path = tmp_path / "m.h5"
    _bridge(precision)[0].h5(str(path))
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
        assert table is not None
        (row,) = table.params_for("opensees/recorder/rl")
        assert row.type == "Ladruno"
        assert row.params["precision"] == precision
        (rec,) = m.recorders()
    assert rec.kind == "ladruno"
    args = tuple(rec.args)
    if precision == "f32":
        assert args[1:3] == ("-precision", "f32"), args
    else:
        assert "-precision" not in args, args
