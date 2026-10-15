"""#1974: spec-emitted ``.out`` recorders carry ``-precision``.

OpenSees writes 6 significant digits unless told otherwise, so every
``.out`` value oracle capped near rtol 1e-5 (measured in the #1962
review). :class:`LogicalRecorder` now carries ``precision`` (default 17,
the digit count that round-trips an IEEE double) and all three
formatters (``format_tcl``, ``format_python``, ``to_ops_args``) emit
``-precision <n>`` for it.

Oracles:

* emit shape: every recorder ``emit_logical`` yields, for every category
  that emits, carries ``-precision 17``, and the three formatters agree
  token for token;
* live round trip: a cantilever whose responses have no short decimal
  form is recorded through ``to_ops_args``; the file, read back with the
  transcoder's own parsers, equals ``ops.nodeDisp`` / ``ops.eleResponse``
  at rtol 1e-12. The same run at ``precision=6`` misses rtol 1e-10, so
  the round trip is not vacuous.
"""
from __future__ import annotations

import ast
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.results.spec._emit import (
    LogicalRecorder,
    emit_logical,
    format_python,
    format_tcl,
    to_ops_args,
)
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord
from apeGmsh.results.transcoders._txt import parse_element_file, parse_node_file

NODAL_FORCE = ("nodal_resisting_force_x", "nodal_resisting_force_y",
               "nodal_resisting_moment_z")


def _records() -> list[ResolvedRecorderRecord]:
    """One record per emitting category (nodes, elements, gauss, line_stations)."""
    return [
        ResolvedRecorderRecord(
            category="nodes", name="n",
            components=("displacement_x", "displacement_y", "velocity_x"),
            dt=None, n_steps=None, node_ids=np.array([1, 2]),
        ),
        ResolvedRecorderRecord(
            category="elements", name="e", components=NODAL_FORCE,
            dt=0.5, n_steps=None, element_ids=np.array([3, 4]),
        ),
        ResolvedRecorderRecord(
            category="gauss", name="g", components=("stress_xx",),
            dt=None, n_steps=None, element_ids=np.array([5]),
        ),
        ResolvedRecorderRecord(
            category="line_stations", name="b", components=("axial_force",),
            dt=None, n_steps=None, element_ids=np.array([6, 7]),
        ),
    ]


def _all_logicals(file_format: str = "out") -> list[LogicalRecorder]:
    out: list[LogicalRecorder] = []
    for rec in _records():
        out.extend(emit_logical(rec, file_format=file_format))
    return out


def _python_args(line: str) -> tuple:
    """Evaluate the argument list of one ``ops.recorder(...)`` source line."""
    call = line.split("  #", 1)[0]
    node = ast.parse(call, mode="eval").body
    assert isinstance(node, ast.Call)
    return tuple(ast.literal_eval(a) for a in node.args)


def _tcl_args(line: str) -> list[str]:
    return shlex.split(line.split("  ;#", 1)[0])[1:]   # drop "recorder"


# =====================================================================
# Emit shape
# =====================================================================

@pytest.mark.parametrize("file_format", ["out", "xml"])
def test_every_emitted_recorder_carries_full_precision(file_format: str) -> None:
    logicals = _all_logicals(file_format)
    # nodes: disp + vel; elements: 1; gauss: 1; line_stations: force + gpx
    assert len(logicals) == 6
    for rec in logicals:
        assert rec.precision == 17
        args = to_ops_args(rec)
        i = args.index("-precision")
        assert args[i + 1] == 17 and isinstance(args[i + 1], int)
        assert "-precision 17" in format_tcl(rec)
        assert "'-precision', 17" in format_python(rec)


def test_precision_sits_before_the_response_tokens() -> None:
    """OpenSees takes the first unrecognised token as the response, so
    ``-precision`` must precede it (it is read as an option there)."""
    for rec in _all_logicals():
        args = to_ops_args(rec)
        n = len(rec.response_tokens)
        assert args[-n:] == rec.response_tokens
        assert args[-n - 2:-n] == ("-precision", 17)


@pytest.mark.parametrize("file_format", ["out", "xml"])
def test_three_formatters_agree_token_for_token(file_format: str) -> None:
    for rec in _all_logicals(file_format):
        live = to_ops_args(rec)
        assert _python_args(format_python(rec)) == live
        assert _tcl_args(format_tcl(rec)) == [str(a) for a in live]


def test_explicit_precision_flows_to_all_formatters() -> None:
    rec = replace(_all_logicals()[0], precision=9)
    assert to_ops_args(rec)[-3:-1] == ("-precision", 9)
    assert "-precision 9 " in format_tcl(rec)
    assert "'-precision', 9," in format_python(rec)


@pytest.mark.parametrize("bad", [0, -3, True, 1.5, "17", None])
def test_non_positive_int_precision_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="precision must be a positive int"):
        LogicalRecorder(
            kind="Node", file_path="x.out", file_format="out",
            target_kind="node", target_ids=(1,), response_tokens=("disp",),
            dofs=(1,), record_name="r", precision=bad,
        )


# =====================================================================
# Live round trip
# =====================================================================

def _run_cantilever(tmp_path: Path, precision: int) -> dict[str, Any]:
    """Record a 2-D elastic cantilever through ``to_ops_args``; return the
    files' parsed rows and the domain's own values per step."""
    import openseespy.opensees as ops

    node_rec = ResolvedRecorderRecord(
        category="nodes", name="n",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None, node_ids=np.array([2, 3]),
    )
    ele_rec = ResolvedRecorderRecord(
        category="elements", name="e", components=NODAL_FORCE,
        dt=None, n_steps=None, element_ids=np.array([1, 2]),
    )
    out = str(tmp_path) + "/"
    [node_lr] = emit_logical(node_rec, output_dir=out)
    [ele_lr] = emit_logical(ele_rec, output_dir=out)
    node_lr = replace(node_lr, precision=precision)
    ele_lr = replace(ele_lr, precision=precision)

    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        # Coordinates, section and loads with no short decimal form, so
        # the 7th+ significant digits of every response are populated.
        ops.node(1, 0.0, 0.0)
        ops.node(2, 1.0 / 3.0, 0.1 * np.sqrt(2.0))
        ops.node(3, 2.0 / 3.0 + 0.01 * np.pi, 0.2 * np.sqrt(2.0))
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        A, E, Iz = 0.0123456789, 2.0e11 / 3.0, 1.23456789e-5
        ops.element("elasticBeamColumn", 1, 1, 2, A, E, Iz, 1)
        ops.element("elasticBeamColumn", 2, 2, 3, A, E, Iz, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(3, 1.0e4 * np.e, -2.0e4 / 7.0, 1.0e3 * np.sqrt(3.0))
        ops.load(2, 0.0, -1.0e4 / 3.0, 0.0)

        ops.recorder(*to_ops_args(node_lr))
        ops.recorder(*to_ops_args(ele_lr))

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-12, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0 / 3.0)
        ops.analysis("Static")

        disp, force = [], []
        for _ in range(3):
            assert ops.analyze(1) == 0
            disp.append([[ops.nodeDisp(n, d) for d in (1, 2)] for n in (2, 3)])
            force.append([ops.eleResponse(e, "globalForce") for e in (1, 2)])
    finally:
        ops.wipe()   # closes and flushes the recorder files

    _, per_dof = parse_node_file(node_lr.file_path, node_lr)
    _, ele_rows = parse_element_file(ele_lr.file_path, ele_lr, 6)
    return {
        "file_disp": np.stack([per_dof[1], per_dof[2]], axis=-1),  # (T, N, 2)
        "domain_disp": np.asarray(disp),
        "file_force": ele_rows,                                    # (T, E, 6)
        "domain_force": np.asarray(force),
    }


@pytest.mark.live
def test_out_round_trip_matches_domain_to_full_precision(tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    r = _run_cantilever(tmp_path, precision=17)
    assert r["file_disp"].shape == r["domain_disp"].shape == (3, 2, 2)
    assert r["file_force"].shape == r["domain_force"].shape == (3, 2, 6)
    np.testing.assert_allclose(r["file_disp"], r["domain_disp"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        r["file_force"], r["domain_force"], rtol=1e-12,
        atol=1e-12 * np.abs(r["domain_force"]).max(),
    )


@pytest.mark.live
def test_six_digit_round_trip_misses_the_oracle(tmp_path: Path) -> None:
    """Negative control: OpenSees's default 6 digits cannot meet rtol 1e-10,
    so the full-precision test above measures the flag, not luck."""
    pytest.importorskip("openseespy.opensees")
    r = _run_cantilever(tmp_path, precision=6)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            r["file_disp"], r["domain_disp"], rtol=1e-10, atol=0,
        )
    np.testing.assert_allclose(r["file_disp"], r["domain_disp"], rtol=1e-5, atol=0)
