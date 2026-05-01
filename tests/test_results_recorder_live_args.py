"""Live-emit argument tuples — ``to_ops_args`` and ``mpco_ops_args``.

These tests cover the new helpers that produce ready-to-splat
argument tuples for ``ops.recorder(*args)`` calls. They are the
foundation for the live recorder strategy
(``ResolvedRecorderSpec.emit_recorders`` / ``emit_mpco``).

The tests do not import openseespy — they verify the tuples have
the right shape, types, and ordering, and that they agree with the
existing source-code formatters where they overlap.
"""
from __future__ import annotations

import numpy as np

from apeGmsh.solvers._recorder_emit import (
    emit_logical,
    emit_mpco_python,
    format_python,
    mpco_ops_args,
    to_ops_args,
)
from apeGmsh.solvers._recorder_specs import ResolvedRecorderRecord


# =====================================================================
# to_ops_args — Node recorder
# =====================================================================

def test_to_ops_args_node_displacement_basic() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    )
    [logical] = emit_logical(rec)
    args = to_ops_args(logical)

    # Shape: ('Node', '-file', '<path>', '-time', '-node', 1, 2,
    #         '-dof', 1, 2, 'disp')
    assert args[0] == "Node"
    assert args[1] == "-file"
    assert args[2] == "r_disp.out"
    assert args[3] == "-time"
    assert args[4] == "-node"
    # Node IDs are int, not str
    assert args[5] == 1 and isinstance(args[5], int)
    assert args[6] == 2 and isinstance(args[6], int)
    assert args[7] == "-dof"
    assert args[8] == 1 and isinstance(args[8], int)
    assert args[9] == 2 and isinstance(args[9], int)
    assert args[-1] == "disp"


def test_to_ops_args_node_with_dt() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([5]),
    )
    [logical] = emit_logical(rec)
    args = to_ops_args(logical)

    # -dT must be present, value must be float
    assert "-dT" in args
    dt_idx = args.index("-dT")
    assert args[dt_idx + 1] == 0.01
    assert isinstance(args[dt_idx + 1], float)


def test_to_ops_args_node_xml_format() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    [logical] = emit_logical(rec, file_format="xml")
    args = to_ops_args(logical)

    assert "-xml" in args
    assert "-file" not in args
    xml_idx = args.index("-xml")
    assert args[xml_idx + 1] == "r_disp.xml"


def test_to_ops_args_node_with_output_dir() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    [logical] = emit_logical(rec, output_dir="out/")
    args = to_ops_args(logical)

    file_idx = args.index("-file")
    assert args[file_idx + 1] == "out/r_disp.out"


# =====================================================================
# to_ops_args — Element recorder
# =====================================================================

def test_to_ops_args_element_gauss_stress() -> None:
    rec = ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([100, 101]),
    )
    [logical] = emit_logical(rec)
    args = to_ops_args(logical)

    assert args[0] == "Element"
    assert "-ele" in args
    ele_idx = args.index("-ele")
    assert args[ele_idx + 1] == 100 and isinstance(args[ele_idx + 1], int)
    assert args[ele_idx + 2] == 101
    # No -dof for element recorders
    assert "-dof" not in args
    # Last token is the gauss keyword
    assert args[-1] == "stresses"


def test_to_ops_args_element_line_stations_pair() -> None:
    """line_stations records produce two logical recorders — section
    force + integrationPoints. Both should round-trip through
    to_ops_args cleanly."""
    rec = ResolvedRecorderRecord(
        category="line_stations", name="beam",
        components=("axial_force",),
        dt=None, n_steps=None,
        element_ids=np.array([1, 2, 3]),
    )
    logicals = list(emit_logical(rec))
    assert len(logicals) == 2

    args_force = to_ops_args(logicals[0])
    args_gpx = to_ops_args(logicals[1])

    # First recorder: section force tokens
    assert args_force[0] == "Element"
    assert "section" in args_force and "force" in args_force

    # Second recorder: integrationPoints
    assert args_gpx[0] == "Element"
    assert "integrationPoints" in args_gpx


# =====================================================================
# Cross-check: to_ops_args agrees with format_python
# =====================================================================

def test_to_ops_args_round_trip_with_format_python() -> None:
    """The arg tuple, when passed through repr-equivalent, should
    reproduce the format_python source exactly. Catches drift between
    the two."""
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "velocity_y"),
        dt=0.005, n_steps=None,
        node_ids=np.array([10, 20, 30]),
    )
    for logical in emit_logical(rec):
        py_source = format_python(logical)
        args = to_ops_args(logical)

        # Every element of the tuple should appear in the source
        # (string args wrapped in quotes via repr; numbers as-is).
        for a in args:
            if isinstance(a, str):
                assert repr(a) in py_source, (
                    f"{a!r} not in {py_source!r}"
                )
            else:
                assert str(a) in py_source, (
                    f"{a} not in {py_source!r}"
                )


# =====================================================================
# mpco_ops_args
# =====================================================================

def test_mpco_ops_args_nodal_only() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    args = mpco_ops_args([rec])

    # ('mpco', '<path>', '-N', 'displacement')
    assert args[0] == "mpco"
    assert args[1] == "run.mpco"
    assert args[2] == "-N"
    assert "displacement" in args
    assert "-E" not in args


def test_mpco_ops_args_element_only() -> None:
    rec = ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    )
    args = mpco_ops_args([rec])

    assert args[0] == "mpco"
    assert "-N" not in args
    assert "-E" in args
    assert "stresses" in args


def test_mpco_ops_args_mixed_with_dt() -> None:
    n_rec = ResolvedRecorderRecord(
        category="nodes", name="rn",
        components=("displacement_x",),
        dt=0.02, n_steps=None,
        node_ids=np.array([1]),
    )
    e_rec = ResolvedRecorderRecord(
        category="gauss", name="re",
        components=("strain_xx",),
        dt=0.01, n_steps=None,
        element_ids=np.array([1]),
    )
    args = mpco_ops_args([n_rec, e_rec])

    # Both -N and -E
    assert "-N" in args
    assert "-E" in args
    # Cadence: smallest dt wins → 0.01
    assert "-T" in args
    t_idx = args.index("-T")
    assert args[t_idx + 1] == "dt"
    assert args[t_idx + 2] == 0.01
    assert isinstance(args[t_idx + 2], float)


def test_mpco_ops_args_output_dir_and_filename() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    args = mpco_ops_args([rec], output_dir="results/", filename="my_run.mpco")
    assert args[1] == "results/my_run.mpco"


def test_mpco_ops_args_round_trip_with_emit_mpco_python() -> None:
    """Tuple agrees with the source-code form."""
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    )
    py_source = emit_mpco_python([rec])
    args = mpco_ops_args([rec])

    for a in args:
        if isinstance(a, str):
            assert repr(a) in py_source
        else:
            assert str(a) in py_source


def test_mpco_ops_args_modal_record_emits_modes_token() -> None:
    rec = ResolvedRecorderRecord(
        category="modal", name="modes",
        components=(),
        dt=None, n_steps=None,
        n_modes=10,
    )
    args = mpco_ops_args([rec])
    # Modal records contribute the modes-of-vibration tokens on -N.
    assert "modesOfVibration" in args
