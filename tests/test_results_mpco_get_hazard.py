"""F6/F7a — MPCO reader optional-child hazard sweep + step-cadence guard.

F6: the MPCO readers probed optional HDF5 children with ``Group.get()``,
which returns ``None`` ambiguously whether an intermediate path segment
is missing or a link is broken (project h5py optional-child hazard rule).
They now use membership (``name in group``) via the ``_child`` helper.
This module unit-tests ``_child`` and adds an AST guard that fails if any
string-literal ``Group.get(...)`` probe creeps back into ``_mpco*.py``.

F7a: ``_build_time_vector_for_mpco_stage`` picked the first node-result
group's step cadence with no validation. It now cross-checks every
node-result group and raises on a step-count mismatch (STKO records all
results on one cadence; a mismatch means step-indexed reads would
misalign).
"""
from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._mpco import MPCOReader, _child


# =====================================================================
# F6 — _child membership semantics
# =====================================================================

def test_child_returns_none_for_missing_intermediate(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "m.h5", "w") as f:
        g = f.create_group("g")
        assert _child(g, "RESULTS/ON_NODES") is None        # no RESULTS
        g.create_group("RESULTS")
        assert _child(g, "RESULTS/ON_NODES") is None        # RESULTS, no ON_NODES
        g["RESULTS"].create_group("ON_NODES")
        assert _child(g, "RESULTS/ON_NODES") is not None     # full path resolves
        assert _child(g, "DATA") is None                     # single missing seg


# =====================================================================
# F6/F7a — time-vector builder
# =====================================================================

def _stage_with_node_results(
    f: "h5py.File", result_steps: dict[str, int],
) -> "h5py.Group":
    stage = f.create_group("MODEL_STAGE[1]")
    on_nodes = stage.create_group("RESULTS").create_group("ON_NODES")
    for name, n in result_steps.items():
        data = on_nodes.create_group(name).create_group("DATA")
        for k in range(n):
            ds = data.create_dataset(f"STEP_{k}", data=np.zeros((2, 3)))
            ds.attrs["STEP"] = k
            ds.attrs["TIME"] = float(k + 1)
    return stage


def test_time_vector_empty_when_on_nodes_absent(tmp_path: Path) -> None:
    # Missing optional child must degrade to an empty time vector, not
    # crash and not silently mask a malformed file as something else.
    with h5py.File(tmp_path / "m.h5", "w") as f:
        stage = f.create_group("MODEL_STAGE[1]")   # no RESULTS/ON_NODES
        reader = MPCOReader.__new__(MPCOReader)
        out = reader._build_time_vector_for_mpco_stage(stage)
    assert out.size == 0


def test_time_vector_builds_from_steps(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "m.h5", "w") as f:
        stage = _stage_with_node_results(f, {"DISPLACEMENT": 5})
        reader = MPCOReader.__new__(MPCOReader)
        out = reader._build_time_vector_for_mpco_stage(stage)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_zero_step_group_is_skipped(tmp_path: Path) -> None:
    # A result group with no STEP_<k> datasets must be skipped (not
    # treated as a 0-step cadence that mismatches the real one).
    with h5py.File(tmp_path / "m.h5", "w") as f:
        stage = _stage_with_node_results(f, {"EMPTY": 0, "DISPLACEMENT": 5})
        reader = MPCOReader.__new__(MPCOReader)
        out = reader._build_time_vector_for_mpco_stage(stage)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_step_count_mismatch_raises(tmp_path: Path) -> None:
    # Two node results disagree on step count → fail loud (F7a).
    with h5py.File(tmp_path / "m.h5", "w") as f:
        stage = _stage_with_node_results(
            f, {"DISPLACEMENT": 5, "ACCELERATION": 4},
        )
        reader = MPCOReader.__new__(MPCOReader)
        with pytest.raises(ValueError, match="step-count mismatch"):
            reader._build_time_vector_for_mpco_stage(stage)


# =====================================================================
# F6 — AST guard: no string-literal Group.get() probe in _mpco*.py
# =====================================================================

_READERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "apeGmsh" / "results" / "readers"
)


def _is_attrs_receiver(value: ast.expr) -> bool:
    """True for ``<x>.attrs`` — ``.attrs.get(...)`` is a Mapping, allowed."""
    return isinstance(value, ast.Attribute) and value.attr == "attrs"


def _literal_group_get_offences(source: str, filename: str) -> list[tuple[int, str]]:
    """Find ``<group>.get("<literal>")`` calls (the forbidden probe).

    A string-literal ``.get`` argument is the on-disk path of an h5py
    child probe; dict ``.get`` sites in these modules all key off a
    variable, and ``.attrs.get`` is a Mapping read — both allowed.
    """
    tree = ast.parse(source, filename=filename)
    offences: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or not node.args:
            continue
        arg0 = node.args[0]
        if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
            continue
        if _is_attrs_receiver(node.func.value):
            continue
        offences.append((node.lineno, f'.get({arg0.value!r})'))
    return offences


def test_no_literal_group_get_in_mpco_readers() -> None:
    files = sorted(_READERS_DIR.glob("_mpco*.py"))
    assert files, f"no _mpco*.py readers found under {_READERS_DIR}"
    all_offences: list[str] = []
    for path in files:
        for ln, why in _literal_group_get_offences(
            path.read_text(encoding="utf-8"), str(path),
        ):
            all_offences.append(f"  {path.name}:{ln}  {why}")
    assert not all_offences, (
        "MPCO readers must probe optional HDF5 children with "
        "`name in group` (the `_child` helper), never `Group.get(\"...\")` "
        "— a multi-segment get masks a missing intermediate / broken link "
        "as 'absent'. Offences:\n" + "\n".join(all_offences)
    )


def test_positive_control_catches_literal_group_get() -> None:
    src = 'x = grp.get("RESULTS/ON_NODES")\n'
    assert _literal_group_get_offences(src, "<test>")


def test_positive_control_allows_attrs_get() -> None:
    src = 'x = grp.attrs.get("TIME", 0.0)\n'
    assert not _literal_group_get_offences(src, "<test>")


def test_positive_control_allows_dict_get_with_variable() -> None:
    src = "x = some_dict.get(key)\n"
    assert not _literal_group_get_offences(src, "<test>")
