"""Phase 6 — TXT parser for OpenSees ``.out`` recorder files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.transcoders._txt import parse_node_file
from apeGmsh.solvers._recorder_emit import LogicalRecorder


def _write_out(path: Path, time: list[float], rows: list[list[float]]) -> None:
    """Write a fake OpenSees ``-time`` recorder file."""
    lines = []
    for t, vals in zip(time, rows):
        all_cols = [t] + list(vals)
        lines.append(" ".join(f"{v:.10g}" for v in all_cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================
# Single DOF, single node
# =====================================================================

def test_single_node_single_dof(tmp_path: Path) -> None:
    path = tmp_path / "r_disp.out"
    _write_out(path, [0.0, 0.1, 0.2], [[1.5], [1.6], [1.7]])

    lr = LogicalRecorder(
        kind="Node", file_path=str(path), file_format="out",
        target_kind="node", target_ids=(7,),
        response_tokens=("disp",), dofs=(1,),
        record_name="r",
    )
    time, per_dof = parse_node_file(path, lr)
    np.testing.assert_allclose(time, [0.0, 0.1, 0.2])
    assert set(per_dof) == {1}
    np.testing.assert_allclose(per_dof[1], [[1.5], [1.6], [1.7]])


# =====================================================================
# Multiple nodes × multiple DOFs (node-major layout)
# =====================================================================

def test_multi_node_multi_dof(tmp_path: Path) -> None:
    path = tmp_path / "r_disp.out"
    # 3 nodes × 3 dofs (1, 2, 3) per row, plus time:
    # t  ux1 uy1 uz1   ux2 uy2 uz2   ux3 uy3 uz3
    rows = [
        [0.10, 0.20, 0.30, 1.10, 1.20, 1.30, 2.10, 2.20, 2.30],
        [0.11, 0.22, 0.33, 1.11, 1.22, 1.33, 2.11, 2.22, 2.33],
    ]
    _write_out(path, [0.0, 1.0], rows)

    lr = LogicalRecorder(
        kind="Node", file_path=str(path), file_format="out",
        target_kind="node", target_ids=(1, 2, 3),
        response_tokens=("disp",), dofs=(1, 2, 3),
        record_name="r",
    )
    time, per_dof = parse_node_file(path, lr)
    assert time.shape == (2,)
    assert set(per_dof) == {1, 2, 3}
    # DOF 1 (ux): columns 1, 4, 7 → [0.10, 1.10, 2.10] at step 0
    np.testing.assert_allclose(per_dof[1][0], [0.10, 1.10, 2.10])
    # DOF 2 (uy): columns 2, 5, 8
    np.testing.assert_allclose(per_dof[2][0], [0.20, 1.20, 2.20])
    # DOF 3 (uz): columns 3, 6, 9
    np.testing.assert_allclose(per_dof[3][0], [0.30, 1.30, 2.30])
    # Step 1
    np.testing.assert_allclose(per_dof[1][1], [0.11, 1.11, 2.11])


# =====================================================================
# Sparse DOFs (e.g. only z direction)
# =====================================================================

def test_sparse_dofs(tmp_path: Path) -> None:
    """Recorder with only DOFs 1, 3 (skipping 2)."""
    path = tmp_path / "r_disp.out"
    # 2 nodes × 2 dofs (1, 3): ux1 uz1 ux2 uz2
    rows = [
        [0.1, 0.3, 1.1, 1.3],
        [0.11, 0.33, 1.11, 1.33],
    ]
    _write_out(path, [0.0, 1.0], rows)

    lr = LogicalRecorder(
        kind="Node", file_path=str(path), file_format="out",
        target_kind="node", target_ids=(1, 2),
        response_tokens=("disp",), dofs=(1, 3),
        record_name="r",
    )
    time, per_dof = parse_node_file(path, lr)
    assert set(per_dof) == {1, 3}
    np.testing.assert_allclose(per_dof[1][0], [0.1, 1.1])
    np.testing.assert_allclose(per_dof[3][0], [0.3, 1.3])


# =====================================================================
# Single time step (1-D file)
# =====================================================================

def test_single_step(tmp_path: Path) -> None:
    path = tmp_path / "r_disp.out"
    path.write_text("0.0 1.5 2.5\n", encoding="utf-8")

    lr = LogicalRecorder(
        kind="Node", file_path=str(path), file_format="out",
        target_kind="node", target_ids=(1, 2),
        response_tokens=("disp",), dofs=(1,),
        record_name="r",
    )
    time, per_dof = parse_node_file(path, lr)
    assert time.shape == (1,)
    assert per_dof[1].shape == (1, 2)
    np.testing.assert_allclose(per_dof[1], [[1.5, 2.5]])


# =====================================================================
# Error paths
# =====================================================================

def test_missing_file_raises(tmp_path: Path) -> None:
    lr = LogicalRecorder(
        kind="Node", file_path=str(tmp_path / "nope.out"), file_format="out",
        target_kind="node", target_ids=(1,),
        response_tokens=("disp",), dofs=(1,),
        record_name="r",
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_node_file(tmp_path / "nope.out", lr)


def test_column_count_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "r_disp.out"
    # File has 3 cols (1 time + 2 vals), but spec says 2 nodes × 2 DOFs = 4 cols.
    path.write_text("0.0 1.0 2.0\n", encoding="utf-8")
    lr = LogicalRecorder(
        kind="Node", file_path=str(path), file_format="out",
        target_kind="node", target_ids=(1, 2),
        response_tokens=("disp",), dofs=(1, 2),
        record_name="r",
    )
    with pytest.raises(ValueError, match="expected"):
        parse_node_file(path, lr)


def test_non_node_kind_raises(tmp_path: Path) -> None:
    lr = LogicalRecorder(
        kind="Element", file_path="x.out", file_format="out",
        target_kind="ele", target_ids=(1,),
        response_tokens=("stress",), dofs=None,
        record_name="r",
    )
    with pytest.raises(ValueError, match="expects kind='Node'"):
        parse_node_file("x.out", lr)
