"""Phase 1 — modes-as-stages: kind='mode', T=1, eigenvalue/freq/period attrs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.writers import NativeWriter


def test_single_mode_stage_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    # Mode shape: 1 step, 3 nodes
    shape_x = np.array([[0.10, 0.50, 0.30]])
    shape_y = np.array([[0.05, -0.20, 0.10]])

    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(
            name="mode_1",
            kind="mode",
            time=np.array([0.0]),
            eigenvalue=158.7,
            frequency_hz=2.005,
            period_s=0.499,
            mode_index=1,
        )
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": shape_x,
                                   "displacement_y": shape_y})
        w.end_stage()

    with NativeReader(path) as r:
        stages = r.stages()
        assert len(stages) == 1
        s = stages[0]
        assert s.kind == "mode"
        assert s.n_steps == 1
        assert s.mode_index == 1
        assert s.eigenvalue == pytest.approx(158.7)
        assert s.frequency_hz == pytest.approx(2.005)
        assert s.period_s == pytest.approx(0.499)

        slab = r.read_nodes(s.id, "displacement_x")
        assert slab.values.shape == (1, 3)
        np.testing.assert_allclose(slab.values, shape_x)


def test_multiple_modes_in_one_file(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    node_ids = np.array([1, 2], dtype=np.int64)
    modes = [
        (1, 100.0, np.sqrt(100.0) / (2 * np.pi), 1.0 / (np.sqrt(100.0) / (2 * np.pi))),
        (2, 400.0, np.sqrt(400.0) / (2 * np.pi), 1.0 / (np.sqrt(400.0) / (2 * np.pi))),
        (3, 900.0, np.sqrt(900.0) / (2 * np.pi), 1.0 / (np.sqrt(900.0) / (2 * np.pi))),
    ]

    with NativeWriter(path) as w:
        w.open()
        for idx, eig, freq, per in modes:
            sid = w.begin_stage(
                name=f"mode_{idx}", kind="mode",
                time=np.array([0.0]),
                eigenvalue=eig, frequency_hz=freq, period_s=per,
                mode_index=idx,
            )
            shape = np.array([[float(idx) * 0.1, float(idx) * 0.2]])
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": shape})
            w.end_stage()

    with NativeReader(path) as r:
        stages = r.stages()
        assert len(stages) == 3
        # All are mode-kind
        assert all(s.kind == "mode" for s in stages)
        # Sort by mode_index for stable comparison
        by_idx = sorted(stages, key=lambda s: s.mode_index)
        assert [s.mode_index for s in by_idx] == [1, 2, 3]

        slab = r.read_nodes(by_idx[2].id, "displacement_x")
        np.testing.assert_allclose(slab.values, [[0.3, 0.6]])


def test_mode_requires_eigenvalue(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open()
        with pytest.raises(ValueError, match="kind='mode' requires"):
            w.begin_stage(
                name="bad_mode", kind="mode",
                time=np.array([0.0]),
                # missing eigenvalue
            )
