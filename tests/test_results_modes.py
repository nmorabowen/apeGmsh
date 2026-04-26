"""Phase 2 — modes accessor and mode-only properties on scoped Results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


def _write_modes_file(tmp_path: Path, *, n_modes: int = 3) -> Path:
    path = tmp_path / "modes.h5"
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    with NativeWriter(path) as w:
        w.open()
        for k in range(1, n_modes + 1):
            eig = float(k * 100.0)
            f = float(k * 2.0)
            T = 1.0 / f
            sid = w.begin_stage(
                name=f"mode_{k}", kind="mode",
                time=np.array([0.0]),
                eigenvalue=eig, frequency_hz=f, period_s=T, mode_index=k,
            )
            shape = np.array([[float(k) * 0.1,
                                float(k) * 0.2,
                                float(k) * 0.3]])
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": shape})
            w.end_stage()
    return path


def test_modes_accessor_returns_scoped_results(tmp_path: Path) -> None:
    path = _write_modes_file(tmp_path, n_modes=3)
    with Results.from_native(path) as r:
        modes = r.modes
        assert len(modes) == 3
        for m in modes:
            assert m.kind == "mode"


def test_mode_indexing_and_attrs(tmp_path: Path) -> None:
    path = _write_modes_file(tmp_path, n_modes=3)
    with Results.from_native(path) as r:
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert [m.mode_index for m in modes] == [1, 2, 3]
        m2 = modes[1]
        assert m2.eigenvalue == pytest.approx(200.0)
        assert m2.frequency_hz == pytest.approx(4.0)
        assert m2.period_s == pytest.approx(0.25)
        assert m2.name == "mode_2"


def test_mode_shape_is_single_step(tmp_path: Path) -> None:
    path = _write_modes_file(tmp_path, n_modes=2)
    with Results.from_native(path) as r:
        m1 = sorted(r.modes, key=lambda m: m.mode_index)[0]
        slab = m1.nodes.get(component="displacement_x")
        assert slab.values.shape == (1, 3)
        np.testing.assert_allclose(slab.values, [[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(slab.time, [0.0])


def test_mode_props_raise_on_non_mode_stage(tmp_path: Path) -> None:
    """A scoped non-mode stage doesn't expose eigenvalue / frequency_hz."""
    path = tmp_path / "mixed.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="static", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0", node_ids=np.array([1]),
                      components={"displacement_x": np.array([[0.0]])})
        w.end_stage()

    with Results.from_native(path) as r:
        s = r.stage("static")
        with pytest.raises(AttributeError, match="not 'mode'"):
            _ = s.eigenvalue
        with pytest.raises(AttributeError, match="not 'mode'"):
            _ = s.frequency_hz


def test_mode_props_raise_on_unscoped(tmp_path: Path) -> None:
    path = _write_modes_file(tmp_path, n_modes=2)
    with Results.from_native(path) as r:
        # Unscoped → stage-scoped check fires first (correct behavior).
        with pytest.raises(AttributeError, match="stage-scoped"):
            _ = r.eigenvalue


def test_modes_empty_when_no_mode_stages(tmp_path: Path) -> None:
    path = tmp_path / "no_modes.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="grav", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0", node_ids=np.array([1]),
                      components={"displacement_x": np.array([[0.0]])})
        w.end_stage()
    with Results.from_native(path) as r:
        assert r.modes == []


def test_mixed_stages_and_modes_in_one_file(tmp_path: Path) -> None:
    path = tmp_path / "mixed.h5"
    node_ids = np.array([1, 2], dtype=np.int64)
    with NativeWriter(path) as w:
        w.open()
        # Transient
        sid = w.begin_stage(name="dynamic", kind="transient",
                             time=np.array([0.0, 1.0]))
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": np.zeros((2, 2))})
        w.end_stage()
        # Two modes
        for k in (1, 2):
            sid = w.begin_stage(
                name=f"mode_{k}", kind="mode",
                time=np.array([0.0]),
                eigenvalue=float(k * 50.0),
                frequency_hz=float(k * 1.0),
                period_s=1.0 / float(k),
                mode_index=k,
            )
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x":
                                       np.array([[float(k), float(k)]])})
            w.end_stage()

    with Results.from_native(path) as r:
        assert len(r.stages) == 3
        assert len(r.modes) == 2
        # Non-mode stage stays accessible by name
        dyn = r.stage("dynamic")
        assert dyn.kind == "transient"
        assert dyn.n_steps == 2
