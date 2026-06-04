"""Reader tests for the Ladruno Monitor sink (``read_monitor`` / ``tail_monitor``).

Fork-free against the committed ``monitor.h5`` fixture (a real fork-built
sink: 2 nodes x 2 dofs disp, 12 frames). A ``@pytest.mark.live`` parity test
runs a monitored analysis on the fork build and checks the sink against
``ops.nodeDisp``.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import MonitorData, read_monitor, tail_monitor

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
MONITOR = FIXTURES / "monitor.h5"
LADRUNO = FIXTURES / "truss2d.ladruno"  # a .ladruno, NOT a monitor sink

_COLS = (
    "node2.disp.dof1", "node2.disp.dof2",
    "node3.disp.dof1", "node3.disp.dof2",
)


def test_read_monitor_basic() -> None:
    m = read_monitor(MONITOR)
    assert isinstance(m, MonitorData)
    assert m.columns == _COLS
    assert m.n_frames == 12
    assert m.step.tolist() == list(range(12))
    assert m.frames.shape == (12, 4)
    # node 2/3 dof2 (transverse) stay zero; node 3 dof1 grows.
    np.testing.assert_allclose(m.channel("node2.disp.dof2"), 0.0)
    assert m.channel("node3.disp.dof1")[-1] > m.channel("node3.disp.dof1")[0]


def test_channel_unknown_raises() -> None:
    m = read_monitor(MONITOR)
    with pytest.raises(KeyError, match="not in monitor sink"):
        m.channel("node9.disp.dof1")


def test_to_dataframe_time_and_step_index() -> None:
    m = read_monitor(MONITOR)
    df_t = m.to_dataframe()  # default index="time"
    assert list(df_t.columns) == list(_COLS)
    assert df_t.index.name == "time"
    assert len(df_t) == 12
    df_s = m.to_dataframe(index="step")
    assert df_s.index.name == "step"
    assert df_s.index.tolist() == list(range(12))


def test_to_dataframe_bad_index_raises() -> None:
    with pytest.raises(ValueError, match="index must be 'time' or 'step'"):
        read_monitor(MONITOR).to_dataframe(index="frame")


def test_read_monitor_rejects_non_monitor() -> None:
    with pytest.raises(ValueError, match="not a Ladruno monitor sink"):
        read_monitor(LADRUNO)


def test_tail_drains_completed_file() -> None:
    # On a finished sink, tail yields every frame then stops at the timeout.
    got = list(tail_monitor(MONITOR, timeout=0.2, poll=0.02))
    assert len(got) == 12
    steps = [g[0] for g in got]
    assert steps == list(range(12))
    # frame rows match the at-rest read.
    m = read_monitor(MONITOR)
    np.testing.assert_allclose(got[-1][2], m.frames[-1])


def test_tail_start_offset() -> None:
    got = list(tail_monitor(MONITOR, start=10, timeout=0.2, poll=0.02))
    assert [g[0] for g in got] == [10, 11]


def _write_swmr_monitor(path: Path, ready: threading.Event,
                        n: int = 6) -> None:
    """Write a valid SWMR monitor sink incrementally (test writer)."""
    import h5py

    with h5py.File(path, "w", libver="latest") as f:
        f.attrs["FORMAT"] = "ladruno-monitor"
        f.attrs["FORMAT_VERSION"] = 1
        f.attrs["GENERATOR"] = "Ladruno"
        f.create_dataset(
            "COLUMNS", data=np.array(["node1.disp.dof1"], dtype="S32"),
        )
        step = f.create_dataset("STEP", shape=(0,), maxshape=(None,),
                                dtype="i4", chunks=(8,))
        tvec = f.create_dataset("TIME", shape=(0,), maxshape=(None,),
                                dtype="f8", chunks=(8,))
        frames = f.create_dataset("FRAMES", shape=(0, 1), maxshape=(None, 1),
                                  dtype="f8", chunks=(8, 1))
        f.swmr_mode = True
        ready.set()
        import time as _t
        for k in range(n):
            step.resize((k + 1,))
            step[k] = k
            tvec.resize((k + 1,))
            tvec[k] = 0.01 * (k + 1)
            frames.resize((k + 1, 1))
            frames[k, 0] = float(k) * 2.0
            for ds in (step, tvec, frames):
                ds.flush()
            _t.sleep(0.03)


def test_tail_follows_growing_file(tmp_path: Path) -> None:
    # A writer thread streams frames into a SWMR sink; tail collects them
    # live (proves the refresh/wait-for-new-frame path, not just drain).
    path = tmp_path / "live.h5"
    ready = threading.Event()
    writer = threading.Thread(
        target=_write_swmr_monitor, args=(path, ready, 6),
    )
    writer.start()
    try:
        ready.wait(timeout=5.0)
        got = list(tail_monitor(path, timeout=1.0, poll=0.02))
    finally:
        writer.join(timeout=5.0)
    assert [g[0] for g in got] == list(range(6))
    np.testing.assert_allclose([g[2][0] for g in got],
                               [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


# ---------------------------------------------------------------------------
# Live parity — run a monitored analysis and check the sink vs ops.nodeDisp
# ---------------------------------------------------------------------------

ops = pytest.importorskip("openseespy.opensees")


@pytest.mark.live
def test_monitor_live_parity(tmp_path: Path) -> None:
    sink = str(tmp_path / "mon.h5")
    try:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for i, x in enumerate([0.0, 1.0, 2.0], start=1):
            ops.node(i, x, 0.0)
        ops.fix(1, 1, 1)
        ops.fix(2, 0, 1)
        ops.fix(3, 0, 1)
        ops.uniaxialMaterial("Elastic", 1, 1000.0)
        ops.element("Truss", 1, 1, 2, 1.0, 1)
        ops.element("Truss", 2, 2, 3, 1.0, 1)
        ops.mass(2, 1.0, 1.0)
        ops.mass(3, 1.0, 1.0)
        ops.recorder("Monitor", "-node", 3, "-dof", 1, "-resp", "disp",
                     "-sink", sink, "-every", 1)
    except Exception:
        pytest.skip("running build has no 'Monitor' recorder (needs the fork)")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1e-10, 25)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    live = []
    for _ in range(10):
        ops.analyze(1, 0.02)
        live.append(ops.nodeDisp(3, 1))
    ops.wipe()

    m = read_monitor(sink)
    assert m.columns == ("node3.disp.dof1",)
    np.testing.assert_allclose(m.channel("node3.disp.dof1"), live, atol=1e-9)
