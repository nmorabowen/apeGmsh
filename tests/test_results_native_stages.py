"""Phase 1 — multiple stages in one file (transient + static + mode)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.writers import NativeWriter


def test_two_transient_stages_independent(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    node_ids = np.array([1, 2], dtype=np.int64)

    grav_time = np.array([0.0, 1.0])
    grav_ux = np.array([[0.0, 0.0], [0.01, 0.02]])

    dyn_time = np.linspace(0.0, 5.0, 11)
    dyn_ux = np.outer(np.sin(dyn_time), np.array([0.1, 0.2]))

    with NativeWriter(path) as w:
        w.open()
        sid_g = w.begin_stage(name="gravity", kind="static", time=grav_time)
        w.write_nodes(sid_g, "partition_0", node_ids=node_ids,
                      components={"displacement_x": grav_ux})
        w.end_stage()

        sid_d = w.begin_stage(name="dynamic", kind="transient", time=dyn_time)
        w.write_nodes(sid_d, "partition_0", node_ids=node_ids,
                      components={"displacement_x": dyn_ux})
        w.end_stage()

    with NativeReader(path) as r:
        stages = r.stages()
        names = {s.name: s for s in stages}
        assert set(names) == {"gravity", "dynamic"}
        assert names["gravity"].kind == "static"
        assert names["dynamic"].kind == "transient"
        assert names["gravity"].n_steps == 2
        assert names["dynamic"].n_steps == 11

        g_slab = r.read_nodes(names["gravity"].id, "displacement_x")
        np.testing.assert_allclose(g_slab.values, grav_ux)

        d_slab = r.read_nodes(names["dynamic"].id, "displacement_x")
        np.testing.assert_allclose(d_slab.values, dyn_ux)
        np.testing.assert_allclose(d_slab.time, dyn_time)


def test_cannot_begin_two_stages_at_once(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    import pytest
    with NativeWriter(path) as w:
        w.open()
        w.begin_stage(name="a", kind="static", time=np.array([0.0]))
        with pytest.raises(RuntimeError, match="still open"):
            w.begin_stage(name="b", kind="static", time=np.array([0.0]))


def test_stages_ordered_numerically_past_ten(tmp_path: Path) -> None:
    # HDF5 returns group names alphabetically; "stage_10" must NOT sort
    # before "stage_2". stages() orders by the integer suffix (F2).
    path = tmp_path / "run.h5"
    node_ids = np.array([1, 2], dtype=np.int64)
    time = np.array([0.0, 1.0])
    ux = np.array([[0.0, 0.0], [0.01, 0.02]])

    with NativeWriter(path) as w:
        w.open()
        for i in range(11):
            sid = w.begin_stage(name=f"s{i}", kind="transient", time=time)
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": ux})
            w.end_stage()

    with NativeReader(path) as r:
        assert [s.id for s in r.stages()] == [
            f"stage_{i}" for i in range(11)
        ]
        # And the names track the same order.
        assert [s.name for s in r.stages()] == [f"s{i}" for i in range(11)]


def test_stages_custom_id_does_not_break_ordering(tmp_path: Path) -> None:
    # begin_stage accepts an arbitrary stage_id; a non-numeric one must
    # not raise from the integer-suffix sort. Numeric "stage_<int>" ids
    # come first (by suffix), custom ids fall back to lexical order.
    path = tmp_path / "run.h5"
    node_ids = np.array([1], dtype=np.int64)
    time = np.array([0.0])
    ux = np.array([[0.0]])

    with NativeWriter(path) as w:
        w.open()
        for sid_arg in ("custom_gravity", None, None):
            sid = w.begin_stage(
                name="s", kind="static", time=time, stage_id=sid_arg,
            )
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": ux})
            w.end_stage()

    with NativeReader(path) as r:
        ids = [s.id for s in r.stages()]
    # The stage counter advances on every begin_stage (even the custom
    # one), so the two auto ids are stage_1 / stage_2; they sort first by
    # suffix, and the non-numeric custom id falls back to lexical last.
    assert ids == ["stage_1", "stage_2", "custom_gravity"]
