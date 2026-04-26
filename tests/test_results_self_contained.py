"""Phase 2 — self-contained native HDF5: viewer can open without a session.

The architecture promise: a ``run.h5`` shipped to a colleague carries
everything needed (geometry + results + PGs) so they can plot it
without the apeGmsh session that produced it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


def test_native_file_self_contained_after_session_ends(g, tmp_path: Path) -> None:
    """Build a small model, write results, then end the session and read."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    n_nodes = len(fem.nodes.ids)
    snapshot_before = fem.snapshot_id

    path = tmp_path / "self_contained.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="grav", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0",
                      node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
                      components={"displacement_x": np.full((1, n_nodes), 0.5)})
        w.end_stage()

    # Drop our reference to fem — the file is the only truth now.
    del fem

    # Open as a fresh user with no apeGmsh session in scope here.
    with Results.from_native(path) as r:
        assert r.fem is not None
        assert r.fem.snapshot_id == snapshot_before
        # PG queries work — even though we never had to bind anything.
        slab = r.nodes.get(pg="Body", component="displacement_x")
        assert slab.node_ids.size == n_nodes
        np.testing.assert_allclose(slab.values, 0.5)


def test_native_file_without_embedded_fem_works_for_id_queries(
    tmp_path: Path,
) -> None:
    """No embedded FEMData → PG queries fail, ID queries succeed."""
    import pytest
    path = tmp_path / "no_fem.h5"
    with NativeWriter(path) as w:
        w.open()         # no fem= passed
        sid = w.begin_stage(name="g", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0",
                      node_ids=np.array([1, 2, 3]),
                      components={"displacement_x":
                                   np.array([[0.1, 0.2, 0.3]])})
        w.end_stage()

    with Results.from_native(path) as r:
        # No fem → r.fem is None.
        assert r.fem is None
        # ID-based query works.
        slab = r.nodes.get(ids=[1, 3], component="displacement_x")
        assert slab.node_ids.tolist() == [1, 3]
        # PG-based query raises with a helpful message.
        with pytest.raises(RuntimeError, match="bound FEMData"):
            r.nodes.get(pg="Top", component="displacement_x")


def test_inspect_summary_includes_fem_and_stages(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "summary.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="grav", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0",
                      node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
                      components={"displacement_x":
                                   np.zeros((1, len(fem.nodes.ids)))})
        w.end_stage()

    with Results.from_native(path) as r:
        s = r.inspect.summary()
        assert "FEM" in s
        assert "snapshot_id=" in s
        assert "grav" in s
