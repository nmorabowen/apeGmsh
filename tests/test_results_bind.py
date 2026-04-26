"""Phase 2 — bind validation.

The bind contract: candidate FEMData and embedded snapshot must
share ``snapshot_id``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import BindError, Results
from apeGmsh.results.writers import NativeWriter


@pytest.fixture
def grav_results_with_fem(g, tmp_path: Path) -> tuple[Path, "object"]:
    """Build a tiny mesh, write a results file with FEMData embedded."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "grav.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="grav", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0",
                      node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
                      components={"displacement_x":
                                   np.zeros((1, len(fem.nodes.ids)))})
        w.end_stage()
    return path, fem


def test_auto_bind_from_embedded(grav_results_with_fem) -> None:
    path, fem = grav_results_with_fem
    with Results.from_native(path) as r:
        assert r.fem is not None
        assert r.fem.snapshot_id == fem.snapshot_id


def test_explicit_bind_matching_hash(grav_results_with_fem) -> None:
    path, fem = grav_results_with_fem
    # Pass the same fem explicitly — should succeed and use it.
    with Results.from_native(path, fem=fem) as r:
        # The candidate fem (with full label/Part info) is preferred.
        assert r.fem is fem


def test_bind_after_construction(grav_results_with_fem) -> None:
    path, fem = grav_results_with_fem
    with Results.from_native(path) as r:
        rebound = r.bind(fem)
        assert rebound.fem is fem
        # The original is unchanged (we returned a new instance).
        assert r.fem is not None
        assert r.fem is not fem


def test_bind_mismatch_raises(grav_results_with_fem, g, tmp_path: Path) -> None:
    """A different mesh produces a different snapshot_id → BindError."""
    path, _orig_fem = grav_results_with_fem

    # Use a fresh session for the second model (the fixture's `g` is
    # already used; we'll add another body to the same session — the
    # fem we extract will have a different snapshot_id.)
    g.model.geometry.add_box(2, 0, 0, 1, 1, 1, label="box2")
    g.physical.add_volume("box2", name="Body2")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    other_fem = g.mesh.queries.get_fem_data(dim=3)

    with Results.from_native(path) as r:
        with pytest.raises(BindError, match="snapshot_id mismatch"):
            r.bind(other_fem)


def test_pg_query_works_after_bind(grav_results_with_fem, g) -> None:
    """PG selection resolves through the bound FEMData."""
    path, fem = grav_results_with_fem
    with Results.from_native(path, fem=fem) as r:
        slab = r.nodes.get(pg="Body", component="displacement_x")
        # All nodes are in 'Body' — should match the full set.
        assert slab.node_ids.size == len(fem.nodes.ids)
