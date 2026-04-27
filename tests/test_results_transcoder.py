"""Phase 6 — RecorderTranscoder end-to-end (synthetic .out files)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.transcoders import RecorderTranscoder
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Mock FEMData (real hash via compute_snapshot_id)
# =====================================================================

class _MockFem:
    def __init__(self, node_ids, salt: int = 0) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        coords = np.zeros((ids.size, 3), dtype=np.float64) + float(salt) * 1e-9
        self.nodes = SimpleNamespace(ids=ids, coords=coords)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 3
        group.attrs["ndf"] = 6
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


# =====================================================================
# Helpers — emit synthetic .out files matching spec emission
# =====================================================================

def _write_synthetic_node_out(
    path: Path,
    *,
    time: list[float],
    node_ids: list[int],
    dofs: list[int],
    value_fn,
) -> None:
    """Write a node recorder file with a deterministic value function.

    ``value_fn(step_idx, node_id, dof) -> float``.
    """
    lines = []
    for k, t in enumerate(time):
        cols = [t]
        for nid in node_ids:
            for dof in dofs:
                cols.append(value_fn(k, nid, dof))
        lines.append(" ".join(f"{v:.10g}" for v in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================
# Single nodes record: displacement
# =====================================================================

def test_transcode_single_node_record(tmp_path: Path) -> None:
    fem = _MockFem([1, 2, 3])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Spec — record name "all_disp", three displacement components
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes",
            name="all_disp",
            components=("displacement_x", "displacement_y", "displacement_z"),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2, 3]),
        ),),
    )

    # Synthetic .out file matching what Phase 5 emit would produce.
    # Components → ops_type "disp" with dofs [1, 2, 3] (sorted).
    _write_synthetic_node_out(
        output_dir / "all_disp_disp.out",
        time=[0.0, 0.1, 0.2],
        node_ids=[1, 2, 3],
        dofs=[1, 2, 3],
        value_fn=lambda k, nid, dof: 0.01 * nid + 0.001 * dof + 0.0001 * k,
    )

    target = tmp_path / "out.h5"
    RecorderTranscoder(spec, output_dir, target, fem).run()

    with Results.from_native(target, fem=fem) as r:
        slab_x = r.nodes.get(component="displacement_x")
        assert slab_x.values.shape == (3, 3)        # 3 steps × 3 nodes
        np.testing.assert_allclose(slab_x.time, [0.0, 0.1, 0.2])
        # Step 0, node 1, dof 1: 0.01*1 + 0.001*1 = 0.011
        np.testing.assert_allclose(
            slab_x.values[0],
            [0.011, 0.021, 0.031],
        )


# =====================================================================
# Multi-record merge (different node sets per record)
# =====================================================================

def test_transcode_two_records_disjoint_nodes(tmp_path: Path) -> None:
    fem = _MockFem([1, 2, 3, 4])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="top",
                components=("displacement_z",),
                dt=None, n_steps=None,
                node_ids=np.array([3, 4]),
            ),
            ResolvedRecorderRecord(
                category="nodes", name="bot",
                components=("reaction_force_z",),
                dt=None, n_steps=None,
                node_ids=np.array([1, 2]),
            ),
        ),
    )

    _write_synthetic_node_out(
        output_dir / "top_disp.out",
        time=[0.0, 1.0],
        node_ids=[3, 4],
        dofs=[3],
        value_fn=lambda k, nid, dof: -0.001 * (nid + k),
    )
    _write_synthetic_node_out(
        output_dir / "bot_reaction.out",
        time=[0.0, 1.0],
        node_ids=[1, 2],
        dofs=[3],
        value_fn=lambda k, nid, dof: 100.0 * (nid + k),
    )

    target = tmp_path / "out.h5"
    RecorderTranscoder(spec, output_dir, target, fem).run()

    with Results.from_native(target, fem=fem) as r:
        # Master node IDs are the union of both records' node sets.
        slab_z = r.nodes.get(component="displacement_z")
        # 4 nodes total, 2 steps
        assert slab_z.values.shape == (2, 4)
        # Top record had nodes 3, 4 — those should be populated.
        # Bot's nodes 1, 2 should be NaN for displacement.
        sorted_ids = slab_z.node_ids
        assert list(sorted_ids) == [1, 2, 3, 4]
        # Step 0: nodes 1, 2 are NaN; nodes 3, 4 have values
        assert np.isnan(slab_z.values[0, 0])
        assert np.isnan(slab_z.values[0, 1])
        np.testing.assert_allclose(slab_z.values[0, 2], -0.003)
        np.testing.assert_allclose(slab_z.values[0, 3], -0.004)

        slab_rxn = r.nodes.get(component="reaction_force_z")
        # Bot's nodes have reactions; top's are NaN
        np.testing.assert_allclose(slab_rxn.values[0, 0], 100.0)
        np.testing.assert_allclose(slab_rxn.values[0, 1], 200.0)
        assert np.isnan(slab_rxn.values[0, 2])
        assert np.isnan(slab_rxn.values[0, 3])


# =====================================================================
# Snapshot mismatch
# =====================================================================

def test_transcode_snapshot_mismatch_raises(tmp_path: Path) -> None:
    fem_a = _MockFem([1, 2], salt=0)
    fem_b = _MockFem([1, 2], salt=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem_a.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),),
    )

    target = tmp_path / "out.h5"
    with pytest.raises(RuntimeError, match="snapshot_id mismatch"):
        RecorderTranscoder(spec, output_dir, target, fem_b).run()


# =====================================================================
# Element-level records are skipped (Phase 6 v1 limitation)
# =====================================================================

def test_unwired_element_records_skipped_silently(tmp_path: Path) -> None:
    """Element-level records the transcoder doesn't yet handle don't crash.

    Phase 11a wired ``gauss``. Phase 11b wired ``line_stations``. The
    remaining element-level categories (``elements`` / ``fibers`` /
    ``layers``) still skip silently — surface in
    ``transcoder.unsupported`` for inspection.
    """
    fem = _MockFem([1])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="r",
                components=("displacement_x",),
                dt=None, n_steps=None,
                node_ids=np.array([1]),
            ),
            ResolvedRecorderRecord(
                category="elements", name="b",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
        ),
    )

    _write_synthetic_node_out(
        output_dir / "r_disp.out",
        time=[0.0, 1.0],
        node_ids=[1],
        dofs=[1],
        value_fn=lambda k, nid, dof: 0.5,
    )
    # Note: no b_elements.out file; the elements record is skipped.

    target = tmp_path / "out.h5"
    transcoder = RecorderTranscoder(spec, output_dir, target, fem)
    transcoder.run()
    assert "elements:b" in transcoder.unsupported

    with Results.from_native(target, fem=fem) as r:
        # Node data parsed
        np.testing.assert_allclose(
            r.nodes.get(component="displacement_x").values, [[0.5], [0.5]],
        )
        # Line-stations level not present (no components available)
        assert r.elements.gauss.available_components() == []


# =====================================================================
# from_recorders + cache layer
# =====================================================================

def test_from_recorders_caches_result(tmp_path: Path) -> None:
    """First call transcodes; second call uses the cache."""
    fem = _MockFem([1, 2])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),),
    )
    _write_synthetic_node_out(
        output_dir / "r_disp.out",
        time=[0.0, 1.0], node_ids=[1, 2], dofs=[1],
        value_fn=lambda k, nid, dof: 0.1 * (nid + k),
    )

    cache_root = tmp_path / "cache"
    # First call — transcodes
    with Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
    ) as r:
        slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values[0], [0.1, 0.2])

    # Cache directory should now have one HDF5 file
    cached = list(cache_root.glob("*.h5"))
    assert len(cached) == 1
    cached_mtime = cached[0].stat().st_mtime_ns

    # Second call — reuses cache, doesn't re-transcode
    with Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
    ) as r:
        slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values[0], [0.1, 0.2])

    # Cache file mtime unchanged → not re-written
    assert cached[0].stat().st_mtime_ns == cached_mtime


def test_cache_invalidates_on_source_change(tmp_path: Path) -> None:
    """Modify the source .out file → cache rebuilds."""
    fem = _MockFem([1])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),),
    )
    src = output_dir / "r_disp.out"
    src.write_text("0.0 1.0\n", encoding="utf-8")

    cache_root = tmp_path / "cache"
    Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
    ).close()
    cached_files_v1 = sorted(cache_root.glob("*.h5"))
    assert len(cached_files_v1) == 1
    v1_key = cached_files_v1[0].stem

    # Modify the source — different value AND different size
    src.write_text("0.0 2.5\n0.5 3.5\n", encoding="utf-8")
    Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
    ).close()
    cached_files_v2 = sorted(cache_root.glob("*.h5"))
    # Either two cached files coexist, or v1 was overwritten — at minimum
    # the new key differs from v1.
    new_keys = {f.stem for f in cached_files_v2}
    assert v1_key not in new_keys or len(new_keys) > 1


def test_cache_invalidates_on_fem_change(tmp_path: Path) -> None:
    """Different FEMData snapshot_id → different cache entry."""
    fem_a = _MockFem([1], salt=0)
    fem_b = _MockFem([1], salt=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    spec_a = ResolvedRecorderSpec(
        fem_snapshot_id=fem_a.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),),
    )
    spec_b = ResolvedRecorderSpec(
        fem_snapshot_id=fem_b.snapshot_id,
        records=spec_a.records,
    )
    src = output_dir / "r_disp.out"
    src.write_text("0.0 1.0\n", encoding="utf-8")

    cache_root = tmp_path / "cache"
    Results.from_recorders(
        spec_a, output_dir, fem=fem_a, cache_root=cache_root,
    ).close()
    Results.from_recorders(
        spec_b, output_dir, fem=fem_b, cache_root=cache_root,
    ).close()

    keys = {f.stem for f in cache_root.glob("*.h5")}
    assert len(keys) == 2     # different fem → different cache entry
