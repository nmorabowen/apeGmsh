"""Phase 6 — RecorderTranscoder end-to-end (synthetic .out files)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.transcoders import RecorderTranscoder
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.conftest import _open_model_from_h5, _stub_model_h5_path


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
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=6)


# =====================================================================
# Helpers — emit synthetic .out files matching spec emission
# =====================================================================

def _cached_files(cache_root: Path) -> list[Path]:
    """The cached transcodes, sorted: ``<key>.h5`` (native) or
    ``<key>.ladruno`` (nodal records only), never the ``<key>.model.h5``
    archive beside them."""
    return sorted(
        p for p in [*cache_root.glob("*.h5"), *cache_root.glob("*.ladruno")]
        if not p.name.endswith(".model.h5"))


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

    with Results.from_native(target, fem=fem, model=_open_model_from_h5(target)) as r:
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

    with Results.from_native(target, fem=fem, model=_open_model_from_h5(target)) as r:
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
# Element-level records are skipped (Phase 6 v1 limitation)
# =====================================================================

def test_unwired_element_records_skipped_silently(tmp_path: Path) -> None:
    """Element-level records the transcoder doesn't yet handle don't crash.

    Phase 11a wired ``gauss``. Phase 11b wired ``line_stations``
    (Step 2c) and ``elements`` (Step 3c); fibres are read too (#1982).
    A modal record (no ``.out`` emission) surfaces in
    ``transcoder.unsupported`` without a warning; a layers record raises
    (``tests/test_results_out_fiber_material.py``).
    """
    import warnings as _w
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
                category="modal", name="b", components=(),
                dt=None, n_steps=None, n_modes=2,
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
    # Note: no file for the modal record; it is skipped before any
    # file lookup happens.

    target = tmp_path / "out.h5"
    transcoder = RecorderTranscoder(spec, output_dir, target, fem)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        transcoder.run()
    assert "modal:b" in transcoder.unsupported
    assert not [w for w in caught if "modal" in str(w.message)]

    with Results.from_native(target, fem=fem, model=_open_model_from_h5(target)) as r:
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
        model=_open_model_from_h5(_stub_model_h5_path()),
    ) as r:
        slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values[0], [0.1, 0.2])

    # Cache directory should now have one cache file.
    cached = _cached_files(cache_root)
    assert len(cached) == 1
    cached_mtime = cached[0].stat().st_mtime_ns

    # Second call — reuses cache, doesn't re-transcode
    with Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
        model=_open_model_from_h5(_stub_model_h5_path()),
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
        model=_open_model_from_h5(_stub_model_h5_path()),
    ).close()
    cached_files_v1 = _cached_files(cache_root)
    assert len(cached_files_v1) == 1
    v1_key = cached_files_v1[0].stem

    # Modify the source — different value AND different size
    src.write_text("0.0 2.5\n0.5 3.5\n", encoding="utf-8")
    Results.from_recorders(
        spec, output_dir, fem=fem, cache_root=cache_root,
        model=_open_model_from_h5(_stub_model_h5_path()),
    ).close()
    cached_files_v2 = _cached_files(cache_root)
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
        model=_open_model_from_h5(_stub_model_h5_path()),
    ).close()
    Results.from_recorders(
        spec_b, output_dir, fem=fem_b, cache_root=cache_root,
        model=_open_model_from_h5(_stub_model_h5_path()),
    ).close()

    keys = {f.stem for f in _cached_files(cache_root)}
    assert len(keys) == 2     # different fem → different cache entry


# =====================================================================
# ADR 0131 D6: nodal records transcode to .ladruno
# =====================================================================

#: Every node recorder type the read decodes (rotations aside: the stub
#: model's archive masks them, and tests/test_results_transcoder_nodes_ndf.py
#: holds them on both targets).
_ALL_NODAL = (
    "displacement_x", "displacement_y", "displacement_z", "velocity_y",
    "acceleration_z", "reaction_force_x", "reaction_force_z", "force_y",
    "pore_pressure",
)


def _value(k: int, nid: int, dof: int) -> float:
    return 1.0 + 0.01 * nid + 0.001 * dof + 0.0001 * k


def _nodal_spec(fem: Any, components: tuple[str, ...] = _ALL_NODAL) -> ResolvedRecorderSpec:
    """Two records over overlapping nodes, so the merge leaves NaN holes."""
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="a", components=components,
                dt=None, n_steps=None, node_ids=np.array([1, 2, 3])),
            ResolvedRecorderRecord(
                category="nodes", name="b", components=components[:2],
                dt=None, n_steps=None, node_ids=np.array([3, 4])),
        ),
    )


def _write_spec_files(spec: ResolvedRecorderSpec, out: Path,
                      time: tuple[float, ...] = (0.0, 0.5, 1.0),
                      stage_id: "str | None" = None) -> None:
    """Every file the emit names for ``spec``, valued by :func:`_value`."""
    from apeGmsh.results.writers._cache import quiet_emit_logical

    out.mkdir(exist_ok=True)
    for rec in spec.records:
        for lr in quiet_emit_logical(rec, output_dir=str(out), file_format="out",
                                     stage_id=stage_id):
            _write_synthetic_node_out(
                Path(lr.file_path), time=list(time),
                node_ids=list(lr.target_ids), dofs=list(lr.dofs),
                value_fn=_value)


def _native_read(spec: ResolvedRecorderSpec, out: Path, fem: Any, tmp: Path,
                 model: Any, **kw: Any) -> Results:
    """The read every nodal transcode was before: a native target."""
    archive = tmp / "native.model.h5"
    model.to_h5(archive)
    target = tmp / "native.h5"
    RecorderTranscoder(spec, out, target, fem, model_h5_src=archive, **kw).run()
    return Results.from_native(target, fem=fem, model=model)


def _assert_same_reads(got: Results, want: Results) -> None:
    """Stages, times, node ids and values (NaN in place) are the native read's."""
    assert [(s.name, s.kind, s.n_steps) for s in got.stages] \
        == [(s.name, s.kind, s.n_steps) for s in want.stages]
    for g_stage, w_stage in zip(got.stages, want.stages):
        g, w = got.stage(g_stage.id), want.stage(w_stage.id)
        comps = sorted(w.nodes.available_components())
        assert sorted(g.nodes.available_components()) == comps
        assert comps
        for comp in comps:
            gs, ws = g.nodes.get(component=comp), w.nodes.get(component=comp)
            np.testing.assert_array_equal(gs.node_ids, ws.node_ids, err_msg=comp)
            np.testing.assert_array_equal(gs.time, ws.time, err_msg=comp)
            np.testing.assert_array_equal(gs.values, ws.values, err_msg=comp)


def test_a_nodal_run_transcodes_to_ladruno_read_as_the_native_file(tmp_path: Path) -> None:
    """The card's oracle: ``from_recorders`` of nodal records opens a
    ``.ladruno``, every stage, time, node id and value (the merge's NaN
    holes included) equal to the native transcode of the same files, and
    ``Results.from_ladruno`` reads the same file the same way."""
    import h5py

    import apeGmsh
    from apeGmsh.results.readers._ladruno import LadrunoReader

    fem = _MockFem([1, 2, 3, 4])
    spec = _nodal_spec(fem)
    out = tmp_path / "out"
    _write_spec_files(spec, out)
    model = _open_model_from_h5(_stub_model_h5_path())
    cache = tmp_path / "cache"

    with Results.from_recorders(spec, out, fem=fem, cache_root=cache, model=model) as r, \
            _native_read(spec, out, fem, tmp_path, model) as native:
        assert r.path is not None and r.path.suffix == ".ladruno"
        assert isinstance(r._reader, LadrunoReader)
        assert r.model is model
        assert [s.name for s in r.stages] == ["analysis"]
        _assert_same_reads(r, native)
        assert np.isnan(r.nodes.get(component="velocity_y").values[:, -1]).all()
        path = r.path

    with h5py.File(path, "r") as f:
        prov = {k: str(v) for k, v in f["INFO/PROVENANCE"].attrs.items()}
    assert prov["opensees_backend"] == "unknown"
    assert prov["opensees_build_sha"] == "unknown"
    assert prov["writer"] == f"apeGmsh recorders->ladruno {apeGmsh.__version__}"
    assert prov["unsupported"] == '["elements"]'
    assert prov["fem_hash"] == str(model.lineage.fem_hash)
    assert "stage_names" not in prov

    sibling = path.with_suffix(".model.h5")
    (tmp_path / "again").mkdir()
    with Results.from_ladruno(path, fem=fem, model_h5=sibling) as lad, \
            _native_read(spec, out, fem, tmp_path / "again", model) as native:
        for comp in _ALL_NODAL:
            np.testing.assert_array_equal(
                lad.nodes.get(component=comp).values,
                native.nodes.get(component=comp).values, err_msg=comp)


def test_a_cached_ladruno_transcode_reopens_unchanged(tmp_path: Path) -> None:
    """Reload: the second call opens the cached ``.ladruno`` without
    re-writing it, with the same stages, times and node ids; a deleted
    model archive beside it is written again before the read."""
    fem = _MockFem([1, 2, 3, 4])
    spec = _nodal_spec(fem)
    out = tmp_path / "out"
    _write_spec_files(spec, out)
    model = _open_model_from_h5(_stub_model_h5_path())
    cache = tmp_path / "cache"
    with Results.from_recorders(spec, out, fem=fem, cache_root=cache, model=model) as first:
        path = first.path
        assert path is not None
        stamp = path.stat().st_mtime_ns
        before = [(s.name, s.kind, s.n_steps) for s in first.stages]
        slab = first.nodes.get(component="displacement_x")
    path.with_suffix(".model.h5").unlink()
    with Results.from_recorders(spec, out, fem=fem, cache_root=cache, model=model) as again:
        assert again.path == path and path.stat().st_mtime_ns == stamp
        assert [(s.name, s.kind, s.n_steps) for s in again.stages] == before
        reread = again.nodes.get(component="displacement_x")
        np.testing.assert_array_equal(reread.node_ids, slab.node_ids)
        np.testing.assert_array_equal(reread.time, slab.time)
        np.testing.assert_array_equal(reread.values, slab.values)
    assert path.with_suffix(".model.h5").is_file()


def test_a_spec_the_ladruno_cannot_hold_keeps_the_native_target(tmp_path: Path) -> None:
    """``displacement_increment`` has no recorder group: the spec keeps the
    native file, read as before, and no record is dropped."""
    fem = _MockFem([1, 2, 3, 4])
    spec = _nodal_spec(fem, ("displacement_x", "displacement_increment_x"))
    assert not RecorderTranscoder.writes_nodal_history(spec)
    out = tmp_path / "out"
    _write_spec_files(spec, out)
    model = _open_model_from_h5(_stub_model_h5_path())
    with Results.from_recorders(spec, out, fem=fem, cache_root=tmp_path / "c",
                                model=model) as r:
        assert r.path is not None and r.path.suffix == ".h5"
        assert sorted(r.nodes.available_components()) == [
            "displacement_increment_x", "displacement_x"]


def test_element_records_keep_the_native_target() -> None:
    fem = _MockFem([1])
    nodes = ResolvedRecorderRecord(
        category="nodes", name="n", components=("displacement_x",),
        dt=None, n_steps=None, node_ids=np.array([1]))
    modal = ResolvedRecorderRecord(
        category="modal", name="m", components=(), dt=None, n_steps=None, n_modes=2)
    assert RecorderTranscoder.writes_nodal_history(
        ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(nodes, modal)))
    for category, comp in (("gauss", "stress_xx"), ("line_stations", "axial_force"),
                           ("elements", "nodal_resisting_force_x"),
                           ("fibers", "fiber_stress")):
        rec = ResolvedRecorderRecord(
            category=category, name="e", components=(comp,),
            dt=None, n_steps=None, element_ids=np.array([1]))
        spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(nodes, rec))
        assert not RecorderTranscoder.writes_nodal_history(spec), category
        assert RecorderTranscoder.target_for(spec, Path("k.h5")) == Path("k.h5")


def test_a_ladruno_target_refuses_an_element_record_by_name(tmp_path: Path) -> None:
    """Never a silent drop: a ``.ladruno`` target with a Gauss record refuses
    naming it, before any file is read or written."""
    fem = _MockFem([1])
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(
            category="nodes", name="n", components=("displacement_x",),
            dt=None, n_steps=None, node_ids=np.array([1])),
        ResolvedRecorderRecord(
            category="gauss", name="g", components=("stress_xx",),
            dt=None, n_steps=None, element_ids=np.array([1])),
    ))
    target = tmp_path / "t.ladruno"
    with pytest.raises(ValueError, match=r"\['gauss:g'\]"):
        RecorderTranscoder(spec, tmp_path / "missing", target, fem,
                           model_h5_src=_stub_model_h5_path()).run()
    assert list(tmp_path.iterdir()) == []


def test_a_ladruno_target_needs_an_archive_to_stamp(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _nodal_spec(fem)
    with pytest.raises(ValueError, match="model_h5_src= or model_archive="):
        RecorderTranscoder(spec, tmp_path / "missing", tmp_path / "t.ladruno", fem).run()
    assert list(tmp_path.iterdir()) == []
