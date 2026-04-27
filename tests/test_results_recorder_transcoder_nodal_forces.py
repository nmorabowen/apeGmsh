"""Phase 11b Step 3c — nodal-forces transcoder mocked tests.

Hand-builds OpenSees-style ``.out`` files for ``recorder Element ...
globalForce`` and ``... localForce``, runs ``RecorderTranscoder``
over a matching spec, and reads the resulting native HDF5 through
the public ``Results`` API.

Real-Tcl coverage (full emit → run → transcode → read round-trip
with three-way physics agreement) lives in
``test_results_recorder_transcoder_nodal_forces_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.transcoders._recorder import RecorderTranscoder
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


class _MinimalFem:
    def __init__(self, node_ids: np.ndarray) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        self.nodes = SimpleNamespace(
            ids=ids, coords=np.zeros((ids.size, 3), dtype=np.float64),
        )
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
        n.create_dataset("ids", data=np.asarray(self.nodes.ids, dtype=np.int64))
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _make_spec(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


def _write_element_out(
    path: Path, *, times: np.ndarray, flat: np.ndarray,
) -> None:
    """Write `recorder Element ... -file foo.out`-shaped output."""
    T, E, K = flat.shape
    rows = np.column_stack([
        times.reshape(-1, 1),
        flat.reshape(T, E * K),
    ])
    np.savetxt(path, rows)


# =====================================================================
# Emit: frame derivation
# =====================================================================

class TestEmitFrameDerivation:
    """Step 3c.1 changes — emit picks the right keyword by frame."""

    def test_global_components_emit_globalForce(self) -> None:
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id="dummy",
        )
        [line] = spec.to_tcl_commands()
        assert "globalForce" in line
        assert "localForce" not in line

    def test_local_components_emit_localForce(self) -> None:
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_local_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id="dummy",
        )
        [line] = spec.to_tcl_commands()
        assert "localForce" in line
        # Should NOT contain globalForce — the recorder line ends in
        # 'localForce' as the response token.
        assert "globalForce" not in line

    def test_mixed_frames_raises_at_emit(self) -> None:
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=(
                    "nodal_resisting_force_x",
                    "nodal_resisting_force_local_y",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id="dummy",
        )
        with pytest.raises(ValueError, match="mixes global and local"):
            spec.to_tcl_commands()


# =====================================================================
# End-to-end: build .out → transcode → read back
# =====================================================================

class TestElasticBeam3dGlobal:
    def test_full_round_trip(self, tmp_path: Path) -> None:
        fem = _MinimalFem([1, 2])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="beam",
                components=(
                    "nodal_resisting_force_x", "nodal_resisting_force_y",
                    "nodal_resisting_force_z", "nodal_resisting_moment_x",
                    "nodal_resisting_moment_y", "nodal_resisting_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ElasticBeam3d",
            ),
            snapshot_id=fem.snapshot_id,
        )

        # Build a synthetic .out with 12 columns per element (2 nodes × 6).
        # Encoding: t * 1000 + n * 10 + k.
        T, E, K = 2, 1, 12
        flat = np.zeros((T, E, K), dtype=np.float64)
        for t in range(T):
            for n in range(2):
                for k in range(6):
                    flat[t, 0, n * 6 + k] = t * 1000 + n * 10 + k
        _write_element_out(
            tmp_path / "beam_elements.out",
            times=np.array([1.0, 2.0]),
            flat=flat,
        )

        target = tmp_path / "out.h5"
        RecorderTranscoder(
            spec, output_dir=tmp_path, target_path=target, fem=fem,
            stage_name="static", stage_kind="static",
        ).run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            slab_fx = s.elements.get(component="nodal_resisting_force_x")
            assert slab_fx.values.shape == (2, 1, 2)
            # k=0 (Px). t=0,n=0 → 0; t=0,n=1 → 10; t=1,n=0 → 1000.
            np.testing.assert_array_equal(slab_fx.values[0, 0], [0.0, 10.0])
            np.testing.assert_array_equal(slab_fx.values[1, 0], [1000.0, 1010.0])

            # k=5 (Mz). t=0,n=0 → 5; t=0,n=1 → 15.
            slab_mz = s.elements.get(component="nodal_resisting_moment_z")
            np.testing.assert_array_equal(slab_mz.values[0, 0], [5.0, 15.0])


class TestElasticBeam3dLocal:
    def test_local_frame_round_trip(self, tmp_path: Path) -> None:
        fem = _MinimalFem([1, 2])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="beam",
                components=(
                    "nodal_resisting_force_local_x",
                    "nodal_resisting_force_local_y",
                    "nodal_resisting_moment_local_x",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ElasticBeam3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        flat = np.arange(12, dtype=np.float64).reshape(1, 1, 12) + 100
        _write_element_out(
            tmp_path / "beam_elements.out",
            times=np.array([1.0]), flat=flat,
        )
        target = tmp_path / "out.h5"
        RecorderTranscoder(
            spec, output_dir=tmp_path, target_path=target, fem=fem,
            stage_name="g", stage_kind="static",
        ).run()
        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            # k=0 (N) at n=0 → 100; n=1 → 106.
            slab_n = s.elements.get(component="nodal_resisting_force_local_x")
            np.testing.assert_array_equal(slab_n.values[0, 0], [100.0, 106.0])
            # k=3 (T): n=0 → 103; n=1 → 109.
            slab_t = s.elements.get(component="nodal_resisting_moment_local_x")
            np.testing.assert_array_equal(slab_t.values[0, 0], [103.0, 109.0])


class TestElasticBeam2d:
    def test_2d_round_trip(self, tmp_path: Path) -> None:
        fem = _MinimalFem([1, 2])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="beam2d",
                components=("nodal_resisting_force_x", "nodal_resisting_moment_z"),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ElasticBeam2d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        # 2 nodes × 3 comps = 6 cols per element.
        flat = np.array([[10, 20, 30, 40, 50, 60]], dtype=np.float64).reshape(1, 1, 6)
        _write_element_out(
            tmp_path / "beam2d_elements.out",
            times=np.array([1.0]), flat=flat,
        )
        target = tmp_path / "out.h5"
        RecorderTranscoder(
            spec, output_dir=tmp_path, target_path=target, fem=fem,
            stage_name="g", stage_kind="static",
        ).run()
        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            slab_fx = s.elements.get(component="nodal_resisting_force_x")
            np.testing.assert_array_equal(slab_fx.values[0, 0], [10.0, 40.0])
            slab_mz = s.elements.get(component="nodal_resisting_moment_z")
            np.testing.assert_array_equal(slab_mz.values[0, 0], [30.0, 60.0])


# =====================================================================
# Validation errors
# =====================================================================

class TestValidationErrors:
    def test_missing_class_name_raises(self, tmp_path: Path) -> None:
        fem = _MinimalFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name=None,
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        (tmp_path / "r_elements.out").write_text("")
        with pytest.raises(ValueError, match="element_class_name"):
            RecorderTranscoder(
                spec, output_dir=tmp_path, target_path=target, fem=fem,
                stage_name="g", stage_kind="static",
            ).run()

    def test_uncatalogued_class_raises(self, tmp_path: Path) -> None:
        fem = _MinimalFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="MysteryBeam3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        (tmp_path / "r_elements.out").write_text("")
        with pytest.raises(ValueError, match="NODAL_FORCE_CATALOG"):
            RecorderTranscoder(
                spec, output_dir=tmp_path, target_path=target, fem=fem,
                stage_name="g", stage_kind="static",
            ).run()
