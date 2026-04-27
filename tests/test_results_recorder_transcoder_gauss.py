"""Phase 11a Step D — gauss transcoder mocked tests.

Hand-builds an OpenSees-style ``.out`` file (no Tcl/openseespy needed),
runs ``RecorderTranscoder`` over a matching spec, and reads the
resulting native HDF5 through the public ``Results`` API.

Real-Tcl coverage (emit → run → transcode round-trip) lives in
``test_results_recorder_transcoder_gauss_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.transcoders._recorder import (
    RecorderTranscoder,
    _identify_layout,
    _record_catalog_token,
)
from apeGmsh.solvers._element_response import (
    IntRule,
    flatten,
    lookup,
)
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Synthetic FEMData (mirror of the existing capture-test mock)
# =====================================================================

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
        group.attrs["ndf"] = 3
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=np.asarray(self.nodes.ids, dtype=np.int64))
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _make_spec(*records, snapshot_id):
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Synthetic .out builder
# =====================================================================

def _write_element_out(
    path: Path,
    *,
    times: np.ndarray,
    flat: np.ndarray,           # (T, E, K)
) -> None:
    """Write an OpenSees-shaped .out file: ``time  e1c0 e1c1 ... eMcK``."""
    T, E, K = flat.shape
    rows = np.column_stack([
        times.reshape(-1, 1),
        flat.reshape(T, E * K),
    ])
    np.savetxt(path, rows)


# =====================================================================
# Catalog token derivation
# =====================================================================

class TestCatalogTokenDerivation:
    def test_stress_token(self) -> None:
        rec = ResolvedRecorderRecord(
            category="gauss", name="r",
            components=("stress_xx", "stress_yy"),
            dt=None, n_steps=None, element_ids=np.array([1]),
        )
        assert _record_catalog_token(rec) == "stress"

    def test_strain_token(self) -> None:
        rec = ResolvedRecorderRecord(
            category="gauss", name="r",
            components=("strain_xx",),
            dt=None, n_steps=None, element_ids=np.array([1]),
        )
        assert _record_catalog_token(rec) == "strain"

    def test_mixed_raises(self) -> None:
        rec = ResolvedRecorderRecord(
            category="gauss", name="r",
            components=("stress_xx", "strain_yy"),
            dt=None, n_steps=None, element_ids=np.array([1]),
        )
        with pytest.raises(ValueError, match="mixes work-conjugate families"):
            _record_catalog_token(rec)


# =====================================================================
# Layout sniff via column count
# =====================================================================

class TestLayoutSniff:
    def test_six_columns_ambiguous_without_hint(self) -> None:
        """flat_size=6 matches FourNodeTet (1 GP, tet) AND SSPbrick (1 GP, hex).

        Their shapes differ (coord_system + GP coordinates), so the
        sniff cannot pick one without a hint.
        """
        with pytest.raises(ValueError, match="Ambiguous"):
            _identify_layout("stress", flat_size=6)

    def test_six_columns_with_class_hint_picks_tet(self) -> None:
        layout, cls, rule = _identify_layout(
            "stress", flat_size=6, class_hint="FourNodeTetrahedron",
        )
        assert cls == "FourNodeTetrahedron"
        assert rule == IntRule.Tet_GL_1

    def test_six_columns_with_class_hint_picks_sspbrick(self) -> None:
        layout, cls, rule = _identify_layout(
            "stress", flat_size=6, class_hint="SSPbrick",
        )
        assert cls == "SSPbrick"
        assert rule == IntRule.Hex_GL_1
        assert layout.coord_system == "isoparametric"

    def test_twentyfour_columns_ambiguous_without_hint(self) -> None:
        """flat_size=24 now matches both TenNodeTet (4 GPs × 6 comp,
        barycentric_tet) and ASDShellT3 (3 GPs × 8 comp,
        barycentric_tri). Different shapes → must raise.
        """
        with pytest.raises(ValueError, match="Ambiguous"):
            _identify_layout("stress", flat_size=24)

    def test_twentyfour_with_class_hint_picks_tet(self) -> None:
        layout, cls, rule = _identify_layout(
            "stress", flat_size=24, class_hint="TenNodeTetrahedron",
        )
        assert cls == "TenNodeTetrahedron"
        assert rule == IntRule.Tet_GL_2

    def test_twentyfour_with_class_hint_picks_asdshellt3(self) -> None:
        layout, cls, rule = _identify_layout(
            "stress", flat_size=24, class_hint="ASDShellT3",
        )
        assert cls == "ASDShellT3"
        assert layout.coord_system == "barycentric_tri"

    def test_fortyeight_columns_picks_brick_or_bbarbrick(self) -> None:
        """Brick and BbarBrick are shape-equivalent twins; sniff picks one."""
        layout, cls, rule = _identify_layout("stress", flat_size=48)
        assert cls in ("Brick", "BbarBrick")
        assert rule == IntRule.Hex_GL_2
        assert layout.n_gauss_points == 8

    def test_fortyeight_with_class_hint_picks_bbarbrick(self) -> None:
        layout, cls, rule = _identify_layout(
            "stress", flat_size=48, class_hint="BbarBrick",
        )
        assert cls == "BbarBrick"

    def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="No catalog entry"):
            _identify_layout("stress", flat_size=99)

    def test_unknown_class_hint_raises(self) -> None:
        with pytest.raises(ValueError, match="No catalog entry"):
            _identify_layout("stress", flat_size=6, class_hint="NotAClass")


# =====================================================================
# Transcoder round-trip
# =====================================================================

class TestTranscoderGauss:
    def test_single_class_round_trip(self, tmp_path: Path) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        fem = _MinimalFem([1, 2, 3, 4])

        # Two elements, two time steps.
        T, E, G = 2, 2, 1
        per_comp = {}
        for ki, name in enumerate(layout.component_layout):
            arr = np.zeros((T, E, G), dtype=np.float64)
            for t in range(T):
                for e in range(E):
                    arr[t, e, 0] = ki * 100.0 + t * 10.0 + e
            per_comp[name] = arr
        flat = flatten(per_comp, layout)

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="solid_stress",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([10, 20]),
                # FourNodeTet shares flat_size=6 with SSPbrick — hint
                # disambiguates.
                element_class_name="FourNodeTetrahedron",
            ),
            snapshot_id=fem.snapshot_id,
        )
        # The emit side decides the file path; mirror it via the spec.
        # solid_stress + category=gauss → solid_stress_gauss.out
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_element_out(
            out_dir / "solid_stress_gauss.out",
            times=np.array([0.5, 1.0]),
            flat=flat,
        )

        target = tmp_path / "transcoded.h5"
        transcoder = RecorderTranscoder(
            spec, out_dir, target, fem, stage_name="static", stage_kind="static",
        )
        transcoder.run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            sxx = s.elements.gauss.get(component="stress_xx")
            assert sxx.values.shape == (2, 2)
            # ki=0 (stress_xx); pattern = t*10 + e.
            np.testing.assert_array_equal(
                sxx.values, [[0.0, 1.0], [10.0, 11.0]],
            )
            np.testing.assert_array_equal(sxx.element_index, [10, 20])
            np.testing.assert_allclose(
                sxx.natural_coords,
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
            )

    def test_multi_gp_round_trip(self, tmp_path: Path) -> None:
        layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")
        fem = _MinimalFem([1, 2])

        T, E, G = 3, 2, 4
        per_comp = {}
        for ki, name in enumerate(layout.component_layout):
            arr = np.zeros((T, E, G), dtype=np.float64)
            for t in range(T):
                for e in range(E):
                    for g in range(G):
                        arr[t, e, g] = ki * 1000 + t * 100 + e * 10 + g
            per_comp[name] = arr
        flat = flatten(per_comp, layout)

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="ts2",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([100, 200]),
                # TenNodeTet shares flat_size=24 with ASDShellT3 — hint
                # disambiguates.
                element_class_name="TenNodeTetrahedron",
            ),
            snapshot_id=fem.snapshot_id,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_element_out(
            out_dir / "ts2_gauss.out",
            times=np.array([0.0, 0.5, 1.0]),
            flat=flat,
        )

        target = tmp_path / "transcoded.h5"
        RecorderTranscoder(spec, out_dir, target, fem).run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            sxx = s.elements.gauss.get(component="stress_xx")
            # 2 elements × 4 GPs = 8 cols.
            assert sxx.values.shape == (3, 8)
            # element_index repeats per GP.
            np.testing.assert_array_equal(
                sxx.element_index, [100, 100, 100, 100, 200, 200, 200, 200],
            )
            # ki=0; t=2; e=0 → cols 0..3 = [200, 201, 202, 203].
            np.testing.assert_array_equal(
                sxx.values[2, :4], [200.0, 201.0, 202.0, 203.0],
            )

    def test_class_hint_disambiguates_six_columns(self, tmp_path: Path) -> None:
        """flat_size=6 + ``element_class_name="SSPbrick"`` → SSPbrick layout.

        Without the hint, the 6-col case is ambiguous between
        FourNodeTetrahedron (1-GP tet) and SSPbrick (1-GP hex).
        """
        layout = lookup("SSPbrick", IntRule.Hex_GL_1, "stress")
        fem = _MinimalFem([1, 2, 3, 4, 5, 6, 7, 8])

        T, E, G = 1, 1, 1
        per_comp = {
            name: np.array([[[float(ki * 10)]]])
            for ki, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="ssp",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([42]),
                element_class_name="SSPbrick",
            ),
            snapshot_id=fem.snapshot_id,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_element_out(
            out_dir / "ssp_gauss.out", times=np.array([1.0]), flat=flat,
        )

        target = tmp_path / "transcoded.h5"
        RecorderTranscoder(spec, out_dir, target, fem).run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            assert slab.values.shape == (1, 1)
            # SSPbrick GP at the parent-cube origin.
            np.testing.assert_array_equal(
                slab.natural_coords, [[0.0, 0.0, 0.0]],
            )

    def test_strain_record_uses_strain_layout(self, tmp_path: Path) -> None:
        """Records with strain components route through the strain catalog entry."""
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "strain")
        fem = _MinimalFem([1])

        T, E, G = 1, 1, 1
        per_comp = {
            name: np.array([[[float(ki)]]], dtype=np.float64)
            for ki, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="r",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([7]),
                element_class_name="FourNodeTetrahedron",
            ),
            snapshot_id=fem.snapshot_id,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_element_out(
            out_dir / "r_gauss.out", times=np.array([1.0]), flat=flat,
        )

        target = tmp_path / "transcoded.h5"
        RecorderTranscoder(spec, out_dir, target, fem).run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="strain_xy")
            # ki=3 for strain_xy in the catalog layout.
            assert slab.values[0, 0] == 3.0


# =====================================================================
# Mixed nodes + gauss in one transcode
# =====================================================================

class TestMixedRecords:
    def test_nodes_and_gauss_same_stage(self, tmp_path: Path) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        fem = _MinimalFem([1, 2, 3, 4])

        # Build a node .out file: "all_disp_disp.out" with -dof 1 2 3.
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        T = 2
        n_nodes = 4
        node_rows = np.column_stack([
            np.array([0.5, 1.0]),
            np.tile(np.arange(n_nodes * 3, dtype=np.float64), (T, 1)),
        ])
        np.savetxt(out_dir / "all_disp_disp.out", node_rows)

        # Build a gauss .out file.
        T, E, G = 2, 1, 1
        per_comp = {
            name: np.full((T, E, G), float(ki))
            for ki, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)
        _write_element_out(
            out_dir / "body_gauss.out", times=np.array([0.5, 1.0]), flat=flat,
        )

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="nodes", name="all_disp",
                components=("displacement_x", "displacement_y", "displacement_z"),
                dt=None, n_steps=None,
                node_ids=np.array([1, 2, 3, 4]),
            ),
            ResolvedRecorderRecord(
                category="gauss", name="body",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([5]),
                element_class_name="FourNodeTetrahedron",
            ),
            snapshot_id=fem.snapshot_id,
        )

        target = tmp_path / "merged.h5"
        RecorderTranscoder(spec, out_dir, target, fem).run()

        with Results.from_native(target) as r:
            s = r.stage(r.stages[0].id)
            # Gauss surfaced.
            sxx = s.elements.gauss.get(component="stress_xx")
            assert sxx.values.shape == (2, 1)
            # Nodes surfaced (displacement_x, node 1, t=0).
            ux = r.nodes.get(ids=[1], component="displacement_x")
            assert ux.values.shape == (2, 1)


# =====================================================================
# Error paths
# =====================================================================

class TestTranscoderErrors:
    def test_column_count_mismatch_raises(self, tmp_path: Path) -> None:
        """Heterogeneous-class record (or wrong K) is detected."""
        fem = _MinimalFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="bad",
                components=("stress_xx", "stress_yy", "stress_zz",
                            "stress_xy", "stress_yz", "stress_xz"),
                dt=None, n_steps=None,
                element_ids=np.array([1, 2, 3]),    # 3 elements
            ),
            snapshot_id=fem.snapshot_id,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # 1 time + 17 data cols → not divisible by 3 elements.
        np.savetxt(
            out_dir / "bad_gauss.out",
            np.zeros((1, 18), dtype=np.float64),
        )

        target = tmp_path / "x.h5"
        with pytest.raises(ValueError, match="not divisible"):
            RecorderTranscoder(spec, out_dir, target, fem).run()

    def test_unknown_class_size_raises(self, tmp_path: Path) -> None:
        """Column count that no catalog entry matches surfaces clearly."""
        fem = _MinimalFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="weird",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # 1 time + 11 data cols → no matching layout for token=stress.
        np.savetxt(
            out_dir / "weird_gauss.out",
            np.zeros((1, 12), dtype=np.float64),
        )
        target = tmp_path / "x.h5"
        with pytest.raises(ValueError, match="No catalog entry"):
            RecorderTranscoder(spec, out_dir, target, fem).run()
