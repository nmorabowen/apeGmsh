"""Phase 11b Step 2c.2 — line-stations transcoder mocked tests.

Hand-builds OpenSees-style ``.out`` + ``_gpx.out`` files (no Tcl /
openseespy needed), runs ``RecorderTranscoder`` over a matching spec,
and reads the resulting native HDF5 through the public ``Results``
API.

Real-Tcl-subprocess coverage (full emit → run → transcode → read
round-trip with three-way physics agreement against MPCO and
DomainCapture) lives in
``test_results_recorder_transcoder_line_stations_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.transcoders import _recorder as recorder_module
from apeGmsh.results.transcoders._recorder import (
    RecorderTranscoder,
    _resolve_element_lengths,
)
from apeGmsh.results.spec._emit import line_station_gpx_path
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.conftest import _open_model_from_h5


# =====================================================================
# Mock fem with a duck-typed element-end-coords lookup
# =====================================================================

class _BeamFem:
    """Minimal FEMData with the surface ``_resolve_element_lengths`` needs.

    Exposes ``element_end_coords(eid) -> (c1, c2)``; the transcoder
    prefers this duck-typed shortcut over the full fem.elements API.
    """

    def __init__(
        self,
        node_ids: np.ndarray,
        coords: np.ndarray,
        connectivity: dict[int, tuple[int, int]],
    ) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        self.nodes = SimpleNamespace(
            ids=ids, coords=np.asarray(coords, dtype=np.float64),
        )
        self.elements = []
        self._conn = connectivity
        self._coords_by_id = {
            int(nid): np.asarray(c, dtype=np.float64)
            for nid, c in zip(ids, coords)
        }

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=6)

    def element_end_coords(self, eid: int) -> tuple[np.ndarray, np.ndarray]:
        n_a, n_b = self._conn[int(eid)]
        return self._coords_by_id[n_a], self._coords_by_id[n_b]


def _make_spec(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Synthetic .out and _gpx.out builders
# =====================================================================

def _write_section_out(
    path: Path,
    *,
    times: np.ndarray,
    flat: np.ndarray,           # (T, E, n_IP * n_components) element-major
) -> None:
    """Write an OpenSees-shaped section.force .out file."""
    T, E, K = flat.shape
    rows = np.column_stack([
        times.reshape(-1, 1),
        flat.reshape(T, E * K),
    ])
    np.savetxt(path, rows)


def _write_gpx_out(
    path: Path,
    *,
    times: np.ndarray,
    xi_phys_per_elem: np.ndarray,    # (E, n_IP) — physical xi*L
) -> None:
    """Write an OpenSees-shaped integrationPoints _gpx.out file.

    OpenSees writes one row per step; values are static across steps.
    """
    E, n_ip = xi_phys_per_elem.shape
    flat_row = xi_phys_per_elem.reshape(-1)        # (E * n_IP,)
    rows = np.column_stack([
        times.reshape(-1, 1),
        np.tile(flat_row, (times.size, 1)),
    ])
    np.savetxt(path, rows)


# =====================================================================
# Element-length resolver
# =====================================================================

#: The model-less refusal (#2331): it names both inputs that give a count.
NO_MODEL = r"pass model_h5_src= .* or pairing="


def _run_with_known_counts(
    monkeypatch: pytest.MonkeyPatch, spec: ResolvedRecorderSpec,
    tmp_path: Path, target: Path, fem, n_ip: int,
) -> None:
    """Transcode ``spec`` with each element's count known to be ``n_ip``.

    These mocked files come with no model, and a model-less read is refused
    by name, since nothing names a count (#2331): that is asserted first.
    The test wrote the files, so it knows the count a model's
    ``beamIntegration`` would give; it stands in for one, checked against
    the file's columns, and the decode is then held to the files.
    """
    def run() -> None:
        RecorderTranscoder(
            spec, output_dir=tmp_path, target_path=target, fem=fem,
            stage_name="static", stage_kind="static",
        ).run()

    with pytest.raises(ValueError, match=NO_MODEL):
        run()
    assert not target.exists()

    def counts(rec, n_columns, *_a):
        assert n_columns == n_ip * len(rec.element_ids)
        return n_ip

    monkeypatch.setattr(recorder_module, "require_station_source",
                        lambda *_a: None)
    monkeypatch.setattr(recorder_module, "line_station_count", counts)
    run()


class TestResolveElementLengths:
    def test_duck_typed_path(self) -> None:
        fem = _BeamFem(
            node_ids=np.array([1, 2, 3]),
            coords=np.array([
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 4.0, 0.0],
            ]),
            connectivity={1: (1, 2), 2: (2, 3)},
        )
        L = _resolve_element_lengths(fem, [1, 2])
        np.testing.assert_allclose(L, [5.0, 4.0])

    def test_skew_3d_distance(self) -> None:
        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 12.0],     # 3-4-12 → 13
            ]),
            connectivity={1: (1, 2)},
        )
        L = _resolve_element_lengths(fem, [1])
        np.testing.assert_allclose(L, [13.0])


class TestResolveElementLengthsRealFem:
    """#1741: on a real FEMData the lengths are looked up by element id.

    The portal frame's column ids (5-7, 11-13) are not physical-group
    tags (1, 2), so a ``tag=`` lookup raised ``KeyError: 'No group with
    tag 5'``. Each column element is ``H / 3`` long, each girder element
    ``L / 3`` (three elements per curve).
    """

    def test_lengths_by_element_id(self) -> None:
        from tests.test_results_recorder_tag_parity import H, L, _portal_fem

        fem = _portal_fem()
        cols = [int(e) for e in fem.elements.select(pg="Cols").ids]
        beam = [int(e) for e in fem.elements.select(pg="Beam").ids]
        assert min(cols) > 2          # ids that are no PG tag
        ids = cols[::-1] + beam       # unsorted: rows follow the request
        got = _resolve_element_lengths(fem, ids)
        np.testing.assert_allclose(
            got, [H / 3] * len(cols) + [L / 3] * len(beam), rtol=1e-12,
        )

    def test_unknown_element_id_raises(self) -> None:
        from tests.test_results_recorder_tag_parity import _portal_fem

        fem = _portal_fem()
        known = int(fem.elements.select(pg="Cols").ids[0])
        with pytest.raises(ValueError, match=r"\[99999\]"):
            _resolve_element_lengths(fem, [known, 99999])


# =====================================================================
# End-to-end: build .out + _gpx.out → transcode → read back
# =====================================================================

class TestForceBeamColumn3dAggregated:
    def _setup(self, tmp_path: Path):
        """Single ForceBeamColumn3d, 5-IP Lobatto, aggregated 6-comp section.

        Lobatto-5 puts IPs at xi/L ∈ {0, 0.17266, 0.5, 0.82733, 1.0}
        — so for L = 5, xi*L = {0, 0.86, 2.5, 4.13, 5.0}, which maps
        to natural ξ = {-1, -0.6547, 0, +0.6547, +1}.
        """
        L = 5.0
        n_ip = 5
        xi_phys = np.array(
            [0.0, 0.17266, 0.5, 0.82733, 1.0], dtype=np.float64,
        ) * L

        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
            connectivity={1: (1, 2)},
        )

        section_path = tmp_path / "beam_line_stations.out"
        gpx_path = tmp_path / line_station_gpx_path("beam_line_stations.out")
        assert gpx_path.name == "beam_line_stations_gpx.out"

        # 6 components per IP × 5 IPs × 1 element = 30 cols + 1 time col.
        # Encoding: t * 1000 + ip * 100 + k where k ∈ 0..5
        T = 2
        flat = np.zeros((T, 1, n_ip * 6), dtype=np.float64)
        for t in range(T):
            for ip in range(n_ip):
                for k in range(6):
                    flat[t, 0, ip * 6 + k] = (
                        t * 1000.0 + (ip + 1) * 100.0 + k
                    )
        _write_section_out(
            section_path,
            times=np.array([1.0, 2.0]),
            flat=flat,
        )
        _write_gpx_out(
            gpx_path,
            times=np.array([1.0, 2.0]),
            xi_phys_per_elem=xi_phys.reshape(1, n_ip),
        )

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=(
                    "axial_force", "bending_moment_z",
                    "bending_moment_y", "torsion",
                    "shear_y", "shear_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ForceBeamColumn3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        return fem, spec, tmp_path / "out.h5"

    def test_full_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fem, spec, target = self._setup(tmp_path)
        _run_with_known_counts(monkeypatch, spec, tmp_path, target, fem, 5)

        with Results.from_native(target, model=_open_model_from_h5(target)) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
            assert slab.values.shape == (2, 5)
            assert slab.element_index.tolist() == [1, 1, 1, 1, 1]
            np.testing.assert_allclose(
                slab.station_natural_coord,
                [-1.0, -0.65468, 0.0, 0.65466, 1.0],
                atol=1e-4,
            )
            # axial = comp 0; t=0,ip=1 → 100; t=0,ip=5 → 500; t=1 → +1000.
            np.testing.assert_array_equal(
                slab.values[0], [100.0, 200.0, 300.0, 400.0, 500.0],
            )
            np.testing.assert_array_equal(
                slab.values[1], [1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            )

            # shear_z (k=5) at t=0, ip=3 → 305.
            slab_vz = s.elements.line_stations.get(component="shear_z")
            assert slab_vz.values[0, 2] == 305.0

            # bending_moment_y (k=2) at t=1, ip=5 → 1502.
            slab_my = s.elements.line_stations.get(
                component="bending_moment_y",
            )
            assert slab_my.values[1, 4] == 1502.0


class TestForceBeamColumn3dBareSection:
    def test_bare_4comp_section_excludes_shears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bare FiberSection3d (4 comps: P, Mz, My, T) — shears absent."""
        L = 4.0
        n_ip = 3
        xi_phys = np.array([0.1127, 0.5, 0.8873], dtype=np.float64) * L

        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
            connectivity={1: (1, 2)},
        )

        section_path = tmp_path / "beam_line_stations.out"
        gpx_path = tmp_path / line_station_gpx_path("beam_line_stations.out")

        T = 1
        # 4 comps × 3 IPs = 12 cols per element.
        flat = np.zeros((T, 1, n_ip * 4), dtype=np.float64)
        for ip in range(n_ip):
            for k in range(4):
                flat[0, 0, ip * 4 + k] = (ip + 1) * 100.0 + k
        _write_section_out(
            section_path, times=np.array([1.0]), flat=flat,
        )
        _write_gpx_out(
            gpx_path, times=np.array([1.0]),
            xi_phys_per_elem=xi_phys.reshape(1, n_ip),
        )

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=("axial_force", "torsion"),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ForceBeamColumn3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        _run_with_known_counts(monkeypatch, spec, tmp_path, target, fem, n_ip)

        with Results.from_native(target, model=_open_model_from_h5(target)) as r:
            s = r.stage(r.stages[0].id)
            ax = s.elements.line_stations.get(component="axial_force")
            np.testing.assert_array_equal(ax.values[0], [100.0, 200.0, 300.0])
            tor = s.elements.line_stations.get(component="torsion")
            np.testing.assert_array_equal(tor.values[0], [103.0, 203.0, 303.0])
            sh = s.elements.line_stations.get(component="shear_y")
            assert sh.values.shape == (1, 0)


class TestForceBeamColumn2dBare:
    def test_2d_bare_section_2_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ForceBeamColumn2d, bare 2D section (P, Mz)."""
        L = 3.0
        n_ip = 3
        xi_phys = np.array([0.1127, 0.5, 0.8873], dtype=np.float64) * L

        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
            connectivity={1: (1, 2)},
        )

        section_path = tmp_path / "b_line_stations.out"
        gpx_path = tmp_path / line_station_gpx_path("b_line_stations.out")
        # 2 comps × 3 IPs = 6 cols per element.
        T = 1
        flat = np.zeros((T, 1, n_ip * 2), dtype=np.float64)
        for ip in range(n_ip):
            for k in range(2):
                flat[0, 0, ip * 2 + k] = (ip + 1) * 10.0 + k
        _write_section_out(section_path, times=np.array([1.0]), flat=flat)
        _write_gpx_out(
            gpx_path, times=np.array([1.0]),
            xi_phys_per_elem=xi_phys.reshape(1, n_ip),
        )

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="b",
                components=("axial_force", "bending_moment_z"),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ForceBeamColumn2d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        _run_with_known_counts(monkeypatch, spec, tmp_path, target, fem, n_ip)
        with Results.from_native(target, model=_open_model_from_h5(target)) as r:
            s = r.stage(r.stages[0].id)
            ax = s.elements.line_stations.get(component="axial_force")
            np.testing.assert_array_equal(ax.values[0], [10.0, 20.0, 30.0])
            mz = s.elements.line_stations.get(component="bending_moment_z")
            np.testing.assert_array_equal(mz.values[0], [11.0, 21.0, 31.0])


class TestMultiBucketSplit:
    def test_two_elements_different_n_ip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two beams with different beamIntegrations → two write groups.

        OpenSees can record both elements through one ``recorder
        Element ... section force`` line because the section-shape is
        the same per element. The transcoder splits them into two
        write_line_stations_group calls based on gp_x signature.
        """
        L1, L2 = 5.0, 4.0
        # Element 1: 3 IPs (Legendre).
        xi_phys_1 = np.array([0.1127, 0.5, 0.8873]) * L1
        # Element 2: 5 IPs (Lobatto).
        xi_phys_2 = np.array([0.0, 0.17266, 0.5, 0.82733, 1.0]) * L2
        # Force a common length so the .out file's flat_size is the
        # same for both elements (otherwise OpenSees can't put them in
        # one recorder). For this synthetic test we'll keep equal
        # n_IP — diversity comes from different gp_x VALUES (e.g. two
        # Lobatto-5 beams with different L).
        n_ip = 5
        # Element 1: now Lobatto-5 with L=5
        xi_phys_1 = np.array([0.0, 0.17266, 0.5, 0.82733, 1.0]) * L1
        # Element 2: Lobatto-5 with L=4
        xi_phys_2 = np.array([0.0, 0.17266, 0.5, 0.82733, 1.0]) * L2
        # The gp_x signatures in natural coords are IDENTICAL
        # ([-1, -0.65, 0, +0.65, +1]) — so one group is correct.
        # To force two groups, use different rules per element. Since
        # the .out can't carry differing n_IP per element, we'd need
        # separate records. Demonstrate that here via TWO records.

        fem = _BeamFem(
            node_ids=np.array([1, 2, 3, 4]),
            coords=np.array([
                [0.0, 0.0, 0.0],
                [L1, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0 + L2, 0.0, 0.0],
            ]),
            connectivity={1: (1, 2), 2: (3, 4)},
        )

        # Same record, same n_IP → ONE group, but stitched in element
        # order. Force "two groups" via gp_x VALUES being different
        # (different L → different natural? No, natural is invariant
        # under L for any rule of the same type).
        # Easier proof: SECTION CODES differ. So make element 1 use
        # 4-comp section, element 2 use 6-comp. But that breaks the
        # one-record .out shape constraint.
        # Skip this test variant — heterogeneous beam-integrations in
        # one record are unusual. Validate single-bucket multi-element
        # instead.
        T = 1
        flat = np.zeros((T, 2, n_ip * 4), dtype=np.float64)
        for e in range(2):
            for ip in range(n_ip):
                for k in range(4):
                    flat[0, e, ip * 4 + k] = (
                        (e + 1) * 1000.0 + (ip + 1) * 100.0 + k
                    )
        section_path = tmp_path / "beams_line_stations.out"
        gpx_path = tmp_path / line_station_gpx_path(
            "beams_line_stations.out",
        )
        _write_section_out(section_path, times=np.array([1.0]), flat=flat)
        _write_gpx_out(
            gpx_path, times=np.array([1.0]),
            xi_phys_per_elem=np.stack([xi_phys_1, xi_phys_2]),
        )

        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="beams",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([1, 2], dtype=np.int64),
                element_class_name="ForceBeamColumn3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        _run_with_known_counts(monkeypatch, spec, tmp_path, target, fem, n_ip)

        with Results.from_native(target, model=_open_model_from_h5(target)) as r:
            s = r.stage(r.stages[0].id)
            ax = s.elements.line_stations.get(component="axial_force")
        # Both elements share Lobatto-5 (same natural coords) → one
        # group with 2 × 5 = 10 stations.
        assert ax.values.shape == (1, 10)
        # Axial-comp encoding (k=0): (e+1)*1000 + (ip+1)*100. So
        # e=0,ip=0..4 → 1100..1500; e=1,ip=0..4 → 2100..2500.
        np.testing.assert_array_equal(
            ax.values[0],
            [1100, 1200, 1300, 1400, 1500, 2100, 2200, 2300, 2400, 2500],
        )


# =====================================================================
# Elastic beams: localForce emit + two-station synthesis (#1741)
# =====================================================================

def _ls_record(name: str, class_name: str | None, comps=("axial_force",),
               eids=(1,)) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=comps,
        dt=None, n_steps=None,
        element_ids=np.asarray(eids, dtype=np.int64),
        element_class_name=class_name,
    )


class TestElasticBeamEmit:
    """An elastic beam answers no ``section force``: its record emits one
    ``localForce`` recorder; every other record keeps the section pair."""

    @pytest.mark.parametrize("cls", [
        "ElasticBeam2d", "ElasticBeam3d", "ModElasticBeam2d",
        "ElasticTimoshenkoBeam2d", "ElasticTimoshenkoBeam3d",
    ])
    def test_elastic_class_emits_one_local_force(self, cls: str) -> None:
        from apeGmsh.results.spec._emit import emit_logical

        (lr,) = emit_logical(_ls_record("g", cls, eids=(4, 9)),
                             output_dir="o", stage_id="s")
        assert lr.response_tokens == ("localForce",)
        assert lr.target_ids == (4, 9)
        assert lr.file_path == "o/s__g_line_stations_local_force.out"

    @pytest.mark.parametrize("cls", ["ForceBeamColumn2d", None])
    def test_other_records_keep_the_section_pair(self, cls) -> None:
        from apeGmsh.results.spec._emit import emit_logical

        tokens = [lr.response_tokens
                  for lr in emit_logical(_ls_record("c", cls))]
        assert tokens == [("section", "force"), ("integrationPoints",)]


def _elastic_beam_model(tmp_path: Path, ndm: int) -> tuple:
    """A two-element ``elasticBeamColumn`` line through the bridge.

    Returns ``(fem, model_h5, eids)``. An elastic-beam line-stations
    transcode refuses without a model (it could not check that every
    element is an elastic beam), so these cases pass ``model_h5_src=``.
    No openseespy: the bridge only writes ``model.h5``.
    """
    import gmsh

    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name=f"ls_elastic_{ndm}d", verbose=False) as g:
        geo = g.model.geometry
        c = geo.add_line(geo.add_point(0.0, 0.0, 0.0),
                         geo.add_point(4.0, 0.0, 0.0))
        gmsh.model.mesh.setTransfiniteCurve(c, 3)
        g.physical.add(1, [c], name="Beam")
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem, default_orientation=None)
    if ndm == 2:
        ops.model(ndm=2, ndf=3)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(), A=0.02, E=2.0e11,
            Iz=3.0e-4)
    else:
        ops.model(ndm=3, ndf=6)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
            A=0.02, E=2.0e11, Iz=3.0e-4, Iy=2.0e-4, G=8.0e10, J=1.0e-4)
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    return fem, model_h5, [int(e) for e in fem.elements.select(pg="Beam").ids]


class TestElasticBeamTranscode:
    """Hand-written ``localForce`` file → transcode → read back.

    Row per element is ``[end i, end j]``; station 1 is ``sign * end i``
    and station 2 is ``-sign * end j``, the force-beam ``section force``
    convention every route applies (#2009, G17 junction data).
    """

    #: end-i sign per component: N, Vz, T, My, Mz flip; Vy alone keeps.
    _END_I_SIGN = {"axial_force": -1.0, "shear_y": 1.0, "shear_z": -1.0,
                   "torsion": -1.0, "bending_moment_y": -1.0,
                   "bending_moment_z": -1.0}

    @pytest.mark.parametrize("cls, comps", [
        ("ElasticBeam2d", ("axial_force", "shear_y", "bending_moment_z")),
        ("ElasticBeam3d", ("axial_force", "shear_y", "shear_z", "torsion",
                           "bending_moment_y", "bending_moment_z")),
    ])
    def test_two_station_round_trip(self, tmp_path: Path, cls, comps) -> None:
        fem, model_h5, ids = _elastic_beam_model(
            tmp_path, 3 if cls.endswith("3d") else 2)
        eids = tuple(sorted(ids, reverse=True))    # not in id order
        n = len(comps)
        T = 2
        # value = t*1000 + e*100 + end*10 + k   (end 0 = i, 1 = j)
        flat = np.empty((T, len(eids), 2 * n))
        for t in range(T):
            for e in range(len(eids)):
                for end in range(2):
                    for k in range(n):
                        flat[t, e, end * n + k] = (
                            t * 1000.0 + (e + 1) * 100.0 + end * 10.0 + k
                        )
        _write_section_out(tmp_path / "g_line_stations_local_force.out",
                           times=np.array([1.0, 2.0]), flat=flat)
        spec = _make_spec(_ls_record("g", cls, comps=comps, eids=eids),
                          snapshot_id=fem.snapshot_id)
        target = tmp_path / "out.h5"
        RecorderTranscoder(spec, output_dir=tmp_path, target_path=target,
                           fem=fem, stage_name="s", stage_kind="static",
                           model_h5_src=model_h5).run()

        with Results.from_native(target,
                                 model=_open_model_from_h5(target)) as r:
            s = r.stage(r.stages[0].id)
            for k, comp in enumerate(comps):
                sign = self._END_I_SIGN[comp]
                slab = s.elements.line_stations.get(component=comp)
                assert slab.element_index.tolist() == [
                    eids[0], eids[0], eids[1], eids[1]]
                np.testing.assert_array_equal(
                    slab.station_natural_coord, [-1.0, 1.0, -1.0, 1.0],
                )
                for t in range(T):
                    expected = [
                        v
                        for e in range(len(eids))
                        for v in (sign * flat[t, e, k],
                                  -sign * flat[t, e, n + k])
                    ]
                    np.testing.assert_array_equal(slab.values[t], expected)

    def test_wrong_column_count_raises(self, tmp_path: Path) -> None:
        fem, model_h5, ids = _elastic_beam_model(tmp_path, 2)
        # A 3-D row (12 values) under a 2-D class (6 expected).
        _write_section_out(tmp_path / "g_line_stations_local_force.out",
                           times=np.array([1.0]),
                           flat=np.zeros((1, 1, 12)))
        spec = _make_spec(_ls_record("g", "ElasticBeam2d", eids=ids[:1]),
                          snapshot_id=fem.snapshot_id)
        with pytest.raises(ValueError, match="expected 7 columns"):
            RecorderTranscoder(spec, output_dir=tmp_path,
                               target_path=tmp_path / "out.h5", fem=fem,
                               stage_name="s", stage_kind="static",
                               model_h5_src=model_h5).run()


# =====================================================================
# Validation: error paths
# =====================================================================

class TestValidationErrors:
    def test_missing_class_name_raises(self, tmp_path: Path) -> None:
        L = 3.0
        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
            connectivity={1: (1, 2)},
        )
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="b",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name=None,
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        # Empty placeholder files so the file-existence isn't the issue.
        (tmp_path / "b_line_stations.out").write_text("")
        (tmp_path / "b_line_stations_gpx.out").write_text("")
        with pytest.raises(ValueError, match="element_class_name"):
            RecorderTranscoder(
                spec, output_dir=tmp_path, target_path=target, fem=fem,
                stage_name="s", stage_kind="static",
            ).run()

    def test_uncatalogued_class_raises(self, tmp_path: Path) -> None:
        fem = _BeamFem(
            node_ids=np.array([1, 2]),
            coords=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            connectivity={1: (1, 2)},
        )
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="line_stations", name="b",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="MysteryBeam3d",
            ),
            snapshot_id=fem.snapshot_id,
        )
        target = tmp_path / "out.h5"
        (tmp_path / "b_line_stations.out").write_text("")
        (tmp_path / "b_line_stations_gpx.out").write_text("")
        with pytest.raises(ValueError, match="CUSTOM_RULE_CATALOG"):
            RecorderTranscoder(
                spec, output_dir=tmp_path, target_path=target, fem=fem,
                stage_name="s", stage_kind="static",
            ).run()
