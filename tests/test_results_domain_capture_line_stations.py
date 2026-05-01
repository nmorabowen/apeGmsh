"""Phase 11b Step 2b — line-stations DomainCapture, mocked ops.

Exercises the in-process line-stations capture path for force-/disp-
based beam-columns without spinning up OpenSees. A fake ``ops`` module
provides deterministic ``eleType`` / ``eleResponse`` /  ``eleNodes`` /
``nodeCoord`` so the catalog lookup → resolve_layout_from_gp_x →
unflatten → write_line_stations_group flow can be inspected on disk.

Real-openseespy coverage lives in
``test_results_domain_capture_line_stations_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.capture._domain import (
    DomainCapture,
    _class_dimension,
    _infer_section_codes,
    _normalise_integration_points,
)
from apeGmsh.solvers._element_response import (
    ELE_TAG_ForceBeamColumn3d,
)
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops with line-stations support
# =====================================================================

class _FakeOpsBeams:
    """Stand-in for openseespy that knows about beam-column elements.

    Responses are keyed by (eid, joined_token). For multi-arg
    eleResponse calls (``ops.eleResponse(eid, "section", "1",
    "force")``) the joined token is ``"section.1.force"``.
    """

    def __init__(self) -> None:
        self.ele_class: dict[int, str] = {}
        self.ele_nodes: dict[int, list[int]] = {}
        self.node_coords: dict[int, np.ndarray] = {}
        # (eid, joined_token) → ndarray
        self.ele_response: dict[tuple[int, str], np.ndarray] = {}
        # eids that should raise from eleResponse(integrationPoints) —
        # mimics DispBeamColumn elements.
        self.no_integration_points: set[int] = set()

    def eleType(self, eid: int) -> str:
        return self.ele_class[int(eid)]

    def eleNodes(self, eid: int) -> list[int]:
        return list(self.ele_nodes[int(eid)])

    def nodeCoord(self, node_tag: int) -> list[float]:
        return list(self.node_coords[int(node_tag)])

    def eleResponse(self, eid: int, *args) -> list[float]:
        if (
            len(args) == 1
            and args[0] == "integrationPoints"
            and int(eid) in self.no_integration_points
        ):
            raise RuntimeError(
                f"integrationPoints not available for element {eid}"
            )
        token = ".".join(str(a) for a in args)
        return list(self.ele_response[(int(eid), token)])

    def reactions(self) -> None: ...


# =====================================================================
# Mock fem
# =====================================================================

class _MockFem:
    def __init__(self, node_ids) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        coords = np.zeros((ids.size, 3), dtype=np.float64)
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
        n.create_dataset("ids", data=np.asarray(self.nodes.ids, dtype=np.int64))
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _spec_with(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Helper unit tests
# =====================================================================

class TestClassDimension:
    @pytest.mark.parametrize("class_name, expected_dim", [
        ("ForceBeamColumn2d", 2),
        ("ForceBeamColumn3d", 3),
        ("ForceBeamColumnCBDI2d", 2),
        ("ForceBeamColumnWarping2d", 2),
        ("DispBeamColumn3d", 3),
        ("ElasticForceBeamColumn3D", 3),    # case-insensitive suffix
    ])
    def test_known_classes(self, class_name: str, expected_dim: int) -> None:
        assert _class_dimension(class_name) == expected_dim

    def test_unknown_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer dimension"):
            _class_dimension("SomeOtherElement")


class TestInferSectionCodes:
    @pytest.mark.parametrize("class_name, n_comp, expected_codes", [
        ("ForceBeamColumn2d", 2, (2, 1)),
        ("ForceBeamColumn2d", 3, (2, 1, 3)),
        ("ForceBeamColumn3d", 3, (2, 1, 4)),
        ("ForceBeamColumn3d", 4, (2, 1, 4, 6)),
        ("ForceBeamColumn3d", 5, (2, 1, 4, 6, 3)),
        ("ForceBeamColumn3d", 6, (2, 1, 4, 6, 3, 5)),
    ])
    def test_canonical_aggregations(
        self, class_name: str, n_comp: int, expected_codes: tuple[int, ...],
    ) -> None:
        assert _infer_section_codes(class_name, n_comp) == expected_codes

    def test_non_canonical_n_comp_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer section codes"):
            _infer_section_codes("ForceBeamColumn3d", 7)

    def test_non_canonical_2d_raises(self) -> None:
        with pytest.raises(ValueError):
            _infer_section_codes("ForceBeamColumn2d", 4)


class TestNormaliseIntegrationPoints:
    def test_endpoints_round_trip(self) -> None:
        L = 5.0
        xi_phys = np.array([0.0, L])
        xi_nat = _normalise_integration_points(xi_phys, L)
        np.testing.assert_array_equal(xi_nat, [-1.0, +1.0])

    def test_lobatto5_round_trip(self) -> None:
        # Physical positions for Lobatto-5 on a beam of length 5.
        L = 5.0
        # In [0, 1]: 0, 0.17266, 0.5, 0.82733, 1.0
        xi_norm01 = np.array([0.0, 0.17266, 0.5, 0.82733, 1.0])
        xi_phys = xi_norm01 * L
        xi_nat = _normalise_integration_points(xi_phys, L)
        np.testing.assert_allclose(
            xi_nat, [-1.0, -0.65468, 0.0, 0.65466, 1.0], atol=1e-4,
        )

    def test_zero_length_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _normalise_integration_points(np.array([0.0]), 0.0)


# =====================================================================
# Single-class single-bucket capture
# =====================================================================

class TestForceBeamColumn3dCapture:
    def _build_ops(
        self, eid: int, gp_x_natural: np.ndarray, L: float,
    ) -> _FakeOpsBeams:
        """Set up class, nodes, integrationPoints — but no per-step force yet."""
        ops = _FakeOpsBeams()
        ops.ele_class[eid] = "ForceBeamColumn3d"
        ops.ele_nodes[eid] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([L, 0.0, 0.0])
        # integrationPoints returns physical xi*L = (xi_natural + 1)/2 * L.
        xi_phys = (gp_x_natural + 1.0) * 0.5 * L
        ops.ele_response[(eid, "integrationPoints")] = xi_phys.copy()
        return ops

    def test_aggregated_6comp_3ip_capture(self, tmp_path: Path) -> None:
        """3-IP, aggregated section, 2 steps. Verify shape, names, values."""
        fem = _MockFem([1, 2])
        eid = 7
        gp_x = np.array([-0.7745966, 0.0, +0.7745966])
        L = 5.0
        ops = self._build_ops(eid, gp_x, L)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="beam_forces",
                components=(
                    "axial_force", "bending_moment_z",
                    "bending_moment_y", "torsion",
                    "shear_y", "shear_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("static", kind="static")
            for t in (0.0, 1.0):
                # Per-step section.force values: 6 comps × 3 IPs each.
                # Encoding: t * 1000 + ip * 100 + comp_idx
                for ip in range(1, 4):
                    base = int(t) * 1000 + ip * 100
                    vec = np.array(
                        [base + k for k in range(6)], dtype=np.float64,
                    )
                    ops.ele_response[(eid, f"section.{ip}.force")] = vec
                cap.step(t=t)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
            # 1 element × 3 IPs = 3 stations; 2 steps.
            assert slab.values.shape == (2, 3)
            np.testing.assert_allclose(
                slab.station_natural_coord, gp_x, atol=1e-6,
            )
            assert slab.element_index.tolist() == [eid, eid, eid]
            # axial = comp 0; t=0,ip=1,k=0 → 100; ip=2 → 200; ip=3 → 300.
            np.testing.assert_array_equal(
                slab.values[0], [100.0, 200.0, 300.0],
            )
            np.testing.assert_array_equal(
                slab.values[1], [1100.0, 1200.0, 1300.0],
            )

            # shear_z (k=5) at t=0,ip=2 → 200 + 5 = 205.
            slab_vz = s.elements.line_stations.get(component="shear_z")
            assert slab_vz.values[0, 1] == 205.0
            # bending_moment_y (k=2): t=0,ip=3 → 302.
            slab_my = s.elements.line_stations.get(component="bending_moment_y")
            assert slab_my.values[0, 2] == 302.0

    def test_bare_4comp_section_excludes_shears(self, tmp_path: Path) -> None:
        """Bare 3D section returns 4 components — shear_y/z absent."""
        fem = _MockFem([1, 2])
        eid = 1
        gp_x = np.array([-1.0, +1.0])
        L = 4.0
        ops = self._build_ops(eid, gp_x, L)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=(
                    "axial_force", "bending_moment_z",
                    "bending_moment_y", "torsion",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        for ip in (1, 2):
            ops.ele_response[(eid, f"section.{ip}.force")] = np.array(
                [10.0 * ip, 20.0 * ip, 30.0 * ip, 40.0 * ip],
            )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            ax = s.elements.line_stations.get(component="axial_force")
            np.testing.assert_array_equal(ax.values[0], [10.0, 20.0])
            tor = s.elements.line_stations.get(component="torsion")
            np.testing.assert_array_equal(tor.values[0], [40.0, 80.0])
            # shear_y not in resolved layout for a 4-comp section.
            sh = s.elements.line_stations.get(component="shear_y")
            assert sh.values.shape == (1, 0)

    def test_inconsistent_per_ip_size_raises(self, tmp_path: Path) -> None:
        """If IP 2 returns a different vector size than IP 1, fail loudly."""
        fem = _MockFem([1, 2])
        eid = 1
        gp_x = np.array([-1.0, +1.0])
        L = 4.0
        ops = self._build_ops(eid, gp_x, L)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        # IP 1 has 4 components → infers (P, Mz, My, T)
        ops.ele_response[(eid, "section.1.force")] = np.zeros(4)
        # IP 2 has 5 components — should raise during step().
        ops.ele_response[(eid, "section.2.force")] = np.zeros(5)

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            with pytest.raises(ValueError, match="returned 5 values"):
                cap.step(t=0.0)


# =====================================================================
# Multi-bucket: same class, different gp_x
# =====================================================================

class TestMultiBucketCapture:
    def test_two_force_beams_with_different_n_ip(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2, 3, 4])
        ops = _FakeOpsBeams()
        # Element 7: 3 IPs, L = 5.
        ops.ele_class[7] = "ForceBeamColumn3d"
        ops.ele_nodes[7] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([5.0, 0.0, 0.0])
        gp_x_7 = np.array([-0.7745966, 0.0, +0.7745966])
        ops.ele_response[(7, "integrationPoints")] = (gp_x_7 + 1.0) * 0.5 * 5.0
        # Element 8: 5 IPs, L = 4.
        ops.ele_class[8] = "ForceBeamColumn3d"
        ops.ele_nodes[8] = [3, 4]
        ops.node_coords[3] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[4] = np.array([4.0, 0.0, 0.0])
        gp_x_8 = np.array([-1.0, -0.65465367, 0.0, 0.65465367, 1.0])
        ops.ele_response[(8, "integrationPoints")] = (gp_x_8 + 1.0) * 0.5 * 4.0

        # Both use a 4-comp bare 3D section (P, Mz, My, T).
        for eid, n_ip in ((7, 3), (8, 5)):
            for ip in range(1, n_ip + 1):
                ops.ele_response[(eid, f"section.{ip}.force")] = np.array(
                    [eid * 10.0, eid * 20.0, eid * 30.0, eid * 40.0],
                )

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([7, 8]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        # 3 + 5 = 8 stations.
        assert slab.values.shape == (1, 8)
        # element_index repeats
        assert slab.element_index.tolist() == [7, 7, 7, 8, 8, 8, 8, 8]


# =====================================================================
# Skip behaviours
# =====================================================================

class TestSkipBehaviours:
    def test_disp_beam_skipped_no_integration_points(
        self, tmp_path: Path,
    ) -> None:
        """DispBeamColumn3d-like element: integrationPoints unavailable."""
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[5] = "DispBeamColumn3d"
        ops.ele_nodes[5] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([3.0, 0.0, 0.0])
        ops.no_integration_points.add(5)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([5]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()
            # Skipped element should be tracked.
            (lc,) = cap._line_station_capturers
            assert any(eid == 5 for eid, _ in lc.skipped_elements)

        # Empty native file — read returns empty slab.
        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 0)

    def test_uncatalogued_class_skipped(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[5] = "MysteryBeam2d"
        ops.ele_nodes[5] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([3.0, 0.0, 0.0])

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([5]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()
            (lc,) = cap._line_station_capturers
            assert any(
                "not in CUSTOM_RULE_CATALOG" in reason
                for _, reason in lc.skipped_elements
            )


# =====================================================================
# Mixed: line_stations record alongside gauss record (Phase 11a parity)
# =====================================================================

class TestMixedCategories:
    def test_line_stations_and_gauss_in_same_capture(
        self, tmp_path: Path,
    ) -> None:
        """A capture spec with both gauss and line_stations records works."""
        fem = _MockFem([1, 2, 3, 4])
        ops = _FakeOpsBeams()
        # Beam (line_stations source)
        ops.ele_class[7] = "ForceBeamColumn3d"
        ops.ele_nodes[7] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([2.0, 0.0, 0.0])
        gp_x = np.array([-1.0, +1.0])
        ops.ele_response[(7, "integrationPoints")] = (gp_x + 1.0) * 0.5 * 2.0
        for ip in (1, 2):
            ops.ele_response[(7, f"section.{ip}.force")] = np.array(
                [100.0 * ip, 200.0 * ip, 300.0 * ip, 400.0 * ip],
            )
        # Tet (gauss source)
        ops.ele_class[10] = "FourNodeTetrahedron"
        ops.ele_response[(10, "stresses")] = np.arange(6, dtype=np.float64)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([7]),
            ),
            ResolvedRecorderRecord(
                category="gauss", name="solid",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "mixed.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            ls = s.elements.line_stations.get(component="axial_force")
            assert ls.values.shape == (1, 2)
            np.testing.assert_array_equal(ls.values[0], [100.0, 200.0])
            gs = s.elements.gauss.get(component="stress_xx")
            assert gs.values.shape == (1, 1)
            np.testing.assert_array_equal(gs.values[0], [0.0])


# =====================================================================
# Elastic-beam line_stations synthesis (ops.eleResponse(localForce))
# =====================================================================
#
# Closed-form elastic beams have no integration points; the capturer
# falls back to a 2-station synthesis driven by ``localForce`` end-
# node force vectors. Exercises the same code path that lets
# ``example_buckleUP_v2`` (elasticBeamColumn elements) produce a
# usable line-stations slab.

class TestElasticBeamSynthesis:
    def _build_ops_3d(
        self, eid: int, local_force: np.ndarray, L: float = 2.0,
    ) -> _FakeOpsBeams:
        """ElasticBeam3d at nodes 1, 2 with a fixed localForce vector."""
        ops = _FakeOpsBeams()
        ops.ele_class[eid] = "ElasticBeam3d"
        ops.ele_nodes[eid] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([L, 0.0, 0.0])
        ops.ele_response[(eid, "localForce")] = local_force
        return ops

    def test_elastic_3d_round_trip(self, tmp_path: Path) -> None:
        """A constant-force ElasticBeam3d round-trips through capture."""
        eid = 7
        # 12-DOF localForce: [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i,
        #                     Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j]
        local_force = np.array([
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ])
        fem = _MockFem([1, 2])
        ops = self._build_ops_3d(eid, local_force)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=(
                    "axial_force", "shear_y", "shear_z",
                    "torsion", "bending_moment_y", "bending_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "elastic.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        # Slab at 2 stations × 1 element = 2 column rows per component.
        # Station 1 keeps the localForce sign; station 2 flips.
        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            for comp, dof_idx in [
                ("axial_force",       0),  # Fx
                ("shear_y",           1),  # Fy
                ("shear_z",           2),  # Fz
                ("torsion",           3),  # Mx
                ("bending_moment_y",  4),  # My
                ("bending_moment_z",  5),  # Mz
            ]:
                slab = s.elements.line_stations.get(component=comp)
                assert slab.values.shape == (1, 2)
                expected_station_1 = local_force[dof_idx]
                expected_station_2 = -local_force[6 + dof_idx]
                np.testing.assert_allclose(
                    slab.values[0],
                    [expected_station_1, expected_station_2],
                    err_msg=f"component {comp!r}",
                )
                # Stations sit at ξ ∈ {-1, +1}.
                np.testing.assert_allclose(
                    slab.station_natural_coord, [-1.0, 1.0],
                )

    def test_elastic_2d_round_trip(self, tmp_path: Path) -> None:
        """ElasticBeam2d uses the 3-DOF-per-node layout."""
        eid = 11
        # 6-DOF localForce: [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
        local_force = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[eid] = "ElasticBeam2d"
        ops.ele_nodes[eid] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([2.0, 0.0, 0.0])
        ops.ele_response[(eid, "localForce")] = local_force

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force", "shear_y", "bending_moment_z"),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "elastic2d.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            for comp, dof_idx in [
                ("axial_force",       0),
                ("shear_y",           1),
                ("bending_moment_z",  2),
            ]:
                slab = s.elements.line_stations.get(component=comp)
                np.testing.assert_allclose(
                    slab.values[0],
                    [local_force[dof_idx], -local_force[3 + dof_idx]],
                    err_msg=f"component {comp!r}",
                )

    def test_skipped_element_warning_fires(self, tmp_path: Path) -> None:
        """Disp-beam-style skip emits a UserWarning at end_stage()."""
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[5] = "DispBeamColumn3d"
        ops.ele_nodes[5] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([1.0, 0.0, 0.0])
        ops.no_integration_points.add(5)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([5]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "skip.h5"
        with pytest.warns(UserWarning, match="DomainCapture dropped"):
            with DomainCapture(spec, path, fem, ops=ops) as cap:
                cap.begin_stage("g", kind="static")
                cap.step(t=0.0)
                cap.end_stage()

    def test_no_warning_when_nothing_skipped(self, tmp_path: Path) -> None:
        """Clean force-based recording must not emit the skip warning."""
        eid = 7
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[eid] = "ForceBeamColumn3d"
        ops.ele_nodes[eid] = [1, 2]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([2.0, 0.0, 0.0])
        gp_x = np.array([-1.0, +1.0])
        ops.ele_response[(eid, "integrationPoints")] = (
            (gp_x + 1.0) * 0.5 * 2.0
        )
        for ip in (1, 2):
            ops.ele_response[(eid, f"section.{ip}.force")] = np.array(
                [10.0, 20.0, 30.0, 40.0],
            )

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="line_stations", name="r",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([eid]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        import warnings
        path = tmp_path / "clean.h5"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with DomainCapture(spec, path, fem, ops=ops) as cap:
                cap.begin_stage("g", kind="static")
                cap.step(t=0.0)
                cap.end_stage()
