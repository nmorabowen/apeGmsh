"""Phase 11c Step 2a — mocked MPCO fiber-section reads.

Builds a synthetic MPCO-shaped HDF5 file and exercises the fiber I/O
module + ``MPCOReader.read_fibers`` end-to-end without OpenSees.
Real-MPCO end-to-end coverage (live OpenSees fiber-section runs) is
deferred to a Stage 4 integration test.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_fiber_io as _mfiber
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.solvers._element_response import (
    ELE_TAG_DispBeamColumn3d,
    ELE_TAG_ForceBeamColumn3d,
    IntRule,
)


# =====================================================================
# Synthetic MPCO file builder
# =====================================================================

def _create_skeleton(
    path: Path, *, node_ids: np.ndarray, coords: np.ndarray,
    n_steps: int, dt: float,
) -> tuple["h5py.File", str]:
    f = h5py.File(path, "w")
    info = f.create_group("INFO")
    info.create_dataset("SPATIAL_DIM", data=3)
    info.create_dataset("SOLVER_NAME", data=np.bytes_(b"OpenSees"))
    info.create_dataset("SOLVER_VERSION", data=np.array([3, 7, 2]))

    stage_name = "MODEL_STAGE[1]"
    stage = f.create_group(stage_name)
    stage.attrs["STEP"] = 0
    stage.attrs["TIME"] = 0.0

    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    nodes.create_dataset("COORDINATES", data=coords.astype(np.float64))
    model.create_group("ELEMENTS")
    model.create_group("SECTION_ASSIGNMENTS")

    results = stage.create_group("RESULTS")
    on_nodes = results.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.attrs["DISPLAY_NAME"] = np.bytes_(b"Displacement")
    disp.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    disp_data = disp.create_group("DATA")
    for k in range(n_steps):
        ds = disp_data.create_dataset(
            f"STEP_{k}",
            data=np.zeros((node_ids.size, 3), dtype=np.float64),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * dt

    results.create_group("ON_ELEMENTS")
    return f, stage_name


def _add_connectivity(
    stage_grp: "h5py.Group", *,
    class_tag: int, class_name: str, custom_rule_idx: int,
    element_ids: np.ndarray, gp_x: np.ndarray,
) -> None:
    conn_key = f"{class_tag}-{class_name}[{IntRule.Custom}:{custom_rule_idx}]"
    conn_grp = stage_grp["MODEL/ELEMENTS"]
    conn_ds = conn_grp.create_dataset(
        conn_key,
        data=np.zeros((element_ids.size, 3), dtype=np.int32),
    )
    conn_ds.attrs["GP_X"] = gp_x.astype(np.float64)


def _add_fiber_section(
    stage_grp: "h5py.Group", *,
    section_tag: int, section_class: str,
    element_gp_pairs: list[tuple[int, int]],
    fiber_y: np.ndarray, fiber_z: np.ndarray, fiber_area: np.ndarray,
    fiber_material: np.ndarray,
) -> None:
    sec_grp = stage_grp["MODEL/SECTION_ASSIGNMENTS"].create_group(
        f"SECTION_{section_tag}[{section_class}]",
    )
    sec_grp.create_dataset(
        "ASSIGNMENT",
        data=np.array(element_gp_pairs, dtype=np.int32).reshape(-1, 2),
    )
    fdata = np.column_stack([fiber_y, fiber_z, fiber_area]).astype(np.float64)
    sec_grp.create_dataset("FIBER_DATA", data=fdata)
    sec_grp.create_dataset(
        "FIBER_MATERIALS",
        data=fiber_material.reshape(-1, 1).astype(np.int32),
    )


def _add_fiber_bucket(
    stage_grp: "h5py.Group", *,
    class_tag: int, class_name: str, custom_rule_idx: int,
    element_ids: np.ndarray, n_ip: int, n_fibers: int,
    flat_data: np.ndarray,            # (T, E, n_ip * n_fibers)
    token: str = "section.fiber.stress",
) -> str:
    on_elements = stage_grp["RESULTS/ON_ELEMENTS"]
    token_grp = on_elements.require_group(token)
    bucket_key = (
        f"{class_tag}-{class_name}[{IntRule.Custom}:{custom_rule_idx}:0]"
    )
    bucket = token_grp.create_group(bucket_key)
    num_columns = n_ip * n_fibers
    bucket.attrs["NUM_COLUMNS"] = np.array([num_columns], dtype=np.int32)

    meta = bucket.create_group("META")
    # Fiber sections compress in META by using MULTIPLICITY = n_fibers
    # per IP block (one block per IP, 1 component each).
    meta.create_dataset(
        "MULTIPLICITY",
        data=np.full((n_ip, 1), n_fibers, dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS",
        data=np.arange(n_ip, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset(
        "NUM_COMPONENTS",
        data=np.ones((n_ip, 1), dtype=np.int32),
    )
    meta.create_dataset(
        "COMPONENTS",
        data=np.array([
            ";".join("0.1.2.3.4.stress" for _ in range(n_ip)).encode("ascii"),
        ]),
    )

    bucket.create_dataset(
        "ID", data=element_ids.reshape(-1, 1).astype(np.int32),
    )
    data = bucket.create_group("DATA")
    T = flat_data.shape[0]
    for k in range(T):
        ds = data.create_dataset(
            f"STEP_{k}", data=flat_data[k].astype(np.float64),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * 1.0
    return bucket_key


def _build_unique_flat(
    *, T: int, E: int, n_ip: int, n_fibers: int,
) -> np.ndarray:
    """Each entry encodes (t, e, ip, f) so we can verify packing."""
    flat = np.empty((T, E, n_ip * n_fibers), dtype=np.float64)
    for t in range(T):
        for e in range(E):
            for ip in range(n_ip):
                for f in range(n_fibers):
                    flat[t, e, ip * n_fibers + f] = (
                        t * 1000.0 + e * 100.0 + ip * 10.0 + f
                    )
    return flat


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def simple_fiber_mpco(tmp_path: Path):
    """One bucket with 2 elements × 3 IPs × 4 fibers."""
    path = tmp_path / "fiber_simple.mpco"
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    n_steps = 2
    dt = 0.1
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=n_steps, dt=dt,
    )
    stage = f[stage_name]

    element_ids = np.array([10, 11], dtype=np.int64)
    gp_x = np.array([-0.7745966, 0.0, 0.7745966], dtype=np.float64)
    _add_connectivity(
        stage, class_tag=ELE_TAG_ForceBeamColumn3d,
        class_name="ForceBeamColumn3d", custom_rule_idx=0,
        element_ids=element_ids, gp_x=gp_x,
    )

    fiber_y = np.array([-0.1, 0.1, 0.1, -0.1], dtype=np.float64)
    fiber_z = np.array([-0.05, -0.05, 0.05, 0.05], dtype=np.float64)
    fiber_area = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    fiber_material = np.array([42, 42, 42, 42], dtype=np.int64)
    pairs = [(int(e), gp) for e in element_ids for gp in range(3)]
    _add_fiber_section(
        stage, section_tag=99, section_class="FiberSection3d",
        element_gp_pairs=pairs,
        fiber_y=fiber_y, fiber_z=fiber_z,
        fiber_area=fiber_area, fiber_material=fiber_material,
    )

    flat = _build_unique_flat(T=n_steps, E=2, n_ip=3, n_fibers=4)
    bucket_key = _add_fiber_bucket(
        stage, class_tag=ELE_TAG_ForceBeamColumn3d,
        class_name="ForceBeamColumn3d", custom_rule_idx=0,
        element_ids=element_ids, n_ip=3, n_fibers=4,
        flat_data=flat,
    )

    f.close()
    return {
        "path": path,
        "element_ids": element_ids,
        "gp_x": gp_x,
        "fiber_y": fiber_y,
        "fiber_z": fiber_z,
        "fiber_area": fiber_area,
        "fiber_material": fiber_material,
        "bucket_key": bucket_key,
        "flat": flat,
        "n_steps": n_steps,
    }


# =====================================================================
# Routing
# =====================================================================

class TestCanonicalToFiberToken:
    def test_stress(self) -> None:
        assert (
            _mfiber.canonical_to_fiber_token("fiber_stress")
            == ("section.fiber.stress", "fiber_stress")
        )

    def test_strain(self) -> None:
        assert (
            _mfiber.canonical_to_fiber_token("fiber_strain")
            == ("section.fiber.strain", "fiber_strain")
        )

    def test_non_fiber_returns_none(self) -> None:
        assert _mfiber.canonical_to_fiber_token("stress_xx") is None
        assert _mfiber.canonical_to_fiber_token("displacement_x") is None


# =====================================================================
# SECTION_ASSIGNMENTS lookup
# =====================================================================

class TestFindFiberSection:
    def test_finds_assigned_section(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec = _mfiber.find_fiber_section_for_element(sa, 10, gp_idx=0)
        assert sec.section_tag == 99
        assert sec.section_class == "FiberSection3d"
        assert sec.n_fibers == 4
        np.testing.assert_array_almost_equal(
            sec.fiber_y, simple_fiber_mpco["fiber_y"],
        )
        np.testing.assert_array_almost_equal(
            sec.fiber_area, simple_fiber_mpco["fiber_area"],
        )
        np.testing.assert_array_equal(
            sec.fiber_material_tag, simple_fiber_mpco["fiber_material"],
        )

    def test_unmatched_element_raises(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            with pytest.raises(ValueError, match="No SECTION_ASSIGNMENTS"):
                _mfiber.find_fiber_section_for_element(sa, 999, gp_idx=0)

    def test_match_at_other_gp(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec = _mfiber.find_fiber_section_for_element(sa, 11, gp_idx=2)
        assert sec.n_fibers == 4


# =====================================================================
# Bucket discovery
# =====================================================================

class TestDiscoverFiberBuckets:
    def test_finds_one_bucket(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            mpco_name, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
        assert mpco_name == "section.fiber.stress"
        assert len(buckets) == 1
        assert buckets[0].bracket_key == simple_fiber_mpco["bucket_key"]
        assert buckets[0].fiber_layout.class_tag == ELE_TAG_ForceBeamColumn3d

    def test_strain_token_no_bucket(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_strain",
            )
        # Fixture only writes stress bucket.
        assert buckets == []

    def test_uncatalogued_class_skipped(self, tmp_path: Path) -> None:
        # Unknown beam-column class shouldn't enter the bucket list.
        path = tmp_path / "unknown.mpco"
        f, stage_name = _create_skeleton(
            path,
            node_ids=np.array([1, 2], dtype=np.int64),
            coords=np.zeros((2, 3)), n_steps=1, dt=0.1,
        )
        stage = f[stage_name]
        _add_connectivity(
            stage, class_tag=999, class_name="WeirdBeam",
            custom_rule_idx=0,
            element_ids=np.array([10], dtype=np.int64),
            gp_x=np.array([0.0]),
        )
        _add_fiber_section(
            stage, section_tag=1, section_class="FiberSection3d",
            element_gp_pairs=[(10, 0)],
            fiber_y=np.array([0.0]), fiber_z=np.array([0.0]),
            fiber_area=np.array([1.0]),
            fiber_material=np.array([42], dtype=np.int64),
        )
        flat = np.zeros((1, 1, 1))
        _add_fiber_bucket(
            stage, class_tag=999, class_name="WeirdBeam",
            custom_rule_idx=0,
            element_ids=np.array([10], dtype=np.int64),
            n_ip=1, n_fibers=1, flat_data=flat,
        )
        f.close()

        with h5py.File(path, "r") as g:
            on_elem = g["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
        assert buckets == []


# =====================================================================
# Bucket layout resolution + validation
# =====================================================================

class TestResolveFiberBucketLayout:
    def test_resolves_correctly(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            layout = _mfiber.resolve_fiber_bucket_layout(
                stage["MODEL/ELEMENTS"],
                stage["MODEL/SECTION_ASSIGNMENTS"],
                bucket_grp, bucket,
            )
        assert layout.n_ip == 3
        assert layout.n_fibers == 4
        np.testing.assert_array_almost_equal(
            layout.gp_x, simple_fiber_mpco["gp_x"],
        )

    def test_validate_meta_succeeds(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            layout = _mfiber.resolve_fiber_bucket_layout(
                stage["MODEL/ELEMENTS"],
                stage["MODEL/SECTION_ASSIGNMENTS"],
                bucket_grp, bucket,
            )
            # Should not raise.
            _mfiber.validate_fiber_bucket_meta(
                bucket_grp, layout, bracket_key=bucket.bracket_key,
            )


# =====================================================================
# read_fiber_bucket_slab — column ordering and selection
# =====================================================================

class TestReadFiberBucketSlab:
    def test_full_read_packs_correctly(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            t_idx = np.array([0, 1], dtype=np.int64)
            result = _mfiber.read_fiber_bucket_slab(
                bucket_grp,
                stage["MODEL/ELEMENTS"],
                stage["MODEL/SECTION_ASSIGNMENTS"],
                bucket, t_idx=t_idx,
                element_ids=None, gp_indices=None,
            )

        assert result is not None
        values, ei, gpi, y, z, area, mtag = result

        # 2 steps × (2 elements × 3 IPs × 4 fibers) = (2, 24)
        assert values.shape == (2, 24)
        assert ei.shape == (24,)
        assert gpi.shape == (24,)
        assert y.shape == z.shape == area.shape == mtag.shape == (24,)

        # Verify outer-fastest ordering: element 10 then 11; within each,
        # GP 0..2; within each, fibers 0..3.
        expected_ei = np.repeat([10, 11], 12)
        np.testing.assert_array_equal(ei, expected_ei)

        expected_gpi = np.tile(np.repeat([0, 1, 2], 4), 2)
        np.testing.assert_array_equal(gpi, expected_gpi)

        # Spot-check one value: (t=0, e=0, ip=2, f=3)
        # encoded as 0*1000 + 0*100 + 2*10 + 3 = 23
        # Column for (e=0, ip=2, f=3) is 0*12 + 2*4 + 3 = 11.
        assert values[0, 11] == 23.0

        # Fibers tile per (E × n_ip).
        np.testing.assert_array_almost_equal(
            y[:4], simple_fiber_mpco["fiber_y"],
        )
        np.testing.assert_array_almost_equal(
            y[4:8], simple_fiber_mpco["fiber_y"],
        )

    def test_element_ids_filter(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            t_idx = np.array([0], dtype=np.int64)
            result = _mfiber.read_fiber_bucket_slab(
                bucket_grp,
                stage["MODEL/ELEMENTS"],
                stage["MODEL/SECTION_ASSIGNMENTS"],
                bucket, t_idx=t_idx,
                element_ids=np.array([11]), gp_indices=None,
            )
        assert result is not None
        values, ei, gpi, *_ = result
        # 1 step × (1 element × 3 IPs × 4 fibers) = (1, 12)
        assert values.shape == (1, 12)
        np.testing.assert_array_equal(ei, np.full(12, 11))

    def test_gp_indices_filter(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            t_idx = np.array([0], dtype=np.int64)
            result = _mfiber.read_fiber_bucket_slab(
                bucket_grp,
                stage["MODEL/ELEMENTS"],
                stage["MODEL/SECTION_ASSIGNMENTS"],
                bucket, t_idx=t_idx,
                element_ids=None, gp_indices=np.array([1]),
            )
        assert result is not None
        values, ei, gpi, *_ = result
        # 1 step × (2 elements × 1 GP × 4 fibers) = (1, 8)
        assert values.shape == (1, 8)
        np.testing.assert_array_equal(gpi, np.array([1] * 8))
        # All values should encode ip=1: t=0, e=0/1, ip=1, f=0..3
        # → e=0: 10..13, e=1: 110..113.
        np.testing.assert_array_equal(
            values[0],
            np.array([10, 11, 12, 13, 110, 111, 112, 113], dtype=np.float64),
        )

    def test_gp_indices_out_of_range_raises(self, simple_fiber_mpco) -> None:
        with h5py.File(simple_fiber_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["section.fiber.stress"][bucket.bracket_key]
            with pytest.raises(ValueError, match="out of range"):
                _mfiber.read_fiber_bucket_slab(
                    bucket_grp,
                    stage["MODEL/ELEMENTS"],
                    stage["MODEL/SECTION_ASSIGNMENTS"],
                    bucket, t_idx=np.array([0]),
                    element_ids=None, gp_indices=np.array([5]),
                )


# =====================================================================
# End-to-end via Results facade
# =====================================================================

class TestReadFibersEndToEnd:
    def test_results_elements_fibers_get(self, simple_fiber_mpco) -> None:
        results = Results.from_mpco(str(simple_fiber_mpco["path"]))
        slab = results.elements.fibers.get(component="fiber_stress")
        assert slab.values.shape == (2, 24)
        assert slab.element_index.shape == (24,)
        np.testing.assert_array_almost_equal(
            slab.y[:4], simple_fiber_mpco["fiber_y"],
        )
        results._reader.close()

    def test_available_components_lists_stress_only(
        self, simple_fiber_mpco,
    ) -> None:
        results = Results.from_mpco(str(simple_fiber_mpco["path"]))
        comps = results._reader.available_components(
            results._reader.stages()[0].id, ResultLevel.FIBERS,
        )
        assert comps == ["fiber_stress"]
        results._reader.close()

    def test_filter_via_facade(self, simple_fiber_mpco) -> None:
        results = Results.from_mpco(str(simple_fiber_mpco["path"]))
        slab = results.elements.fibers.get(
            component="fiber_stress",
            ids=[11],
            gp_indices=[0, 2],
        )
        # 1 element × 2 GPs × 4 fibers = 8 columns, T=2.
        assert slab.values.shape == (2, 8)
        np.testing.assert_array_equal(slab.element_index, np.full(8, 11))
        np.testing.assert_array_equal(
            slab.gp_index, np.repeat([0, 2], 4),
        )
        results._reader.close()
