"""Phase 11c Step 2b — mocked MPCO layered-shell reads.

Builds a synthetic MPCO-shaped HDF5 file and exercises the layer I/O
module + ``MPCOReader.read_layers`` end-to-end without OpenSees.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_layer_io as _mlayer
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.solvers._element_response import (
    ELE_TAG_ASDShellQ4,
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


def _add_shell_connectivity(
    stage_grp: "h5py.Group", *,
    class_tag: int, class_name: str, surface_int_rule: int,
    custom_rule_idx: int,
    element_ids: np.ndarray,
) -> None:
    conn_key = f"{class_tag}-{class_name}[{surface_int_rule}:{custom_rule_idx}]"
    conn_grp = stage_grp["MODEL/ELEMENTS"]
    conn_grp.create_dataset(
        conn_key,
        data=np.zeros((element_ids.size, 5), dtype=np.int32),
    )


def _add_local_axes(
    stage_grp: "h5py.Group", *,
    class_tag: int, class_name: str, surface_int_rule: int,
    custom_rule_idx: int,
    element_ids: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    la_grp = stage_grp["MODEL"].require_group("LOCAL_AXES")
    key = f"{class_tag}-{class_name}[{surface_int_rule}:{custom_rule_idx}]"
    g = la_grp.create_group(key)
    g.create_dataset("ID", data=element_ids.reshape(-1, 1).astype(np.int32))
    g.create_dataset("QUATERNIONS", data=quaternions.astype(np.float64))


def _add_layered_section(
    stage_grp: "h5py.Group", *,
    section_tag: int, section_class: str,
    element_sgp_pairs: list[tuple[int, int]],
    layer_thickness: np.ndarray,
    layer_material: np.ndarray,
) -> None:
    sec_grp = stage_grp["MODEL/SECTION_ASSIGNMENTS"].create_group(
        f"SECTION_{section_tag}[{section_class}]",
    )
    sec_grp.create_dataset(
        "ASSIGNMENT",
        data=np.array(element_sgp_pairs, dtype=np.int32).reshape(-1, 2),
    )
    n_layers = layer_thickness.size
    fdata = np.zeros((n_layers, 3), dtype=np.float64)
    fdata[:, 2] = layer_thickness
    sec_grp.create_dataset("FIBER_DATA", data=fdata)
    sec_grp.create_dataset(
        "FIBER_MATERIALS",
        data=layer_material.reshape(-1, 1).astype(np.int32),
    )


def _add_layer_bucket(
    stage_grp: "h5py.Group", *,
    class_tag: int, class_name: str, surface_int_rule: int,
    custom_rule_idx: int,
    element_ids: np.ndarray,
    n_sgp: int, n_layers: int,
    flat_data: np.ndarray,             # (T, E, n_sgp * n_layers)
    token: str = "material.fiber.stress",
) -> str:
    on_elements = stage_grp["RESULTS/ON_ELEMENTS"]
    token_grp = on_elements.require_group(token)
    bucket_key = (
        f"{class_tag}-{class_name}[{surface_int_rule}:{custom_rule_idx}:0]"
    )
    bucket = token_grp.create_group(bucket_key)
    num_columns = n_sgp * n_layers
    bucket.attrs["NUM_COLUMNS"] = np.array([num_columns], dtype=np.int32)

    meta = bucket.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY",
        data=np.full((n_sgp, 1), n_layers, dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS",
        data=np.arange(n_sgp, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset(
        "NUM_COMPONENTS",
        data=np.ones((n_sgp, 1), dtype=np.int32),
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


def _build_unique_layer_flat(
    *, T: int, E: int, n_sgp: int, n_layers: int,
) -> np.ndarray:
    flat = np.empty((T, E, n_sgp * n_layers), dtype=np.float64)
    for t in range(T):
        for e in range(E):
            for sgp in range(n_sgp):
                for lyr in range(n_layers):
                    flat[t, e, sgp * n_layers + lyr] = (
                        t * 10000.0 + e * 1000.0 + sgp * 100.0 + lyr
                    )
    return flat


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def simple_layer_mpco(tmp_path: Path):
    """ASDShellQ4 with 2 elements × 4 surface GPs × 3 layers."""
    path = tmp_path / "layer_simple.mpco"
    node_ids = np.arange(1, 9, dtype=np.int64)
    coords = np.zeros((8, 3), dtype=np.float64)
    n_steps = 2
    dt = 0.1
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=n_steps, dt=dt,
    )
    stage = f[stage_name]

    element_ids = np.array([20, 21], dtype=np.int64)
    surface_rule = IntRule.Quad_GL_2
    _add_shell_connectivity(
        stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
        surface_int_rule=surface_rule, custom_rule_idx=0,
        element_ids=element_ids,
    )

    quats = np.array([
        [1.0, 0.0, 0.0, 0.0],   # element 20: identity
        [0.7071, 0.0, 0.7071, 0.0],  # element 21: 90° about y
    ], dtype=np.float64)
    _add_local_axes(
        stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
        surface_int_rule=surface_rule, custom_rule_idx=0,
        element_ids=element_ids, quaternions=quats,
    )

    thickness = np.array([0.01, 0.02, 0.01], dtype=np.float64)  # 3 layers
    materials = np.array([7, 7, 7], dtype=np.int64)
    pairs = [(int(e), sgp) for e in element_ids for sgp in range(4)]
    _add_layered_section(
        stage, section_tag=5, section_class="LayeredShellFiberSection",
        element_sgp_pairs=pairs,
        layer_thickness=thickness, layer_material=materials,
    )

    flat = _build_unique_layer_flat(T=n_steps, E=2, n_sgp=4, n_layers=3)
    bucket_key = _add_layer_bucket(
        stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
        surface_int_rule=surface_rule, custom_rule_idx=0,
        element_ids=element_ids, n_sgp=4, n_layers=3, flat_data=flat,
    )

    f.close()
    return {
        "path": path,
        "element_ids": element_ids,
        "thickness": thickness,
        "materials": materials,
        "quats": quats,
        "bucket_key": bucket_key,
        "flat": flat,
    }


# =====================================================================
# Routing
# =====================================================================

class TestCanonicalToLayerToken:
    def test_stress_keyword_swap(self) -> None:
        # Shells use ``material.fiber.*`` not ``section.fiber.*``.
        assert (
            _mlayer.canonical_to_layer_token("fiber_stress")
            == ("material.fiber.stress", "fiber_stress")
        )

    def test_strain(self) -> None:
        assert (
            _mlayer.canonical_to_layer_token("fiber_strain")
            == ("material.fiber.strain", "fiber_strain")
        )

    def test_non_layer_returns_none(self) -> None:
        assert _mlayer.canonical_to_layer_token("stress_xx") is None


# =====================================================================
# SECTION_ASSIGNMENTS lookup
# =====================================================================

class TestFindLayeredSection:
    def test_finds_layers(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec = _mlayer.find_layered_section_for_element(
                sa, 20, surface_gp_idx=0,
            )
        assert sec.section_tag == 5
        assert sec.section_class == "LayeredShellFiberSection"
        assert sec.n_layers == 3
        np.testing.assert_array_almost_equal(
            sec.layer_thickness, simple_layer_mpco["thickness"],
        )

    def test_unmatched_raises(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            with pytest.raises(ValueError, match="No SECTION_ASSIGNMENTS"):
                _mlayer.find_layered_section_for_element(
                    sa, 999, surface_gp_idx=0,
                )


# =====================================================================
# LOCAL_AXES lookup
# =====================================================================

class TestReadLocalAxesQuaternions:
    def test_reads_per_element_quaternions(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            quats = _mlayer.read_local_axes_quaternions(
                stage["MODEL/LOCAL_AXES"],
                bucket.elem_key,
                np.array([20, 21]),
            )
        np.testing.assert_array_almost_equal(quats, simple_layer_mpco["quats"])

    def test_missing_local_axes_returns_identity(self, tmp_path: Path) -> None:
        # File without LOCAL_AXES — fallback to identity.
        path = tmp_path / "no_axes.mpco"
        f, stage_name = _create_skeleton(
            path,
            node_ids=np.arange(1, 5, dtype=np.int64),
            coords=np.zeros((4, 3)), n_steps=1, dt=0.1,
        )
        stage = f[stage_name]
        _add_shell_connectivity(
            stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
            surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
            element_ids=np.array([20], dtype=np.int64),
        )
        _add_layered_section(
            stage, section_tag=1, section_class="LayeredShellFiberSection",
            element_sgp_pairs=[(20, sgp) for sgp in range(4)],
            layer_thickness=np.array([0.01]),
            layer_material=np.array([1], dtype=np.int64),
        )
        flat = np.zeros((1, 1, 4))
        _add_layer_bucket(
            stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
            surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
            element_ids=np.array([20]), n_sgp=4, n_layers=1, flat_data=flat,
        )
        f.close()

        with h5py.File(path, "r") as g:
            stage = g["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            quats = _mlayer.read_local_axes_quaternions(
                None, bucket.elem_key, np.array([20]),
            )
        np.testing.assert_array_almost_equal(
            quats, np.array([[1.0, 0.0, 0.0, 0.0]]),
        )


# =====================================================================
# Bucket discovery
# =====================================================================

class TestDiscoverLayerBuckets:
    def test_finds_bucket(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            mpco_name, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
        assert mpco_name == "material.fiber.stress"
        assert len(buckets) == 1
        assert buckets[0].layer_layout.class_tag == ELE_TAG_ASDShellQ4
        assert buckets[0].layer_layout.surface_int_rule == IntRule.Quad_GL_2

    def test_skips_uncatalogued_class(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown_shell.mpco"
        f, stage_name = _create_skeleton(
            path,
            node_ids=np.arange(1, 5, dtype=np.int64),
            coords=np.zeros((4, 3)), n_steps=1, dt=0.1,
        )
        stage = f[stage_name]
        _add_shell_connectivity(
            stage, class_tag=999, class_name="WeirdShell",
            surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
            element_ids=np.array([20]),
        )
        _add_layered_section(
            stage, section_tag=1, section_class="LayeredShellFiberSection",
            element_sgp_pairs=[(20, sgp) for sgp in range(4)],
            layer_thickness=np.array([0.01]),
            layer_material=np.array([1], dtype=np.int64),
        )
        _add_layer_bucket(
            stage, class_tag=999, class_name="WeirdShell",
            surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
            element_ids=np.array([20]), n_sgp=4, n_layers=1,
            flat_data=np.zeros((1, 1, 4)),
        )
        f.close()
        with h5py.File(path, "r") as g:
            on_elem = g["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
        assert buckets == []


# =====================================================================
# Slab read
# =====================================================================

class TestReadLayerBucketSlab:
    def test_full_packing(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["material.fiber.stress"][bucket.bracket_key]
            t_idx = np.array([0, 1], dtype=np.int64)
            result = _mlayer.read_layer_bucket_slab(
                bucket_grp,
                stage["MODEL/SECTION_ASSIGNMENTS"],
                stage["MODEL/LOCAL_AXES"],
                bucket, "fiber_stress", t_idx=t_idx,
                element_ids=None, gp_indices=None, layer_indices=None,
            )

        assert result is not None
        (values, ei, gpi, lyri, subi, thick, quat) = result

        # 2 steps × (2 elements × 4 sgp × 3 layers × 1 sub_gp) = (2, 24)
        assert values.shape == (2, 24)
        assert ei.shape == (24,)
        np.testing.assert_array_equal(ei, np.repeat([20, 21], 12))

        # gp_index cycles 0..3, each repeated 3 times (one per layer).
        np.testing.assert_array_equal(
            gpi, np.tile(np.repeat([0, 1, 2, 3], 3), 2),
        )
        # layer_index cycles 0..2 within each surface GP.
        np.testing.assert_array_equal(
            lyri, np.tile([0, 1, 2], 8),
        )
        # sub_gp_index always 0 (v1).
        np.testing.assert_array_equal(subi, np.zeros(24, dtype=np.int64))

        # Thickness: per layer, broadcast.
        np.testing.assert_array_almost_equal(
            thick[:3], simple_layer_mpco["thickness"],
        )

        # Quaternions: per element, broadcast across (sgp × lyr × sub).
        np.testing.assert_array_almost_equal(
            quat[:12], np.tile([1.0, 0.0, 0.0, 0.0], (12, 1)),
        )
        np.testing.assert_array_almost_equal(
            quat[12:], np.tile([0.7071, 0.0, 0.7071, 0.0], (12, 1)),
        )

        # Spot value: t=0, e=0, sgp=2, lyr=1 → 0*10000 + 0*1000 + 2*100 + 1 = 201
        # Column for (e=0, sgp=2, lyr=1) is 0*12 + 2*3 + 1 = 7
        assert values[0, 7] == 201.0

    def test_layer_indices_filter(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["material.fiber.stress"][bucket.bracket_key]
            result = _mlayer.read_layer_bucket_slab(
                bucket_grp,
                stage["MODEL/SECTION_ASSIGNMENTS"],
                stage["MODEL/LOCAL_AXES"],
                bucket, "fiber_stress", t_idx=np.array([0]),
                element_ids=None, gp_indices=None,
                layer_indices=np.array([0, 2]),  # skip the middle layer
            )
        assert result is not None
        (values, ei, gpi, lyri, _, thick, _) = result
        # 1 step × (2 elements × 4 sgp × 2 layers × 1 sub) = (1, 16)
        assert values.shape == (1, 16)
        np.testing.assert_array_equal(
            lyri, np.tile([0, 2], 8),
        )
        # Middle thickness (0.02) should NOT appear.
        assert 0.02 not in thick.tolist()

    def test_layer_index_out_of_range_raises(self, simple_layer_mpco) -> None:
        with h5py.File(simple_layer_mpco["path"], "r") as f:
            stage = f["MODEL_STAGE[1]"]
            on_elem = stage["RESULTS/ON_ELEMENTS"]
            _, buckets = _mlayer.discover_layer_buckets(
                on_elem, canonical_component="fiber_stress",
            )
            bucket = buckets[0]
            bucket_grp = on_elem["material.fiber.stress"][bucket.bracket_key]
            with pytest.raises(ValueError, match="out of range"):
                _mlayer.read_layer_bucket_slab(
                    bucket_grp,
                    stage["MODEL/SECTION_ASSIGNMENTS"],
                    stage["MODEL/LOCAL_AXES"],
                    bucket, "fiber_stress", t_idx=np.array([0]),
                    element_ids=None, gp_indices=None,
                    layer_indices=np.array([5]),
                )


# =====================================================================
# End-to-end via Results facade
# =====================================================================

class TestReadLayersEndToEnd:
    def test_results_elements_layers_get(self, simple_layer_mpco) -> None:
        results = Results.from_mpco(str(simple_layer_mpco["path"]))
        slab = results.elements.layers.get(component="fiber_stress")
        assert slab.values.shape == (2, 24)
        assert slab.local_axes_quaternion.shape == (24, 4)
        results._reader.close()

    def test_available_components(self, simple_layer_mpco) -> None:
        results = Results.from_mpco(str(simple_layer_mpco["path"]))
        comps = results._reader.available_components(
            results._reader.stages()[0].id, ResultLevel.LAYERS,
        )
        assert comps == ["fiber_stress"]
        results._reader.close()
