"""Mock-based tests for the ZeroLength spring result reader.

All tests use synthetic HDF5 files written to ``tmp_path``.
No external fixture files are required.  The mock MPCO structure mirrors what
STKO writes for ZeroLength elements:

- ``ON_ELEMENTS/basicForce/<19-ZeroLength[1:0:0]>/`` — per-spring force bucket
- ``ON_ELEMENTS/deformation/<19-ZeroLength[1:0:0]>/`` — per-spring deformation bucket

Each bucket's META has a single block with ``NUM_COMPONENTS = N`` (N springs),
``GAUSS_IDS = [-1]`` (the recorder uses ``-1`` for "no Gauss point" since
the element itself is a point), ``MULTIPLICITY = [1]``.  DATA rows have
shape ``(n_elements, N)`` per step.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._mpco_spring_io import (
    canonical_to_spring_token,
    discover_spring_buckets,
    read_spring_bucket_slab,
    resolve_n_springs,
    spring_canonicals_in_bucket,
)
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.results import Results


# =====================================================================
# MPCO fixture builders
# =====================================================================

def _make_meta(grp: "h5py.Group", n_springs: int) -> None:
    """Write a ZeroLength-style META block for ``n_springs`` springs."""
    meta = grp.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int32))
    meta.create_dataset("GAUSS_IDS",    data=np.array([[-1]], dtype=np.int32))
    meta.create_dataset("NUM_COMPONENTS", data=np.array([[n_springs]], dtype=np.int32))
    # COMPONENTS: "0.F_0,F_1,...,F_{N-1}"
    comp_str = "0." + ",".join(f"F_{i}" for i in range(n_springs))
    meta.create_dataset("COMPONENTS", data=np.array([comp_str.encode()]))
    grp.attrs["NUM_COLUMNS"] = n_springs


def _make_spring_mpco(
    path: Path,
    *,
    element_ids: list[int],
    n_springs: int,
    force_values: np.ndarray,      # (n_steps, n_elements, n_springs)
    deform_values: np.ndarray | None = None,  # same shape or None
    class_name: str = "ZeroLength",
    class_tag: int = 19,
) -> Path:
    """Write a minimal MPCO file with ZeroLength buckets to ``path``."""
    f = h5py.File(str(path), "w")
    stage = f.create_group("MODEL_STAGE[1]")

    # Minimal MODEL
    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset("ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1))
    nodes.create_dataset("COORDINATES", data=np.zeros((len(element_ids), 3)))
    model.create_group("ELEMENTS")
    model.create_group("SECTION_ASSIGNMENTS")

    # TIME — stored as attributes on ON_NODES/DISPLACEMENT/DATA/STEP_<k>
    results = stage.create_group("RESULTS")
    n_steps = force_values.shape[0]
    time_arr = np.linspace(0.0, 1.0, n_steps)
    # Add a dummy ON_NODES group so MPCOReader._build_time_vector_for_mpco_stage
    # can find the step count and time values.
    on_nodes = results.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.create_dataset("ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1))
    n_nodes = len(element_ids)
    data_grp_nd = disp.create_group("DATA")
    for i in range(n_steps):
        ds = data_grp_nd.create_dataset(
            f"STEP_{i}", data=np.zeros((n_nodes, 3), dtype=np.float64),
        )
        ds.attrs["TIME"] = time_arr[i]

    # ON_ELEMENTS
    on_elem = results.create_group("ON_ELEMENTS")
    bracket = f"{class_tag}-{class_name}[1:0:0]"

    def _write_bucket(token_name: str, data: np.ndarray) -> None:
        token_grp = on_elem.require_group(token_name)
        bkt = token_grp.create_group(bracket)
        _make_meta(bkt, n_springs)
        bkt.create_dataset(
            "ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1),
        )
        data_grp = bkt.create_group("DATA")
        for i in range(n_steps):
            data_grp.create_dataset(f"STEP_{i}", data=data[i].astype(np.float64))

    _write_bucket("basicForce", force_values)
    if deform_values is not None:
        _write_bucket("deformation", deform_values)

    f.close()
    return path


# =====================================================================
# Routing helpers
# =====================================================================

class TestRouting:
    def test_spring_force_root_routes_to_basicForce(self) -> None:
        assert canonical_to_spring_token("spring_force") == ("basicForce", "spring_force")

    def test_spring_force_indexed_routes_to_basicForce(self) -> None:
        assert canonical_to_spring_token("spring_force_0") == ("basicForce", "spring_force")

    def test_spring_force_index_3_routes_to_basicForce(self) -> None:
        assert canonical_to_spring_token("spring_force_3") == ("basicForce", "spring_force")

    def test_spring_deformation_root_routes(self) -> None:
        assert canonical_to_spring_token("spring_deformation") == (
            "deformation", "spring_deformation",
        )

    def test_spring_deformation_indexed_routes(self) -> None:
        assert canonical_to_spring_token("spring_deformation_2") == (
            "deformation", "spring_deformation",
        )

    def test_non_spring_canonical_returns_none(self) -> None:
        assert canonical_to_spring_token("stress_xx") is None
        assert canonical_to_spring_token("fiber_stress_0") is None
        assert canonical_to_spring_token("force_x") is None


# =====================================================================
# Discovery
# =====================================================================

class TestDiscovery:
    def test_discovers_zerolength_force_bucket(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 3, 2, 4
        vals = np.random.randn(n_steps, n_el, n_sp)
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2, 3], n_springs=n_sp, force_values=vals,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            grp_name, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force_0",
            )
        assert grp_name == "basicForce"
        assert len(buckets) == 1
        assert buckets[0].elem_key.class_name == "ZeroLength"

    def test_discovers_deformation_bucket(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 1, 2
        fv = np.random.randn(n_steps, n_el, n_sp)
        dv = np.random.randn(n_steps, n_el, n_sp)
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[10, 20], n_springs=n_sp,
            force_values=fv, deform_values=dv,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            grp_name, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_deformation",
            )
        assert grp_name == "deformation"
        assert len(buckets) == 1

    def test_no_bucket_for_non_spring_canonical(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 1, 1, 1
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            grp_name, buckets = discover_spring_buckets(
                on_elem, canonical_component="stress_xx",
            )
        assert grp_name is None
        assert buckets == []

    def test_missing_group_returns_empty(self, tmp_path) -> None:
        """Request deformation when only force bucket exists."""
        n_el, n_sp, n_steps = 2, 2, 2
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            grp_name, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_deformation",
            )
        assert grp_name == "deformation"
        assert buckets == []


# =====================================================================
# META resolution
# =====================================================================

class TestMetaResolution:
    def test_resolve_n_springs_single(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 1, 2
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            n = resolve_n_springs(bkt_grp, buckets[0])
        assert n == 1

    def test_resolve_n_springs_multi(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 3, 5, 2
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[10, 20, 30], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            n = resolve_n_springs(bkt_grp, buckets[0])
        assert n == 5

    def test_spring_canonicals_single(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 1, 1, 1
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            canonicals = spring_canonicals_in_bucket(bkt_grp, buckets[0])
        # 1-spring → bare root name only
        assert canonicals == ("spring_force",)

    def test_spring_canonicals_multi(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 3, 1
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            canonicals = spring_canonicals_in_bucket(bkt_grp, buckets[0])
        assert canonicals == ("spring_force_0", "spring_force_1", "spring_force_2")


# =====================================================================
# Slab reading
# =====================================================================

class TestSlabRead:
    def test_single_spring_all_elements(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 3, 1, 4
        rng = np.random.default_rng(42)
        vals = rng.standard_normal((n_steps, n_el, n_sp))
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[10, 20, 30], n_springs=n_sp, force_values=vals,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            result = read_spring_bucket_slab(
                bkt_grp, buckets[0], "spring_force",
                t_idx=np.arange(n_steps, dtype=np.int64),
                element_ids=None,
            )
        assert result is not None
        read_vals, read_ids = result
        assert read_vals.shape == (n_steps, n_el)
        np.testing.assert_array_equal(read_ids, [10, 20, 30])
        # Column 0 of the raw data is spring 0
        np.testing.assert_array_almost_equal(read_vals, vals[:, :, 0])

    def test_indexed_spring_selects_correct_column(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 3, 2
        vals = np.arange(n_steps * n_el * n_sp, dtype=np.float64).reshape(
            n_steps, n_el, n_sp,
        )
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp, force_values=vals,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force_2",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            result = read_spring_bucket_slab(
                bkt_grp, buckets[0], "spring_force_2",
                t_idx=np.arange(n_steps, dtype=np.int64),
                element_ids=None,
            )
        assert result is not None
        read_vals, _ = result
        # Spring index 2 is column 2 in the raw data
        np.testing.assert_array_almost_equal(read_vals, vals[:, :, 2])

    def test_out_of_range_spring_index_returns_none(self, tmp_path) -> None:
        """Requesting spring_force_5 when the bucket has only 3 springs → None."""
        n_el, n_sp, n_steps = 2, 3, 1
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force_5",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            result = read_spring_bucket_slab(
                bkt_grp, buckets[0], "spring_force_5",
                t_idx=np.arange(n_steps, dtype=np.int64),
                element_ids=None,
            )
        assert result is None

    def test_element_id_filter(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 4, 2, 3
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((n_steps, n_el, n_sp))
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[10, 20, 30, 40], n_springs=n_sp, force_values=vals,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force_1",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            result = read_spring_bucket_slab(
                bkt_grp, buckets[0], "spring_force_1",
                t_idx=np.arange(n_steps, dtype=np.int64),
                element_ids=np.array([10, 30], dtype=np.int64),
            )
        assert result is not None
        read_vals, read_ids = result
        assert read_vals.shape == (n_steps, 2)
        np.testing.assert_array_equal(read_ids, [10, 30])
        np.testing.assert_array_almost_equal(read_vals, vals[:, [0, 2], 1])

    def test_single_timestep(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 3, 2, 5
        rng = np.random.default_rng(0)
        vals = rng.standard_normal((n_steps, n_el, n_sp))
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2, 3], n_springs=n_sp, force_values=vals,
        )
        with h5py.File(str(path), "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = discover_spring_buckets(
                on_elem, canonical_component="spring_force_0",
            )
            bkt_grp = on_elem["basicForce"][buckets[0].bracket_key]
            t_idx = np.array([2], dtype=np.int64)  # step 2 only
            result = read_spring_bucket_slab(
                bkt_grp, buckets[0], "spring_force_0",
                t_idx=t_idx, element_ids=None,
            )
        assert result is not None
        read_vals, _ = result
        assert read_vals.shape == (1, n_el)
        np.testing.assert_array_almost_equal(read_vals[0], vals[2, :, 0])


# =====================================================================
# End-to-end via Results API
# =====================================================================

class TestResultsAPI:
    """Verify that Results.elements.springs.get() works end-to-end."""

    def test_available_components_spring_force(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 3, 2
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        r = Results.from_mpco(str(path))
        sid = r._reader.stages()[0].id
        comps = r._reader.available_components(sid, ResultLevel.SPRINGS)
        r._reader.close()
        assert "spring_force_0" in comps
        assert "spring_force_1" in comps
        assert "spring_force_2" in comps
        assert "spring_force_3" not in comps

    def test_get_spring_force_shape(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 3, 2, 4
        rng = np.random.default_rng(99)
        vals = rng.standard_normal((n_steps, n_el, n_sp))
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2, 3], n_springs=n_sp, force_values=vals,
        )
        r = Results.from_mpco(str(path))
        slab = r.elements.springs.get(component="spring_force_0")
        r._reader.close()
        assert slab.values.shape == (n_steps, n_el)
        assert slab.element_index.shape == (n_el,)

    def test_get_spring_force_values_correct(self, tmp_path) -> None:
        n_el, n_sp, n_steps = 2, 2, 3
        rng = np.random.default_rng(55)
        vals = rng.standard_normal((n_steps, n_el, n_sp))
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp, force_values=vals,
        )
        r = Results.from_mpco(str(path))
        slab = r.elements.springs.get(component="spring_force_1")
        r._reader.close()
        np.testing.assert_array_almost_equal(slab.values, vals[:, :, 1])

    def test_get_single_spring_bare_canonical(self, tmp_path) -> None:
        """1-spring elements use bare 'spring_force' canonical."""
        n_el, n_sp, n_steps = 2, 1, 2
        vals = np.ones((n_steps, n_el, n_sp)) * 5.0
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp, force_values=vals,
        )
        r = Results.from_mpco(str(path))
        slab = r.elements.springs.get(component="spring_force")
        r._reader.close()
        assert slab.values.shape == (n_steps, n_el)
        np.testing.assert_array_almost_equal(slab.values, 5.0)

    def test_empty_slab_when_component_absent(self, tmp_path) -> None:
        """Requesting spring_force_5 when only 2 springs → empty slab."""
        n_el, n_sp, n_steps = 2, 2, 2
        path = _make_spring_mpco(
            tmp_path / "s.mpco",
            element_ids=[1, 2], n_springs=n_sp,
            force_values=np.zeros((n_steps, n_el, n_sp)),
        )
        r = Results.from_mpco(str(path))
        slab = r.elements.springs.get(component="spring_force_5")
        r._reader.close()
        assert slab.values.shape == (n_steps, 0)
        assert slab.element_index.size == 0
