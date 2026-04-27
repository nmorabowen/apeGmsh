"""Phase 11a Step B — mocked MPCO element-level reads.

Builds a synthetic MPCO-shaped HDF5 file in a tmpdir mimicking the
on-disk layout that ``MPCOReader.read_gauss`` walks, then exercises
both single-GP (FourNodeTetrahedron) and multi-GP (TenNodeTetrahedron)
buckets without spinning up OpenSees. Real-MPCO end-to-end coverage
lives in ``test_results_mpco_element_real.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.solvers._element_response import (
    ELE_TAG_FourNodeTetrahedron,
    ELE_TAG_TenNodeTetrahedron,
    IntRule,
    flatten,
    lookup,
)


# =====================================================================
# Synthetic MPCO file builder
# =====================================================================

def _create_mpco_skeleton(
    path: Path,
    *,
    node_ids: np.ndarray,
    node_coords: np.ndarray,
    n_steps: int,
    dt: float,
) -> tuple["h5py.File", str]:
    """Write the minimum MODEL_STAGE skeleton MPCOReader's stage discovery
    needs (INFO + MODEL/NODES + a placeholder ON_NODES result with
    STEP_<k> attributes that drive the time vector).

    Returns the open file handle and the stage group name.
    """
    f = h5py.File(path, "w")
    info = f.create_group("INFO")
    info.create_dataset("SPATIAL_DIM", data=3)
    info.create_dataset("SOLVER_NAME", data=np.bytes_(b"OpenSees"))
    info.create_dataset("SOLVER_VERSION", data=np.array([3, 5, 1]))

    stage_name = "MODEL_STAGE[0]"
    stage = f.create_group(stage_name)
    stage.attrs["STEP"] = 0
    stage.attrs["TIME"] = 0.0

    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    nodes.create_dataset(
        "COORDINATES", data=node_coords.astype(np.float64),
    )

    # ON_NODES driving the per-stage time vector. Single dummy result.
    results = stage.create_group("RESULTS")
    on_nodes = results.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.attrs["DISPLAY_NAME"] = np.bytes_(b"Displacement")
    disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy,Uz"])
    disp.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    disp_data = disp.create_group("DATA")
    for k in range(n_steps):
        step = disp_data.create_dataset(
            f"STEP_{k}",
            data=np.zeros((node_ids.size, 3), dtype=np.float64),
        )
        step.attrs["STEP"] = k
        step.attrs["TIME"] = (k + 1) * dt

    return f, stage_name


def _add_stress_bucket(
    stage_grp: "h5py.Group",
    *,
    bracket_key: str,
    class_tag: int,
    int_rule: int,
    element_ids: np.ndarray,
    flat_data: np.ndarray,        # (T, E, flat_size)
    n_gauss_points: int,
    n_components: int = 6,
    component_labels: tuple[str, ...] = (
        "sigma_xx", "sigma_yy", "sigma_zz",
        "sigma_xy", "sigma_yz", "sigma_xz",
    ),
) -> None:
    """Add one ON_ELEMENTS/stresses/<bracket_key>/ bucket to a stage."""
    on_elements = stage_grp["RESULTS"].require_group("ON_ELEMENTS")
    stress = on_elements.require_group("stresses")
    bucket = stress.create_group(bracket_key)

    flat_size = n_gauss_points * n_components
    bucket.attrs["NUM_COLUMNS"] = flat_size

    # META: one block per GP.
    meta = bucket.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY",
        data=np.ones((n_gauss_points, 1), dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS",
        data=np.arange(n_gauss_points, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset(
        "NUM_COMPONENTS",
        data=np.full((n_gauss_points, 1), n_components, dtype=np.int32),
    )
    components_str = ";".join(
        f"1.{','.join(component_labels)}" for _ in range(n_gauss_points)
    )
    meta.create_dataset("COMPONENTS", data=np.bytes_(components_str.encode()))

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


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def tet1_mpco(tmp_path: Path) -> Path:
    """Synthetic MPCO with 2 FourNodeTetrahedron elements, 2 steps."""
    layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")

    # Two tets sharing a face. Coords are arbitrary — only the bucket
    # data matters for read correctness.
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
    ])

    path = tmp_path / "synthetic_tet1.mpco"
    f, stage_name = _create_mpco_skeleton(
        path, node_ids=node_ids, node_coords=coords, n_steps=2, dt=0.5,
    )
    try:
        T, E = 2, 2
        # Per-component synthetic data: stress_xx[t, e, gp] = e * 100 + t * 10 + 0
        per_comp = {
            name: np.full((T, E, 1), float(k), dtype=np.float64)
            + np.arange(E, dtype=np.float64).reshape(1, E, 1) * 100.0
            + np.arange(T, dtype=np.float64).reshape(T, 1, 1) * 10.0
            for k, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)

        _add_stress_bucket(
            f[stage_name],
            bracket_key=f"{layout.class_tag}-FourNodeTetrahedron[300:0:0]",
            class_tag=layout.class_tag,
            int_rule=IntRule.Tet_GL_1,
            element_ids=np.array([10, 20], dtype=np.int32),
            flat_data=flat,
            n_gauss_points=1,
        )
    finally:
        f.close()
    return path


@pytest.fixture
def tet2_mpco(tmp_path: Path) -> Path:
    """Synthetic MPCO with 2 TenNodeTetrahedron elements, 3 steps, 4 GPs each."""
    layout = lookup("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress")

    node_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    coords = np.eye(4, 3)

    path = tmp_path / "synthetic_tet2.mpco"
    f, stage_name = _create_mpco_skeleton(
        path, node_ids=node_ids, node_coords=coords, n_steps=3, dt=1.0,
    )
    try:
        T, E, G = 3, 2, 4
        per_comp = {}
        # Encode a unique value per (component, t, e, gp) so misordering
        # would surface immediately.
        for ki, name in enumerate(layout.component_layout):
            arr = np.zeros((T, E, G), dtype=np.float64)
            for t in range(T):
                for e in range(E):
                    for g in range(G):
                        arr[t, e, g] = ki * 1000.0 + t * 100.0 + e * 10.0 + g
            per_comp[name] = arr
        flat = flatten(per_comp, layout)

        _add_stress_bucket(
            f[stage_name],
            bracket_key=f"{layout.class_tag}-TenNodeTetrahedron[301:0:0]",
            class_tag=layout.class_tag,
            int_rule=IntRule.Tet_GL_2,
            element_ids=np.array([100, 200], dtype=np.int32),
            flat_data=flat,
            n_gauss_points=4,
        )
    finally:
        f.close()
    return path


# =====================================================================
# Tests — single-GP tet
# =====================================================================

class TestFourNodeTetStress:
    def test_full_read_stress_xx(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            # 2 elements × 1 GP = 2 columns; 2 steps.
            assert slab.values.shape == (2, 2)
            # Encoded as comp_idx * 1 + e * 100 + t * 10. stress_xx is
            # comp_idx 0; e=0,1; t=0,1.
            np.testing.assert_array_equal(
                slab.values, [[0.0, 100.0], [10.0, 110.0]],
            )

    def test_element_index_and_natural_coords(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            np.testing.assert_array_equal(slab.element_index, [10, 20])
            # 1 GP per tet at the volume centroid.
            np.testing.assert_allclose(
                slab.natural_coords,
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
            )

    def test_all_six_components_distinct(self, tet1_mpco: Path) -> None:
        """Each component returns its own column, never mixed."""
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            seen: dict[float, str] = {}
            for k, name in enumerate([
                "stress_xx", "stress_yy", "stress_zz",
                "stress_xy", "stress_yz", "stress_xz",
            ]):
                slab = s.elements.gauss.get(component=name)
                # Encoded value at (t=0, e=0) equals k.
                v = float(slab.values[0, 0])
                assert v == k, f"{name}: expected {k}, got {v}"
                seen[v] = name
            assert len(seen) == 6

    def test_filter_by_element_ids(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(
                component="stress_yy", ids=np.array([20]),
            )
            assert slab.values.shape == (2, 1)
            np.testing.assert_array_equal(slab.element_index, [20])
            # comp_idx=1, e=1 (=100), t=0..1 (=0,10) → [101, 111]
            np.testing.assert_array_equal(slab.values, [[101.0], [111.0]])

    def test_time_slice_single_step(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx", time=0)
            assert slab.values.shape == (1, 2)

    def test_available_components_lists_stress(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            comps = set(s.elements.gauss.available_components())
            for name in [
                "stress_xx", "stress_yy", "stress_zz",
                "stress_xy", "stress_yz", "stress_xz",
            ]:
                assert name in comps

    def test_unknown_component_returns_empty(self, tet1_mpco: Path) -> None:
        with Results.from_mpco(tet1_mpco) as r:
            s = r.stage(r.stages[0].id)
            # 'displacement_x' has no gauss-token mapping.
            slab = s.elements.gauss.get(component="displacement_x")
            assert slab.values.shape == (2, 0)


# =====================================================================
# Tests — multi-GP tet (the unflatten ordering test)
# =====================================================================

class TestTenNodeTetStress:
    def test_shapes(self, tet2_mpco: Path) -> None:
        with Results.from_mpco(tet2_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            # 2 elements × 4 GPs = 8 columns; 3 steps.
            assert slab.values.shape == (3, 8)
            assert slab.element_index.shape == (8,)
            assert slab.natural_coords.shape == (8, 3)

    def test_element_index_repeats_per_gp(self, tet2_mpco: Path) -> None:
        with Results.from_mpco(tet2_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            # Element 100 covers cols 0..3, element 200 covers cols 4..7.
            np.testing.assert_array_equal(
                slab.element_index, [100, 100, 100, 100, 200, 200, 200, 200],
            )

    def test_natural_coords_tiled_per_element(self, tet2_mpco: Path) -> None:
        with Results.from_mpco(tet2_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            alpha = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
            beta = (5.0 - math.sqrt(5.0)) / 20.0
            # First 4 GPs (element 100) and next 4 (element 200) repeat
            # the same alpha/beta arrangement.
            np.testing.assert_allclose(
                slab.natural_coords[0], [alpha, beta, beta], atol=1e-14,
            )
            np.testing.assert_allclose(
                slab.natural_coords[3], [beta, beta, beta], atol=1e-14,
            )
            np.testing.assert_allclose(
                slab.natural_coords[4], [alpha, beta, beta], atol=1e-14,
            )

    def test_gp_slowest_unflatten_ordering(self, tet2_mpco: Path) -> None:
        """The decisive ordering test.

        Encoded value at (comp_idx, t, e, g) is
        ``ki*1000 + t*100 + e*10 + g``. If unflatten stride is wrong
        (component-slowest instead of GP-slowest), this would expose
        it immediately.
        """
        with Results.from_mpco(tet2_mpco) as r:
            s = r.stage(r.stages[0].id)
            # stress_yy (ki=1), step t=2, element e=0 (cols 0..3 hold GPs 0..3).
            slab = s.elements.gauss.get(component="stress_yy")
            row = slab.values[2, :4]   # (4,)
            expected = np.array(
                [1 * 1000 + 2 * 100 + 0 * 10 + g for g in range(4)],
                dtype=np.float64,
            )
            np.testing.assert_array_equal(row, expected)


# =====================================================================
# META validation — drift detection
# =====================================================================

class TestMetaValidation:
    def test_wrong_num_columns_raises(self, tmp_path: Path) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        path = tmp_path / "bad_num_cols.mpco"
        f, stage_name = _create_mpco_skeleton(
            path, node_ids=np.array([1]), node_coords=np.zeros((1, 3)),
            n_steps=1, dt=1.0,
        )
        try:
            # Build a bucket with 5 columns instead of 6 — META and
            # data agree internally, but the catalog says 6.
            on_elements = f[stage_name]["RESULTS"].require_group("ON_ELEMENTS")
            stress = on_elements.create_group("stresses")
            bucket = stress.create_group(
                f"{layout.class_tag}-FourNodeTetrahedron[300:0:0]"
            )
            bucket.attrs["NUM_COLUMNS"] = 5      # ← wrong
            meta = bucket.create_group("META")
            meta.create_dataset("MULTIPLICITY",
                                data=np.array([[1]], dtype=np.int32))
            meta.create_dataset("GAUSS_IDS",
                                data=np.array([[0]], dtype=np.int32))
            meta.create_dataset("NUM_COMPONENTS",
                                data=np.array([[5]], dtype=np.int32))
            meta.create_dataset("COMPONENTS",
                                data=np.bytes_(b"1.s_xx,s_yy,s_zz,s_xy,s_yz"))
            bucket.create_dataset("ID", data=np.array([[1]], dtype=np.int32))
            data = bucket.create_group("DATA")
            ds = data.create_dataset("STEP_0", data=np.zeros((1, 5)))
            ds.attrs["STEP"] = 0
            ds.attrs["TIME"] = 1.0
        finally:
            f.close()

        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            with pytest.raises(ValueError, match="NUM_COLUMNS"):
                s.elements.gauss.get(component="stress_xx")


# =====================================================================
# Bucket filtering — out-of-scope buckets are skipped, not crashed
# =====================================================================

class TestBucketFiltering:
    def test_uncatalogued_class_returns_empty(self, tmp_path: Path) -> None:
        """A bucket with an unknown class name contributes nothing."""
        path = tmp_path / "uncatalogued.mpco"
        f, stage_name = _create_mpco_skeleton(
            path, node_ids=np.array([1]), node_coords=np.zeros((1, 3)),
            n_steps=1, dt=1.0,
        )
        try:
            on_elements = f[stage_name]["RESULTS"].require_group("ON_ELEMENTS")
            stress = on_elements.create_group("stresses")
            stress.create_group("999-NotCatalogued[300:0:0]")
        finally:
            f.close()

        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            assert slab.values.shape == (1, 0)

    def test_custom_rule_bucket_is_skipped(self, tmp_path: Path) -> None:
        """Custom integration rule (1000) is out of v1 scope — skip it."""
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        path = tmp_path / "custom_rule.mpco"
        f, stage_name = _create_mpco_skeleton(
            path, node_ids=np.array([1]), node_coords=np.zeros((1, 3)),
            n_steps=1, dt=1.0,
        )
        try:
            # Bucket with custom_rule_idx=3 — must be ignored.
            on_elements = f[stage_name]["RESULTS"].require_group("ON_ELEMENTS")
            stress = on_elements.create_group("stresses")
            stress.create_group(
                f"{layout.class_tag}-FourNodeTetrahedron[1000:3:0]"
            )
        finally:
            f.close()

        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            assert slab.values.shape == (1, 0)
