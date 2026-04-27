"""Phase 11b Step 3a — mocked MPCO nodal-forces (per-element-node) reads.

Builds a synthetic MPCO-shaped HDF5 file in a tmpdir mimicking the
on-disk layout that ``MPCOReader.read_elements`` walks for closed-
form elastic beams (``ElasticBeam{2d,3d}``,
``ElasticTimoshenkoBeam{2d,3d}``, ``ModElasticBeam2d``).

Real-MPCO end-to-end coverage lives in
``test_results_mpco_nodal_real.py``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_nodal_io as _mn
from apeGmsh.solvers._element_response import (
    ELE_TAG_ElasticBeam2d,
    ELE_TAG_ElasticBeam3d,
    ELE_TAG_ElasticTimoshenkoBeam2d,
    ELE_TAG_ElasticTimoshenkoBeam3d,
    ELE_TAG_ModElasticBeam2d,
)


# =====================================================================
# Synthetic MPCO file builder
# =====================================================================

def _create_skeleton(
    path: Path, *, node_ids: np.ndarray, coords: np.ndarray,
    n_steps: int, dt: float,
) -> tuple["h5py.File", str]:
    """INFO + MODEL_STAGE/MODEL/NODES + dummy ON_NODES (drives time vector)."""
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

    results = stage.create_group("RESULTS")
    on_nodes = results.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.attrs["DISPLAY_NAME"] = np.bytes_(b"Displacement")
    disp.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    disp_data = disp.create_group("DATA")
    for k in range(n_steps):
        ds = disp_data.create_dataset(
            f"STEP_{k}", data=np.zeros((node_ids.size, 3), dtype=np.float64),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * dt

    results.create_group("ON_ELEMENTS")
    return f, stage_name


def _add_nodal_force_bucket(
    stage_grp: "h5py.Group",
    *,
    frame: str,                        # "globalForce" | "localForce"
    class_tag: int,
    class_name: str,
    int_rule: int,
    element_ids: np.ndarray,
    flat_data: np.ndarray,             # (T, E, n_nodes * n_components)
    components_str: str,               # MPCO META/COMPONENTS body
) -> None:
    """Write a connectivity dataset + a matching results bucket."""
    n_total = flat_data.shape[2]

    conn_key = f"{class_tag}-{class_name}[{int_rule}:0]"
    conn = stage_grp["MODEL/ELEMENTS"]
    if conn_key not in conn:
        conn_ds = conn.create_dataset(
            conn_key,
            data=np.zeros((element_ids.size, 3), dtype=np.int32),
        )
        conn_ds.attrs["INTEGRATION_RULE"] = np.array([int_rule], dtype=np.int32)
        conn_ds.attrs["GEOMETRY"] = np.array([1], dtype=np.int32)

    on_elements = stage_grp["RESULTS/ON_ELEMENTS"].require_group(frame)
    bucket_key = f"{class_tag}-{class_name}[{int_rule}:0:0]"
    bucket = on_elements.create_group(bucket_key)
    bucket.attrs["NUM_COLUMNS"] = np.array([n_total], dtype=np.int32)

    meta = bucket.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY", data=np.array([[1]], dtype=np.int32),
    )
    # GAUSS_IDS = [-1] — sentinel for "no integration point"
    meta.create_dataset(
        "GAUSS_IDS", data=np.array([[-1]], dtype=np.int32),
    )
    meta.create_dataset(
        "NUM_COMPONENTS", data=np.array([[n_total]], dtype=np.int32),
    )
    meta.create_dataset(
        "COMPONENTS",
        data=np.array([components_str.encode("ascii")]),
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


_GLOBAL_3D_COMPS = (
    "0.Px_1,Py_1,Pz_1,Mx_1,My_1,Mz_1,"
    "Px_2,Py_2,Pz_2,Mx_2,My_2,Mz_2"
)
_LOCAL_3D_COMPS = (
    "0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,"
    "N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
)
_GLOBAL_2D_COMPS = "0.Px_1,Py_1,Mz_1,Px_2,Py_2,Mz_2"
_LOCAL_2D_COMPS = "0.N_1,Vy_1,Mz_1,N_2,Vy_2,Mz_2"


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def eb3d_global_mpco(tmp_path: Path) -> Path:
    """ElasticBeam3d, globalForce, 2 elements, 2 steps."""
    path = tmp_path / "eb3d_global.mpco"
    node_ids = np.array([1, 2, 3], dtype=np.int32)
    coords = np.zeros((3, 3))
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=2, dt=1.0,
    )
    try:
        # 2 elements × 2 nodes × 6 comps = 24 doubles per step.
        # Encoding per slot: t*1000 + e*100 + n*10 + k
        T, E, n_nodes, K = 2, 2, 2, 6
        flat = np.zeros((T, E, n_nodes * K), dtype=np.float64)
        for t in range(T):
            for e in range(E):
                for n in range(n_nodes):
                    for k in range(K):
                        flat[t, e, n * K + k] = (
                            t * 1000 + e * 100 + n * 10 + k
                        )
        _add_nodal_force_bucket(
            f[stage_name],
            frame="globalForce",
            class_tag=ELE_TAG_ElasticBeam3d,
            class_name="ElasticBeam3d",
            int_rule=1,
            element_ids=np.array([7, 11], dtype=np.int32),
            flat_data=flat,
            components_str=_GLOBAL_3D_COMPS,
        )
    finally:
        f.close()
    return path


@pytest.fixture
def eb3d_global_and_local_mpco(tmp_path: Path) -> Path:
    """ElasticBeam3d with both globalForce and localForce buckets."""
    path = tmp_path / "eb3d_both.mpco"
    node_ids = np.array([1, 2], dtype=np.int32)
    coords = np.zeros((2, 3))
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
    )
    try:
        T, E, n_nodes, K = 1, 1, 2, 6
        flat_global = np.zeros((T, E, n_nodes * K), dtype=np.float64)
        flat_local = np.zeros((T, E, n_nodes * K), dtype=np.float64)
        for n in range(n_nodes):
            for k in range(K):
                flat_global[0, 0, n * K + k] = 1000 + n * 10 + k
                flat_local[0, 0, n * K + k] = 2000 + n * 10 + k
        for frame, comps_str, flat in (
            ("globalForce", _GLOBAL_3D_COMPS, flat_global),
            ("localForce", _LOCAL_3D_COMPS, flat_local),
        ):
            _add_nodal_force_bucket(
                f[stage_name],
                frame=frame,
                class_tag=ELE_TAG_ElasticBeam3d,
                class_name="ElasticBeam3d",
                int_rule=1,
                element_ids=np.array([1], dtype=np.int32),
                flat_data=flat,
                components_str=comps_str,
            )
    finally:
        f.close()
    return path


@pytest.fixture
def eb2d_global_mpco(tmp_path: Path) -> Path:
    """ElasticBeam2d, globalForce, 1 element, 1 step."""
    path = tmp_path / "eb2d.mpco"
    node_ids = np.array([1, 2], dtype=np.int32)
    coords = np.zeros((2, 3))
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
    )
    try:
        # 2 nodes × 3 comps = 6.
        T, E, n_nodes, K = 1, 1, 2, 3
        flat = np.zeros((T, E, n_nodes * K), dtype=np.float64)
        for n in range(n_nodes):
            for k in range(K):
                flat[0, 0, n * K + k] = 100 + n * 10 + k
        _add_nodal_force_bucket(
            f[stage_name],
            frame="globalForce",
            class_tag=ELE_TAG_ElasticBeam2d,
            class_name="ElasticBeam2d",
            int_rule=1,
            element_ids=np.array([5], dtype=np.int32),
            flat_data=flat,
            components_str=_GLOBAL_2D_COMPS,
        )
    finally:
        f.close()
    return path


# =====================================================================
# Read tests
# =====================================================================

class TestReadGlobalForce3D:
    def test_force_x_full_read(self, eb3d_global_mpco: Path) -> None:
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_force_x")
        # 2 elements, 2 nodes per element, 2 steps.
        assert slab.values.shape == (2, 2, 2)
        assert slab.element_ids.tolist() == [7, 11]
        # k=0 (Px / nodal_resisting_force_x). Values:
        # t=0,e=0,n=0 → 0; t=0,e=0,n=1 → 10; t=0,e=1,n=0 → 100; ...
        # t=1,e=0,n=0 → 1000; t=1,e=1,n=1 → 1110.
        np.testing.assert_array_equal(
            slab.values[0],
            [[0.0, 10.0], [100.0, 110.0]],
        )
        np.testing.assert_array_equal(
            slab.values[1],
            [[1000.0, 1010.0], [1100.0, 1110.0]],
        )

    def test_moment_z_picks_correct_column(
        self, eb3d_global_mpco: Path,
    ) -> None:
        # Mz is k=5 in (Px, Py, Pz, Mx, My, Mz).
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_moment_z")
        # k=5: t=0,e=0,n=0 → 5; t=0,e=1,n=1 → 115.
        assert slab.values[0, 0, 0] == 5.0
        assert slab.values[0, 1, 1] == 115.0
        assert slab.values[1, 1, 1] == 1115.0

    def test_local_component_returns_empty(
        self, eb3d_global_mpco: Path,
    ) -> None:
        # globalForce-only bucket: localForce component has no data.
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_force_local_x")
        assert slab.values.shape[1:] == (0, 0)

    def test_element_id_filter(
        self, eb3d_global_mpco: Path,
    ) -> None:
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(
                component="nodal_resisting_force_x",
                ids=np.array([11]),
            )
        assert slab.values.shape == (2, 1, 2)
        assert slab.element_ids.tolist() == [11]
        # Element e=1, n=0 → 100; n=1 → 110.
        np.testing.assert_array_equal(slab.values[0, 0], [100.0, 110.0])

    def test_element_id_filter_empty(
        self, eb3d_global_mpco: Path,
    ) -> None:
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(
                component="nodal_resisting_force_x",
                ids=np.array([999]),
            )
        assert slab.element_ids.size == 0


class TestReadLocalForce3D:
    def test_force_local_x_routes_through_localForce(
        self, eb3d_global_and_local_mpco: Path,
    ) -> None:
        # The reader routes the canonical name to the localForce bucket.
        with Results.from_mpco(eb3d_global_and_local_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab_loc = s.elements.get(
                component="nodal_resisting_force_local_x",
            )
            slab_glob = s.elements.get(component="nodal_resisting_force_x")
        # Local k=0 (N) at e=0,n=0 → 2000.
        assert slab_loc.values[0, 0, 0] == 2000.0
        # Global k=0 (Px) at e=0,n=0 → 1000.
        assert slab_glob.values[0, 0, 0] == 1000.0

    def test_moment_local_x_is_torsion(
        self, eb3d_global_and_local_mpco: Path,
    ) -> None:
        # Local layout: (N, Vy, Vz, T, My, Mz) → (force_x, force_y, force_z,
        # moment_x, moment_y, moment_z) under the apeGmsh canonical names.
        # So nodal_resisting_moment_local_x maps to T (k=3), value = 2003 + n*10.
        with Results.from_mpco(eb3d_global_and_local_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_moment_local_x")
        np.testing.assert_array_equal(slab.values[0, 0], [2003.0, 2013.0])


class TestReadGlobalForce2D:
    def test_force_y_2d(self, eb2d_global_mpco: Path) -> None:
        with Results.from_mpco(eb2d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_force_y")
        assert slab.values.shape == (1, 1, 2)
        # Py is k=1 in (Px, Py, Mz). n=0 → 101; n=1 → 111.
        np.testing.assert_array_equal(slab.values[0, 0], [101.0, 111.0])

    def test_moment_z_2d(self, eb2d_global_mpco: Path) -> None:
        with Results.from_mpco(eb2d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_moment_z")
        # Mz is k=2 in 2D layout.
        np.testing.assert_array_equal(slab.values[0, 0], [102.0, 112.0])

    def test_force_z_not_in_2d(self, eb2d_global_mpco: Path) -> None:
        # 2D ElasticBeam2d has no force_z component.
        with Results.from_mpco(eb2d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_force_z")
        assert slab.values.shape[1:] == (0, 0)


# =====================================================================
# available_components
# =====================================================================

class TestAvailableComponents:
    def test_global_only(self, eb3d_global_mpco: Path) -> None:
        with Results.from_mpco(eb3d_global_mpco) as r:
            s = r.stage(r.stages[0].id)
            comps = set(s.elements.available_components())
        # All 6 global names present; no local.
        for n in ("nodal_resisting_force_x", "nodal_resisting_force_y",
                  "nodal_resisting_force_z", "nodal_resisting_moment_x",
                  "nodal_resisting_moment_y", "nodal_resisting_moment_z"):
            assert n in comps
        for n in ("nodal_resisting_force_local_x",
                  "nodal_resisting_moment_local_z"):
            assert n not in comps

    def test_global_and_local(self, eb3d_global_and_local_mpco: Path) -> None:
        with Results.from_mpco(eb3d_global_and_local_mpco) as r:
            s = r.stage(r.stages[0].id)
            comps = set(s.elements.available_components())
        # Both frames' components should appear.
        assert "nodal_resisting_force_x" in comps
        assert "nodal_resisting_force_local_x" in comps


# =====================================================================
# Bucket discovery
# =====================================================================

class TestBucketDiscovery:
    def test_skips_uncatalogued_class(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("globalForce")
            sf.create_group("999-MysteryBeam3d[1:0:0]")
        with h5py.File(path, "r") as f:
            _, buckets = _mn.discover_nodal_force_buckets(
                f["ON_ELEMENTS"], canonical_component="nodal_resisting_force_x",
            )
        assert buckets == []

    def test_skips_header_idx_nonzero(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("globalForce")
            sf.create_group(f"{ELE_TAG_ElasticBeam3d}-ElasticBeam3d[1:0:1]")
        with h5py.File(path, "r") as f:
            _, buckets = _mn.discover_nodal_force_buckets(
                f["ON_ELEMENTS"], canonical_component="nodal_resisting_force_x",
            )
        assert buckets == []

    def test_returns_none_token_for_non_nodal_component(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            f.create_group("ON_ELEMENTS")
        with h5py.File(path, "r") as f:
            token, buckets = _mn.discover_nodal_force_buckets(
                f["ON_ELEMENTS"], canonical_component="axial_force",
            )
        assert token is None
        assert buckets == []

    def test_picks_up_eb3d(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("globalForce")
            sf.create_group(f"{ELE_TAG_ElasticBeam3d}-ElasticBeam3d[1:0:0]")
        with h5py.File(path, "r") as f:
            _, buckets = _mn.discover_nodal_force_buckets(
                f["ON_ELEMENTS"], canonical_component="nodal_resisting_force_x",
            )
        assert len(buckets) == 1
        assert buckets[0].elem_key.class_name == "ElasticBeam3d"
        assert buckets[0].layout.class_tag == ELE_TAG_ElasticBeam3d
        assert buckets[0].layout.frame == "global"


# =====================================================================
# META validation errors
# =====================================================================

class TestMetaValidation:
    def _bucket_with_meta(
        self, path: Path,
        *, num_columns: int, multiplicity, gauss_ids, num_components,
    ) -> "h5py.Group":
        """Helper: build a synthetic bucket group for direct META checks."""
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            grp.attrs["NUM_COLUMNS"] = np.array([num_columns], dtype=np.int32)
            meta = grp.create_group("META")
            meta.create_dataset("MULTIPLICITY", data=np.asarray(multiplicity))
            meta.create_dataset("GAUSS_IDS", data=np.asarray(gauss_ids))
            meta.create_dataset("NUM_COMPONENTS", data=np.asarray(num_components))
        return None  # unused — caller re-opens to read

    def test_num_columns_mismatch_raises(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import lookup_nodal_force
        layout = lookup_nodal_force("ElasticBeam3d", "global_force")
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            grp.attrs["NUM_COLUMNS"] = np.array([99], dtype=np.int32)  # wrong
            meta = grp.create_group("META")
            meta.create_dataset("MULTIPLICITY", data=np.array([[1]]))
            meta.create_dataset("GAUSS_IDS", data=np.array([[-1]]))
            meta.create_dataset("NUM_COMPONENTS", data=np.array([[12]]))
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="NUM_COLUMNS"):
                _mn.validate_nodal_bucket_meta(
                    f["bucket"], layout, bracket_key="b",
                )

    def test_non_sentinel_gauss_id_raises(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import lookup_nodal_force
        layout = lookup_nodal_force("ElasticBeam3d", "global_force")
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            grp.attrs["NUM_COLUMNS"] = np.array([12], dtype=np.int32)
            meta = grp.create_group("META")
            meta.create_dataset("MULTIPLICITY", data=np.array([[1]]))
            meta.create_dataset("GAUSS_IDS", data=np.array([[0]]))   # wrong
            meta.create_dataset("NUM_COMPONENTS", data=np.array([[12]]))
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="GAUSS_IDS"):
                _mn.validate_nodal_bucket_meta(
                    f["bucket"], layout, bracket_key="b",
                )

    def test_multi_block_meta_raises(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import lookup_nodal_force
        layout = lookup_nodal_force("ElasticBeam3d", "global_force")
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            grp.attrs["NUM_COLUMNS"] = np.array([12], dtype=np.int32)
            meta = grp.create_group("META")
            # Two-block META — out of scope (must be single block).
            meta.create_dataset(
                "MULTIPLICITY", data=np.array([[1], [1]]),
            )
            meta.create_dataset(
                "GAUSS_IDS", data=np.array([[-1], [-1]]),
            )
            meta.create_dataset(
                "NUM_COMPONENTS", data=np.array([[6], [6]]),
            )
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="single META block"):
                _mn.validate_nodal_bucket_meta(
                    f["bucket"], layout, bracket_key="b",
                )
