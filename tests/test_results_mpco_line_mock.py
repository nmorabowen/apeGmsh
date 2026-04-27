"""Phase 11b Step 2a — mocked MPCO line-stations reads.

Builds a synthetic MPCO-shaped HDF5 file in a tmpdir mimicking the
on-disk layout that ``MPCOReader.read_line_stations`` walks, then
exercises force-/disp-based beam-column buckets without spinning up
OpenSees. Real-MPCO end-to-end coverage lives in
``test_results_mpco_line_real.py``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_line_io as _mline
from apeGmsh.solvers._element_response import (
    ELE_TAG_DispBeamColumn3d,
    ELE_TAG_ForceBeamColumn2d,
    ELE_TAG_ForceBeamColumn3d,
    IntRule,
    flatten,
    lookup_custom_rule,
    resolve_layout_from_gp_x,
)


# =====================================================================
# Synthetic MPCO file builder
# =====================================================================

def _create_skeleton(
    path: Path,
    *,
    node_ids: np.ndarray,
    coords: np.ndarray,
    n_steps: int,
    dt: float,
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
    model.create_group("ELEMENTS")  # populated per-bucket

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


def _add_line_station_bucket(
    stage_grp: "h5py.Group",
    *,
    class_tag: int,
    class_name: str,
    custom_rule_idx: int,
    header_idx: int,
    element_ids: np.ndarray,
    gp_x: np.ndarray,
    section_names: tuple[str, ...],
    flat_data: np.ndarray,    # (T, E, n_IP * n_components)
) -> None:
    """Write a connectivity dataset (with GP_X attr) + matching results bucket."""
    n_ip = gp_x.size
    n_comp = len(section_names)

    conn_key = f"{class_tag}-{class_name}[{IntRule.Custom}:{custom_rule_idx}]"
    conn_grp = stage_grp["MODEL/ELEMENTS"]
    conn_ds = conn_grp.create_dataset(
        conn_key,
        data=np.zeros((element_ids.size, 3), dtype=np.int32),
    )
    conn_ds.attrs["GP_X"] = gp_x.astype(np.float64)
    conn_ds.attrs["INTEGRATION_RULE"] = np.array([IntRule.Custom], dtype=np.int32)
    conn_ds.attrs["CUSTOM_INTEGRATION_RULE"] = np.array([1], dtype=np.int32)
    conn_ds.attrs["CUSTOM_INTEGRATION_RULE_DIMENSION"] = np.array([1], dtype=np.int32)
    conn_ds.attrs["GEOMETRY"] = np.array([1], dtype=np.int32)

    section_force = stage_grp["RESULTS/ON_ELEMENTS"].require_group("section.force")
    bucket_key = (
        f"{class_tag}-{class_name}"
        f"[{IntRule.Custom}:{custom_rule_idx}:{header_idx}]"
    )
    bucket = section_force.create_group(bucket_key)
    bucket.attrs["NUM_COLUMNS"] = np.array([n_ip * n_comp], dtype=np.int32)

    meta = bucket.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY",
        data=np.ones((n_ip, 1), dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS",
        data=np.arange(n_ip, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset(
        "NUM_COMPONENTS",
        data=np.full((n_ip, 1), n_comp, dtype=np.int32),
    )
    components_str = ";".join(
        f"0.1.2.{','.join(section_names)}" for _ in range(n_ip)
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


def _build_flat(
    *,
    component_layout: tuple[str, ...],
    n_ip: int,
    element_ids: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Synthesise per-component arrays + their flattened packing.

    Each scalar value is unique: ``ki * 1000 + t * 100 + e * 10 + g``.
    """
    layout = resolve_layout_from_gp_x(
        lookup_custom_rule("ForceBeamColumn3d", "section_force"),
        np.linspace(-1.0, 1.0, n_ip),
        # The codes don't matter for this helper — caller picks them
        # via the catalog and we just use the layout shape. We pass
        # (2, 1) here because resolve_layout_from_gp_x requires SOME
        # codes; the actual catalog layout is constructed by the
        # caller and passed back through ``component_layout``.
        (2, 1),
    )
    # Replace the synthetic layout's component_layout with the
    # caller's actual section component_layout so the flatten reuses
    # the apeGmsh canonical names.
    from apeGmsh.solvers._element_response import ResponseLayout
    real_layout = ResponseLayout(
        n_gauss_points=n_ip,
        natural_coords=layout.natural_coords,
        coord_system=layout.coord_system,
        n_components_per_gp=len(component_layout),
        component_layout=component_layout,
        class_tag=layout.class_tag,
    )

    T = n_steps
    E = element_ids.size
    G = n_ip
    per_comp: dict[str, np.ndarray] = {}
    for ki, name in enumerate(component_layout):
        arr = np.zeros((T, E, G), dtype=np.float64)
        for t in range(T):
            for e in range(E):
                for g in range(G):
                    arr[t, e, g] = ki * 1000.0 + t * 100.0 + e * 10.0 + g
        per_comp[name] = arr
    flat = flatten(per_comp, real_layout)
    return flat, per_comp


# =====================================================================
# Fixtures
# =====================================================================

# Mapping from MPCO section names (in their on-disk order for a
# given section) to apeGmsh canonical component names. Used to
# build matching ``component_layout`` arguments to ``_build_flat``.
_MPCO_TO_CANONICAL = {
    "P": "axial_force",
    "Mz": "bending_moment_z",
    "My": "bending_moment_y",
    "T": "torsion",
    "Vy": "shear_y",
    "Vz": "shear_z",
}


@pytest.fixture
def fbc3d_aggregated_5ip_mpco(tmp_path: Path) -> Path:
    """ForceBeamColumn3d, 5-IP Lobatto, aggregated 6-comp section."""
    path = tmp_path / "fbc3d_5ip.mpco"
    node_ids = np.array([1, 2], dtype=np.int32)
    coords = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float64)
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=2, dt=1.0,
    )
    try:
        gp_x = np.array(
            [-1.0, -0.65465367, 0.0, 0.65465367, 1.0],  # Lobatto-5
            dtype=np.float64,
        )
        section_names = ("P", "Mz", "My", "T", "Vy", "Vz")
        component_layout = tuple(_MPCO_TO_CANONICAL[n] for n in section_names)
        element_ids = np.array([1, 2], dtype=np.int32)
        flat, _ = _build_flat(
            component_layout=component_layout,
            n_ip=gp_x.size,
            element_ids=element_ids,
            n_steps=2,
        )
        _add_line_station_bucket(
            f[stage_name],
            class_tag=ELE_TAG_ForceBeamColumn3d,
            class_name="ForceBeamColumn3d",
            custom_rule_idx=1,
            header_idx=0,
            element_ids=element_ids,
            gp_x=gp_x,
            section_names=section_names,
            flat_data=flat,
        )
    finally:
        f.close()
    return path


@pytest.fixture
def fbc3d_bare_3ip_mpco(tmp_path: Path) -> Path:
    """ForceBeamColumn3d, 3-IP Legendre, bare 4-comp section [P, Mz, My, T]."""
    path = tmp_path / "fbc3d_3ip.mpco"
    node_ids = np.array([1, 2], dtype=np.int32)
    coords = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float64)
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
    )
    try:
        gp_x = np.array([-0.7745966, 0.0, +0.7745966], dtype=np.float64)
        section_names = ("P", "Mz", "My", "T")
        component_layout = tuple(_MPCO_TO_CANONICAL[n] for n in section_names)
        element_ids = np.array([10], dtype=np.int32)
        flat, _ = _build_flat(
            component_layout=component_layout,
            n_ip=gp_x.size,
            element_ids=element_ids,
            n_steps=1,
        )
        _add_line_station_bucket(
            f[stage_name],
            class_tag=ELE_TAG_ForceBeamColumn3d,
            class_name="ForceBeamColumn3d",
            custom_rule_idx=2,
            header_idx=0,
            element_ids=element_ids,
            gp_x=gp_x,
            section_names=section_names,
            flat_data=flat,
        )
    finally:
        f.close()
    return path


@pytest.fixture
def fbc2d_bare_3ip_mpco(tmp_path: Path) -> Path:
    """ForceBeamColumn2d, 3-IP, bare 2D 2-comp section [P, Mz]."""
    path = tmp_path / "fbc2d_3ip.mpco"
    node_ids = np.array([1, 2], dtype=np.int32)
    coords = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64)
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
    )
    try:
        gp_x = np.array([-0.7745966, 0.0, +0.7745966], dtype=np.float64)
        section_names = ("P", "Mz")
        component_layout = tuple(_MPCO_TO_CANONICAL[n] for n in section_names)
        element_ids = np.array([5], dtype=np.int32)
        flat, _ = _build_flat(
            component_layout=component_layout,
            n_ip=gp_x.size,
            element_ids=element_ids,
            n_steps=1,
        )
        _add_line_station_bucket(
            f[stage_name],
            class_tag=ELE_TAG_ForceBeamColumn2d,
            class_name="ForceBeamColumn2d",
            custom_rule_idx=1,
            header_idx=0,
            element_ids=element_ids,
            gp_x=gp_x,
            section_names=section_names,
            flat_data=flat,
        )
    finally:
        f.close()
    return path


# =====================================================================
# Tests — full-shape (aggregated 6-component) bucket
# =====================================================================

class TestForceBeamColumn3dAggregated:
    def test_axial_force_full_read(self, fbc3d_aggregated_5ip_mpco: Path) -> None:
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
            # 2 elements × 5 IPs = 10 stations, 2 steps.
            assert slab.values.shape == (2, 10)
            assert slab.element_index.tolist() == [
                1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            ]
            np.testing.assert_allclose(
                slab.station_natural_coord,
                np.tile(
                    [-1.0, -0.65465367, 0.0, 0.65465367, 1.0], 2,
                ),
                atol=1e-7,
            )

    def test_axial_force_values_match_unflatten(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        # axial_force is component k=0 in (P, Mz, My, T, Vy, Vz) order.
        # Synthetic encoding: ki * 1000 + t * 100 + e * 10 + g.
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        # ki=0; t=0,e=0,g=0..4 then t=0,e=1,g=0..4
        expected_t0 = np.concatenate([
            np.arange(5, dtype=np.float64),         # e=0
            10.0 + np.arange(5, dtype=np.float64),  # e=1
        ])
        expected_t1 = expected_t0 + 100.0
        np.testing.assert_array_equal(slab.values[0], expected_t0)
        np.testing.assert_array_equal(slab.values[1], expected_t1)

    def test_shear_z_picks_correct_column(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        # shear_z (Vz) is the LAST column (k=5) of the (P,Mz,My,T,Vy,Vz)
        # layout — verifies we route names correctly through the
        # resolved component_layout.
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="shear_z")
        # ki=5; t=0,e=0,g=0 → 5000.
        # t=0,e=0,g=4 → 5004; t=0,e=1,g=0 → 5010; t=1,e=1,g=4 → 5114.
        assert slab.values[0, 0] == 5000.0
        assert slab.values[0, 4] == 5004.0
        assert slab.values[0, 5] == 5010.0
        assert slab.values[1, 9] == 5114.0

    def test_bending_moment_y_picks_correct_column(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        # bending_moment_y is k=2 (My).
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="bending_moment_y")
        # ki=2 → 2000 + t*100 + e*10 + g.
        assert slab.values[0, 0] == 2000.0
        assert slab.values[1, 9] == 2114.0

    def test_element_id_filter(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(
                component="axial_force",
                ids=np.array([2]),
            )
        # Only element 2: 1 element × 5 IPs = 5 stations.
        assert slab.values.shape == (2, 5)
        assert slab.element_index.tolist() == [2, 2, 2, 2, 2]

    def test_element_id_filter_empty(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(
                component="axial_force",
                ids=np.array([999]),
            )
        assert slab.values.shape == (2, 0)
        assert slab.element_index.size == 0


class TestForceBeamColumn3dBareSection:
    def test_axial_force(self, fbc3d_bare_3ip_mpco: Path) -> None:
        with Results.from_mpco(fbc3d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        # 1 element × 3 IPs = 3 stations, 1 step.
        assert slab.values.shape == (1, 3)
        assert slab.element_index.tolist() == [10, 10, 10]
        np.testing.assert_allclose(
            slab.station_natural_coord,
            [-0.7745966, 0.0, +0.7745966], atol=1e-7,
        )
        # ki=0 (axial), t=0, e=0, g=0..2.
        np.testing.assert_array_equal(slab.values[0], [0.0, 1.0, 2.0])

    def test_torsion_picks_4th_column(self, fbc3d_bare_3ip_mpco: Path) -> None:
        # Bare section is (P, Mz, My, T) — torsion is k=3.
        with Results.from_mpco(fbc3d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="torsion")
        # ki=3 → 3000.
        np.testing.assert_array_equal(slab.values[0], [3000.0, 3001.0, 3002.0])

    def test_shear_y_not_in_bare_section_returns_empty(
        self, fbc3d_bare_3ip_mpco: Path,
    ) -> None:
        # Bare 3D fiber section has no shear → component absent from
        # the resolved layout → empty slab.
        with Results.from_mpco(fbc3d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="shear_y")
        assert slab.values.shape == (1, 0)


class TestForceBeamColumn2d:
    def test_axial_force(self, fbc2d_bare_3ip_mpco: Path) -> None:
        with Results.from_mpco(fbc2d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 3)
        assert slab.element_index.tolist() == [5, 5, 5]

    def test_bending_moment_z(self, fbc2d_bare_3ip_mpco: Path) -> None:
        # 2D bare section is (P, Mz) — Mz is k=1.
        with Results.from_mpco(fbc2d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="bending_moment_z")
        np.testing.assert_array_equal(slab.values[0], [1000.0, 1001.0, 1002.0])

    def test_shear_y_not_in_2d_bare_returns_empty(
        self, fbc2d_bare_3ip_mpco: Path,
    ) -> None:
        with Results.from_mpco(fbc2d_bare_3ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="shear_y")
        assert slab.values.shape == (1, 0)


# =====================================================================
# Module-level helpers — direct unit tests
# =====================================================================

class TestParseSectionCodesFromMeta:
    def test_aggregated_segment(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            meta = grp.create_group("META")
            meta.create_dataset(
                "COMPONENTS",
                data=np.array([b"0.1.2.P,Mz,My,T,Vy,Vz;0.1.2.P,Mz,My,T,Vy,Vz"]),
            )
        with h5py.File(path, "r") as f:
            codes = _mline.parse_section_codes_from_meta(
                f["bucket"], bracket_key="dummy",
            )
        assert codes == (2, 1, 4, 6, 3, 5)

    def test_bare_3d_segment(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            meta = grp.create_group("META")
            meta.create_dataset(
                "COMPONENTS",
                data=np.array([b"0.1.2.P,Mz,My,T;0.1.2.P,Mz,My,T;0.1.2.P,Mz,My,T"]),
            )
        with h5py.File(path, "r") as f:
            codes = _mline.parse_section_codes_from_meta(
                f["bucket"], bracket_key="dummy",
            )
        assert codes == (2, 1, 4, 6)

    def test_heterogeneous_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            meta = grp.create_group("META")
            # IP 0 has 4 comps, IP 1 has 6 — heterogeneous.
            meta.create_dataset(
                "COMPONENTS",
                data=np.array([b"0.1.2.P,Mz,My,T;0.1.2.P,Mz,My,T,Vy,Vz"]),
            )
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="heterogeneous"):
                _mline.parse_section_codes_from_meta(
                    f["bucket"], bracket_key="hetero",
                )

    def test_unknown_section_name_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            meta = grp.create_group("META")
            meta.create_dataset(
                "COMPONENTS",
                data=np.array([b"0.1.2.P,Mzzz"]),
            )
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="unknown section component"):
                _mline.parse_section_codes_from_meta(
                    f["bucket"], bracket_key="bad",
                )

    def test_missing_components_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("bucket")
            grp.create_group("META")  # no COMPONENTS
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="META/COMPONENTS"):
                _mline.parse_section_codes_from_meta(
                    f["bucket"], bracket_key="missing",
                )


class TestReadGpXFromConnectivity:
    def test_round_trip(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import parse_mpco_element_key
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("model_elements")
            ds = grp.create_dataset(
                "74-ForceBeamColumn3d[1000:3]",
                data=np.zeros((1, 3), dtype=np.int32),
            )
            ds.attrs["GP_X"] = np.array([-1.0, 0.0, +1.0])
        elem_key = parse_mpco_element_key("74-ForceBeamColumn3d[1000:3]")
        with h5py.File(path, "r") as f:
            gp_x = _mline.read_gp_x_from_connectivity(
                f["model_elements"], elem_key,
            )
        np.testing.assert_array_equal(gp_x, [-1.0, 0.0, +1.0])

    def test_missing_connectivity_raises(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import parse_mpco_element_key
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            f.create_group("model_elements")
        elem_key = parse_mpco_element_key("74-ForceBeamColumn3d[1000:9]")
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="connectivity"):
                _mline.read_gp_x_from_connectivity(
                    f["model_elements"], elem_key,
                )

    def test_missing_gp_x_attribute_raises(self, tmp_path: Path) -> None:
        from apeGmsh.solvers._element_response import parse_mpco_element_key
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("model_elements")
            grp.create_dataset(
                "74-ForceBeamColumn3d[1000:3]",
                data=np.zeros((1, 3), dtype=np.int32),
            )
            # No GP_X attribute.
        elem_key = parse_mpco_element_key("74-ForceBeamColumn3d[1000:3]")
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="GP_X"):
                _mline.read_gp_x_from_connectivity(
                    f["model_elements"], elem_key,
                )


# =====================================================================
# Bucket-discovery filter tests
# =====================================================================

class TestDiscoverLineStationBuckets:
    def test_skips_non_custom_rule(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("section.force")
            # Standard-rule bucket — should be skipped (only Custom=1000
            # buckets enter the line-stations path).
            sf.create_group("74-ForceBeamColumn3d[1:0:0]")
        with h5py.File(path, "r") as f:
            _, buckets = _mline.discover_line_station_buckets(
                f["ON_ELEMENTS"], canonical_component="axial_force",
            )
        assert buckets == []

    def test_skips_header_idx_nonzero(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("section.force")
            # Heterogeneous-section bucket (hdr=1) — out of scope.
            sf.create_group("74-ForceBeamColumn3d[1000:1:1]")
        with h5py.File(path, "r") as f:
            _, buckets = _mline.discover_line_station_buckets(
                f["ON_ELEMENTS"], canonical_component="axial_force",
            )
        assert buckets == []

    def test_skips_uncatalogued_class(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("section.force")
            sf.create_group("999-FancyNewBeam[1000:1:0]")
        with h5py.File(path, "r") as f:
            _, buckets = _mline.discover_line_station_buckets(
                f["ON_ELEMENTS"], canonical_component="axial_force",
            )
        assert buckets == []

    def test_returns_none_token_for_non_line_station_component(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            f.create_group("ON_ELEMENTS")
        with h5py.File(path, "r") as f:
            token, buckets = _mline.discover_line_station_buckets(
                f["ON_ELEMENTS"], canonical_component="stress_xx",
            )
        assert token is None
        assert buckets == []

    def test_picks_up_force_beam_3d(self, tmp_path: Path) -> None:
        path = tmp_path / "h.h5"
        with h5py.File(path, "w") as f:
            on_elements = f.create_group("ON_ELEMENTS")
            sf = on_elements.create_group("section.force")
            sf.create_group("74-ForceBeamColumn3d[1000:1:0]")
        with h5py.File(path, "r") as f:
            _, buckets = _mline.discover_line_station_buckets(
                f["ON_ELEMENTS"], canonical_component="axial_force",
            )
        assert len(buckets) == 1
        assert buckets[0].elem_key.class_name == "ForceBeamColumn3d"
        assert buckets[0].custom.class_tag == ELE_TAG_ForceBeamColumn3d


# =====================================================================
# Empty-state behaviours
# =====================================================================

class TestEmptySlabReturns:
    def test_no_section_force_group(self, tmp_path: Path) -> None:
        path = tmp_path / "h.mpco"
        node_ids = np.array([1], dtype=np.int32)
        coords = np.zeros((1, 3))
        f, _ = _create_skeleton(
            path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
        )
        f.close()
        # File has MODEL/ELEMENTS but no section.force group.
        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 0)

    def test_non_line_station_component(
        self, fbc3d_aggregated_5ip_mpco: Path,
    ) -> None:
        # ``stress_xx`` has no line-stations routing → empty slab even
        # when the file does have line-stations buckets.
        with Results.from_mpco(fbc3d_aggregated_5ip_mpco) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="stress_xx")
        assert slab.values.shape == (2, 0)


# =====================================================================
# Multi-bucket stitching
# =====================================================================

class TestMultiBucketStitching:
    def test_two_buckets_different_n_ip(self, tmp_path: Path) -> None:
        """Stitch a 5-IP bucket with a 3-IP bucket sharing the same class."""
        path = tmp_path / "stitched.mpco"
        node_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        coords = np.array(
            [[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0]], dtype=np.float64,
        )
        f, stage_name = _create_skeleton(
            path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
        )
        try:
            section_names = ("P", "Mz", "My", "T", "Vy", "Vz")
            component_layout = tuple(_MPCO_TO_CANONICAL[n] for n in section_names)

            # Bucket 1: 5 IPs, custom_rule_idx = 1, elements 1, 2.
            gp_x_1 = np.linspace(-1.0, 1.0, 5)
            ids_1 = np.array([1, 2], dtype=np.int32)
            flat_1, _ = _build_flat(
                component_layout=component_layout,
                n_ip=5, element_ids=ids_1, n_steps=1,
            )
            _add_line_station_bucket(
                f[stage_name],
                class_tag=ELE_TAG_ForceBeamColumn3d,
                class_name="ForceBeamColumn3d",
                custom_rule_idx=1, header_idx=0,
                element_ids=ids_1, gp_x=gp_x_1,
                section_names=section_names, flat_data=flat_1,
            )

            # Bucket 2: 3 IPs, custom_rule_idx = 2, element 3.
            gp_x_2 = np.array([-0.7745966, 0.0, +0.7745966])
            ids_2 = np.array([3], dtype=np.int32)
            flat_2, _ = _build_flat(
                component_layout=component_layout,
                n_ip=3, element_ids=ids_2, n_steps=1,
            )
            _add_line_station_bucket(
                f[stage_name],
                class_tag=ELE_TAG_ForceBeamColumn3d,
                class_name="ForceBeamColumn3d",
                custom_rule_idx=2, header_idx=0,
                element_ids=ids_2, gp_x=gp_x_2,
                section_names=section_names, flat_data=flat_2,
            )
        finally:
            f.close()

        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        # 2 elements × 5 IPs + 1 element × 3 IPs = 13 stations.
        assert slab.values.shape == (1, 13)
        # Element_index repeats the ids (per bucket).
        assert sorted(set(slab.element_index.tolist())) == [1, 2, 3]


# =====================================================================
# DispBeamColumn3d catalogued (read-side parity)
# =====================================================================

class TestDispBeamColumnReadable:
    def test_disp_beam_3d_via_mpco(self, tmp_path: Path) -> None:
        """DispBeamColumn3d is in the catalog; MPCO reads work even
        though DomainCapture (Step 2b) cannot drive it directly."""
        path = tmp_path / "dbc3d.mpco"
        node_ids = np.array([1, 2], dtype=np.int32)
        coords = np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64)
        f, stage_name = _create_skeleton(
            path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
        )
        try:
            gp_x = np.array([-0.7745966, 0.0, +0.7745966])
            section_names = ("P", "Mz", "My", "T")
            component_layout = tuple(
                _MPCO_TO_CANONICAL[n] for n in section_names
            )
            element_ids = np.array([1], dtype=np.int32)
            flat, _ = _build_flat(
                component_layout=component_layout,
                n_ip=gp_x.size, element_ids=element_ids, n_steps=1,
            )
            _add_line_station_bucket(
                f[stage_name],
                class_tag=ELE_TAG_DispBeamColumn3d,
                class_name="DispBeamColumn3d",
                custom_rule_idx=1, header_idx=0,
                element_ids=element_ids, gp_x=gp_x,
                section_names=section_names, flat_data=flat,
            )
        finally:
            f.close()
        with Results.from_mpco(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 3)
