"""Mocked MPCO material-state (damage / equivalent plastic strain) reads.

Exercises the META-driven layout resolution against synthetic MPCO
files: a single-component "damage" bucket and a two-component
``d+,d-`` bucket (mirroring ASDConcrete).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_material_io as _mmat
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.solvers._element_response import (
    ELE_TAG_FourNodeTetrahedron, IntRule,
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


def _add_tet_connectivity(
    stage_grp: "h5py.Group", *, element_ids: np.ndarray,
) -> None:
    conn_key = (
        f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron"
        f"[{IntRule.Tet_GL_1}:0]"
    )
    bucket = np.column_stack([
        element_ids.astype(np.int32),
        np.zeros((element_ids.size, 4), dtype=np.int32),
    ])
    stage_grp["MODEL/ELEMENTS"].create_dataset(conn_key, data=bucket)


def _add_material_state_bucket(
    stage_grp: "h5py.Group", *,
    element_ids: np.ndarray, n_components: int,
    component_symbols: tuple[str, ...],
    flat_data: np.ndarray,
    token: str = "material.damage",
) -> str:
    """Write one material-state bucket with given component symbols.

    ``flat_data`` shape: (T, E, n_GP * n_components). For
    FourNodeTetrahedron with Tet_GL_1, n_GP=1, so shape becomes
    (T, E, n_components).
    """
    on_elements = stage_grp["RESULTS/ON_ELEMENTS"]
    token_grp = on_elements.require_group(token)
    bucket_key = (
        f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron"
        f"[{IntRule.Tet_GL_1}:0:0]"
    )
    bg = token_grp.create_group(bucket_key)
    bg.attrs["NUM_COLUMNS"] = np.array([n_components], dtype=np.int32)

    n_gp = 1   # Tet_GL_1
    meta = bg.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY",
        data=np.ones((n_gp, 1), dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS",
        data=np.arange(n_gp, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset(
        "NUM_COMPONENTS",
        data=np.full((n_gp, 1), n_components, dtype=np.int32),
    )
    meta.create_dataset(
        "COMPONENTS",
        data=np.array([
            ";".join(
                f"0.1.4.{','.join(component_symbols)}"
                for _ in range(n_gp)
            ).encode("ascii"),
        ]),
    )

    bg.create_dataset(
        "ID", data=element_ids.reshape(-1, 1).astype(np.int32),
    )
    data = bg.create_group("DATA")
    T = flat_data.shape[0]
    for k in range(T):
        ds = data.create_dataset(
            f"STEP_{k}", data=flat_data[k].astype(np.float64),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * 1.0
    return bucket_key


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def two_component_damage(tmp_path: Path):
    """Two FourNodeTetrahedrons, each carrying ``d+,d-`` damage."""
    path = tmp_path / "damage_2c.mpco"
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    coords = np.zeros((5, 3), dtype=np.float64)
    n_steps = 2
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=n_steps, dt=0.1,
    )
    stage = f[stage_name]
    eids = np.array([10, 11], dtype=np.int64)
    _add_tet_connectivity(stage, element_ids=eids)

    # 1 GP × 2 components (d+, d-).
    flat = np.empty((n_steps, eids.size, 2), dtype=np.float64)
    for t in range(n_steps):
        for e_idx in range(eids.size):
            # d+ encodes as t*100 + e_idx*10 + 1, d- as +2.
            flat[t, e_idx, 0] = t * 100.0 + e_idx * 10.0 + 1.0
            flat[t, e_idx, 1] = t * 100.0 + e_idx * 10.0 + 2.0
    _add_material_state_bucket(
        stage, element_ids=eids, n_components=2,
        component_symbols=("d+", "d-"),
        flat_data=flat,
        token="material.damage",
    )
    f.close()
    return path, eids, flat


@pytest.fixture
def single_component_damage(tmp_path: Path):
    """Single ``damage`` value per GP (e.g. simpler damage models)."""
    path = tmp_path / "damage_1c.mpco"
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.zeros((4, 3), dtype=np.float64)
    n_steps = 2
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=n_steps, dt=0.1,
    )
    stage = f[stage_name]
    eids = np.array([42], dtype=np.int64)
    _add_tet_connectivity(stage, element_ids=eids)
    flat = np.array([[[0.5]], [[0.9]]], dtype=np.float64)   # T=2, E=1, comp=1
    _add_material_state_bucket(
        stage, element_ids=eids, n_components=1,
        component_symbols=("d",),
        flat_data=flat,
        token="material.damage",
    )
    f.close()
    return path, eids, flat


@pytest.fixture
def equivalent_plastic_strain_2c(tmp_path: Path):
    """``PLE+, PLE-`` two-component plastic strain."""
    path = tmp_path / "eps_pl_2c.mpco"
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.zeros((4, 3), dtype=np.float64)
    n_steps = 1
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=n_steps, dt=0.1,
    )
    stage = f[stage_name]
    eids = np.array([7], dtype=np.int64)
    _add_tet_connectivity(stage, element_ids=eids)
    flat = np.array([[[0.001, 0.002]]], dtype=np.float64)
    _add_material_state_bucket(
        stage, element_ids=eids, n_components=2,
        component_symbols=("PLE+", "PLE-"),
        flat_data=flat,
        token="material.equivalentPlasticStrain",
    )
    f.close()
    return path, eids, flat


# =====================================================================
# Token / canonical routing
# =====================================================================

class TestParentTokenForCanonical:
    @pytest.mark.parametrize("canonical, expected", [
        ("damage", "damage"),
        ("damage_tension", "damage"),
        ("damage_compression", "damage"),
        ("damage_3", "damage"),
        ("equivalent_plastic_strain", "equivalent_plastic_strain"),
        ("equivalent_plastic_strain_tension", "equivalent_plastic_strain"),
        ("stress_xx", None),
        ("displacement_x", None),
    ])
    def test_canonical_to_parent(
        self, canonical: str, expected: str | None,
    ) -> None:
        assert _mmat.parent_token_for_canonical(canonical) == expected


# =====================================================================
# META segment parsing
# =====================================================================

class TestParseMetaComponents:
    def test_two_component_damage(self) -> None:
        # Single GP, two segments per IP repeated for multiple IPs.
        blob = b"0.1.4.d+,d-;0.1.4.d+,d-"
        layout, n = _mmat.parse_meta_components(
            blob, parent_token="damage", bracket_key="dummy",
        )
        assert layout == ("damage_tension", "damage_compression")
        assert n == 2

    def test_single_component_damage_uses_bare_canonical(self) -> None:
        blob = b"0.1.4.d"
        layout, n = _mmat.parse_meta_components(
            blob, parent_token="damage", bracket_key="dummy",
        )
        assert layout == ("damage",)
        assert n == 1

    def test_unknown_symbol_falls_back_to_index(self) -> None:
        blob = b"0.1.4.foo,bar"
        layout, n = _mmat.parse_meta_components(
            blob, parent_token="damage", bracket_key="dummy",
        )
        assert layout == ("damage_0", "damage_1")
        assert n == 2

    def test_heterogeneous_ips_raise(self) -> None:
        blob = b"0.1.4.d+,d-;0.1.4.d+"
        with pytest.raises(ValueError, match="heterogeneous"):
            _mmat.parse_meta_components(
                blob, parent_token="damage", bracket_key="dummy",
            )

    def test_empty_blob_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _mmat.parse_meta_components(
                b"", parent_token="damage", bracket_key="dummy",
            )

    def test_eps_pl_naming(self) -> None:
        blob = b"0.1.4.PLE+,PLE-"
        layout, _ = _mmat.parse_meta_components(
            blob, parent_token="equivalent_plastic_strain",
            bracket_key="dummy",
        )
        assert layout == (
            "equivalent_plastic_strain_tension",
            "equivalent_plastic_strain_compression",
        )


# =====================================================================
# Bucket discovery
# =====================================================================

class TestDiscoverMaterialStateBuckets:
    def test_finds_damage_bucket(self, two_component_damage) -> None:
        path, _, _ = two_component_damage
        with h5py.File(path, "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            mpco_name, buckets = _mmat.discover_material_state_buckets(
                on_elem, canonical_component="damage_tension",
            )
        assert mpco_name == "material.damage"
        assert len(buckets) == 1
        assert buckets[0].parent_token == "damage"

    def test_eps_pl_canonical_finds_eps_pl_bucket(
        self, equivalent_plastic_strain_2c,
    ) -> None:
        path, _, _ = equivalent_plastic_strain_2c
        with h5py.File(path, "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = _mmat.discover_material_state_buckets(
                on_elem, canonical_component="equivalent_plastic_strain_tension",
            )
        assert len(buckets) == 1

    def test_non_material_canonical_returns_none(
        self, two_component_damage,
    ) -> None:
        path, _, _ = two_component_damage
        with h5py.File(path, "r") as f:
            on_elem = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            mpco_name, buckets = _mmat.discover_material_state_buckets(
                on_elem, canonical_component="stress_xx",
            )
        assert mpco_name is None
        assert buckets == []


# =====================================================================
# End-to-end via Results facade
# =====================================================================

class TestEndToEnd:
    def test_two_component_damage_separates_canonicals(
        self, two_component_damage,
    ) -> None:
        path, eids, flat = two_component_damage
        r = Results.from_mpco(str(path))
        # Available components surfaces both per-segment canonicals.
        sid = r._reader.stages()[0].id
        comps = r._reader.available_components(sid, ResultLevel.GAUSS)
        assert "damage_tension" in comps
        assert "damage_compression" in comps

        # Tension picks d+ column.
        slab_t = r.elements.gauss.get(component="damage_tension", time=0)
        # 2 elements * 1 GP = 2 columns.
        assert slab_t.values.shape == (1, 2)
        # Encoded values: e=0 d+ = 1, e=1 d+ = 11.
        np.testing.assert_array_almost_equal(slab_t.values[0], [1.0, 11.0])

        # Compression picks d- column.
        slab_c = r.elements.gauss.get(component="damage_compression", time=0)
        np.testing.assert_array_almost_equal(slab_c.values[0], [2.0, 12.0])

        r._reader.close()

    def test_late_step_picks_correct_columns(
        self, two_component_damage,
    ) -> None:
        path, _, _ = two_component_damage
        r = Results.from_mpco(str(path))
        slab_t = r.elements.gauss.get(component="damage_tension", time=1)
        # t=1: e=0 → 101, e=1 → 111.
        np.testing.assert_array_almost_equal(slab_t.values[0], [101.0, 111.0])
        r._reader.close()

    def test_single_component_uses_bare_canonical(
        self, single_component_damage,
    ) -> None:
        path, _, _ = single_component_damage
        r = Results.from_mpco(str(path))
        sid = r._reader.stages()[0].id
        comps = r._reader.available_components(sid, ResultLevel.GAUSS)
        # Bare ``damage`` (no suffix) since META has one symbol.
        assert "damage" in comps
        # And split canonicals are NOT present.
        assert "damage_tension" not in comps
        slab = r.elements.gauss.get(component="damage", time=1)
        np.testing.assert_array_almost_equal(slab.values[0], [0.9])
        r._reader.close()

    def test_bare_damage_on_two_component_returns_empty(
        self, two_component_damage,
    ) -> None:
        # Asking for ``damage`` (bare) on a multi-component bucket
        # should return an empty slab — the bucket's META exposes
        # ``damage_tension`` / ``damage_compression``, not ``damage``.
        path, _, _ = two_component_damage
        r = Results.from_mpco(str(path))
        slab = r.elements.gauss.get(component="damage", time=0)
        assert slab.values.shape[1] == 0
        r._reader.close()

    def test_eps_pl_two_component(
        self, equivalent_plastic_strain_2c,
    ) -> None:
        path, _, _ = equivalent_plastic_strain_2c
        r = Results.from_mpco(str(path))
        slab_t = r.elements.gauss.get(
            component="equivalent_plastic_strain_tension", time=0,
        )
        np.testing.assert_array_almost_equal(slab_t.values[0], [0.001])
        slab_c = r.elements.gauss.get(
            component="equivalent_plastic_strain_compression", time=0,
        )
        np.testing.assert_array_almost_equal(slab_c.values[0], [0.002])
        r._reader.close()


# =====================================================================
# Error paths
# =====================================================================

class TestMisshapedBucketSkipped:
    def test_num_columns_mismatch_raises_inside_resolver(
        self, tmp_path: Path,
    ) -> None:
        """A NUM_COLUMNS that disagrees with META should raise from the
        resolver — the dispatch path catches this and returns an empty
        slab in the user-facing read, but the resolver itself is strict."""
        path = tmp_path / "bad.mpco"
        f, stage_name = _create_skeleton(
            path, node_ids=np.array([1], dtype=np.int64),
            coords=np.zeros((1, 3)), n_steps=1, dt=0.1,
        )
        stage = f[stage_name]
        _add_tet_connectivity(stage, element_ids=np.array([10]))
        # Write bucket where NUM_COLUMNS says 5 but META says 2.
        bg = stage["RESULTS/ON_ELEMENTS"].require_group("damage").create_group(
            f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron"
            f"[{IntRule.Tet_GL_1}:0:0]"
        )
        bg.attrs["NUM_COLUMNS"] = np.array([5], dtype=np.int32)
        meta = bg.create_group("META")
        meta.create_dataset(
            "MULTIPLICITY", data=np.array([[1]], dtype=np.int32),
        )
        meta.create_dataset(
            "GAUSS_IDS", data=np.array([[0]], dtype=np.int32),
        )
        meta.create_dataset(
            "NUM_COMPONENTS", data=np.array([[2]], dtype=np.int32),
        )
        meta.create_dataset(
            "COMPONENTS", data=np.array([b"0.1.4.d+,d-"]),
        )
        bg.create_dataset("ID", data=np.array([[10]], dtype=np.int32))
        bg.create_group("DATA").create_dataset(
            "STEP_0", data=np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        )
        f.close()

        with h5py.File(path, "r") as g:
            on_elem = g["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
            _, buckets = _mmat.discover_material_state_buckets(
                on_elem, canonical_component="damage_tension",
            )
            assert len(buckets) == 1
            bucket_grp = on_elem[buckets[0].mpco_group_name][
                buckets[0].bracket_key
            ]
            with pytest.raises(ValueError, match="NUM_COLUMNS"):
                _mmat.resolve_material_state_layout(bucket_grp, buckets[0])
