"""Phase 11c Step 2b — mocked MPCO layered-shell reads.

Builds a synthetic MPCO-shaped HDF5 file and exercises the layer I/O
module + ``MPCOReader.read_layers`` end-to-end without OpenSees.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_layer_io as _mlayer
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.opensees._response_catalog import (
    ELE_TAG_ASDShellQ4,
    LAYER_CATALOG,
    RESPONSE_CATALOG,
    IntRule,
)


from tests.conftest import _stub_model_h5_path

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
    nodes: np.ndarray | None = None,
) -> None:
    """MPCO's flat ``(elemTag, n1..nK)`` rows; ``nodes`` is ``(E, K)``.

    Without ``nodes`` the rows are zeros (no element 0 is ever read).
    """
    conn_key = f"{class_tag}-{class_name}[{surface_int_rule}:{custom_rule_idx}]"
    conn_grp = stage_grp["MODEL/ELEMENTS"]
    if nodes is None:
        data = np.zeros((element_ids.size, 5), dtype=np.int32)
    else:
        data = np.column_stack([element_ids, nodes]).astype(np.int32)
    conn_grp.create_dataset(conn_key, data=data)


def _add_local_axes(
    stage_grp: "h5py.Group", *,
    element_ids: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    """The writer's layout (``MPCORecorder::writeModelLocalAxes``): one
    flat ``ID`` + ``LOCAL_AXES`` pair per stage, scalar-first rows."""
    la_grp = stage_grp["MODEL"].require_group("LOCAL_AXES")
    la_grp.create_dataset("ID", data=element_ids.astype(np.int32))
    la_grp.create_dataset("LOCAL_AXES", data=quaternions.astype(np.float64))


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
    _add_local_axes(stage, element_ids=element_ids, quaternions=quats)

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
                model_grp=stage["MODEL"],
            )
        np.testing.assert_array_almost_equal(quats, simple_layer_mpco["quats"])


# =====================================================================
# Shell frames the file does not store (#1996): computed from the file's
# connectivity and coordinates, NaN + a warning when they cannot be
# =====================================================================

#: Every class the MPCO layer reader discovers; a class joining
#: ``LAYER_CATALOG`` joins this matrix.
LAYER_CLASSES = sorted({c for c, _tok in LAYER_CATALOG})

_C30 = np.cos(np.radians(30.0))
_S30 = np.sin(np.radians(30.0))
#: ``(u, v)``: the unit first edge and the unit in-plane vector normal to
#: it. The oracle local axes are written out by hand: ``x = u``,
#: ``y = v``, ``z = u x v``.
ORIENTATIONS: dict[str, tuple[tuple[float, ...], tuple[float, ...],
                              tuple[float, ...]]] = {
    "xy": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "tilted_30_about_x": ((1, 0, 0), (0, _C30, _S30), (0, -_S30, _C30)),
    "vertical_xz": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "vertical_yz": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "skew": ((2 / 3, 1 / 3, 2 / 3), (-2 / 3, 2 / 3, 1 / 3),
             (-1 / 3, -2 / 3, 2 / 3)),
}


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Textbook unit quaternion ``(w, x, y, z)`` -> rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _element_coords(class_name: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Corner nodes in connectivity order: a 2 x 1 quad, a 2 x 1 right
    triangle, or a quad's 9 nodes, off the origin."""
    o = np.array([3.0, -1.0, 2.0])
    if LAYER_CATALOG[(class_name, "fiber_stress")].coord_system.startswith(
            "simplex_tri"):
        return np.array([o, o + 2 * u, o + v])
    corners = [o, o + 2 * u, o + 2 * u + v, o + v]
    if class_name == "ShellMITC9":
        mids = [(corners[i] + corners[(i + 1) % 4]) / 2 for i in range(4)]
        return np.array(corners + mids + [o + u + v / 2])
    return np.array(corners)


def _frame_file(
    path: Path, class_name: str, coords: np.ndarray, *,
    element_ids: np.ndarray, stored: np.ndarray | None = None,
    node_ids: np.ndarray | None = None,
) -> str:
    """A one-stage MPCO file whose elements all share ``coords``'s nodes,
    with a one-layer bucket; ``stored`` adds the writer's LOCAL_AXES."""
    layout = LAYER_CATALOG[(class_name, "fiber_stress")]
    n_sgp = RESPONSE_CATALOG[
        (class_name, layout.surface_int_rule, "stress")].n_gauss_points
    if node_ids is None:
        node_ids = np.arange(101, 101 + coords.shape[0], dtype=np.int64)
    f, stage_name = _create_skeleton(
        path, node_ids=node_ids, coords=coords, n_steps=1, dt=1.0,
    )
    stage = f[stage_name]
    kw = dict(class_tag=layout.class_tag, class_name=class_name,
              surface_int_rule=layout.surface_int_rule, custom_rule_idx=0)
    _add_shell_connectivity(
        stage, element_ids=element_ids,
        nodes=np.tile(np.arange(101, 101 + coords.shape[0]),
                      (element_ids.size, 1)), **kw,
    )
    if stored is not None:
        _add_local_axes(stage, element_ids=element_ids[:len(stored)],
                        quaternions=stored)
    _add_layered_section(
        stage, section_tag=1, section_class="LayeredShellFiberSection",
        element_sgp_pairs=[(int(e), g) for e in element_ids
                           for g in range(n_sgp)],
        layer_thickness=np.array([0.01]),
        layer_material=np.array([1], dtype=np.int64),
    )
    _add_layer_bucket(
        stage, element_ids=element_ids, n_sgp=n_sgp, n_layers=1,
        flat_data=np.zeros((1, element_ids.size, n_sgp)), **kw,
    )
    f.close()
    return stage_name


def _read_frames(path: Path, element_ids: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as f:
        stage = f["MODEL_STAGE[1]"]
        _, (bucket,) = _mlayer.discover_layer_buckets(
            stage["RESULTS/ON_ELEMENTS"], canonical_component="fiber_stress",
        )
        local_axes = stage["MODEL/LOCAL_AXES"] if "LOCAL_AXES" in stage["MODEL"] else None
        return _mlayer.read_local_axes_quaternions(
            local_axes, bucket.elem_key, element_ids,
            model_grp=stage["MODEL"],
        )


class TestComputedShellFrame:
    @pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
    @pytest.mark.parametrize("class_name", LAYER_CLASSES)
    def test_frame_is_the_closed_form_local_axes(
        self, tmp_path: Path, class_name: str, orientation: str,
    ) -> None:
        u, v, z = (np.asarray(a, dtype=float) for a in ORIENTATIONS[orientation])
        np.testing.assert_allclose(np.cross(u, v), z, atol=1e-15)
        eids = np.array([7, 9], dtype=np.int64)
        path = tmp_path / "frame.mpco"
        _frame_file(path, class_name, _element_coords(class_name, u, v),
                    element_ids=eids)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quats = _read_frames(path, eids)
        assert quats.shape == (2, 4)
        for q in quats:
            # Rows of the rotation global -> local are the local axes.
            np.testing.assert_allclose(
                _quat_to_matrix(q), np.stack([u, v, z]), atol=1e-12,
                err_msg=f"{class_name} {orientation}")

    def test_tilted_frame_is_the_c7_closed_form(self, tmp_path: Path) -> None:
        """``(cos 15, -sin 15, 0, 0)`` is what DomainCapture (C7) writes and
        what the MPCO writer stores for a beam with the same axes."""
        u, v, _z = (np.asarray(a, dtype=float)
                    for a in ORIENTATIONS["tilted_30_about_x"])
        path = tmp_path / "tilted.mpco"
        _frame_file(path, "ASDShellQ4", _element_coords("ASDShellQ4", u, v),
                    element_ids=np.array([5]))
        q = _read_frames(path, np.array([5]))[0]
        half = np.radians(15.0)
        np.testing.assert_allclose(
            q * np.sign(q[0]), [np.cos(half), -np.sin(half), 0, 0], atol=1e-14)

    @pytest.mark.parametrize("class_name", LAYER_CLASSES)
    def test_results_layer_slab_carries_the_tilted_frame(
        self, tmp_path: Path, class_name: str,
    ) -> None:
        """End to end through ``Results.from_mpco``: no identity row."""
        u, v, z = (np.asarray(a, dtype=float)
                   for a in ORIENTATIONS["tilted_30_about_x"])
        eids = np.array([11, 12], dtype=np.int64)
        path = tmp_path / "e2e.mpco"
        _frame_file(path, class_name, _element_coords(class_name, u, v),
                    element_ids=eids)
        results = Results.from_mpco(str(path), model_h5=_stub_model_h5_path())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _mlayer.ShellFrameUnavailableWarning)
                slab = results.elements.layers.get(component="fiber_stress")
        finally:
            results._reader.close()
        q = np.asarray(slab.local_axes_quaternion)
        assert q.shape[0] == np.asarray(slab.values).shape[1] > 0
        for row in q:
            np.testing.assert_allclose(
                _quat_to_matrix(row), np.stack([u, v, z]), atol=1e-12)

    def test_stored_frame_wins_and_the_rest_are_computed(
        self, tmp_path: Path,
    ) -> None:
        u, v, z = (np.asarray(a, dtype=float) for a in ORIENTATIONS["vertical_xz"])
        eids = np.array([3, 4], dtype=np.int64)
        stored = np.array([[0.0, 0.0, 0.0, 1.0]])     # element 3 only
        path = tmp_path / "mixed.mpco"
        _frame_file(path, "ShellMITC4", _element_coords("ShellMITC4", u, v),
                    element_ids=eids, stored=stored)
        quats = _read_frames(path, eids)
        np.testing.assert_array_equal(quats[0], stored[0])
        np.testing.assert_allclose(
            _quat_to_matrix(quats[1]), np.stack([u, v, z]), atol=1e-12)


class TestShellFrameUnavailable:
    """Never a synthesised identity: NaN, and a warning naming the rows."""

    def _assert_nan_and_warned(
        self, path: Path, eids: np.ndarray, match: str,
    ) -> np.ndarray:
        with pytest.warns(_mlayer.ShellFrameUnavailableWarning, match=match):
            quats = _read_frames(path, eids)
        assert np.isnan(quats).all()
        return quats

    def test_node_missing_from_the_file(self, tmp_path: Path) -> None:
        u, v, _z = (np.asarray(a, dtype=float) for a in ORIENTATIONS["xy"])
        coords = _element_coords("ASDShellQ4", u, v)
        path = tmp_path / "no_node.mpco"
        # Node 104 (the fourth corner) is renumbered away.
        _frame_file(path, "ASDShellQ4", coords, element_ids=np.array([20]),
                    node_ids=np.array([101, 102, 103, 999]))
        self._assert_nan_and_warned(
            path, np.array([20]), r"node missing.*elements \[20\]")

    def test_element_missing_from_the_connectivity(self, tmp_path: Path) -> None:
        u, v, _z = (np.asarray(a, dtype=float) for a in ORIENTATIONS["xy"])
        path = tmp_path / "no_row.mpco"
        _frame_file(path, "ShellDKGT", _element_coords("ShellDKGT", u, v),
                    element_ids=np.array([20]))
        self._assert_nan_and_warned(
            path, np.array([21]), r"no row in MODEL/ELEMENTS.*\[21\]")

    def test_degenerate_geometry(self, tmp_path: Path) -> None:
        coords = np.zeros((4, 3))
        path = tmp_path / "degenerate.mpco"
        _frame_file(path, "ASDShellQ4", coords, element_ids=np.array([20]))
        self._assert_nan_and_warned(path, np.array([20]), "zero length")

    def test_no_connectivity_dataset(self, simple_layer_mpco) -> None:
        # No stored frames and no connectivity dataset for the class.
        with h5py.File(simple_layer_mpco["path"], "a") as f:
            del f["MODEL_STAGE[1]/MODEL/LOCAL_AXES"]
            key = next(iter(f["MODEL_STAGE[1]/MODEL/ELEMENTS"]))
            del f[f"MODEL_STAGE[1]/MODEL/ELEMENTS/{key}"]
        self._assert_nan_and_warned(
            simple_layer_mpco["path"], np.array([20, 21]),
            r"no MODEL/ELEMENTS/.* connectivity.*\[20, 21\]")

    def test_unrecognised_local_axes_layout_warns(self, tmp_path: Path) -> None:
        """A LOCAL_AXES group not in the writer's flat layout is not read
        as a frame; the shells' frames are computed instead."""
        u, v, z = (np.asarray(a, dtype=float) for a in ORIENTATIONS["skew"])
        path = tmp_path / "odd_layout.mpco"
        stage_name = _frame_file(
            path, "ASDShellT3", _element_coords("ASDShellT3", u, v),
            element_ids=np.array([20]))
        with h5py.File(path, "a") as f:
            odd = f[stage_name]["MODEL"].create_group("LOCAL_AXES").create_group("X")
            odd.create_dataset("QUATERNIONS", data=np.array([[1.0, 0, 0, 0]]))
        with pytest.warns(_mlayer.ShellFrameUnavailableWarning,
                          match="expected the writer's ID and LOCAL_AXES"):
            quats = _read_frames(path, np.array([20]))
        np.testing.assert_allclose(
            _quat_to_matrix(quats[0]), np.stack([u, v, z]), atol=1e-12)


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
            with pytest.warns(
                _mlayer.LayerBucketUnreadWarning,
                match=r"999-WeirdShell\[201:0:0\]: class 'WeirdShell' has "
                      r"no layered-shell entry",
            ):
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
        results = Results.from_mpco(str(simple_layer_mpco["path"]), model_h5=_stub_model_h5_path())
        slab = results.elements.layers.get(component="fiber_stress")
        assert slab.values.shape == (2, 24)
        assert slab.local_axes_quaternion.shape == (24, 4)
        results._reader.close()

    def test_available_components(self, simple_layer_mpco) -> None:
        results = Results.from_mpco(str(simple_layer_mpco["path"]), model_h5=_stub_model_h5_path())
        comps = results._reader.available_components(
            results._reader.stages()[0].id, ResultLevel.LAYERS,
        )
        assert comps == ["fiber_stress"]
        results._reader.close()
