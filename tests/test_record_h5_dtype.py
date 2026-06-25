"""Unit tests for the symmetric record-compound dtype helper.

Covers :mod:`apeGmsh.mesh._record_h5`:

* :func:`make_record_dtype` produces the 4-field outer compound for
  any payload dtype.
* Per-record-type payload dtype factories produce the expected
  field names and shapes.
* A round-trip of a numpy compound row through an in-memory HDF5
  file preserves every field bit-identically (the symmetric
  contract has to survive write → read).
"""
from __future__ import annotations

import io

import h5py
import numpy as np

from apeGmsh.mesh._record_h5 import (
    element_load_payload_dtype,
    interpolation_payload_dtype,
    make_record_dtype,
    mass_payload_dtype,
    node_group_payload_dtype,
    node_pair_payload_dtype,
    node_to_surface_payload_dtype,
    nodal_load_payload_dtype,
    sp_payload_dtype,
    surface_coupling_payload_dtype,
)


def test_make_record_dtype_outer_compound_shape() -> None:
    """Outer compound has the four contract fields in the documented order."""
    dt = make_record_dtype(node_pair_payload_dtype())
    assert dt.names == ("target_kind", "target", "payload_kind", "payload")


def test_node_pair_payload_fields() -> None:
    dt = node_pair_payload_dtype()
    # ``name`` (utf-8) added in neutral schema 2.5.0 (Phase 2,
    # additive); old 2.4.0 files lack the trailing field and the
    # reader probes presence.
    assert dt.names == (
        "master_node", "slave_node", "dofs", "offset", "penalty_stiffness",
        "name",
        # Retained-node DOFs for equal_dof_mixed (ADR 0069, schema 2.17.0).
        "master_dofs",
    )
    assert dt["master_node"] == np.dtype(np.int64)
    assert dt["offset"].shape == (3,)


# CouplingControl columns shared by node_group + interpolation payloads
# (neutral schema 2.12.0; host auto-scalers 2.13.0; EmbeddedNodeControl
# pressure tie 2.18.0).
_CPL_FIELDS = (
    "cpl_has", "cpl_k", "cpl_kr", "cpl_enforce", "cpl_dtcr", "cpl_absolute",
    "cpl_k_auto", "cpl_k_alpha", "cpl_host", "cpl_wcap",
    "cpl_pressure", "cpl_kp",
)

#: Default (control=None) values for the _CPL_FIELDS tail when building
#: payload rows by hand.
_CPL_NONE = (
    np.uint8(0), float("nan"), float("nan"), np.uint8(0), float("nan"),
    np.uint8(0), np.uint8(0), float("nan"), np.int64(-1), float("nan"),
    np.uint8(0), float("nan"),
)


def test_node_group_payload_fields() -> None:
    dt = node_group_payload_dtype()
    assert dt.names == (
        "master_node", "slave_nodes", "dofs", "offsets", "plane_normal",
        "name",
        # Fork coupling knobs (neutral schema 2.12.0 / 2.13.0)
        *_CPL_FIELDS,
        # LadrunoRigidBody emission (ADR 0071; schema 2.19.0 as_element/mass,
        # 2.20.0 omega).
        "as_element", "mass", "omega",
    )
    assert dt["plane_normal"].shape == (3,)
    assert dt["omega"].shape == (3,)
    assert dt["cpl_has"] == np.dtype(np.uint8)
    assert dt["cpl_k"] == np.dtype(np.float64)
    assert dt["cpl_host"] == np.dtype(np.int64)


def test_interpolation_payload_fields() -> None:
    dt = interpolation_payload_dtype()
    assert dt.names == (
        "slave_node", "master_nodes", "weights", "dofs",
        "projected_point", "parametric_coords",
        "name",
        # ASDEmbeddedNodeElement options (neutral schema 2.8.0)
        "stiffness", "stiffness_p", "has_stiffness_p",
        "rotational", "pressure", "excess",
        # enforce route (neutral schema 2.14.0, ADR 0068)
        "enforce",
        # Fork coupling knobs (neutral schema 2.12.0 / 2.13.0)
        *_CPL_FIELDS,
    )
    assert dt["projected_point"].shape == (3,)
    assert dt["parametric_coords"].shape == (2,)
    assert dt["stiffness"] == np.dtype(np.float64)
    assert dt["has_stiffness_p"] == np.dtype(np.uint8)
    assert dt["rotational"] == np.dtype(np.uint8)


def test_surface_coupling_payload_fields() -> None:
    dt = surface_coupling_payload_dtype()
    assert dt.names == (
        "master_nodes", "slave_nodes", "dofs",
        "mortar_operator_shape", "mortar_operator",
        # slave_records, CSR-flattened (tied_contact/mortar lossless
        # round-trip); older files lack these and decode to [].
        "sr_slave_nodes", "sr_master_counts", "sr_master_nodes",
        "sr_weights", "sr_dof_counts", "sr_dofs",
        "sr_projected", "sr_parametric",
        "name",
        # ASDEmbeddedNodeElement options per slave (neutral schema 2.8.0)
        "sr_stiffness", "sr_stiffness_p", "sr_has_stiffness_p",
        "sr_rotational", "sr_pressure", "sr_excess",
        # enforce route per slave (neutral schema 2.14.0, ADR 0068)
        "sr_enforce",
        # CouplingControl knobs per slave (neutral schema 2.12.0;
        # host auto-scalers 2.13.0)
        "sr_cpl_has", "sr_cpl_k", "sr_cpl_kr", "sr_cpl_enforce",
        "sr_cpl_dtcr", "sr_cpl_absolute",
        "sr_cpl_k_auto", "sr_cpl_k_alpha", "sr_cpl_host", "sr_cpl_wcap",
        # EmbeddedNodeControl pressure tie per slave (schema 2.18.0 mirror).
        "sr_cpl_pressure", "sr_cpl_kp",
    )
    assert dt["mortar_operator_shape"].shape == (2,)


def test_node_to_surface_payload_fields() -> None:
    dt = node_to_surface_payload_dtype()
    assert dt.names == (
        "master_node", "slave_nodes", "phantom_nodes",
        "phantom_coords", "dofs",
        "name",
    )


def test_nodal_load_payload_fields() -> None:
    dt = nodal_load_payload_dtype()
    assert dt.names == ("node_id", "force_xyz", "moment_xyz", "name")
    assert dt["force_xyz"].shape == (3,)
    assert dt["moment_xyz"].shape == (3,)


def test_element_load_payload_fields() -> None:
    dt = element_load_payload_dtype()
    assert dt.names == ("element_id", "load_type", "params_json", "name")


def test_sp_payload_fields() -> None:
    dt = sp_payload_dtype()
    assert dt.names == (
        "node_id", "dof", "value", "is_homogeneous", "name",
    )


def test_mass_payload_fields() -> None:
    dt = mass_payload_dtype()
    assert dt.names == ("node_id", "mass", "name")
    assert dt["mass"].shape == (6,)


def _h5_roundtrip(rows: np.ndarray, dataset_name: str = "rows") -> np.ndarray:
    """Write ``rows`` to an in-memory HDF5 file and read them back."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset(dataset_name, data=rows)
    buf.seek(0)
    with h5py.File(buf, "r") as f:
        out: np.ndarray = f[dataset_name][:]
    return out


def test_node_pair_record_roundtrips_through_h5() -> None:
    """A NodePair row survives write → read with all fields intact."""
    dt = make_record_dtype(node_pair_payload_dtype())
    rows = np.empty(1, dtype=dt)
    rows[0] = (
        "node", "42", "rigid_beam",
        (
            10,                          # master_node
            42,                          # slave_node
            np.array([1, 2, 3], dtype=np.int64),
            (0.5, 1.0, -0.25),           # offset
            float("nan"),                # penalty_stiffness (absent)
            "",                          # name (2.5.0; empty = unset)
            np.array([], dtype=np.int64),  # master_dofs (2.17.0; empty = none)
        ),
    )
    out = _h5_roundtrip(rows)
    assert out[0]["target_kind"] == b"node"
    assert out[0]["target"] == b"42"
    assert out[0]["payload_kind"] == b"rigid_beam"
    payload = out[0]["payload"]
    assert int(payload["master_node"]) == 10
    assert int(payload["slave_node"]) == 42
    np.testing.assert_array_equal(payload["dofs"], [1, 2, 3])
    np.testing.assert_array_equal(payload["offset"], [0.5, 1.0, -0.25])
    assert np.isnan(payload["penalty_stiffness"])


def test_sp_record_roundtrips_through_h5() -> None:
    dt = make_record_dtype(sp_payload_dtype())
    rows = np.empty(2, dtype=dt)
    rows[0] = ("node", "5", "sp", (5, 1, 0.0, 1, ""))
    rows[1] = ("node", "5", "sp", (5, 3, 0.002, 0, ""))
    out = _h5_roundtrip(rows)
    assert int(out[0]["payload"]["is_homogeneous"]) == 1
    assert float(out[1]["payload"]["value"]) == 0.002


def test_mass_record_roundtrips_through_h5() -> None:
    dt = make_record_dtype(mass_payload_dtype())
    rows = np.empty(1, dtype=dt)
    rows[0] = (
        "node", "7", "mass",
        (7, (100.0, 100.0, 100.0, 0.0, 0.0, 0.0), ""),
    )
    out = _h5_roundtrip(rows)
    np.testing.assert_array_equal(
        out[0]["payload"]["mass"],
        [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
    )


def test_nodal_load_payload_with_nan_for_absent_moment() -> None:
    """NodalLoad rows can encode "force only, no moment" via NaN-fill."""
    dt = make_record_dtype(nodal_load_payload_dtype())
    rows = np.empty(1, dtype=dt)
    nan = float("nan")
    rows[0] = (
        "node", "12", "nodal",
        (12, (1.0e3, 0.0, 0.0), (nan, nan, nan), ""),
    )
    out = _h5_roundtrip(rows)
    assert np.isnan(out[0]["payload"]["moment_xyz"]).all()
    np.testing.assert_array_equal(
        out[0]["payload"]["force_xyz"], [1.0e3, 0.0, 0.0],
    )


def test_node_group_vlen_offsets_packed_flat() -> None:
    """offsets is a flat vlen-float (3*n_slaves); shape is implied by slaves."""
    dt = make_record_dtype(node_group_payload_dtype())
    rows = np.empty(1, dtype=dt)
    nan = float("nan")
    slaves = np.array([20, 21, 22], dtype=np.int64)
    offsets_flat = np.array(
        [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float64,
    )
    dofs = np.array([1, 2, 3], dtype=np.int64)
    rows[0] = (
        "node", "10", "rigid_diaphragm",
        # Trailing values = the cpl_* CouplingControl columns (neutral schema
        # 2.12.0 / 2.13.0) encoded as "no control", then the LadrunoRigidBody
        # as_element/mass (2.19.0) + omega (2.20.0) tail, encoded as "unset".
        (
            10, slaves, dofs, offsets_flat, (nan, nan, 1.0), "", *_CPL_NONE,
            np.uint8(0), nan, (nan, nan, nan),
        ),
    )
    out = _h5_roundtrip(rows)
    payload = out[0]["payload"]
    np.testing.assert_array_equal(payload["slave_nodes"], slaves)
    np.testing.assert_array_equal(payload["offsets"], offsets_flat)
    # Reshape on read works as documented.
    np.testing.assert_array_equal(
        payload["offsets"].reshape(-1, 3),
        np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )


def test_make_record_dtype_accepts_arbitrary_payload() -> None:
    """The helper does not depend on a specific payload shape — anyone can plug in."""
    payload = np.dtype([("anything", np.int32)])
    outer = make_record_dtype(payload)
    assert outer["payload"] == payload
