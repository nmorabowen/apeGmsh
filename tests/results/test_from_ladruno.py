"""``Results.from_ladruno`` factory + public-API tests (recorder-plan L2b).

Exercises the self-sufficient path: ``from_ladruno`` with **no** ``model_h5``
builds the broker from the ``.ladruno`` itself, so these run against the
committed fork fixtures with no sibling ``model.h5`` and no fork at test time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"
BEAM = FIXTURES / "beam3d.ladruno"
QUAD = FIXTURES / "quad2d.ladruno"


def test_self_sufficient_no_model_h5() -> None:
    r = Results.from_ladruno(TRUSS)
    assert r.fem is not None
    # A broker is bound even without a model_h5 (built from the file).
    assert r.model is not None
    assert r.model.fem is not None


def test_nodes_get_displacement_x() -> None:
    r = Results.from_ladruno(TRUSS)
    slab = r.nodes.get(component="displacement_x")
    assert slab.component == "displacement_x"
    assert slab.values.shape == (4, 3)
    assert slab.node_ids.tolist() == [1, 2, 3]
    # Node 1 fixed in x → zero; tip node 3 grows with the load ramp.
    i3 = slab.node_ids.tolist().index(3)
    assert np.all(np.diff(slab.values[:, i3]) > 0)


def test_minimal_broker_ndm_ndf() -> None:
    # The self-sufficient broker uses ndm from INFO/SPATIAL_DIM and
    # ndf=ndm (a .ladruno doesn't record ndf — see opensees_model docs).
    rt = Results.from_ladruno(TRUSS)
    assert rt.model.ndm == 2 and rt.model.ndf == 2
    rb = Results.from_ladruno(BEAM)
    assert rb.model.ndm == 3 and rb.model.ndf == 3


def test_rejects_non_ladruno(tmp_path: Path) -> None:
    import h5py

    bad = tmp_path / "x.ladruno"
    with h5py.File(bad, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "MPCO"
        info.attrs["FORMAT_VERSION"] = 1
    with pytest.raises(ValueError, match="expected 'Ladruno'"):
        Results.from_ladruno(bad)


def test_time_slice_last_step() -> None:
    r = Results.from_ladruno(TRUSS)
    slab = r.nodes.get(component="displacement_x", time=-1)
    assert slab.values.shape == (1, 3)


# ---------------------------------------------------------------------------
# Energy balance (L4) — recorder -G energy verb
# ---------------------------------------------------------------------------

ENERGY = FIXTURES / "energy.ladruno"
_ENERGY_COLS = ["KE", "IE", "DW", "ULW", "RES", "ERR"]


def test_energy_whole_domain() -> None:
    df = Results.from_ladruno(ENERGY).energy()
    assert list(df.columns) == _ENERGY_COLS
    assert df.index.name == "time"
    assert len(df) == 5                       # energy fixture: 5 transient steps
    assert "ERR" in df.columns                # the headline quality diagnostic


def test_energy_per_region() -> None:
    df = Results.from_ladruno(ENERGY).energy(region=1)
    assert list(df.columns) == _ENERGY_COLS
    assert len(df) == 5


def test_energy_unknown_region_raises() -> None:
    with pytest.raises(ValueError, match="region 999 is not"):
        Results.from_ladruno(ENERGY).energy(region=999)


def test_energy_absent_raises() -> None:
    # truss2d was recorded without -G energy → no ON_DOMAIN/energyBalance.
    with pytest.raises(ValueError, match="no ON_DOMAIN/energyBalance"):
        Results.from_ladruno(TRUSS).energy()


# ---------------------------------------------------------------------------
# Finding B — node envelopes (recorder -envelope) via the public API
# ---------------------------------------------------------------------------

NODE_ENVELOPE = FIXTURES / "node_envelope.ladruno"


def test_node_envelope_dataframe() -> None:
    df = Results.from_ladruno(NODE_ENVELOPE).node_envelope("displacement_x")
    assert list(df.columns) == ["min", "max", "absmax", "arg_step"]
    assert df.index.name == "node_id"
    assert df.index.tolist() == [1, 2, 3]
    # node 3 (tip) extremes from the cyclic pushover path.
    row = df.loc[3]
    np.testing.assert_allclose(row["min"], -0.03, atol=1e-9)
    np.testing.assert_allclose(row["max"], 0.02, atol=1e-9)
    np.testing.assert_allclose(row["absmax"], 0.03, atol=1e-9)
    # arg_step is the recorder's session commitTag (regeneration-relative).
    assert int(row["arg_step"]) >= 0


def test_node_envelope_absent_raises() -> None:
    # truss2d was recorded as a plain time series (no -envelope).
    with pytest.raises(ValueError, match="not recorded with the '-envelope'"):
        Results.from_ladruno(TRUSS).node_envelope("displacement_x")


# ---------------------------------------------------------------------------
# L2b-2 — element value channels via the public API
# ---------------------------------------------------------------------------

def test_public_gauss_stress() -> None:
    r = Results.from_ladruno(QUAD)
    slab = r.elements.gauss.get(component="stress_xx")
    assert slab.values.shape == (2, 4)
    assert slab.element_index.tolist() == [1, 1, 1, 1]


def test_public_element_token_driven() -> None:
    # Token-driven: results.elements.get(component=<file token>).
    r = Results.from_ladruno(QUAD)
    slab = r.elements.get(component="force")
    assert slab.values.shape == (2, 1, 8)
    np.testing.assert_allclose(slab.values[-1].sum(), 0.0, atol=1e-9)


def test_public_line_stations_beam() -> None:
    r = Results.from_ladruno(BEAM)
    slab = r.elements.line_stations.get(component="axial_force")
    assert slab.values.shape == (1, 2)
    np.testing.assert_allclose(slab.station_natural_coord, [-1.0, 1.0])


# ---------------------------------------------------------------------------
# L2b-2 — multi-partition merge (synthesized .part-N fixtures)
# ---------------------------------------------------------------------------

PART0 = FIXTURES / "truss2d.part-0.ladruno"


def test_partition_auto_discovery_merges_fem() -> None:
    # Passing one partition path discovers its sibling and merges.
    r = Results.from_ladruno(PART0)
    assert r._reader.partitions("stage_0") == ["partition_0", "partition_1"]
    assert r.fem.info.n_nodes == 3      # node 2 (boundary) deduplicated
    assert r.fem.info.n_elems == 2      # elements concatenated


def test_partition_node_union_read() -> None:
    r = Results.from_ladruno(PART0)
    slab = r.nodes.get(component="displacement_x")
    assert slab.node_ids.tolist() == [1, 2, 3]


def test_partition_line_station_concat() -> None:
    r = Results.from_ladruno(PART0)
    slab = r.elements.line_stations.get(component="axial_force")
    assert sorted(slab.element_index.tolist()) == [1, 2]
    np.testing.assert_allclose(slab.values[-1], [10.0, 10.0])


def test_partition_explicit_list_no_merge_flag() -> None:
    # merge_partitions=False on a single partition path reads only it.
    r = Results.from_ladruno(PART0, merge_partitions=False)
    assert r._reader.partitions("stage_0") == ["partition_0"]
    assert r.fem.info.n_elems == 1


# ---------------------------------------------------------------------------
# Local axes / beam orientation (L3) — MODEL/LOCAL_AXES
# ---------------------------------------------------------------------------

def test_local_axes_beam_x_axis_is_beam_direction() -> None:
    # beam3d: node1 (0,0,0) → node2 (3,1,2); the local x-axis must point
    # along the beam axis. This validates BOTH the read and the
    # rows-are-axes (transpose) quaternion convention end-to-end.
    la = Results.from_ladruno(BEAM).elements.local_axes()
    assert la.element_ids.tolist() == [1]
    axis = np.array([3.0, 1.0, 2.0])
    axis /= np.linalg.norm(axis)
    np.testing.assert_allclose(la.x_axis[0], axis, atol=1e-6)
    # frame is non-identity (skew beam) and the matrix is orthonormal.
    assert not np.allclose(la.quaternions[0], [1.0, 0.0, 0.0, 0.0])
    R = la.matrices[0]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_local_axes_element_filter() -> None:
    la = Results.from_ladruno(BEAM).elements.local_axes(ids=[1])
    assert la.element_ids.tolist() == [1]
    assert la.quaternions.shape == (1, 4)


def test_local_axes_absent_returns_empty() -> None:
    # truss elements have no local frame → no MODEL/LOCAL_AXES group.
    la = Results.from_ladruno(TRUSS).elements.local_axes()
    assert la.element_ids.size == 0
    assert la.quaternions.shape == (0, 4)
