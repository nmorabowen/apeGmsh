"""``LadrunoReader`` core tests (recorder-plan L2a).

Drives the reader **directly** against committed fork-generated fixtures
(`tests/fixtures/ladruno/*.ladruno`) — no fork at test time, no `Results`
factory, no `model_h5`. Covers identity validation, stage/time discovery,
the self-describing FEM, and chunked nodal reads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._protocol import ResultLevel, ResultsReader

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"
BEAM = FIXTURES / "beam3d.ladruno"
QUAD = FIXTURES / "quad2d.ladruno"
BEZIER = FIXTURES / "bezier_tri6.ladruno"


def test_satisfies_results_reader_protocol() -> None:
    with LadrunoReader(TRUSS) as r:
        assert isinstance(r, ResultsReader)


def test_identity_rejects_non_ladruno(tmp_path: Path) -> None:
    import h5py

    bad = tmp_path / "not.ladruno"
    with h5py.File(bad, "w") as h:
        h.create_group("INFO")  # no GENERATOR
    with pytest.raises(ValueError, match="not a Ladruno file"):
        LadrunoReader(bad)


def test_identity_rejects_wrong_generator(tmp_path: Path) -> None:
    import h5py

    bad = tmp_path / "mpco_like.ladruno"
    with h5py.File(bad, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "MPCO"
        info.attrs["FORMAT_VERSION"] = 1
    with pytest.raises(ValueError, match="expected 'Ladruno'"):
        LadrunoReader(bad)


def test_identity_rejects_unsupported_version(tmp_path: Path) -> None:
    import h5py

    bad = tmp_path / "future.ladruno"
    with h5py.File(bad, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "Ladruno"
        info.attrs["FORMAT_VERSION"] = 999
    with pytest.raises(ValueError, match="not supported"):
        LadrunoReader(bad)


def test_stages_single_static_stage() -> None:
    with LadrunoReader(TRUSS) as r:
        stages = r.stages()
        assert len(stages) == 1
        s = stages[0]
        assert s.id == "stage_0"
        assert s.kind == "static"
        assert s.n_steps == 4  # truss2d fixture runs 4 LoadControl steps


def test_time_vector() -> None:
    with LadrunoReader(TRUSS) as r:
        t = r.time_vector("stage_0")
        assert t.shape == (4,)
        # LoadControl(0.25) over 4 steps → pseudo-time 0.25 .. 1.0
        np.testing.assert_allclose(t, [0.25, 0.5, 0.75, 1.0])


def test_partitions_single() -> None:
    with LadrunoReader(TRUSS) as r:
        assert r.partitions("stage_0") == ["partition_0"]


def test_fem_self_describing() -> None:
    with LadrunoReader(TRUSS) as r:
        fem = r.fem()
        assert fem is not None
        # 3 nodes, 2 truss (line) elements
        assert fem.info.n_nodes == 3
        assert fem.info.n_elems == 2
        # Truss → dim 1 from BASIS TOPOLOGY="line"
        assert all(t.dim == 1 for t in fem.info.types)


def test_available_components_nodes() -> None:
    with LadrunoReader(TRUSS) as r:
        comps = r.available_components("stage_0", ResultLevel.NODES)
        assert "displacement_x" in comps
        assert "displacement_y" in comps


def test_read_nodes_displacement_x() -> None:
    with LadrunoReader(TRUSS) as r:
        slab = r.read_nodes("stage_0", "displacement_x")
        assert slab.component == "displacement_x"
        assert slab.values.shape == (4, 3)        # (T=4, N=3)
        assert slab.node_ids.tolist() == [1, 2, 3]
        # Node 1 is fixed in x → zero displacement across all steps.
        n1 = slab.values[:, slab.node_ids.tolist().index(1)]
        np.testing.assert_allclose(n1, 0.0)
        # Tip node 3 displacement grows monotonically with the load ramp.
        n3 = slab.values[:, slab.node_ids.tolist().index(3)]
        assert np.all(np.diff(n3) > 0)


def test_read_nodes_node_filter() -> None:
    with LadrunoReader(TRUSS) as r:
        slab = r.read_nodes("stage_0", "displacement_x", node_ids=np.array([3]))
        assert slab.node_ids.tolist() == [3]
        assert slab.values.shape == (4, 1)


def test_read_nodes_time_slice_scalar() -> None:
    with LadrunoReader(TRUSS) as r:
        slab = r.read_nodes("stage_0", "displacement_x", time_slice=-1)
        assert slab.values.shape == (1, 3)   # last step only
        assert slab.time.shape == (1,)


def test_read_nodes_unknown_component_empty() -> None:
    with LadrunoReader(TRUSS) as r:
        slab = r.read_nodes("stage_0", "temperature")
        assert slab.values.shape[1] == 0
        assert slab.node_ids.size == 0


def test_beam3d_fem_and_kind() -> None:
    with LadrunoReader(BEAM) as r:
        stages = r.stages()
        assert stages[0].kind == "static"
        fem = r.fem()
        assert fem is not None
        assert fem.info.n_elems == 1
        # ElasticBeam3d → line element, dim 1
        assert all(t.dim == 1 for t in fem.info.types)


# ---------------------------------------------------------------------------
# L2b-2 — element value channels (Gauss / line-station / element)
# ---------------------------------------------------------------------------

def test_gauss_available_and_read() -> None:
    with LadrunoReader(QUAD) as r:
        comps = r.available_components("stage_0", ResultLevel.GAUSS)
        assert {"stress_xx", "stress_yy", "stress_xy",
                "strain_xx", "strain_yy", "strain_xy"} <= set(comps)
        slab = r.read_gauss("stage_0", "stress_xx")
        # FourNodeQuad: 1 element × 4 Gauss points, 2 steps.
        assert slab.values.shape == (2, 4)
        assert slab.element_index.tolist() == [1, 1, 1, 1]
        # natural coords are the 2×2 Gauss-Legendre points (±1/√3).
        assert slab.natural_coords.shape == (4, 2)
        np.testing.assert_allclose(
            np.abs(slab.natural_coords), 1.0 / np.sqrt(3.0), atol=1e-9,
        )


def test_gauss_unknown_component_empty() -> None:
    with LadrunoReader(QUAD) as r:
        slab = r.read_gauss("stage_0", "stress_zz")  # not present in 2D
        assert slab.values.shape[1] == 0


def test_elements_token_driven() -> None:
    # Token-driven element reads: the component IS the file's ON_ELEMENTS
    # token; the slab is the raw NUM_COLUMNS block. Gauss tokens
    # (stress/strain, LEVELS=4) are NOT listed under ELEMENTS.
    with LadrunoReader(QUAD) as r:
        comps = r.available_components("stage_0", ResultLevel.ELEMENTS)
        assert comps == ["force"]                  # quad's element-level token
        assert "stress" not in comps and "strain" not in comps
        slab = r.read_elements("stage_0", "force")
        # quad ``force`` = 4 nodes × 2 dof = 8 raw columns; (T=2, E=1, 8).
        assert slab.values.shape == (2, 1, 8)
        assert slab.element_ids.tolist() == [1]
        # P1_*+P2_* nodal forces self-equilibrate (ΣFx=ΣFy=0) → full block sum 0.
        np.testing.assert_allclose(slab.values[-1].sum(), 0.0, atol=1e-9)


def test_elements_token_unknown_empty() -> None:
    with LadrunoReader(QUAD) as r:
        slab = r.read_elements("stage_0", "basicForce")  # not in quad2d
        assert slab.values.shape[1] == 0


def test_elements_beam_localforce_block() -> None:
    with LadrunoReader(BEAM) as r:
        assert "localForce" in r.available_components(
            "stage_0", ResultLevel.ELEMENTS,
        )
        slab = r.read_elements("stage_0", "localForce")
        # ElasticBeam3d localForce = 12 raw columns (N,Vy,Vz,T,My,Mz ×2 ends).
        assert slab.values.shape == (1, 1, 12)
        assert slab.element_ids.tolist() == [1]


def test_line_stations_beam_two_stations() -> None:
    with LadrunoReader(BEAM) as r:
        comps = r.available_components("stage_0", ResultLevel.LINE_STATIONS)
        assert set(comps) == {
            "axial_force", "shear_y", "shear_z",
            "torsion", "bending_moment_y", "bending_moment_z",
        }
        slab = r.read_line_stations("stage_0", "axial_force")
        # 1 beam × 2 stations.
        assert slab.values.shape == (1, 2)
        assert slab.element_index.tolist() == [1, 1]
        np.testing.assert_allclose(slab.station_natural_coord, [-1.0, 1.0])
        # localForce end-force sign flip → a continuous internal-force
        # diagram: both station values agree for an axially-balanced beam.
        np.testing.assert_allclose(slab.values[-1, 0], slab.values[-1, 1])


def test_line_stations_truss_basic_force() -> None:
    with LadrunoReader(TRUSS) as r:
        assert r.available_components(
            "stage_0", ResultLevel.LINE_STATIONS,
        ) == ["axial_force"]
        slab = r.read_line_stations("stage_0", "axial_force")
        # 2 truss elements × 1 station (basicForce ξ=0).
        assert slab.values.shape == (4, 2)
        np.testing.assert_allclose(slab.station_natural_coord, [0.0, 0.0])
        # Tip load 10 → axial force 10 in both members at the last step.
        np.testing.assert_allclose(slab.values[-1], [10.0, 10.0])


def test_line_stations_element_filter() -> None:
    with LadrunoReader(TRUSS) as r:
        slab = r.read_line_stations(
            "stage_0", "axial_force", element_ids=np.array([2]),
        )
        assert slab.element_index.tolist() == [2]
        assert slab.values.shape == (4, 1)


def test_bezier_tri6_gauss_axis_token_naming() -> None:
    # BezierTri6 emits the axis-form continuum tokens (sigma_xx / eps_xx /
    # gamma_xy), not the digit form (sigma11) — the reader maps both.
    with LadrunoReader(BEZIER) as r:
        comps = r.available_components("stage_0", ResultLevel.GAUSS)
        assert {"stress_xx", "stress_yy", "stress_xy",
                "strain_xx", "strain_yy", "strain_xy"} <= set(comps)
        slab = r.read_gauss("stage_0", "stress_xx")
        # 1 BezierTri6 × 3 Gauss points; natural coords are the 2 free
        # area coords (PARAM_DOMAIN="bary").
        assert slab.values.shape == (1, 3)
        assert slab.element_index.tolist() == [1, 1, 1]
        assert slab.natural_coords.shape == (3, 2)
