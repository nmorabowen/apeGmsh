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
FIBERBEAM = FIXTURES / "fiberbeam.ladruno"
NODE_ENVELOPE = FIXTURES / "node_envelope.ladruno"


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


def test_line_stations_carry_beam_quaternion() -> None:
    # L3 follow-up: line-station slabs carry the recorder's per-row beam
    # frame so the diagram can orient by true cross-section roll.
    with LadrunoReader(BEAM) as r:
        slab = r.read_line_stations("stage_0", "axial_force")
        assert slab.local_axes_quaternion is not None
        assert slab.local_axes_quaternion.shape == (2, 4)   # 2 stations
        la = r.read_local_axes("stage_0", element_ids=np.array([1]))
        np.testing.assert_allclose(slab.local_axes_quaternion[0], la.quaternions[0])
        # skew beam → non-identity frame on every station row.
        assert not np.allclose(slab.local_axes_quaternion[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            slab.local_axes_quaternion[0], slab.local_axes_quaternion[1],
        )


def test_line_stations_no_frame_quaternion_none() -> None:
    # Truss has no MODEL/LOCAL_AXES → slab carries no frame (None), so the
    # plot falls back to node geometry.
    with LadrunoReader(TRUSS) as r:
        slab = r.read_line_stations("stage_0", "axial_force")
        assert slab.local_axes_quaternion is None


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


# =====================================================================
# L2b-3 — section-level line stations + fibers (force-based fiber beam)
# =====================================================================

def test_section_force_line_stations() -> None:
    # forceBeamColumn (Lobatto, 3 stations) fiber section: section.force
    # (LEVELS=2) → axial_force / bending_moment_z line stations whose
    # natural coords come from QUADRATURE/GP_PARAM keyed by GAUSS_ID.
    with LadrunoReader(FIBERBEAM) as r:
        comps = r.available_components("stage_0", ResultLevel.LINE_STATIONS)
        assert {"axial_force", "bending_moment_z",
                "axial_strain", "curvature_z"} <= set(comps)

        axial = r.read_line_stations("stage_0", "axial_force")
        # 1 beam × 3 Lobatto stations at ξ = -1, 0, +1.
        assert axial.values.shape == (2, 3)
        assert axial.element_index.tolist() == [1, 1, 1]
        np.testing.assert_allclose(axial.station_natural_coord, [-1.0, 0.0, 1.0])
        # Constant section axial force == applied axial load (3.0) at full load.
        np.testing.assert_allclose(axial.values[-1], [3.0, 3.0, 3.0], atol=1e-9)

        mz = r.read_line_stations("stage_0", "bending_moment_z")
        # Tip transverse load 2.0 on a unit cantilever → linear Mz: 2 at the
        # base station, ~0 at the tip station.
        np.testing.assert_allclose(mz.values[-1], [2.0, 1.0, 0.0], atol=1e-9)


def test_section_deformation_line_stations() -> None:
    with LadrunoReader(FIBERBEAM) as r:
        kappa = r.read_line_stations("stage_0", "curvature_z")
        assert kappa.values.shape == (2, 3)
        np.testing.assert_allclose(
            kappa.station_natural_coord, [-1.0, 0.0, 1.0],
        )
        # curvature_z is the work-conjugate of Mz → same per-station profile
        # shape (zero at the tip station).
        np.testing.assert_allclose(kappa.values[-1, 2], 0.0, atol=1e-9)


def test_section_force_station_element_filter() -> None:
    with LadrunoReader(FIBERBEAM) as r:
        slab = r.read_line_stations(
            "stage_0", "axial_force", element_ids=np.array([1]),
        )
        assert slab.element_index.tolist() == [1, 1, 1]
        slab_empty = r.read_line_stations(
            "stage_0", "axial_force", element_ids=np.array([999]),
        )
        assert slab_empty.values.shape[1] == 0


def test_fibers_available_and_read() -> None:
    with LadrunoReader(FIBERBEAM) as r:
        comps = r.available_components("stage_0", ResultLevel.FIBERS)
        assert set(comps) == {"fiber_stress", "fiber_strain"}

        slab = r.read_fibers("stage_0", "fiber_stress")
        # 1 beam × 3 GPs × 4 fibers = 12 columns, 2 steps.
        assert slab.values.shape == (2, 12)
        assert slab.element_index.tolist() == [1] * 12
        # GP-major, fiber-minor ordering.
        assert slab.gp_index.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        # Fiber geometry from MODEL/SECTION_ASSIGNMENTS (2×2 patch over
        # [-0.05, 0.05]² → fibers at y = ±0.025, area 0.0025, material 1).
        np.testing.assert_allclose(np.unique(slab.y), [-0.025, 0.025])
        np.testing.assert_allclose(slab.area, 0.0025)
        assert set(slab.material_tag.tolist()) == {1}
        # Station ξ from QUADRATURE/GP_PARAM (same source as the
        # line-stations path) — Lobatto-3 stations at -1, 0, +1,
        # repeated per fiber.
        assert slab.station_natural_coord is not None
        np.testing.assert_allclose(
            slab.station_natural_coord, np.repeat([-1.0, 0.0, 1.0], 4),
        )


def test_fibers_gp_filter() -> None:
    with LadrunoReader(FIBERBEAM) as r:
        slab = r.read_fibers(
            "stage_0", "fiber_strain", gp_indices=np.array([2]),
        )
        # Only the tip station's 4 fibers.
        assert slab.values.shape == (2, 4)
        assert set(slab.gp_index.tolist()) == {2}
        np.testing.assert_allclose(slab.station_natural_coord, 1.0)


def test_fibers_unknown_component_empty() -> None:
    with LadrunoReader(FIBERBEAM) as r:
        slab = r.read_fibers("stage_0", "fiber_stress_zz")  # not a fiber name
        assert slab.values.shape[1] == 0


# -- material.fiber.* alias (layered shells, recorder PR #200) ----------
#
# Layered shells emit per-layer stress under ``material.fiber.<resp>``
# (the recorder swaps ``section``→``material`` for shells; the bucket
# layout is byte-identical to ``section.fiber.<resp>``). We synthesise that
# by renaming the fiber-beam fixture's buckets, so the read is provable
# fork-free; a live layered-shell round-trip is deferred to a fork build.


def _rename_fiber_buckets_to_material(src: Path, dst: Path) -> None:
    """Copy ``src``→``dst`` with ``section.fiber.*`` renamed to
    ``material.fiber.*`` (the shell spelling)."""
    import shutil

    import h5py

    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        for resp in ("stress", "strain"):
            on_e.move(f"section.fiber.{resp}", f"material.fiber.{resp}")


def test_fibers_material_spelling_reads_like_section(tmp_path: Path) -> None:
    shell = tmp_path / "shell.ladruno"
    _rename_fiber_buckets_to_material(FIBERBEAM, shell)
    with LadrunoReader(FIBERBEAM) as ref, LadrunoReader(shell) as r:
        comps = r.available_components("stage_0", ResultLevel.FIBERS)
        assert set(comps) == {"fiber_stress", "fiber_strain"}
        got = r.read_fibers("stage_0", "fiber_stress")
        want = ref.read_fibers("stage_0", "fiber_stress")
        # The material.fiber.* bucket reads identically to section.fiber.*.
        np.testing.assert_array_equal(got.values, want.values)
        assert got.element_index.tolist() == want.element_index.tolist()
        assert got.gp_index.tolist() == want.gp_index.tolist()
        # EXCEPT station ξ: a layered shell's gauss id is a SURFACE GP,
        # not a beam station — the shell spelling carries NaN.
        assert np.isnan(got.station_natural_coord).all()
        assert np.isfinite(want.station_natural_coord).all()


def test_fibers_gathers_both_spellings(tmp_path: Path) -> None:
    # A model carrying both fiber-section beams (section.fiber.*) and
    # layered shells (material.fiber.*) emits both buckets; the read gathers
    # from every present spelling. Synthesised by duplicating the beam
    # bucket under the material spelling (same element — contrived, but it
    # locks the gather-from-all-spellings loop).
    import shutil

    import h5py

    both = tmp_path / "both.ladruno"
    shutil.copy(FIBERBEAM, both)
    with h5py.File(both, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        on_e.copy("section.fiber.stress", "material.fiber.stress")
    with LadrunoReader(both) as r:
        slab = r.read_fibers("stage_0", "fiber_stress")
        # 12 columns from each spelling.
        assert slab.values.shape == (2, 24)


def test_fiber_stress_not_leaked_into_gauss() -> None:
    # section.fiber.stress carries sigma11 (→ stress_xx under the digit map)
    # but as MULTIPLICITY>1 fiber blocks. read_gauss must NOT surface them
    # as continuum Gauss stress.
    with LadrunoReader(FIBERBEAM) as r:
        assert r.available_components("stage_0", ResultLevel.GAUSS) == []
        slab = r.read_gauss("stage_0", "stress_xx")
        assert slab.values.shape[1] == 0


def test_layers_and_springs_empty_by_design() -> None:
    # A .ladruno has no distinct layer/spring level (layered shells are
    # fiber sections; zeroLength state flows through element/gauss reads).
    with LadrunoReader(FIBERBEAM) as r:
        assert r.available_components("stage_0", ResultLevel.LAYERS) == []
        assert r.available_components("stage_0", ResultLevel.SPRINGS) == []
        assert r.read_layers("stage_0", "fiber_stress").values.shape[1] == 0
        assert r.read_springs("stage_0", "spring_force_0").values.shape[1] == 0


# -- node envelopes (recorder -envelope, Finding B) ---------------------
#
# node_envelope.ladruno is a static cyclic pushover recorded with
# ``-envelope``: node 3's Ux path is +0.02 → -0.03 → +0.01, so the
# time-reduced extremes are MIN=-0.03, MAX=0.02, ABSMAX=0.03 at step 8.


def test_node_envelope_available_components() -> None:
    with LadrunoReader(NODE_ENVELOPE) as r:
        comps = r.available_node_envelope_components("stage_0")
        assert set(comps) == {"displacement_x", "displacement_y"}
        # The time-series ON_NODES path is empty under -envelope.
        assert r.available_components("stage_0", ResultLevel.NODES) == []


def test_node_envelope_read_extremes() -> None:
    with LadrunoReader(NODE_ENVELOPE) as r:
        env = r.read_node_envelope("stage_0", "displacement_x")
        assert env.node_ids.tolist() == [1, 2, 3]
        # node 3 (tip): the cyclic path's extremes.
        i3 = env.node_ids.tolist().index(3)
        np.testing.assert_allclose(env.min[i3], -0.03, atol=1e-9)
        np.testing.assert_allclose(env.max[i3], 0.02, atol=1e-9)
        np.testing.assert_allclose(env.absmax[i3], 0.03, atol=1e-9)
        # arg_step is the recorder's session commitTag (regeneration-relative,
        # not a fresh 0-based index) — assert it's plumbed as a valid index.
        assert env.arg_step.dtype.kind == "i"
        assert env.arg_step[i3] >= 0
        # ABSMAX is componentwise max(|MIN|, |MAX|) by construction.
        np.testing.assert_allclose(
            env.absmax, np.maximum(np.abs(env.min), np.abs(env.max)),
        )


def test_node_envelope_node_filter() -> None:
    with LadrunoReader(NODE_ENVELOPE) as r:
        env = r.read_node_envelope(
            "stage_0", "displacement_x", node_ids=np.array([3]),
        )
        assert env.node_ids.tolist() == [3]
        np.testing.assert_allclose(env.absmax, [0.03], atol=1e-9)


def test_node_envelope_on_timeseries_file_raises() -> None:
    # A plain (non-envelope) .ladruno has no ENVELOPES tree.
    with LadrunoReader(TRUSS) as r:
        with pytest.raises(ValueError, match="not recorded with the '-envelope'"):
            r.read_node_envelope("stage_0", "displacement_x")


def test_node_envelope_unknown_component_raises() -> None:
    with LadrunoReader(NODE_ENVELOPE) as r:
        with pytest.raises(ValueError, match="not in this .ladruno's node"):
            r.read_node_envelope("stage_0", "temperature")
