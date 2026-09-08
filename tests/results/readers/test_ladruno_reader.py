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
from apeGmsh.results.readers._ladruno_element_io import (
    ElementWidthMismatchWarning,
    GaussColumnDroppedWarning,
    read_element_slab,
)
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


# -- missing element results are LOUD ----------------------------------
#
# A recorder ``-E <token>`` that matched no element writes a file with no
# ON_ELEMENTS group at all. Silently answering "empty" there cost a field
# team a whole capture, so every element-level read now raises. Scope is
# the missing *group*, never a missing component — the empty-slab answers
# asserted above (stress_zz on a 2-D quad, basicForce on a quad, …) stay
# exactly as they were.


def _fixture_without_element_results(src: Path, dst: Path) -> Path:
    """Copy a ``.ladruno`` fixture with its ON_ELEMENTS group deleted."""
    import shutil

    import h5py

    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        for k in (k for k in f if k.startswith("MODEL_STAGE[")):
            results = f[k]["RESULTS"]
            if "ON_ELEMENTS" in results:
                del results["ON_ELEMENTS"]
    return dst


@pytest.mark.parametrize(
    ("method", "component"),
    [
        ("read_gauss", "stress_xx"),
        ("read_elements", "force"),
        ("read_line_stations", "axial_force"),
        ("read_fibers", "fiber_stress"),
    ],
)
def test_missing_element_results_raises(
    tmp_path: Path, method: str, component: str,
) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        MissingElementResults,
    )

    stripped = _fixture_without_element_results(
        QUAD, tmp_path / "no_elements.ladruno",
    )
    with LadrunoReader(stripped) as r:
        # Probing stays quiet — only an actual read is loud.
        assert r.available_components("stage_0", ResultLevel.GAUSS) == []
        with pytest.raises(MissingElementResults) as exc:
            getattr(r, method)("stage_0", component)
    msg = str(exc.value)
    assert "ON_ELEMENTS" in msg
    assert component in msg
    assert "no_elements.ladruno" in msg
    assert "-E token matched no element" in msg


def test_missing_element_results_only_when_group_absent() -> None:
    # The file HAS element results, just not this component → still the
    # long-standing empty slab (regression guard for the scope rule).
    with LadrunoReader(QUAD) as r:
        assert r.read_gauss("stage_0", "stress_zz").values.shape[1] == 0


def test_partitioned_tolerates_one_rank_without_element_results(
    tmp_path: Path,
) -> None:
    # A rank owning none of the recorded elements legitimately writes no
    # ON_ELEMENTS group: the stitch drops it, and only an all-ranks-missing
    # read is loud.
    import shutil

    from apeGmsh.results.readers._ladruno_element_io import (
        MissingElementResults,
    )
    from apeGmsh.results.readers._ladruno_multi import (
        LadrunoMultiPartitionReader,
    )

    part0 = tmp_path / "truss2d.part-0.ladruno"
    part1 = tmp_path / "truss2d.part-1.ladruno"
    shutil.copy(FIXTURES / "truss2d.part-0.ladruno", part0)
    _fixture_without_element_results(
        FIXTURES / "truss2d.part-1.ladruno", part1,
    )
    with LadrunoMultiPartitionReader([part0, part1]) as r:
        comps = r.available_components("stage_0", ResultLevel.ELEMENTS)
        assert comps  # partition 0 still carries element results
        slab = r.read_elements("stage_0", comps[0])
        assert slab.values.shape[1] > 0

    # Strip BOTH ranks → nothing survives, so the read is loud again.
    _fixture_without_element_results(
        FIXTURES / "truss2d.part-0.ladruno", part0,
    )
    with LadrunoMultiPartitionReader([part0, part1]) as r:
        with pytest.raises(MissingElementResults):
            r.read_elements("stage_0", "basicForce")


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


# ---------------------------------------------------------------------------
# Generic ``C1..Cn`` columns — named from RESPONSE_CATALOG
# ---------------------------------------------------------------------------
#
# The fork's plain ``stress`` / ``strain`` responses on its plane elements
# tag no ResponseType, so the recorder writes anonymous columns. These pin
# the resolver's outcomes without needing the fork; the live counterpart is
# tests/opensees/integration_ladruno/test_ladruno_gauss_generic_columns.py.

def _generic_block(width: int) -> list:
    from apeGmsh.results.readers._ladruno_element_io import _Block

    return [_Block(
        level=0, gauss_id=-1,
        comp_names=tuple(f"C{i + 1}" for i in range(width)), col_start=0,
    )]


def test_generic_columns_named_from_catalog() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        resolve_generic_gauss_blocks,
    )

    out = resolve_generic_gauss_blocks(
        _generic_block(9), token="stress",
        bucket_key="33016-LadrunoLST[0:0:0]",
    )
    assert [b.gauss_id for b in out] == [0, 1, 2]
    assert [b.col_start for b in out] == [0, 3, 6]
    assert all(
        b.comp_names == ("stress_xx", "stress_yy", "stress_xy") for b in out
    )


def test_generic_columns_wrong_width_raises() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
        resolve_generic_gauss_blocks,
    )

    # A wrong component name is worse than a missing one: 8 columns fit no
    # LadrunoLST layout, so the reader refuses rather than mis-labelling.
    with pytest.raises(GaussLayoutMismatch, match="Refusing to guess"):
        resolve_generic_gauss_blocks(
            _generic_block(8), token="stress",
            bucket_key="33016-LadrunoLST[0:0:0]",
        )


def test_generic_columns_unknown_class_left_alone() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        resolve_generic_gauss_blocks,
    )

    blocks = _generic_block(9)
    assert resolve_generic_gauss_blocks(
        blocks, token="stress", bucket_key="99999-NotACatalogedClass[0:0:0]",
    ) is blocks


def test_named_columns_untouched_by_the_resolver() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        _Block,
        resolve_generic_gauss_blocks,
    )

    blocks = [_Block(
        level=1, gauss_id=0,
        comp_names=("sigma11", "sigma22", "sigma12", "sigma33"), col_start=0,
    )]
    assert resolve_generic_gauss_blocks(
        blocks, token="stressesPlaneStrain",
        bucket_key="33016-LadrunoLST[0:0:0]",
    ) is blocks


# ---------------------------------------------------------------------------
# Overlapping tokens — one (element, GP) slot covered twice
# ---------------------------------------------------------------------------
#
# `stress` (anonymous, catalog-named) and `stressesPlaneStrain` (self-named,
# and a superset — it carries sigma33) both report the in-plane components
# for the same elements at the same Gauss points. The live counterpart is
# test_both_stress_tokens_in_one_recorder_do_not_double_count.

def _overlap(v_named: float, v_generic: float):
    """Two columns for element 7 / GP 0 — one file-named, one catalog."""
    return (
        np.array([[v_generic, v_named]]),      # values (T=1, 2)
        np.array([7, 7]),                      # element_index
        np.array([0, 0]),                      # gauss_index
        np.array([False, True]),               # named
    )


def test_overlapping_tokens_keep_the_file_named_column() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        _dedupe_gauss_columns,
    )

    keep = _dedupe_gauss_columns(*_overlap(5.0, 5.0), component="stress_xx")
    assert keep.tolist() == [1]                # the file-named column


def test_no_overlap_leaves_the_slab_alone() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        _dedupe_gauss_columns,
    )

    assert _dedupe_gauss_columns(
        np.array([[1.0, 2.0]]), np.array([7, 7]), np.array([0, 1]),
        np.array([True, True]), component="stress_xx",
    ) is None


def test_overlapping_tokens_that_disagree_raise() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
        _dedupe_gauss_columns,
    )

    # Same material state read twice cannot differ — if it does, one of
    # the two buckets' columns is mis-labelled. Do not pick one.
    with pytest.raises(GaussLayoutMismatch, match="element 7 Gauss point 0"):
        _dedupe_gauss_columns(*_overlap(5.0, -3.0), component="stress_xx")


# ---------------------------------------------------------------------------
# ADR 0105 Amendment 1 — material-level buckets, and the loud drop
# ---------------------------------------------------------------------------
#
# ``ASDPlasticMaterial3D`` writes one bucket per ``material.<token>``
# request, each a per-Gauss-point block labelled with the MATERIAL's own
# private names (``p``, ``J2stress``, ``BackStress_1``…). The reader maps
# those buckets by TOKEN and by column POSITION — the labels are not
# unique across levels (``p`` is the section axial force ``P`` under
# case-insensitive matching), so they cannot be the key. Synthesised here
# from the quad fixture — same bucket shape the fork writes, provably
# fork-free — because the mapping and the drop-warning are reader
# behaviour, not fork behaviour. The live round-trip is
# ``tests/opensees/integration_ladruno/test_ladruno_gauss_generic_columns.py``.


def _add_material_bucket(
    path: Path, token: str, labels: "tuple[str, ...]",
) -> None:
    """Append a per-GP ``material.<token>`` bucket to a ``.ladruno`` file.

    One block per Gauss point (``LEVELS == 4``, the Nd-material depth the
    recorder writes), each carrying ``labels``. DATA is filled with an
    arange so a read can be checked column by column.
    """
    import h5py

    with h5py.File(path, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        key = next(iter(on_e["stress"]))
        src = on_e["stress"][key]
        n_t, n_e, _ = src["DATA"].shape
        n_gp = int(np.asarray(src["COLUMN_MAP"]["GAUSS_ID"][...]).size)
        width = n_gp * len(labels)
        grp = on_e.create_group(f"{token}/{key}")
        grp.create_dataset("DATA", data=np.arange(
            n_t * n_e * width, dtype=np.float64,
        ).reshape(n_t, n_e, width))
        for name in ("ID", "STEP", "TIME"):
            if name in src:
                grp.create_dataset(name, data=np.asarray(src[name][...]))
        cm = grp.create_group("COLUMN_MAP")
        cm.create_dataset("LEVELS", data=np.full(n_gp, 4, dtype=np.int64))
        cm.create_dataset(
            "GAUSS_ID", data=np.arange(n_gp, dtype=np.int64),
        )
        cm.attrs["COMP_NAMES"] = "\n".join([",".join(labels)] * n_gp)


def _quad_with(tmp_path: Path, buckets: dict) -> Path:
    import shutil

    dst = tmp_path / "material.ladruno"
    shutil.copy(QUAD, dst)
    for token, labels in buckets.items():
        _add_material_bucket(dst, token, labels)
    return dst


def test_material_level_buckets_reach_the_gauss_level(tmp_path: Path) -> None:
    path = _quad_with(tmp_path, {
        "material.PStress": ("p",),
        "material.J2Stress": ("J2stress",),
        "material.VolStrain": ("epsVol",),
        "material.J2Strain": ("J2strain",),
        "material.BackStress": tuple(f"BackStress_{i}" for i in range(1, 7)),
        "material.YieldStress": ("YieldStress",),
    })
    with LadrunoReader(path) as r:
        comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
        assert {
            "material_mean_stress", "material_j2_stress",
            "material_volumetric_strain", "material_j2_strain",
            "back_stress_xx", "back_stress_yy", "back_stress_zz",
            "back_stress_xy", "back_stress_yz", "back_stress_xz",
            "yield_stress",
        } <= comps
        # One column per (element, GP) — 1 element × 4 GPs, 2 steps.
        slab = r.read_gauss("stage_0", "material_mean_stress")
        assert slab.values.shape == (2, 4)
        # The BackStress block is 6 wide per GP; the xy column is offset 3.
        slab = r.read_gauss("stage_0", "back_stress_xy")
        assert slab.values.shape == (2, 4)
        np.testing.assert_array_equal(
            slab.values[0], np.array([3.0, 9.0, 15.0, 21.0]),
        )


def test_sanisand_buckets_reach_the_gauss_level_with_real_names(
    tmp_path: Path,
) -> None:
    # TIMs A12 — the fork's own COMP_NAMES (PR #820).
    import warnings

    path = _quad_with(tmp_path, {
        "material.psi": ("psi",),
        "material.implexDetail": (
            "implexDetail_total", "implexDetail_dev", "implexDetail_vol",
            "implexDetail_clampFired", "implexDetail_clampCount",
            "implexDetail_f",
        ),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
        assert {
            "state_parameter",
            "implex_detail_total", "implex_detail_dev",
            "implex_detail_vol", "implex_detail_clamp_fired",
            "implex_detail_clamp_count", "implex_detail_f",
        } <= comps
        slab = r.read_gauss("stage_0", "state_parameter")
        assert slab.values.shape == (2, 4)


def test_sanisand_buckets_reach_the_gauss_level_with_generic_names(
    tmp_path: Path,
) -> None:
    # Older fork builds write C1..Cn instead of the named COMP_NAMES —
    # registering by token (positional) has to work on both.
    import warnings

    path = _quad_with(tmp_path, {
        "material.psi": ("C1",),
        "material.implexDetail": tuple(f"C{i}" for i in range(1, 7)),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
        assert {
            "state_parameter",
            "implex_detail_total", "implex_detail_dev",
            "implex_detail_vol", "implex_detail_clamp_fired",
            "implex_detail_clamp_count", "implex_detail_f",
        } <= comps


def test_implex_guards_seven_slots_resolve_by_position(tmp_path: Path) -> None:
    # ADR 92 P2-9. Unlike every other fork response, implexGuards carries no
    # ResponseType at all (LadrunoSANISAND.cpp:3947-3951), so C1..C7 is not
    # an "older build" case here -- it is what every build writes, and the
    # by-position map is the only thing that can name these columns.
    import warnings

    path = _quad_with(tmp_path, {
        "material.implexGuards": tuple(f"C{i}" for i in range(1, 8)),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert {
        "implex_guards_floor_fallback", "implex_guards_f0_guard",
        "implex_guards_hold_preserved", "implex_guards_reversal_noise",
        "implex_guards_trial_f0_guard", "implex_guards_hold_skip_commit",
        "implex_guards_control_backoff",
    } <= comps


def test_implex_guards_slot_order_matches_the_fork_fill_site() -> None:
    # A by-position map is only as good as its order, and a wrong order
    # mislabels data silently. Pinned to LadrunoSANISAND.cpp:3996-4002.
    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    assert material_bucket_canonicals("material.implexGuards") == (
        "implex_guards_floor_fallback",     # out4g(0) getFloorFallbacks
        "implex_guards_f0_guard",           # out4g(1) getGuardsFired
        "implex_guards_hold_preserved",     # out4g(2) getHoldsPreserved
        "implex_guards_reversal_noise",     # out4g(3) getReversalNoiseGuards
        "implex_guards_trial_f0_guard",     # out4g(4) getTrialGuardF0
        "implex_guards_hold_skip_commit",   # out4g(5) getHoldSkipCommits
        "implex_guards_control_backoff",    # out4g(6) getControlFactorBackoffs
    )
    # Both fork spellings reach the same map (the table keys are lowered).
    assert (material_bucket_canonicals("material.ImplexGuards")
            == material_bucket_canonicals("material.implexGuards"))


def test_ladruno_branch_eight_slots_resolve_by_position(tmp_path: Path) -> None:
    # Fork ADR-95 (PR #803). Like implexGuards, the fork returns a bare
    # MaterialResponse with no ResponseType (DruckerPrager.cpp:1004-1005),
    # so C1..C8 is what every build writes and the by-position map is the
    # only thing that can name these columns.
    import warnings

    path = _quad_with(tmp_path, {
        "material.ladrunoBranch": tuple(f"C{i}" for i in range(1, 9)),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert {
        "dp_branch", "dp_gamma_cone", "dp_gamma_cutoff",
        "dp_f1_trial", "dp_f2_trial", "dp_forced_accept",
        "dp_i1", "dp_det_a_min",
    } <= comps


def test_ladruno_branch_slot_order_matches_the_fork_fill_site() -> None:
    # Pinned to DruckerPrager::getLadrunoBranch(), DruckerPrager.cpp:960-970.
    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    assert material_bucket_canonicals("material.ladrunoBranch") == (
        "dp_branch",            # mLadBranch
        "dp_gamma_cone",        # mLadGamma0
        "dp_gamma_cutoff",      # mLadGamma1
        "dp_f1_trial",          # mLadF1Trial
        "dp_f2_trial",          # mLadF2Trial
        "dp_forced_accept",     # mLadForcedAccept
        "dp_i1",                # I1 of the returned stress
        "dp_det_a_min",         # detAmin
    )


def test_ladruno_tangent_is_not_named_yet(tmp_path: Path) -> None:
    # The 36-entry `ladrunoTangent` (responseID 96) is refused bare by the
    # recorder but has no by-position map: nothing reads it yet, so it takes
    # the documented unknown-bucket route rather than inventing 36 names.
    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    assert material_bucket_canonicals("material.ladrunoTangent") is None


def test_unknown_material_bucket_warns_and_names_it(tmp_path: Path) -> None:
    path = _quad_with(tmp_path, {"material.Mystery": ("WhoKnows",)})
    with LadrunoReader(path) as r:
        with pytest.warns(GaussColumnDroppedWarning) as rec:
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert "WhoKnows" not in comps
    msg = str(rec[0].message)
    assert "WhoKnows" in msg
    assert "material.Mystery" in msg
    assert "FourNodeQuad" in msg


def test_fully_mapped_buckets_do_not_warn(tmp_path: Path) -> None:
    # The quad fixture's own stress/strain buckets map completely — and a
    # beam's section.force / section.deformation stations are not Gauss
    # columns at all. Neither may warn, or the warning is noise.
    import warnings

    for fixture in (QUAD, FIBERBEAM):
        with LadrunoReader(fixture) as r:
            with warnings.catch_warnings():
                warnings.simplefilter("error", GaussColumnDroppedWarning)
                r.available_components("stage_0", ResultLevel.GAUSS)


def test_one_warning_per_bucket(tmp_path: Path) -> None:
    path = _quad_with(tmp_path, {
        "material.Mystery": ("WhoKnows", "NorMe"),
        "material.Other": ("Neither",),
    })
    with LadrunoReader(path) as r:
        with pytest.warns(GaussColumnDroppedWarning) as rec:
            r.available_components("stage_0", ResultLevel.GAUSS)
    assert len(rec) == 2                       # two buckets, not three labels
    joined = " ".join(str(w.message) for w in rec)
    for label in ("WhoKnows", "NorMe", "Neither"):
        assert label in joined


def test_material_bucket_labels_are_not_matched_as_labels(tmp_path: Path) -> None:
    # Same labels, a token nobody maps: the columns must NOT resolve.
    # Proves the mapping is keyed by token, not by label.
    path = _quad_with(tmp_path, {"material.Mystery": ("p", "J2stress")})
    with LadrunoReader(path) as r:
        with pytest.warns(GaussColumnDroppedWarning):
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert "material_mean_stress" not in comps
    assert "material_j2_stress" not in comps


def test_section_axial_force_is_not_a_gauss_mean_stress() -> None:
    # The collision the token key exists to avoid: a section.force station
    # labels its axial force ``P``, and canonicalisation is
    # case-insensitive, so a label-keyed ``p`` would surface every
    # force-based beam station as material_mean_stress at the Gauss level.
    with LadrunoReader(FIBERBEAM) as r:
        assert r.available_components("stage_0", ResultLevel.GAUSS) == []
        # …while the station itself still reads at its own level.
        assert "axial_force" in r.available_components(
            "stage_0", ResultLevel.LINE_STATIONS,
        )


# =====================================================================
# Mixed-width buckets under one response token
#
# The fork sizes a zeroLength `force` response by ndf1 + ndf2, so a 3-D
# model pairing (3,3) nodes with u-p (3,4) nodes writes 6- and 7-column
# buckets under ONE token. They cannot share a dense (T, E, ncol) slab;
# the first width wins. That is a deliberate choice -- but it used to be
# a silent one, and a short read is indistinguishable from elements that
# never recorded the quantity.
# =====================================================================


def _mixed_width_on_elements(path: Path) -> "object":
    """An ON_ELEMENTS group with 6- and 7-column buckets under `force`."""
    import h5py

    f = h5py.File(path, "w")
    tok = f.create_group("ON_ELEMENTS").create_group("force")
    # Alphabetical iteration puts the 6-column bucket first, so it wins.
    a = tok.create_group("a_pairs_3_3")
    a.create_dataset("ID", data=np.array([11, 12], dtype=np.int64))
    a.create_dataset("DATA", data=np.zeros((2, 2, 6), dtype=np.float64))
    b = tok.create_group("b_pairs_3_4")
    b.create_dataset("ID", data=np.array([21, 22], dtype=np.int64))
    b.create_dataset("DATA", data=np.ones((2, 2, 7), dtype=np.float64))
    return f


def test_mixed_width_buckets_drop_elements_and_say_so(tmp_path: Path) -> None:
    f = _mixed_width_on_elements(tmp_path / "mixed.h5")
    try:
        with pytest.warns(ElementWidthMismatchWarning) as rec:
            out = read_element_slab(
                f["ON_ELEMENTS"], "force",
                t_idx=np.array([0, 1]), element_ids=None,
            )
        assert out is not None
        values, ids = out
        # The drop itself: the 7-column pair is gone from the result.
        assert values.shape == (2, 2, 6)
        assert ids.tolist() == [11, 12]
        assert 21 not in ids.tolist() and 22 not in ids.tolist()
        # …and it is no longer silent. The message must carry what the
        # caller needs to act: both widths, and WHICH elements are absent.
        msg = str(rec[0].message)
        assert "6-column" in msg and "has 7" in msg
        assert "21" in msg and "22" in msg
        assert "force" in msg
    finally:
        f.close()


def test_uniform_width_buckets_do_not_warn(tmp_path: Path) -> None:
    # The common case stays quiet -- a warning on every homogeneous read
    # would be noise, and noise is how a real one gets ignored.
    import warnings

    import h5py

    with h5py.File(tmp_path / "uniform.h5", "w") as f:
        tok = f.create_group("ON_ELEMENTS").create_group("force")
        for name, eids in (("a", [11, 12]), ("b", [21, 22])):
            g = tok.create_group(name)
            g.create_dataset("ID", data=np.array(eids, dtype=np.int64))
            g.create_dataset("DATA", data=np.zeros((2, 2, 6), dtype=np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ElementWidthMismatchWarning)
            out = read_element_slab(
                f["ON_ELEMENTS"], "force",
                t_idx=np.array([0, 1]), element_ids=None,
            )
        assert out is not None
        assert out[1].tolist() == [11, 12, 21, 22]

