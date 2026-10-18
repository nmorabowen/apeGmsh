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

from apeGmsh.opensees._response_catalog import LAYER_CATALOG, RESPONSE_CATALOG
from apeGmsh.results._shell_geometry import shell_quaternion
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.readers._mpco_layer_io import ShellFrameUnavailableWarning
from apeGmsh.results.readers._mpco_nodal_io import AbsentDofWarning
from apeGmsh.results.readers._ladruno_element_io import (
    ElementWidthMismatchWarning,
    GaussColumnDroppedWarning,
    LayerBucketUnreadWarning,
    LayerSectionUnresolved,
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


#: #2009: end-i sign per component of the force-beam ``section force``
#: convention; station 1 = sign * end i, station 2 = -sign * end j.
_END_I_SIGN_3D = (
    ("axial_force", 0, -1.0), ("shear_y", 1, 1.0), ("shear_z", 2, -1.0),
    ("torsion", 3, -1.0), ("bending_moment_y", 4, -1.0),
    ("bending_moment_z", 5, -1.0),
)


def test_line_stations_beam_force_beam_signs() -> None:
    """Each of the six 3-D components carries the force-beam sign (#2009)."""
    with LadrunoReader(BEAM) as r:
        raw = r.read_elements("stage_0", "localForce").values[-1, 0]
        assert np.count_nonzero(raw) >= 2    # the pin can tell a sign apart
        for comp, k, sign in _END_I_SIGN_3D:
            slab = r.read_line_stations("stage_0", comp)
            np.testing.assert_allclose(
                slab.values[-1], [sign * raw[k], -sign * raw[6 + k]],
                rtol=1e-12, err_msg=comp,
            )


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


def test_fiber_beam_has_no_layers_and_springs_are_empty_by_design() -> None:
    # A fiber-section beam's scalar fiber blocks are fibers, never layers;
    # a .ladruno has no distinct spring level (zeroLength state flows
    # through element/gauss reads).
    with LadrunoReader(FIBERBEAM) as r:
        assert r.available_components("stage_0", ResultLevel.LAYERS) == []
        assert r.available_components("stage_0", ResultLevel.SPRINGS) == []
        assert r.read_layers("stage_0", "fiber_stress").values.shape[1] == 0
        assert r.read_layers("stage_0", "fiber_stress_0").values.shape[1] == 0
        assert r.read_springs("stage_0", "spring_force_0").values.shape[1] == 0


# -- layered shells: layer state into the MPCO LayerSlab (#1995) --------
#
# The fork writes a layered shell under the fiber tokens, one COLUMN_MAP
# row per SURFACE Gauss point with NUM_COMP = the layer material's
# components (5 for PlateFiber) and MULTIPLICITY = the layer count, columns
# layer-major and component-fast (measured on fork build 79e0623670 and on
# the pinned build by the live-fork lane). The file below is that layout,
# built on the beam fixture's identity and time axis; every DATA value
# encodes its own (element, surface GP, layer, component), so a read that
# mis-orders or mis-slices names the cell it got wrong. The live
# round-trip against eleResponse is
# tests/opensees/integration_ladruno/test_fiber_layer_readback_live.py.

LAYER_SHELLS = sorted({c for c, _token in LAYER_CATALOG})
_TRI_SHELLS = {"ASDShellT3", "ShellDKGT", "ShellNLDKGT"}
SHELL_TAGS = (7, 9)
SHELL_THICKNESS = {7: (0.1, 0.2, 0.3), 9: (0.4, 0.5, 0.6)}
N_LAYERS, N_COMP = 3, 5


def _layer_value(etag: int, gp: int, layer: int, comp: int) -> float:
    return etag * 10000.0 + gp * 100.0 + layer * 10.0 + comp + 1.0


def _shell_corners(cls: str) -> np.ndarray:
    """A unit cell in the xy plane turned 30 deg about x, plus a second
    one shifted along x; corner order is the element's connectivity."""
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    if cls in _TRI_SHELLS:
        cell = [[0, 0, 0], [1, 0, 0], [0, c, s]]
    else:
        cell = [[0, 0, 0], [1, 0, 0], [1, c, s], [0, c, s]]
        if cls == "ShellMITC9":           # mid-side and centre nodes
            cell += [[0.5, 0, 0], [1, c / 2, s / 2], [0.5, c, s],
                     [0, c / 2, s / 2], [0.5, c / 2, s / 2]]
    return np.asarray(cell, dtype=float)


def _layered_shell_file(
    tmp_path: Path, cls: str = "ASDShellQ4", *,
    tokens: "tuple[str, ...]" = ("section.fiber.stress", "section.fiber.strain"),
    class_name: "str | None" = None,
    assign: bool = True,
    connectivity: bool = True,
) -> Path:
    """The beam fixture plus a two-element layered shell of class ``cls``."""
    import shutil

    import h5py

    layout = LAYER_CATALOG[(cls, "fiber_stress")]
    rule = layout.surface_int_rule
    n_gp = RESPONSE_CATALOG[(cls, rule, "stress")].n_gauss_points
    name = class_name or cls
    key = f"{layout.class_tag}-{name}[{rule}:0:0]"
    corners = _shell_corners(cls)
    dst = tmp_path / f"{cls}.ladruno"
    shutil.copy(BEAM, dst)
    with h5py.File(dst, "r+") as f:
        stage = f[next(k for k in f if k.startswith("MODEL_STAGE["))]
        model, on_e = stage["MODEL"], stage["RESULTS/ON_ELEMENTS"]
        ids = np.asarray(model["NODES/ID"][...], dtype=np.int64)
        xyz = np.asarray(model["NODES/COORDINATES"][...], dtype=float)
        conn = []
        for e, etag in enumerate(SHELL_TAGS):
            tags = 100 * (e + 1) + np.arange(len(corners))
            ids = np.concatenate([ids, tags])
            xyz = np.concatenate([xyz, corners + [2.0 * e, 0.0, 0.0]])
            conn.append([etag, *tags])
        del model["NODES/ID"], model["NODES/COORDINATES"]
        model["NODES"].create_dataset("ID", data=ids.astype(np.int32))
        model["NODES"].create_dataset("COORDINATES", data=xyz)
        if connectivity:
            model.create_dataset(
                f"ELEMENTS/{layout.class_tag}-{name}[{rule}:0]/CONNECTIVITY",
                data=np.asarray(conn, dtype=np.int32),
            )
        if assign:
            for etag in SHELL_TAGS:
                sec = model.create_group(
                    f"SECTION_ASSIGNMENTS/SECTION_{etag}[UnknownClassType]")
                sec.create_dataset("ASSIGNMENT", data=np.asarray(
                    [[etag, g] for g in range(n_gp)], dtype=np.int32))
                sec.create_dataset("FIBER_DATA", data=np.asarray(
                    [[0.0, 0.0, t] for t in SHELL_THICKNESS[etag]]))
                sec.create_dataset("FIBER_MATERIALS", data=np.asarray(
                    [1, 2, 1], dtype=np.int32))
        time_src = on_e[next(iter(on_e))]
        time_src = time_src[next(iter(time_src))]
        n_t = int(time_src["TIME"].shape[0])
        data = np.array([[[
            _layer_value(etag, g, k, c) + 0.5 * t
            for g in range(n_gp) for k in range(N_LAYERS) for c in range(N_COMP)
        ] for etag in SHELL_TAGS] for t in range(n_t)])
        for token in tokens:
            grp = on_e.create_group(f"{token}/{key}")
            grp.create_dataset("DATA", data=data)
            grp.create_dataset("ID", data=np.asarray(
                SHELL_TAGS, dtype=np.int32).reshape(-1, 1))
            for ds in ("STEP", "TIME"):
                grp.create_dataset(ds, data=np.asarray(time_src[ds][...]))
            cm = grp.create_group("COLUMN_MAP")
            for ds, val in (("LEVELS", 4), ("SECTION_TAG", -1),
                            ("FIBER_ID", -1), ("NUM_COMP", N_COMP),
                            ("MULTIPLICITY", N_LAYERS)):
                cm.create_dataset(ds, data=np.full((n_gp, 1), val, np.int32))
            cm.create_dataset("GAUSS_ID", data=np.arange(
                n_gp, dtype=np.int32).reshape(-1, 1))
            word = "Stress" if token.endswith("stress") else "Strain"
            names = ",".join(
                [f"Unknown{word}"] + [f"Unknown{word}({i})" for i in range(1, N_COMP)])
            cm.attrs["COMP_NAMES"] = "\n".join([names] * n_gp)
    return dst


def _frame(cls: str, e: int) -> np.ndarray:
    return shell_quaternion(_shell_corners(cls) + [2.0 * e, 0.0, 0.0], cls)


@pytest.mark.parametrize("cls", LAYER_SHELLS)
def test_layered_shell_reads_every_layer_component(
    tmp_path: Path, cls: str,
) -> None:
    path = _layered_shell_file(tmp_path, cls)
    n_gp = RESPONSE_CATALOG[(
        cls, LAYER_CATALOG[(cls, "fiber_stress")].surface_int_rule, "stress",
    )].n_gauss_points
    with LadrunoReader(path) as r:
        # The fiber level lists none of it; the layer level lists all of it.
        assert r.available_components("stage_0", ResultLevel.FIBERS) == []
        assert r.available_components("stage_0", ResultLevel.LAYERS) == sorted(
            f"{p}_{c}" for p in ("fiber_strain", "fiber_stress")
            for c in range(N_COMP))
        for parent in ("fiber_stress", "fiber_strain"):
            for comp in range(N_COMP):
                slab = r.read_layers("stage_0", f"{parent}_{comp}")
                rows = [(e, g, k) for e in SHELL_TAGS for g in range(n_gp)
                        for k in range(N_LAYERS)]
                assert list(zip(slab.element_index.tolist(),
                                slab.gp_index.tolist(),
                                slab.layer_index.tolist())) == rows
                want = [_layer_value(e, g, k, comp) for e, g, k in rows]
                np.testing.assert_array_equal(slab.values[0], want)
                assert slab.values.shape == (1, len(rows))
                assert not slab.sub_gp_index.any()
                np.testing.assert_array_equal(
                    slab.thickness,
                    [SHELL_THICKNESS[e][k] for e, _g, k in rows])
                np.testing.assert_allclose(
                    slab.local_axes_quaternion,
                    [_frame(cls, SHELL_TAGS.index(e)) for e, _g, _k in rows],
                    rtol=0, atol=1e-14)
        # The fiber read of the same token returns nothing and warns nothing.
        assert r.read_fibers("stage_0", "fiber_stress").values.shape[1] == 0


def test_layered_shell_quaternion_is_the_tilted_frame(tmp_path: Path) -> None:
    """The frame oracle in closed form: a cell turned 30 deg about x reads
    the rotation global -> local, a turn of -30 deg about x."""
    half = np.radians(30.0) / 2
    with LadrunoReader(_layered_shell_file(tmp_path)) as r:
        q = r.read_layers("stage_0", "fiber_stress_0").local_axes_quaternion
    assert q.shape == (2 * 4 * N_LAYERS, 4)
    np.testing.assert_allclose(
        q, np.tile([np.cos(half), -np.sin(half), 0.0, 0.0], (q.shape[0], 1)),
        atol=1e-14)


def test_layered_shell_filters(tmp_path: Path) -> None:
    with LadrunoReader(_layered_shell_file(tmp_path)) as r:
        slab = r.read_layers(
            "stage_0", "fiber_strain_2", element_ids=np.array([9]),
            gp_indices=np.array([3, 1]), layer_indices=np.array([2]),
        )
        assert slab.element_index.tolist() == [9, 9]
        assert slab.gp_index.tolist() == [3, 1]
        assert slab.layer_index.tolist() == [2, 2]
        np.testing.assert_array_equal(slab.values[0], [
            _layer_value(9, 3, 2, 2), _layer_value(9, 1, 2, 2)])
        np.testing.assert_array_equal(slab.thickness, [0.6, 0.6])
        with pytest.raises(ValueError, match="gp_indices"):
            r.read_layers("stage_0", "fiber_stress_0", gp_indices=np.array([4]))
        with pytest.raises(ValueError, match="layer_indices"):
            r.read_layers(
                "stage_0", "fiber_stress_0", layer_indices=np.array([3]))
        # A component past the layer material's five reads empty, unlisted.
        assert r.read_layers("stage_0", "fiber_stress_5").values.shape[1] == 0


def test_layered_shell_material_spelling(tmp_path: Path) -> None:
    path = _layered_shell_file(tmp_path, tokens=("material.fiber.stress",))
    with LadrunoReader(path) as r:
        assert r.available_components("stage_0", ResultLevel.LAYERS) == [
            f"fiber_stress_{c}" for c in range(N_COMP)]
        slab = r.read_layers("stage_0", "fiber_stress_4")
        assert slab.values[0, 0] == _layer_value(7, 0, 0, 4)


def test_layered_shell_of_an_uncatalogued_class_warns_by_name(
    tmp_path: Path,
) -> None:
    path = _layered_shell_file(tmp_path, class_name="FooShell")
    with LadrunoReader(path) as r:
        with pytest.warns(LayerBucketUnreadWarning, match="FooShell"):
            assert r.available_components("stage_0", ResultLevel.LAYERS) == []
        with pytest.warns(LayerBucketUnreadWarning, match="LAYER_CATALOG"):
            assert r.read_fibers(
                "stage_0", "fiber_stress").values.shape[1] == 0
        assert r.available_components("stage_0", ResultLevel.FIBERS) == []


def test_layered_shell_without_its_section_refuses_by_name(
    tmp_path: Path,
) -> None:
    with LadrunoReader(_layered_shell_file(tmp_path, assign=False)) as r:
        assert "fiber_stress_0" in r.available_components(
            "stage_0", ResultLevel.LAYERS)
        with pytest.raises(LayerSectionUnresolved, match=r"ASDShellQ4\[201"):
            r.read_layers("stage_0", "fiber_stress_0")


def test_layered_shell_without_connectivity_reads_nan_frames_and_warns(
    tmp_path: Path,
) -> None:
    with LadrunoReader(_layered_shell_file(tmp_path, connectivity=False)) as r:
        with pytest.warns(ShellFrameUnavailableWarning, match=r"\[7, 9\]"):
            slab = r.read_layers("stage_0", "fiber_stress_0")
    assert np.isnan(slab.local_axes_quaternion).all()
    assert slab.values.shape == (1, 2 * 4 * N_LAYERS)


def test_layered_shell_stored_frame_wins(tmp_path: Path) -> None:
    import h5py

    path = _layered_shell_file(tmp_path)
    stored = np.array([0.0, 0.0, 0.0, 1.0])
    tag = LAYER_CATALOG[("ASDShellQ4", "fiber_stress")].class_tag
    with h5py.File(path, "r+") as f:
        stage = f[next(k for k in f if k.startswith("MODEL_STAGE["))]
        grp = stage["MODEL/LOCAL_AXES"].create_group(f"{tag}-ASDShellQ4")
        grp.create_dataset("ID", data=np.array([9], dtype=np.int32))
        grp.create_dataset("FRAME", data=stored.reshape(1, 4))
    with LadrunoReader(path) as r:
        q = r.read_layers("stage_0", "fiber_stress_0").local_axes_quaternion
    per_e = 4 * N_LAYERS
    np.testing.assert_allclose(q[:per_e], np.tile(_frame("ASDShellQ4", 0),
                                                  (per_e, 1)), atol=1e-14)
    np.testing.assert_array_equal(q[per_e:], np.tile(stored, (per_e, 1)))


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
# #1969 — element-level records that used to read back as an empty Gauss slab
# ---------------------------------------------------------------------------
#
# Three silent empties at the same seam, found by the DomainCapture parity
# slice (#1962). The buckets are synthesised in the shape fork build
# 79e062367 writes (dumped from its live runs), so these hold without the
# fork; the live counterparts are the #1969 cases in
# tests/opensees/integration_ladruno/test_element_families_ladruno_live.py.

#: LadrunoLST's own 3-point interior rule in its GP order (LadrunoLST.cpp,
#: the SixNodeTri anchors). Spelled out: the reader takes the layout from
#: RESPONSE_CATALOG, so reading it from there would assert nothing.
LST_GP = np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])
LST_KEY = "33016-LadrunoLST[0:0:0]"


def _lst_file(
    tmp_path: Path, token: str, *, gp_param: "np.ndarray | None" = None,
) -> "tuple[Path, np.ndarray]":
    """The quad fixture with its Gauss buckets replaced by one LST bucket.

    The bucket is what LadrunoLST's untagged ``stress`` response writes:
    ONE element-level block of nine anonymous columns (``C1..C9``,
    ``NUM_COMP 9 x MULTIPLICITY 1``), GP-major, element 7. The element
    answers no ``integrationPoints``, so MODEL/ELEMENTS has no GP_PARAM
    for it unless ``gp_param`` adds one. Returns ``(path, DATA)``.
    """
    import shutil

    import h5py

    dst = tmp_path / "lst.ladruno"
    shutil.copy(QUAD, dst)
    data = np.arange(2 * 1 * 9, dtype=np.float64).reshape(2, 1, 9) + 0.5
    with h5py.File(dst, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        src = on_e["stress"][next(iter(on_e["stress"]))]
        step, time = src["STEP"][...], src["TIME"][...]
        del on_e["stress"], on_e["strain"]
        grp = on_e.create_group(f"{token}/{LST_KEY}")
        grp.create_dataset("DATA", data=data)
        grp.create_dataset("ID", data=np.array([[7]], dtype=np.int64))
        grp.create_dataset("STEP", data=step)
        grp.create_dataset("TIME", data=time)
        cm = grp.create_group("COLUMN_MAP")
        cm.create_dataset("LEVELS", data=np.array([[0]], dtype=np.int64))
        cm.create_dataset("GAUSS_ID", data=np.array([[-1]], dtype=np.int64))
        cm.create_dataset("NUM_COMP", data=np.array([[9]], dtype=np.int64))
        cm.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int64))
        cm.attrs["COMP_NAMES"] = ",".join(f"C{i + 1}" for i in range(9))
        if gp_param is not None:
            quad = f[stage]["MODEL"]["ELEMENTS"].create_group(
                "33016-LadrunoLST[201:0]/QUADRATURE",
            )
            quad.create_dataset("GP_PARAM", data=gp_param)
    return dst, data


def test_lst_stresses_token_reads_the_catalog_layout(tmp_path: Path) -> None:
    # The deck's keyword names the bucket, so `stresses` (not the catalog's
    # `stress`) must still reach the catalog: it used to read (1, 0).
    path, data = _lst_file(tmp_path, "stresses")
    with LadrunoReader(path) as r:
        assert {"stress_xx", "stress_yy", "stress_xy"} <= set(
            r.available_components("stage_0", ResultLevel.GAUSS),
        )
        for k, comp in enumerate(("stress_xx", "stress_yy", "stress_xy")):
            slab = r.read_gauss("stage_0", comp)
            assert slab.element_index.tolist() == [7, 7, 7]
            # GP-major columns: GP g, component k is DATA column 3g + k.
            np.testing.assert_array_equal(
                slab.values, data[:, 0, [k, 3 + k, 6 + k]],
            )
            np.testing.assert_allclose(slab.natural_coords, LST_GP, atol=1e-15)


def test_lst_stress_token_places_its_gauss_points(tmp_path: Path) -> None:
    # Values read under `stress` already; the GPs came back with no natural
    # coordinates, shape (3, 0), because the file carries no GP_PARAM.
    path, data = _lst_file(tmp_path, "stress")
    with LadrunoReader(path) as r:
        slab = r.read_gauss("stage_0", "stress_yy")
    np.testing.assert_array_equal(slab.values, data[:, 0, [1, 4, 7]])
    np.testing.assert_allclose(slab.natural_coords, LST_GP, atol=1e-15)


def test_file_gp_param_wins_over_the_catalog_rule(tmp_path: Path) -> None:
    # The file is self-describing: when it does carry GP_PARAM, that is
    # where the points are, and the catalog's rule is only the fallback.
    own = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]])
    path, _ = _lst_file(tmp_path, "stresses", gp_param=own)
    with LadrunoReader(path) as r:
        slab = r.read_gauss("stage_0", "stress_xx")
    np.testing.assert_array_equal(slab.natural_coords, own)


def test_aliased_token_with_a_wrong_width_still_refuses() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
        resolve_generic_gauss_blocks,
    )

    # The alias reaches the catalog, and with it the refusal: 8 anonymous
    # columns fit no LadrunoLST layout.
    with pytest.raises(GaussLayoutMismatch, match="Refusing to guess"):
        resolve_generic_gauss_blocks(
            _generic_block(8), token="stresses", bucket_key=LST_KEY,
        )


def _truss_axial_force_file(tmp_path: Path) -> Path:
    """The truss fixture with its ``basicForce`` bucket renamed
    ``axialForce`` — the token Truss writes for the deck's ``axialForce``
    (same one ``N`` column, same element-level block)."""
    import shutil

    import h5py

    dst = tmp_path / "truss_axial.ladruno"
    shutil.copy(TRUSS, dst)
    with h5py.File(dst, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        on_e.move("basicForce", "axialForce")
    return dst


def test_truss_axial_force_is_a_gauss_record(tmp_path: Path) -> None:
    # DomainCapture and .out put the truss axialForce under Gauss; the
    # .ladruno reader gave it only as a line station, and Gauss read empty.
    with LadrunoReader(_truss_axial_force_file(tmp_path)) as r:
        assert "axial_force" in r.available_components(
            "stage_0", ResultLevel.GAUSS,
        )
        gauss = r.read_gauss("stage_0", "axial_force")
        line = r.read_line_stations("stage_0", "axial_force")
    assert gauss.values.shape == (4, 2)
    assert gauss.element_index.tolist() == line.element_index.tolist()
    np.testing.assert_array_equal(gauss.values, line.values)
    # Tip load 10 -> axial force 10 in both members at the last step.
    np.testing.assert_allclose(gauss.values[-1], [10.0, 10.0])
    # One GP at the bar's midpoint (the file's own GP_PARAM).
    np.testing.assert_array_equal(gauss.natural_coords, [[0.0], [0.0]])


def test_named_element_block_disagreeing_with_the_catalog_is_left_alone(
) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        _Block,
        resolve_generic_gauss_blocks,
    )

    # A named block is re-read as Gauss data only when its own labels ARE
    # the catalog layout; `Vy` is not a Truss axial force.
    blocks = [_Block(level=0, gauss_id=-1, comp_names=("Vy",), col_start=0)]
    assert resolve_generic_gauss_blocks(
        blocks, token="axialForce", bucket_key="12-Truss[1:0:0]",
    ) is blocks


# ---------------------------------------------------------------------------
# #1990 / #1998 — named per-GP buckets: shell resultants, rule-0 placement
# ---------------------------------------------------------------------------
#
# The buckets are synthesised in the shape fork build 79e062367 writes
# (dumped from its live runs): one block per GP, LEVELS 2 for the shells'
# ``p11..q2`` and 4 for SixNodeTri / EightNodeQuad's ``sigma11..``. The quad
# shells write their rule (201) and a GP_PARAM; the triangle shells write
# their rule (102, 103) and no GP_PARAM; SixNodeTri and EightNodeQuad write
# rule 0 and no GP_PARAM. The live counterpart, with the MPCO parity check,
# is tests/opensees/integration_ladruno/test_results_catalog_hdf5_live.py.
#
# The expected names and coordinates are spelled out from the element
# sources, not read from RESPONSE_CATALOG, which is what the reader uses.

#: The shell element's source order (ASDShellQ4.cpp, ShellMITC4.cpp, ...).
SHELL_LABELS = ("p11", "p22", "p12", "m11", "m22", "m12", "q1", "q2")
SHELL_CANON = (
    "membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
    "bending_moment_xx", "bending_moment_yy", "bending_moment_xy",
    "transverse_shear_xz", "transverse_shear_yz",
)
_G2 = 1.0 / np.sqrt(3.0)
#: The GP_PARAM the fork writes for the quad shells.
QUAD_GL_2 = np.array([[-_G2, -_G2], [_G2, -_G2], [_G2, _G2], [-_G2, _G2]])
#: ASDShellT3.cpp mid-edge (XI, ETA).
TRI_GL_2B = np.array([[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
#: ShellDKGT.cpp sg/tg: the centroid, then the three 1/5-3/5 points.
TRI_GL_2C = np.array([[1 / 3, 1 / 3], [0.2, 0.6], [0.6, 0.2], [0.2, 0.2]])
#: SixNodeTri.cpp's 3-point interior rule.
TRI_GL_2 = np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])
_G3 = np.sqrt(0.6)
#: EightNodeQuad.cpp: corners, edge midpoints, centroid.
QUAD8_GL_3 = np.array([
    [-_G3, -_G3], [_G3, -_G3], [_G3, _G3], [-_G3, _G3],
    [0.0, -_G3], [_G3, 0.0], [0.0, _G3], [-_G3, 0.0], [0.0, 0.0],
])
PLANE_LABELS = ("sigma11", "sigma22", "sigma12")
PLANE_CANON = ("stress_xx", "stress_yy", "stress_xy")


def _named_gp_file(
    tmp_path: Path, key: str, labels: "tuple[str, ...]", n_gp: int, *,
    level: int, token: str = "stresses",
    gp_param: "np.ndarray | None" = None,
) -> "tuple[Path, np.ndarray]":
    """The quad fixture with its Gauss buckets replaced by one named bucket.

    ``n_gp`` blocks of ``labels``, one per GP (``GAUSS_ID`` 0..n-1), for
    element 7. ``gp_param`` adds a MODEL/ELEMENTS GP_PARAM for the class.
    Returns ``(path, DATA)``.
    """
    import shutil

    import h5py

    width = n_gp * len(labels)
    dst = tmp_path / "named.ladruno"
    shutil.copy(QUAD, dst)
    data = np.arange(2 * width, dtype=np.float64).reshape(2, 1, width) + 0.5
    with h5py.File(dst, "r+") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE["))
        on_e = f[stage]["RESULTS"]["ON_ELEMENTS"]
        src = on_e["stress"][next(iter(on_e["stress"]))]
        step, time = src["STEP"][...], src["TIME"][...]
        del on_e["stress"], on_e["strain"]
        grp = on_e.create_group(f"{token}/{key}")
        grp.create_dataset("DATA", data=data)
        grp.create_dataset("ID", data=np.array([[7]], dtype=np.int64))
        grp.create_dataset("STEP", data=step)
        grp.create_dataset("TIME", data=time)
        cm = grp.create_group("COLUMN_MAP")
        col = np.ones((n_gp, 1), dtype=np.int64)
        cm.create_dataset("LEVELS", data=level * col)
        cm.create_dataset(
            "GAUSS_ID", data=np.arange(n_gp, dtype=np.int64).reshape(-1, 1),
        )
        cm.create_dataset("NUM_COMP", data=len(labels) * col)
        cm.create_dataset("MULTIPLICITY", data=col)
        cm.attrs["COMP_NAMES"] = "\n".join([",".join(labels)] * n_gp)
        if gp_param is not None:
            prefix, _, rest = key.partition("[")
            quad = f[stage]["MODEL"]["ELEMENTS"].create_group(
                f"{prefix}[{rest.split(':')[0]}:0]/QUADRATURE",
            )
            quad.create_dataset("GP_PARAM", data=gp_param)
    return dst, data


def _assert_reads(
    path: Path, data: np.ndarray, canon: "tuple[str, ...]", n_gp: int,
    coords: np.ndarray,
) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussCoordsUnavailableWarning,
    )

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", GaussCoordsUnavailableWarning)
        with LadrunoReader(path) as r:
            assert set(canon) <= set(
                r.available_components("stage_0", ResultLevel.GAUSS),
            )
            for k, comp in enumerate(canon):
                slab = r.read_gauss("stage_0", comp)
                assert slab.element_index.tolist() == [7] * n_gp
                # GP-major: GP g, component k is DATA column g*ncomp + k.
                np.testing.assert_array_equal(
                    slab.values,
                    data[:, 0, [g * len(canon) + k for g in range(n_gp)]],
                )
                np.testing.assert_allclose(
                    slab.natural_coords, coords, rtol=0, atol=1e-15,
                )


@pytest.mark.parametrize("key,n_gp,gp_param,coords", [
    ("203-ASDShellQ4[201:0:0]", 4, QUAD_GL_2, QUAD_GL_2),
    ("53-ShellMITC4[201:0:0]", 4, QUAD_GL_2, QUAD_GL_2),
    ("156-ShellDKGQ[201:0:0]", 4, QUAD_GL_2, QUAD_GL_2),
    ("157-ShellNLDKGQ[201:0:0]", 4, QUAD_GL_2, QUAD_GL_2),
    ("204-ASDShellT3[102:0:0]", 3, None, TRI_GL_2B),
    ("167-ShellDKGT[103:0:0]", 4, None, TRI_GL_2C),
    ("168-ShellNLDKGT[103:0:0]", 4, None, TRI_GL_2C),
], ids=["ASDShellQ4", "ShellMITC4", "ShellDKGQ", "ShellNLDKGQ", "ASDShellT3",
        "ShellDKGT", "ShellNLDKGT"])
def test_shell_resultants_read_in_source_order(
    tmp_path: Path, key: str, n_gp: int,
    gp_param: "np.ndarray | None", coords: np.ndarray,
) -> None:
    # They read as an empty slab, with no warning (#1990).
    path, data = _named_gp_file(
        tmp_path, key, SHELL_LABELS, n_gp, level=2, gp_param=gp_param,
    )
    _assert_reads(path, data, SHELL_CANON, n_gp, coords)


@pytest.mark.parametrize("key,n_gp,coords", [
    ("209-SixNodeTri[0:0:0]", 3, TRI_GL_2),
    ("208-EightNodeQuad[0:0:0]", 9, QUAD8_GL_3),
], ids=["SixNodeTri", "EightNodeQuad"])
def test_rule_zero_named_bucket_places_its_gauss_points(
    tmp_path: Path, key: str, n_gp: int, coords: np.ndarray,
) -> None:
    # Values read already; the GPs came back with coords (n, 0) (#1998).
    path, data = _named_gp_file(tmp_path, key, PLANE_LABELS, n_gp, level=4)
    _assert_reads(path, data, PLANE_CANON, n_gp, coords)


def test_rule_zero_file_gp_param_wins(tmp_path: Path) -> None:
    own = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]])
    path, data = _named_gp_file(
        tmp_path, "209-SixNodeTri[0:0:0]", PLANE_LABELS, 3, level=4,
        gp_param=own,
    )
    _assert_reads(path, data, PLANE_CANON, 3, own)


def test_reordered_shell_resultants_refuse(tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
    )

    swapped = ("p22", "p11") + SHELL_LABELS[2:]
    path, _ = _named_gp_file(
        tmp_path, "203-ASDShellQ4[201:0:0]", swapped, 4, level=2,
        gp_param=QUAD_GL_2,
    )
    with LadrunoReader(path) as r:
        with pytest.raises(GaussLayoutMismatch, match="not the full source"):
            r.available_components("stage_0", ResultLevel.GAUSS)
        with pytest.raises(GaussLayoutMismatch, match="not the full source"):
            r.read_gauss("stage_0", "membrane_force_xx")


def test_shell_resultants_the_catalog_contradicts_refuse(
    tmp_path: Path,
) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
    )

    # ASDShellQ4 is catalogued with 4 GPs of shell resultants, not 3.
    path, _ = _named_gp_file(
        tmp_path, "203-ASDShellQ4[201:0:0]", SHELL_LABELS, 3, level=2,
    )
    with LadrunoReader(path) as r:
        with pytest.raises(GaussLayoutMismatch, match="catalog contradicts"):
            r.read_gauss("stage_0", "bending_moment_xx")


def test_rule_zero_uncatalogued_class_warns_by_name(tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussCoordsUnavailableWarning,
    )

    key = "99999-NotACatalogedClass[0:0:0]"
    path, data = _named_gp_file(tmp_path, key, PLANE_LABELS, 3, level=4)
    with LadrunoReader(path) as r:
        with pytest.warns(GaussCoordsUnavailableWarning, match=key.replace(
                "[", r"\[").replace("]", r"\]")):
            slab = r.read_gauss("stage_0", "stress_yy")
    np.testing.assert_array_equal(slab.values, data[:, 0, [1, 4, 7]])
    assert slab.natural_coords.shape == (3, 0)


def test_rule_zero_ambiguous_placement_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    from apeGmsh.opensees._response_catalog import IntRule
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussLayoutMismatch,
    )

    # A second SixNodeTri rule with the same points in another order: rule 0
    # names no rule, so the reader cannot tell which one the file used.
    base = RESPONSE_CATALOG[("SixNodeTri", IntRule.Triangle_GL_2, "stress")]
    monkeypatch.setitem(
        RESPONSE_CATALOG, ("SixNodeTri", IntRule.Triangle_GL_1, "stress"),
        dataclasses.replace(base, natural_coords=TRI_GL_2[::-1].copy()),
    )
    path, _ = _named_gp_file(
        tmp_path, "209-SixNodeTri[0:0:0]", PLANE_LABELS, 3, level=4,
    )
    with LadrunoReader(path) as r:
        with pytest.raises(GaussLayoutMismatch, match="place them differently"):
            r.read_gauss("stage_0", "stress_xx")


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


def test_implex_refusals_slot_order_matches_the_fork(tmp_path: Path) -> None:
    # Fork PR #838 (WP-99, F7) widened implexRefusals 4 -> 6: slot 4
    # (commitLatched) is the per-instance commit-refusal latch, slot 5
    # (latched) the process-wide post-latch count. Pinned in slot order.
    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    assert material_bucket_canonicals("material.implexRefusals") == (
        "implex_refusals_total", "implex_refusals_sign_change",
        "implex_refusals_control", "implex_refusals_companion",
        "implex_refusals_commit_latched", "implex_refusals_latched",
    )


def test_implex_refusals_six_column_block_resolves_with_real_names(
    tmp_path: Path,
) -> None:
    # Before the widening, a 6-column implexRefusals block mismatched the
    # (then 4-wide) table and fell through to the label map, which knows
    # none of these spellings -- the whole bucket was dropped in silence.
    import warnings

    path = _quad_with(tmp_path, {
        "material.implexRefusals": (
            "implexRefusals_total", "implexRefusals_signChange",
            "implexRefusals_control", "implexRefusals_companion",
            "implexRefusals_commitLatched", "implexRefusals_latched",
        ),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert {
        "implex_refusals_total", "implex_refusals_sign_change",
        "implex_refusals_control", "implex_refusals_companion",
        "implex_refusals_commit_latched", "implex_refusals_latched",
    } <= comps


def test_implex_refusals_width_mismatch_still_silently_drops() -> None:
    # A block whose width does not match the table's (e.g. a pre-#838
    # build still writing the old 4-column implexRefusals) falls through
    # _block_canonicals to the label map, which has no entry for these
    # material-private spellings, and _name_mismatch declines to compare
    # (it only checks when the widths already agree) -- so the column is
    # dropped with no mismatch warning at this layer. Confirmed by
    # reading _block_canonicals/_name_mismatch directly, not through the
    # full gauss_available() pipeline (which does separately warn once a
    # bucket lands in "dropped" -- that is a different code path).
    from apeGmsh.results.readers._ladruno_element_io import (
        _Block,
        _block_canonicals,
        _name_mismatch,
    )

    old_format_block = _Block(
        level=4, gauss_id=0,
        comp_names=(
            "implexRefusals_total", "implexRefusals_signChange",
            "implexRefusals_control", "implexRefusals_companion",
        ),
        col_start=0,
    )
    assert _block_canonicals("material.implexRefusals", old_format_block) == (
        [None, None, None, None]
    )
    assert _name_mismatch("material.implexRefusals", old_format_block) is None


def test_ladruno_tangent_is_not_named_yet(tmp_path: Path) -> None:
    # The 36-entry `ladrunoTangent` (responseID 96) is refused bare by the
    # recorder but has no by-position map: nothing reads it yet, so it takes
    # the documented unknown-bucket route rather than inventing 36 names.
    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    assert material_bucket_canonicals("material.ladrunoTangent") is None


def test_cp_iterations_is_a_scalar_gauss_component(tmp_path: Path) -> None:
    """ADR 0107 / fork ADR-97 — one scalar column per Gauss point.

    Shape pinned against a real ``.ladruno`` written by fork build
    ``ff47275fd``: ``NUM_COMP`` 1, ``MULTIPLICITY`` 1, ``FIBER_ID`` -1,
    one column per ``GAUSS_ID``. Before this entry the bucket was
    dropped with a :class:`GaussColumnDroppedWarning`.
    """
    import warnings

    from apeGmsh.results.readers._ladruno_element_io import (
        material_bucket_canonicals,
    )

    # A scalar bucket maps to exactly one canonical name...
    assert material_bucket_canonicals("material.cp_iterations") == (
        "cp_iterations",
    )
    # ...case-insensitively, like every other token in the table.
    assert (material_bucket_canonicals("material.CP_Iterations")
            == material_bucket_canonicals("material.cp_iterations"))

    # ...and it reaches available_components() without warning.
    path = _quad_with(tmp_path, {"material.cp_iterations": ("cp_iterations",)})
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnDroppedWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert "cp_iterations" in comps


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


# =====================================================================
# Real-name mismatch (runway A1)
#
# _MATERIAL_BUCKET_TOKENS resolves material-level Gauss buckets by
# COLUMN POSITION, which means a real-name build whose columns are
# reordered or renamed relative to the fork's documented order is
# mislabelled in SILENCE rather than merely dropped -- a measured probe
# found exactly that for material.substeps (file order
# substeps_capHit, substeps_me read as substeps_me, substeps_capHit).
# _MATERIAL_BUCKET_EXPECTED_NAMES lets the reader compare and warn.
# =====================================================================


def test_swapped_real_names_warn_once_and_keep_positional_canonicals(
    tmp_path: Path,
) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        GaussColumnNameMismatchWarning,
    )

    path = _quad_with(tmp_path, {
        "material.substeps": ("substeps_capHit", "substeps_me"),
    })
    with LadrunoReader(path) as r:
        with pytest.warns(GaussColumnNameMismatchWarning) as rec:
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "substeps_capHit" in msg and "substeps_me" in msg
        assert "material.substeps" in msg
        # The table still wins -- both positional canonicals are present,
        # the warning is a heads-up, not a fix.
        assert {"substeps_me", "substeps_cap_hit"} <= comps
        slab = r.read_gauss("stage_0", "substeps_me")
        assert slab.values.shape == (2, 4)


def test_correct_real_names_do_not_warn_mismatch(tmp_path: Path) -> None:
    import warnings

    from apeGmsh.results.readers._ladruno_element_io import (
        GaussColumnNameMismatchWarning,
    )

    path = _quad_with(tmp_path, {
        "material.substeps": ("substeps_me", "substeps_capHit"),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnNameMismatchWarning)
            r.available_components("stage_0", ResultLevel.GAUSS)


def test_generic_names_never_warn_mismatch(tmp_path: Path) -> None:
    import warnings

    from apeGmsh.results.readers._ladruno_element_io import (
        GaussColumnNameMismatchWarning,
    )

    path = _quad_with(tmp_path, {
        "material.substeps": ("C1", "C2"),
    })
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnNameMismatchWarning)
            r.available_components("stage_0", ResultLevel.GAUSS)


def test_token_without_documented_names_never_warns_mismatch(
    tmp_path: Path,
) -> None:
    # YieldStress is in the positional table but its exact COMP_NAMES
    # spelling is not documented anywhere -- no expectation, no check.
    import warnings

    from apeGmsh.results.readers._ladruno_element_io import (
        GaussColumnNameMismatchWarning,
    )

    path = _quad_with(tmp_path, {"material.YieldStress": ("Fy",)})
    with LadrunoReader(path) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GaussColumnNameMismatchWarning)
            comps = set(r.available_components("stage_0", ResultLevel.GAUSS))
    assert "yield_stress" in comps


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


# ---------------------------------------------------------------------------
# #1971 — rotations a node does not have: NaN with a model, a warning without
# ---------------------------------------------------------------------------
#
# Neither format records a node's ndf, so over a mixed model the recorder
# pads a solid node's rotation columns with 0.0. Every synthetic file below
# holds a non-zero closed-form value in EVERY column, padding included, so a
# column the reader keeps reads its exact value and a masked one reads NaN.
# MPCO sits beside .ladruno here because both readers share the mask.

ROT_3D = ("rotation_x", "rotation_y", "rotation_z")
#: ``(tag, class, nodes per element)``: continuum and structural classes.
TET = (179, "FourNodeTetrahedron", 4)
QUAD4 = (31, "FourNodeQuad", 4)
SHELL = (53, "ShellMITC4", 4)
BEAM2D = (3, "ElasticBeam2d", 2)
BEAM3D = (5, "ElasticBeam3d", 2)
FORMATS = ["ladruno", "mpco"]


def _nodal_value(t: int, node: int, col: int) -> float:
    return 1.0 + 10.0 * t + 0.5 * node + 0.01 * col


def _mixed_file(
    path: Path, fmt: str, node_ids: "list[int]", ndm: int,
    classes: "list[tuple[int, str, int]]", *, mode: bool = False,
    envelope: bool = False,
) -> Path:
    """A ``.ladruno`` or ``.mpco`` with ``DISPLACEMENT`` and ``ROTATION``
    for every node (two steps), or one mode of ``MODES_OF_VIBRATION(U|R)``,
    or (``.ladruno``, ``envelope=True``) their ``-envelope`` extremes:
    ``MIN``/``MAX``/``ABSMAX`` are step 0/1/1's values, ``ARG_STEP`` is 1.

    ``U`` holds ``ndm`` translations, ``R`` the model's rotations (``Rx,Ry,
    Rz`` in 3-D, ``Rz`` in 2-D); column ``c`` of ``R`` holds
    ``_nodal_value(t, node, 3 + c)``. ``MODEL/ELEMENTS`` lists one element
    of each class over the leading nodes, as each format writes it.
    """
    import h5py

    n = len(node_ids)
    groups = {
        "U": ("Ux,Uy,Uz" if ndm == 3 else "Ux,Uy"),
        "R": ("Rx,Ry,Rz" if ndm == 3 else "Rz"),
    }
    ids = np.array(node_ids, dtype=np.int32).reshape(-1, 1)
    with h5py.File(path, "w") as f:
        if fmt == "ladruno":
            info = f.create_group("INFO")
            info.attrs["GENERATOR"] = np.array([b"Ladruno"])
            info.attrs["FORMAT_VERSION"] = np.array([1], dtype=np.int32)
            info.attrs["SPATIAL_DIM"] = np.array([ndm], dtype=np.int32)
        stage = f.create_group("MODEL_STAGE[1]")
        if fmt == "ladruno":
            stage.attrs["KIND"] = np.array([b"eigen" if mode else b"static"])
        nodes = stage.create_group("MODEL/NODES")
        nodes.create_dataset("ID", data=ids)
        nodes.create_dataset("COORDINATES", data=np.zeros((n, ndm)))
        for k, (tag, cls, npe) in enumerate(classes):
            conn = np.array([[100 + k, *node_ids[:npe]]], dtype=np.int32)
            key = f"MODEL/ELEMENTS/{tag}-{cls}[0:0]"
            if fmt == "ladruno":
                stage.create_group(key).create_dataset("CONNECTIVITY", data=conn)
            else:
                stage.create_dataset(key, data=conn)
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        for kind, labels in groups.items():
            width = len(labels.split(","))
            name = (f"MODES_OF_VIBRATION({kind})" if mode
                    else {"U": "DISPLACEMENT", "R": "ROTATION"}[kind])
            grp = on_nodes.create_group(name)
            grp.attrs["COMPONENTS"] = np.array([labels.encode()])
            grp.create_dataset("ID", data=ids)
            off = 0 if kind == "U" else 3
            block = np.array([
                [[_nodal_value(t, nd, off + c) for c in range(width)]
                 for nd in node_ids]
                for t in range(2)
            ])
            if envelope:
                del on_nodes[name]
                grp = stage.require_group("RESULTS/ENVELOPES/ON_NODES"
                                          ).create_group(name)
                grp.attrs["COMPONENTS"] = np.array([labels.encode()])
                grp.create_dataset("ID", data=ids)
                grp.create_dataset("MIN", data=block[0])
                grp.create_dataset("MAX", data=block[1])
                grp.create_dataset("ABSMAX", data=block[1])
                grp.create_dataset("ARG_STEP",
                                   data=np.ones(block[0].shape, dtype=np.int64))
            elif mode:
                ds = grp.create_dataset("DATA/STEP_0/MODE_0", data=block[0])
                ds.attrs["LAMBDA"] = np.array([4.0])
            elif fmt == "ladruno":
                grp.create_dataset("DATA", data=block)
                grp.create_dataset("STEP", data=np.array([1, 2], dtype=np.int32))
                grp.create_dataset("TIME", data=np.array([0.5, 1.0]))
            else:
                for t in range(2):
                    ds = grp.create_dataset(f"DATA/STEP_{t}", data=block[t])
                    ds.attrs["TIME"] = np.array([0.5 * (t + 1)])
    return path


def _open_reader(fmt: str, path: Path):
    return LadrunoReader(path) if fmt == "ladruno" else MPCOReader(path)


def _layout_only():
    """DOF roles that refuse no rotation, so these tests pin the ndf
    layout mask alone; the roles' own mask is
    ``tests/results/test_rotation_roles_masks.py`` (#2029)."""
    from apeGmsh.opensees._readback import rotation_roles_from_tokens

    return rotation_roles_from_tokens(())


def _stage(reader, mode: bool) -> str:
    return [s.id for s in reader.stages() if (s.kind == "mode") == mode][0]


def _last(reader, stage: str, comp: str) -> "dict[int, float]":
    slab = reader.read_nodes(stage, comp)
    return {int(i): float(v) for i, v in zip(slab.node_ids, slab.values[-1])}


@pytest.mark.parametrize("fmt", FORMATS)
def test_rotation_beyond_ndf_is_nan_3d(tmp_path: Path, fmt: str) -> None:
    """Node 1: tet (ndf 3); 2: shell (6); 3: 3-D u-p (4); 4: ndf 5.

    Only the 6-dof node has rotations; the rest read NaN. A u-p node's
    fourth DOF is pressure, so a rotation there is fabricated.
    Translations stay exact everywhere.
    """
    ndf = {1: 3, 2: 6, 3: 4, 4: 5}
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2, 3, 4], 3, [TET, SHELL])
    with _open_reader(fmt, path) as r:
        r.attach_nodes_ndf(ndf, 3, roles=_layout_only)
        stage = _stage(r, False)
        for c, comp in enumerate(ROT_3D):
            got = _last(r, stage, comp)
            assert got[2] == _nodal_value(1, 2, 3 + c), comp
            assert np.isnan([got[1], got[3], got[4]]).all(), (comp, got)
        for c, comp in enumerate(("displacement_x", "displacement_y",
                                  "displacement_z")):
            got = _last(r, stage, comp)
            assert got == {nd: _nodal_value(1, nd, c) for nd in ndf}, comp


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mode", [False, True], ids=["steps", "mode"])
def test_rotation_beyond_ndf_is_nan_2d(
    tmp_path: Path, fmt: str, mode: bool,
) -> None:
    """2-D: quad node 1 (ndf 2) reads NaN ``rotation_z``; frame node 2
    (ndf 3) keeps its real rotation, in a step stage and a mode stage.

    The mask keys on the node's DOF layout: a ``dof > ndf`` rule on the
    file's labels (``rotation_z`` is the sixth DOF) would drop node 2's
    real rotation too.
    """
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], 2, [QUAD4, BEAM2D],
                       mode=mode)
    with _open_reader(fmt, path) as r:
        r.attach_nodes_ndf({1: 2, 2: 3}, 2, roles=_layout_only)
        got = _last(r, _stage(r, mode), "rotation_z")
        assert np.isnan(got[1])
        assert got[2] == _nodal_value(0 if mode else 1, 2, 3)


@pytest.mark.parametrize("fmt", FORMATS)
def test_model_missing_a_file_node_is_not_used(tmp_path: Path, fmt: str) -> None:
    """A model whose nodes_ndf misses a node of the file is another model:
    its ndf is not used, so no real rotation is NaN'd on its say-so, and
    the read warns as a model-less one does, naming the missing node."""
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], 3, [TET, SHELL])
    with _open_reader(fmt, path) as r:
        r.attach_nodes_ndf({1: 3}, 3, roles=_layout_only)
        with pytest.warns(AbsentDofWarning, match=r"misses nodes \[2\]"):
            got = _last(r, _stage(r, False), "rotation_x")
        assert got == {1: _nodal_value(1, 1, 3), 2: _nodal_value(1, 2, 3)}


@pytest.mark.parametrize("fmt", FORMATS)
def test_result_node_outside_the_model_raises(tmp_path: Path, fmt: str) -> None:
    """A result row for a node neither MODEL/NODES nor nodes_ndf lists:
    whether its rotation is a value or padding is unknown, so it raises."""
    import h5py

    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], 3, [TET, SHELL])
    with h5py.File(path, "r+") as f:
        nodes = f["MODEL_STAGE[1]/MODEL/NODES"]
        del nodes["ID"], nodes["COORDINATES"]
        nodes.create_dataset("ID", data=np.array([[1]], dtype=np.int32))
        nodes.create_dataset("COORDINATES", data=np.zeros((1, 3)))
    with _open_reader(fmt, path) as r:
        r.attach_nodes_ndf({1: 3}, 3, roles=_layout_only)
        with pytest.raises(ValueError, match=r"nodes \[2\]"):
            r.read_nodes(_stage(r, False), "rotation_x")


@pytest.mark.parametrize("fmt", FORMATS)
def test_rotation_without_ndf_over_solids_warns_once(
    tmp_path: Path, fmt: str,
) -> None:
    import warnings

    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], 3, [TET, SHELL])
    with _open_reader(fmt, path) as r, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stage = _stage(r, False)
        r.read_nodes(stage, "displacement_x")
        assert not w                       # translations never warn
        for c, comp in enumerate(ROT_3D):
            assert _last(r, stage, comp)[1] == _nodal_value(1, 1, 3 + c)
    hits = [x for x in w if issubclass(x.category, AbsentDofWarning)]
    assert len(hits) == 1, [str(x.message) for x in w]
    assert "model_h5=" in str(hits[0].message)
    assert "#2015" in str(hits[0].message)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("ndm,classes", [
    (3, [BEAM3D, SHELL]), (2, [BEAM2D]),
], ids=["frame-shell-3d", "frame-2d"])
def test_frame_only_file_reads_rotations_silently(
    tmp_path: Path, fmt: str, ndm: int, classes: list,
) -> None:
    """No continuum class: every node has rotations, so no warning."""
    import warnings

    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], ndm, classes)
    with _open_reader(fmt, path) as r, warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for comp in (ROT_3D if ndm == 3 else ("rotation_z",)):
            assert len(_last(r, _stage(r, False), comp)) == 2


@pytest.mark.parametrize("fmt", FORMATS)
def test_up_element_without_ndf_warns(tmp_path: Path, fmt: str) -> None:
    """A u-p element (no ``stress`` catalog row) still warns: its nodes'
    fourth DOF is pressure, which the recorder writes as a rotation."""
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, list(range(1, 9)), 3,
                       [(33019, "LadrunoUP", 8)])
    with _open_reader(fmt, path) as r:
        with pytest.warns(AbsentDofWarning, match="u-p"):
            r.read_nodes(_stage(r, False), "rotation_x")


@pytest.mark.parametrize("fmt", FORMATS)
def test_archive_older_than_nodes_ndf_says_so(tmp_path: Path, fmt: str) -> None:
    """An archive without /opensees/nodes_ndf: the warning names that, not
    a missing ``model_h5=``, which the caller did pass."""
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, [1, 2], 3, [TET, SHELL])
    with _open_reader(fmt, path) as r:
        r.attach_nodes_ndf(None, 3, roles=_layout_only)
        with pytest.warns(AbsentDofWarning) as w:
            r.read_nodes(_stage(r, False), "rotation_x")
    msg = str(w[0].message)
    assert "older than /opensees/nodes_ndf" in msg
    assert "Pass model_h5=" not in msg


# -- node envelopes (``.ladruno`` only: no other reader has them) ------------


def test_envelope_rotation_beyond_ndf_is_nan(tmp_path: Path) -> None:
    """``read_node_envelope`` masks as ``read_nodes`` does: NaN extremes and
    ``arg_step == -1`` where the node has no such rotation."""
    path = _mixed_file(tmp_path / "m.ladruno", "ladruno", [1, 2, 3], 3,
                       [TET, SHELL], envelope=True)
    with LadrunoReader(path) as r:
        r.attach_nodes_ndf({1: 3, 2: 6, 3: 4}, 3, roles=_layout_only)
        stage = _stage(r, False)
        for c, comp in enumerate(ROT_3D):
            env = r.read_node_envelope(stage, comp)
            assert env.node_ids.tolist() == [1, 2, 3]
            for got, t in ((env.min, 0), (env.max, 1), (env.absmax, 1)):
                assert got[1] == _nodal_value(t, 2, 3 + c), comp
                assert np.isnan(got[[0, 2]]).all(), comp
            assert env.arg_step.tolist() == [-1, 1, -1], comp
        disp = r.read_node_envelope(stage, "displacement_x")
        assert disp.max.tolist() == [_nodal_value(1, n, 0) for n in (1, 2, 3)]
        assert disp.arg_step.tolist() == [1, 1, 1]


def test_envelope_rotation_without_ndf_warns_once(tmp_path: Path) -> None:
    path = _mixed_file(tmp_path / "m.ladruno", "ladruno", [1, 2], 3,
                       [TET, SHELL], envelope=True)
    with LadrunoReader(path) as r, pytest.warns(AbsentDofWarning) as w:
        for comp in ROT_3D:
            env = r.read_node_envelope(_stage(r, False), comp)
            assert np.isfinite(env.max).all()
    assert len([x for x in w if issubclass(x.category, AbsentDofWarning)]) == 1


# -- the factories: ``model_h5=`` attaches /opensees/nodes_ndf ---------------


def _tet_beside_shell_archive(g, tmp_path: Path):
    """A real bridge archive: a tet block (3-dof nodes) and a detached
    MITC4 plate (6-dof nodes) under ``ops.model(ndm=3, ndf=6)``."""
    from apeGmsh.opensees import apeSees

    G = g.model.geometry
    block = G.add_box(0, 0, -1.0, 1.0, 1.0, 1.0)
    plate = G.add_rectangle(0.0, 0.0, 0.5, 1.0, 1.0)
    g.model.sync()
    g.physical.add(3, [block], name="Block")
    g.physical.add(2, [plate], name="Plate")
    g.mesh.structured.set_recombine("Plate", dim=2)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    solid = sorted({int(n) for n in fem.nodes.select(pg="Block").ids})
    shell = sorted({int(n) for n in fem.nodes.select(pg="Plate").ids})
    assert solid and shell and not set(solid) & set(shell)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Block", material=ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Plate",
        section=ops.section.ElasticMembranePlateSection(E=1e6, nu=0.0, h=0.1),
    )
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    return fem, solid, shell, model_h5


@pytest.mark.parametrize("fmt", FORMATS)
def test_factory_with_model_reads_nan_on_solid_nodes(
    g, tmp_path: Path, fmt: str,
) -> None:
    from apeGmsh.results import Results

    fem, solid, shell, model_h5 = _tet_beside_shell_archive(g, tmp_path)
    path = _mixed_file(tmp_path / f"m.{fmt}", fmt, solid + shell, 3,
                       [TET, SHELL])
    factory = Results.from_ladruno if fmt == "ladruno" else Results.from_mpco
    r = factory(path, fem=fem, model_h5=model_h5)
    for c, comp in enumerate(ROT_3D):
        slab = r.nodes.get(component=comp)
        got = {int(i): float(v) for i, v in zip(slab.node_ids, slab.values[-1])}
        assert np.isnan([got[n] for n in solid]).all(), comp
        assert [got[n] for n in shell] == [
            _nodal_value(1, n, 3 + c) for n in shell], comp


def test_factory_node_envelope_with_model_reads_nan(g, tmp_path: Path) -> None:
    """``Results.node_envelope`` goes through the same mask."""
    from apeGmsh.results import Results

    fem, solid, shell, model_h5 = _tet_beside_shell_archive(g, tmp_path)
    path = _mixed_file(tmp_path / "m.ladruno", "ladruno", solid + shell, 3,
                       [TET, SHELL], envelope=True)
    df = Results.from_ladruno(path, fem=fem, model_h5=model_h5).node_envelope(
        "rotation_y")
    assert df.loc[solid, "absmax"].isna().all()
    assert (df.loc[solid, "arg_step"] == -1).all()
    assert df.loc[shell, "absmax"].tolist() == [
        _nodal_value(1, n, 4) for n in shell]


def test_factory_without_model_warns_once(g, tmp_path: Path) -> None:
    """The self-sufficient ``.ladruno`` route has no per-node ndf: the
    padded 0.0 stays, and the first rotation read says so."""
    from apeGmsh.results import Results

    fem, solid, shell, _model_h5 = _tet_beside_shell_archive(g, tmp_path)
    path = _mixed_file(tmp_path / "m.ladruno", "ladruno", solid + shell, 3,
                       [TET, SHELL])
    r = Results.from_ladruno(path, fem=fem)
    with pytest.warns(AbsentDofWarning) as w:
        for comp in ROT_3D:
            assert np.isfinite(r.nodes.get(component=comp).values).all()
    assert len([x for x in w if issubclass(x.category, AbsentDofWarning)]) == 1

