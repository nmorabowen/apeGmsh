"""ADR 0077 Tier 1 — ``ParallelModalResult`` harvest + guards.

The eigenvalue write-out format is pinned by
``TclEmitter.eigen_feast_parallel`` (a single whitespace-separated line of
``λ_i = ω_i²``), and the P3 mode-shape harvest format by the same emit
(``mode_shapes.json`` sidecar + one whitespace row per
``mode_shape_<k>.out``), so ``from_job`` parsing is deterministic and
testable without a live distributed run. Modal properties (MPI-blind,
INV-2) fail loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees.analysis.modal import ParallelModalResult


def _write_out(tmp_path, text: str, name: str = "eigenvalues.out"):
    (tmp_path / name).write_text(text)
    return tmp_path


def _write_shape_job(tmp_path):
    """Two modes, three nodes (tags 2, 5, 9), ndf=2 — the pinned P3
    layout: node-major rows, sidecar carries the column map."""
    _write_out(tmp_path, "100.0 400.0")
    (tmp_path / "mode_shapes.json").write_text(
        '{"nodes": [2,5,9], "ndf": 2}'
    )
    (tmp_path / "mode_shape_1.out").write_text(
        "0.1 0.2  0.3 0.4  0.5 0.6\n"
    )
    (tmp_path / "mode_shape_2.out").write_text(
        "1.1 1.2  1.3 1.4  1.5 1.6\n"
    )
    return tmp_path


def test_from_job_reads_eigenvalues_and_derived(tmp_path) -> None:
    # Two degenerate low modes + one higher (whitespace/newline tolerant).
    job = _write_out(tmp_path, "  246.740110  246.740110   1974.00  \n")
    res = ParallelModalResult.from_job(str(job))

    assert res.n_modes == 3
    np.testing.assert_allclose(
        res.eigenvalues, [246.740110, 246.740110, 1974.00]
    )
    np.testing.assert_allclose(res.omega, np.sqrt(res.eigenvalues))
    np.testing.assert_allclose(res.freq, res.omega / (2.0 * np.pi))
    np.testing.assert_allclose(res.periods, 1.0 / res.freq)
    assert res.certified is None


def test_from_job_certified_passthrough(tmp_path) -> None:
    job = _write_out(tmp_path, "100.0 400.0")
    res = ParallelModalResult.from_job(str(job), certified=True)
    assert res.certified is True
    assert res.n_modes == 2


def test_from_job_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="eigenvalue write-out"):
        ParallelModalResult.from_job(str(tmp_path))


def test_from_job_empty_band_is_zero_modes(tmp_path) -> None:
    job = _write_out(tmp_path, "\n")
    res = ParallelModalResult.from_job(str(job))
    assert res.n_modes == 0


def test_from_job_reads_mode_shapes(tmp_path) -> None:
    res = ParallelModalResult.from_job(str(_write_shape_job(tmp_path)))

    np.testing.assert_array_equal(res.shape_nodes, [2, 5, 9])
    field = res.mode_shape_field(1)
    assert field.shape == (3, 2)
    np.testing.assert_allclose(field[1], [0.3, 0.4])
    # Per-node accessor matches the EigenResult (node, mode) convention.
    np.testing.assert_allclose(res.mode_shape(2, 1), [0.1, 0.2])
    np.testing.assert_allclose(res.mode_shape(9, 2), [1.5, 1.6])


def test_mode_shape_unknown_node_and_bad_mode(tmp_path) -> None:
    res = ParallelModalResult.from_job(str(_write_shape_job(tmp_path)))
    with pytest.raises(KeyError, match="not in the harvested field"):
        res.mode_shape(3, 1)
    with pytest.raises(IndexError, match="out of range"):
        res.mode_shape_field(3)
    with pytest.raises(IndexError, match="out of range"):
        res.mode_shape_field(0)


def test_mode_shape_without_sidecar_fails_loud(tmp_path) -> None:
    """A pre-P3 run dir (no sidecar) still harvests eigenvalues; the
    shape accessors fail loud."""
    res = ParallelModalResult.from_job(str(_write_out(tmp_path, "100.0")))
    assert res.n_modes == 1
    with pytest.raises(FileNotFoundError, match="mode_shapes.json"):
        res.mode_shape(1, 1)
    with pytest.raises(FileNotFoundError, match="mode_shapes.json"):
        _ = res.shape_nodes


def test_from_job_missing_mode_file_raises(tmp_path) -> None:
    job = _write_shape_job(tmp_path)
    (job / "mode_shape_2.out").unlink()
    with pytest.raises(FileNotFoundError, match="mode_shape_2.out"):
        ParallelModalResult.from_job(str(job))


def test_from_job_mode_row_width_mismatch_raises(tmp_path) -> None:
    job = _write_shape_job(tmp_path)
    (job / "mode_shape_1.out").write_text("0.1 0.2 0.3\n")
    with pytest.raises(ValueError, match="expected 3 nodes x 2 dofs"):
        ParallelModalResult.from_job(str(job))


def test_modal_properties_fail_loud_mpi_blind(tmp_path) -> None:
    res = ParallelModalResult.from_job(str(_write_out(tmp_path, "100.0")))
    with pytest.raises(NotImplementedError, match="MPI-blind"):
        res.participation_factors("MX")
    with pytest.raises(NotImplementedError, match="MPI-blind"):
        _ = res.mass_ratios


# ---------------------------------------------------------------------------
# Tier 1B — the PARTITIONED (ARPACK) harvest: per-rank sidecars merged
# client-side, with the shared-boundary agreement assertion (ADR 0077
# INV-12). The replicated reader above cannot be reused here: on a
# partitioned deck a rank-0-only read returns a PARTIAL field with no
# error, which is the failure this whole branch exists to make loud.
# ---------------------------------------------------------------------------


def _write_partitioned_job(tmp_path, *, rank1_boundary=(0.5, 0.6)):
    """Two modes, ndf=2, two ranks sharing node 5.

    Rank 0 owns nodes 3, 5; rank 1 owns nodes 5, 8 — so node 5 is written
    twice and must agree. ``rank1_boundary`` lets a test poison rank 1's
    copy of node 5 in mode 1.
    """
    _write_out(tmp_path, "100.0 400.0")
    (tmp_path / "mode_shapes_rank0.json").write_text(
        '{"nodes": [3,5], "ndf": 2, "ndm": 2}'
    )
    (tmp_path / "mode_shapes_rank1.json").write_text(
        '{"nodes": [5,8], "ndf": 2, "ndm": 2}'
    )
    # mode 1: node3=(0.1,0.2) node5=(0.5,0.6) node8=(0.9,1.0)
    (tmp_path / "mode_shape_1_rank0.out").write_text("0.1 0.2  0.5 0.6\n")
    (tmp_path / "mode_shape_1_rank1.out").write_text(
        f"{rank1_boundary[0]} {rank1_boundary[1]}  0.9 1.0\n"
    )
    # mode 2
    (tmp_path / "mode_shape_2_rank0.out").write_text("1.1 1.2  1.5 1.6\n")
    (tmp_path / "mode_shape_2_rank1.out").write_text("1.5 1.6  1.9 2.0\n")
    return tmp_path


def test_from_job_merges_per_rank_shapes(tmp_path) -> None:
    """The merged field is the FULL model in sorted node order — the same
    ``(n_nodes, ndf)`` layout the replicated path produces, so one
    accessor surface serves both deck shapes."""
    res = ParallelModalResult.from_job(str(_write_partitioned_job(tmp_path)))

    np.testing.assert_array_equal(res.shape_nodes, [3, 5, 8])
    field = res.mode_shape_field(1)
    assert field.shape == (3, 2)
    np.testing.assert_allclose(field[0], [0.1, 0.2])   # rank 0 only
    np.testing.assert_allclose(field[1], [0.5, 0.6])   # shared
    np.testing.assert_allclose(field[2], [0.9, 1.0])   # rank 1 only
    np.testing.assert_allclose(res.mode_shape(8, 2), [1.9, 2.0])


def test_from_job_partitioned_boundary_disagreement_raises(tmp_path) -> None:
    """INV-12. Both ranks ran the SAME replicated ARPACK outer loop over
    the same global operator, so a shared node's components must match.
    Disagreement means each rank ran a private Lanczos over its own
    subdomain — the exact failure a pre-#668 binary (or a deck that lost
    its ``system Mumps``) produces, and otherwise completely silent."""
    job = _write_partitioned_job(tmp_path, rank1_boundary=(0.4, 0.6))

    with pytest.raises(ValueError, match="shared boundary node 5") as exc:
        ParallelModalResult.from_job(str(job))
    msg = str(exc.value)
    assert "rank 0" in msg and "rank 1" in msg
    assert "mode 1" in msg
    assert "0.5" in msg and "0.4" in msg      # both values reported
    assert "5a522b03b" in msg                 # points at the likely cause


def test_from_job_partitioned_boundary_tolerance_is_not_wide(
    tmp_path,
) -> None:
    """The check must not be so loose it stops discriminating. A 1e-3
    relative disagreement — three orders TIGHTER than anything a private
    per-rank Lanczos would give — must still raise."""
    job = _write_partitioned_job(tmp_path, rank1_boundary=(0.5005, 0.6))
    with pytest.raises(ValueError, match="shared boundary node 5"):
        ParallelModalResult.from_job(str(job))


def test_from_job_partitioned_boundary_accepts_formatting_noise(
    tmp_path,
) -> None:
    """...but recorder text formatting must not trip it. The two copies
    are the same double, so the only real slack needed is print
    precision."""
    job = _write_partitioned_job(tmp_path, rank1_boundary=(0.5000000001, 0.6))
    res = ParallelModalResult.from_job(str(job))
    np.testing.assert_allclose(res.mode_shape_field(1)[1], [0.5, 0.6])


def test_from_job_partitioned_missing_rank_mode_file_raises(tmp_path) -> None:
    job = _write_partitioned_job(tmp_path)
    (job / "mode_shape_2_rank1.out").unlink()
    with pytest.raises(FileNotFoundError, match="mode_shape_2_rank1.out"):
        ParallelModalResult.from_job(str(job))


def test_from_job_partitioned_ndf_mismatch_raises(tmp_path) -> None:
    job = _write_partitioned_job(tmp_path)
    (job / "mode_shapes_rank1.json").write_text(
        '{"nodes": [5,8], "ndf": 3, "ndm": 2}'
    )
    with pytest.raises(ValueError, match="disagree on ndf"):
        ParallelModalResult.from_job(str(job))


def test_from_job_partitioned_empty_spectrum_harvests_cleanly(
    tmp_path,
) -> None:
    """A partitioned deck run single-process solves only rank 0's
    submodel and can return an EMPTY spectrum with no error (live-
    observed). ``n_modes == 0`` is the visible signal; the reader must
    not crash on the way there."""
    (tmp_path / "eigenvalues.out").write_text("\n")
    (tmp_path / "mode_shapes_rank0.json").write_text(
        '{"nodes": [3,5], "ndf": 2, "ndm": 2}'
    )
    res = ParallelModalResult.from_job(str(tmp_path))
    assert res.n_modes == 0
    np.testing.assert_array_equal(res.shape_nodes, [3, 5])


def test_from_job_partitioned_to_native_roundtrip_uses_merged_field(
    tmp_path,
) -> None:
    """The merged field feeds the existing viewer binding unchanged —
    ``ndm`` comes off the per-rank sidecars like the replicated one."""
    res = ParallelModalResult.from_job(str(_write_partitioned_job(tmp_path)))
    assert res._shape_ndm == 2
    # ndm=2 -> two displacement components, no rotations (ndf=2).
    field = res.mode_shape_field(2)
    np.testing.assert_allclose(field[:, 0], [1.1, 1.5, 1.9])
