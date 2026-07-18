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
