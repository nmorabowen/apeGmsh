"""ADR 0077 Tier 1 — ``ParallelModalResult`` eigenvalue harvest + guards.

The eigenvalue write-out format is pinned by
``TclEmitter.eigen_feast_parallel`` (a single whitespace-separated line of
``λ_i = ω_i²``), so ``from_job`` parsing is deterministic and testable
without a live distributed run. Mode-shape harvest (P3) and modal
properties (MPI-blind, INV-2) fail loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees.analysis.modal import ParallelModalResult


def _write_out(tmp_path, text: str, name: str = "eigenvalues.out"):
    (tmp_path / name).write_text(text)
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


def test_mode_shape_not_implemented_p3(tmp_path) -> None:
    res = ParallelModalResult.from_job(str(_write_out(tmp_path, "100.0")))
    with pytest.raises(NotImplementedError, match="P3"):
        res.mode_shape(1)


def test_modal_properties_fail_loud_mpi_blind(tmp_path) -> None:
    res = ParallelModalResult.from_job(str(_write_out(tmp_path, "100.0")))
    with pytest.raises(NotImplementedError, match="MPI-blind"):
        res.participation_factors("MX")
    with pytest.raises(NotImplementedError, match="MPI-blind"):
        _ = res.mass_ratios
