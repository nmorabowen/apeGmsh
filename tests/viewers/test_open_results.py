"""Pure-layer tests for the viewer's File → Open Results… support.

Covers :func:`sniff_results_format`, :func:`model_requirement`, and
:func:`build_results` from :mod:`apeGmsh.viewers.ui._open_results`. All
headless — no Qt. Real ``.ladruno`` fixtures drive the round-trip; native
and MPCO classification run against fabricated minimal HDF5 files (the
same fabrication pattern as ``tests/results/readers/test_ladruno_reader``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.viewers.ui._open_results import (
    build_results,
    model_requirement,
    sniff_results_format,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"


def _write_native(path: Path, *, with_opensees: bool) -> Path:
    import h5py

    with h5py.File(path, "w") as h:
        h.create_group("model").create_group("meta")
        if with_opensees:
            h.create_group("opensees")
    return path


def _write_mpco(path: Path) -> Path:
    import h5py

    with h5py.File(path, "w") as h:
        h.create_group("MODEL_STAGE[1.000000]")
    return path


def _write_ladruno_marker(path: Path) -> Path:
    import h5py

    with h5py.File(path, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "Ladruno"
        info.attrs["FORMAT_VERSION"] = 1
    return path


# ── sniff_results_format ──────────────────────────────────────────────

def test_sniff_ladruno_real_fixture() -> None:
    assert sniff_results_format(TRUSS) == "ladruno"


def test_sniff_ladruno_by_content_over_extension(tmp_path: Path) -> None:
    # Content (GENERATOR) wins even with a misleading .h5 extension.
    f = _write_ladruno_marker(tmp_path / "mislabelled.h5")
    assert sniff_results_format(f) == "ladruno"


def test_sniff_native(tmp_path: Path) -> None:
    f = _write_native(tmp_path / "run.h5", with_opensees=True)
    assert sniff_results_format(f) == "native"


def test_sniff_mpco_by_content(tmp_path: Path) -> None:
    # .out extension but MPCO content → classified by content.
    f = _write_mpco(tmp_path / "run.out")
    assert sniff_results_format(f) == "mpco"


def test_sniff_unknown_for_foreign_h5(tmp_path: Path) -> None:
    import h5py

    f = tmp_path / "foreign.dat"
    with h5py.File(f, "w") as h:
        h.create_group("SOMETHING_ELSE")
    assert sniff_results_format(f) == "unknown"


def test_sniff_missing_file_falls_back_to_extension(tmp_path: Path) -> None:
    assert sniff_results_format(tmp_path / "ghost.ladruno") == "ladruno"
    assert sniff_results_format(tmp_path / "ghost.mpco") == "mpco"
    assert sniff_results_format(tmp_path / "ghost.h5") == "native"
    assert sniff_results_format(tmp_path / "ghost.xyz") == "unknown"


# ── model_requirement ─────────────────────────────────────────────────

def test_requirement_ladruno_optional() -> None:
    assert model_requirement(TRUSS) == ("ladruno", "optional")


def test_requirement_mpco_required(tmp_path: Path) -> None:
    f = _write_mpco(tmp_path / "run.mpco")
    assert model_requirement(f) == ("mpco", "required")


def test_requirement_native_with_model_none(tmp_path: Path) -> None:
    f = _write_native(tmp_path / "run.h5", with_opensees=True)
    assert model_requirement(f) == ("native", "none")


def test_requirement_native_without_model_required(tmp_path: Path) -> None:
    f = _write_native(tmp_path / "run.h5", with_opensees=False)
    assert model_requirement(f) == ("native", "required")


def test_requirement_unknown_is_required(tmp_path: Path) -> None:
    f = tmp_path / "ghost.xyz"
    assert model_requirement(f) == ("unknown", "required")


# ── build_results ─────────────────────────────────────────────────────

def test_build_results_ladruno_standalone() -> None:
    results = build_results(TRUSS, model_path=None)
    try:
        assert results.fem is not None        # self-sufficient broker
        assert results.model is not None       # always non-None (INV-1)
    finally:
        results.close()


def test_build_results_rejects_unknown(tmp_path: Path) -> None:
    f = tmp_path / "ghost.xyz"
    with pytest.raises(ValueError, match="not a recognised results file"):
        build_results(f, model_path=None)


def test_build_results_mpco_requires_model(tmp_path: Path) -> None:
    f = _write_mpco(tmp_path / "run.mpco")
    with pytest.raises(ValueError, match="needs a model"):
        build_results(f, model_path=None)
