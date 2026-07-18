"""ADR 0077 P4 — ``ParallelModalResult.to_native`` viewer binding.

Writes harvested distributed-FEAST mode shapes as mode-kind stages in a
native results H5 — the exact ``DomainCapture.capture_modes`` layout —
so ``Results.from_native`` / ``r.modes`` / the stage-aware viewer
consume the distributed run with zero new viewer code. Reuses the
capture suite's ``_MockFem`` (the writer needs a real neutral zone for
the bind-contract round trip).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees.analysis.modal import ParallelModalResult

from tests.conftest import _open_model_from_h5
from tests.test_results_domain_capture import _MockFem


def _write_job(
    tmp_path: Path,
    *,
    eigenvalues: str,
    nodes: list[int],
    ndf: int,
    ndm: int | None,
    rows: list[str],
) -> Path:
    (tmp_path / "eigenvalues.out").write_text(eigenvalues)
    meta: dict = {"nodes": nodes, "ndf": ndf}
    if ndm is not None:
        meta["ndm"] = ndm
    (tmp_path / "mode_shapes.json").write_text(json.dumps(meta))
    for k, row in enumerate(rows, start=1):
        (tmp_path / f"mode_shape_{k}.out").write_text(row)
    return tmp_path


def test_to_native_round_trips_through_results_modes(tmp_path: Path) -> None:
    """ndf=6 harvest → mode-kind stages with displacement_* + rotation_*
    readable via ``Results.from_native(...).modes`` (the capture_modes
    surface)."""
    job = _write_job(
        tmp_path,
        eigenvalues="100.0 400.0",
        nodes=[1, 2],
        ndf=6,
        ndm=3,
        rows=[
            # node-major: node 1 dofs 1-6, node 2 dofs 1-6.
            "0.1 0.2 0.3 0.4 0.5 0.6  1.1 1.2 1.3 1.4 1.5 1.6\n",
            "2.1 2.2 2.3 2.4 2.5 2.6  3.1 3.2 3.3 3.4 3.5 3.6\n",
        ],
    )
    res = ParallelModalResult.from_job(str(job))
    fem = _MockFem([1, 2])
    out = tmp_path / "modes.h5"
    res.to_native(out, fem)

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=fem, model=_open_model_from_h5(out),
    ) as r:
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert len(modes) == 2
        assert modes[0].mode_index == 1
        assert modes[0].eigenvalue == pytest.approx(100.0)
        assert modes[0].frequency_hz == pytest.approx(
            np.sqrt(100.0) / (2.0 * np.pi)
        )
        assert modes[1].eigenvalue == pytest.approx(400.0)

        np.testing.assert_allclose(
            modes[0].nodes.get(component="displacement_x").values,
            [[0.1, 1.1]],
        )
        np.testing.assert_allclose(
            modes[0].nodes.get(component="rotation_z").values,
            [[0.6, 1.6]],
        )
        np.testing.assert_allclose(
            modes[1].nodes.get(component="displacement_y").values,
            [[2.2, 3.2]],
        )


def test_to_native_2d_deck_maps_in_plane_components_only(
    tmp_path: Path,
) -> None:
    """A 2-D deck (sidecar ndm=2, ndf=3) writes displacement_x/y only —
    the capture_modes convention (rotations only at ndf>=6; no
    displacement_z for ndm=2)."""
    job = _write_job(
        tmp_path,
        eigenvalues="50.0",
        nodes=[1, 2],
        ndf=3,
        ndm=2,
        rows=["0.1 0.2 0.3  1.1 1.2 1.3\n"],
    )
    res = ParallelModalResult.from_job(str(job))
    fem = _MockFem([1, 2])
    out = tmp_path / "modes.h5"
    res.to_native(out, fem)

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=fem, model=_open_model_from_h5(out),
    ) as r:
        m = r.modes[0]
        comps = m.nodes.available_components()
        assert "displacement_x" in comps
        assert "displacement_y" in comps
        assert "displacement_z" not in comps
        assert not any(c.startswith("rotation") for c in comps)
        np.testing.assert_allclose(
            m.nodes.get(component="displacement_y").values, [[0.2, 1.2]]
        )


def test_to_native_sidecar_without_ndm_defaults_to_3d(tmp_path: Path) -> None:
    """A first-rev sidecar (no "ndm" key) is a 3-D deck — all three
    displacement components come through."""
    job = _write_job(
        tmp_path,
        eigenvalues="100.0",
        nodes=[7],
        ndf=6,
        ndm=None,
        rows=["0.1 0.2 0.3 0.4 0.5 0.6\n"],
    )
    res = ParallelModalResult.from_job(str(job))
    out = tmp_path / "modes.h5"
    res.to_native(out, _MockFem([7]))

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=_MockFem([7]), model=_open_model_from_h5(out),
    ) as r:
        comps = r.modes[0].nodes.available_components()
        assert "displacement_z" in comps


def test_to_native_without_shapes_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "eigenvalues.out").write_text("100.0")
    res = ParallelModalResult.from_job(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mode_shapes.json"):
        res.to_native(tmp_path / "modes.h5", _MockFem([1]))


def test_to_native_nonpositive_eigenvalue_warns(tmp_path: Path) -> None:
    """Mirror of the capture_modes contract: a spurious mode is written
    (eigenvalue preserved) with frequency/period zeroed, loudly."""
    job = _write_job(
        tmp_path,
        eigenvalues="-1.0",
        nodes=[1],
        ndf=6,
        ndm=3,
        rows=["0.1 0.2 0.3 0.4 0.5 0.6\n"],
    )
    res = ParallelModalResult.from_job(str(job))
    fem = _MockFem([1])
    out = tmp_path / "modes.h5"
    with pytest.warns(UserWarning, match="non-positive eigenvalue"):
        res.to_native(out, fem)

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=fem, model=_open_model_from_h5(out),
    ) as r:
        m = r.modes[0]
        assert m.eigenvalue == pytest.approx(-1.0)
        assert m.frequency_hz == 0.0
        assert m.period_s == 0.0
