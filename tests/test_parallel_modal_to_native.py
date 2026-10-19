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
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.opensees.analysis.modal import (
    ParallelModalResult,
    RotationRolesUnavailableWarning,
)

from tests.conftest import _open_model_from_h5
from tests.test_results_domain_capture import _MockFem


class _Cells(list):  # type: ignore[type-arg]
    """An empty element list (what ``compute_snapshot_id`` iterates) that
    also answers ``.types`` with gmsh types of the given ``dim``s."""

    def __init__(self, dims: "tuple[int, ...]") -> None:
        super().__init__()
        self.types = [SimpleNamespace(dim=d) for d in dims]


class _NdfFem(_MockFem):
    """``_MockFem`` that carries a per-node ndf (``fem.nodes.ndf_for``):
    one value for every node, or a ``{node: ndf}`` table; a node outside
    the table raises ``LookupError``, as ``NodeComposite.ndf_for`` does
    for an undeclared ndf. ``solid`` says whether its gmsh element types
    include a 3-D cell; it has no ``model_name``, so no conventional
    archive path (#2203)."""

    def __init__(
        self, node_ids: list[int], ndf: "int | dict[int, int]",
        *, solid: bool = True,
    ) -> None:
        super().__init__(node_ids)
        self.model_name = None
        self.elements = _Cells((1, 3) if solid else (1,))
        table = (dict(ndf) if isinstance(ndf, dict)
                 else {int(n): int(ndf) for n in node_ids})

        def ndf_for(nid: int) -> int:
            if int(nid) not in table:
                raise LookupError(f"node {nid}: ndf not present")
            return table[int(nid)]

        self.nodes.ndf_for = ndf_for


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
    surface). Every node's own ndf is 6 in 3-D (``fem`` says so), so
    DOFs 4-6 are rotations without ``model_h5`` (#2203)."""
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
    fem = _NdfFem([1, 2], 6)
    out = tmp_path / "modes.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RotationRolesUnavailableWarning)
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
    """A 2-D deck (sidecar ndm=2, ndf=3) without ``model_h5`` writes
    displacement_x/y only: a node of ndf 3 in 2-D may be a frame's
    (ux, uy, rz) or a u-p node's (ux, uy, p), so its third column is no
    rotation_z, and one named warning says to pass ``model_h5`` (#2203).
    No displacement_z for ndm=2."""
    job = _write_job(
        tmp_path,
        eigenvalues="50.0",
        nodes=[1, 2],
        ndf=3,
        ndm=2,
        rows=["0.1 0.2 0.3  1.1 1.2 1.3\n"],
    )
    res = ParallelModalResult.from_job(str(job))
    fem = _NdfFem([1, 2], 3)
    out = tmp_path / "modes.h5"
    with pytest.warns(RotationRolesUnavailableWarning, match="model_h5="):
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
    res.to_native(out, _NdfFem([7], 6))

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=_NdfFem([7], 6), model=_open_model_from_h5(out),
    ) as r:
        comps = r.modes[0].nodes.available_components()
        assert "displacement_z" in comps


def test_to_native_without_shapes_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "eigenvalues.out").write_text("100.0")
    res = ParallelModalResult.from_job(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mode_shapes.json"):
        res.to_native(tmp_path / "modes.h5", _MockFem([1]))


def test_to_native_round_trips_a_MERGED_partitioned_harvest(
    tmp_path: Path,
) -> None:
    """ADR 0077 Tier 1B — the viewer binding consumes the **partitioned**
    (ARPACK) harvest too, with no viewer-side change.

    The merged field is assembled from per-rank sidecars rather than read
    whole from one, so this exercises the path `to_native` actually gets
    on a Tier-1B run: node tags from ``np.unique`` over every rank rather
    than straight off one sidecar, and rows stitched across ranks.
    Node 2 is the shared boundary — it must appear ONCE in the H5, with
    the agreed values, not twice.
    """
    (tmp_path / "eigenvalues.out").write_text("100.0 400.0")
    per_rank = {
        0: ([1, 2], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]),
        1: ([2, 3], [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                     [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]]),
    }
    for rank, (nodes, rows) in per_rank.items():
        (tmp_path / f"mode_shapes_rank{rank}.json").write_text(
            json.dumps({"nodes": nodes, "ndf": 6, "ndm": 3})
        )
        for k in (1, 2):
            (tmp_path / f"mode_shape_{k}_rank{rank}.out").write_text(
                " ".join(str(round(v * k, 6)) for row in rows for v in row)
            )

    res = ParallelModalResult.from_job(str(tmp_path))
    np.testing.assert_array_equal(res.shape_nodes, [1, 2, 3])

    fem = _NdfFem([1, 2, 3], 6)
    out = tmp_path / "modes.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RotationRolesUnavailableWarning)
        res.to_native(out, fem)

    from apeGmsh.results import Results
    with Results.from_native(
        out, fem=fem, model=_open_model_from_h5(out),
    ) as r:
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert len(modes) == 2
        assert modes[0].eigenvalue == pytest.approx(100.0)
        # One row per node — the shared node 2 merged, not duplicated.
        np.testing.assert_allclose(
            modes[0].nodes.get(component="displacement_x").values,
            [[0.1, 1.1, 2.1]],
        )
        np.testing.assert_allclose(
            modes[1].nodes.get(component="rotation_z").values,
            [[1.2, 3.2, 5.2]],
        )


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
    fem = _NdfFem([1], 6)
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
