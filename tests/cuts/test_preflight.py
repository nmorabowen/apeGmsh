"""Unit tests for ``SectionCutDef.preflight`` / ``SectionSweepDef.preflight``
(v2.3 — drift validator).

Each test constructs a minimal inline FEM stub (richer than the
``test_builders`` stub: iterable element groups + node coordinates so
the W1 AABB check has something to look at) and a minimal ``model.h5``
that the tag map can read. We don't run any meshing here — Phases 2,
3, and 4 already cover their respective inputs.

The point of preflight is to catch drift between a *frozen, pickled
cut* and a *current* FEM, so the tests deliberately construct cuts
that disagree with the FEM in each of the documented ways.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import (
    PreflightError,
    PreflightReport,
    SectionCutDef,
    SectionSweepDef,
)
from apeGmsh.cuts._preflight import run_cut_checks
from apeGmsh.cuts._tag_map import FemToOpsTagMap

from tests.fixtures.schema import OPENSEES_CURRENT


# --------------------------------------------------------------------- #
# Fixtures: inline FEM stub + minimal model.h5
# --------------------------------------------------------------------- #
class _StubGroup:
    """ElementGroup substitute — exposes ``.ids`` and ``.connectivity``."""

    def __init__(
        self,
        ids: list[int] | np.ndarray,
        connectivity: list[list[int]] | np.ndarray,
    ) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.connectivity = np.asarray(connectivity, dtype=np.int64)


class _StubElements:
    def __init__(self, groups: list[_StubGroup]) -> None:
        self._groups = groups

    def __iter__(self):
        return iter(self._groups)

    @property
    def ids(self) -> np.ndarray:
        if not self._groups:
            return np.array([], dtype=np.int64)
        return np.concatenate([g.ids for g in self._groups])


class _StubNodes:
    def __init__(
        self,
        ids: list[int] | np.ndarray,
        coords: list[list[float]] | np.ndarray,
    ) -> None:
        ids_arr = np.asarray(ids, dtype=np.int64)
        coords_arr = np.asarray(coords, dtype=float)
        if ids_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("ids and coords must have matching length.")
        self._ids = ids_arr
        self.coords = coords_arr
        self._idx = {int(n): i for i, n in enumerate(ids_arr)}

    def index(self, nid: int) -> int:
        try:
            return self._idx[int(nid)]
        except KeyError as exc:
            raise KeyError(f"node {nid} not found") from exc


class _StubFEM:
    def __init__(
        self,
        groups: list[_StubGroup],
        node_ids: list[int] | np.ndarray,
        node_coords: list[list[float]] | np.ndarray,
    ) -> None:
        self.elements = _StubElements(groups)
        self.nodes = _StubNodes(node_ids, node_coords)


def _write_minimal_h5(
    path: Path,
    *,
    groups: dict[str, dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = OPENSEES_CURRENT
        meta.attrs["apeGmsh_version"] = "0.0.0-test"
        meta.attrs["created_iso"] = "2026-01-01T00:00:00Z"
        meta.attrs["ndm"] = 3
        meta.attrs["ndf"] = 6
        meta.attrs["snapshot_id"] = "test"
        meta.attrs["model_name"] = "test"
        em = f.create_group("opensees/element_meta")
        for type_token, arrays in groups.items():
            g = em.create_group(type_token)
            g.attrs["type"] = type_token
            g.create_dataset("ids", data=arrays["ids"].astype(np.int64))
            g.create_dataset("fem_eids", data=arrays["fem_eids"].astype(np.int64))


def _baseline_setup(tmp_path: Path) -> tuple[_StubFEM, Path]:
    """Two beam elements straddling z=5.

    Element 101 (ops 10): nodes 1→2 along z, x=0
    Element 102 (ops 11): nodes 3→4 along z, x=1
    All beams run from z=0 to z=10; cut plane at z=5 with +z normal
    therefore straddles every node.
    """
    fem = _StubFEM(
        groups=[
            _StubGroup(
                ids=[101, 102],
                connectivity=[[1, 2], [3, 4]],
            ),
        ],
        node_ids=[1, 2, 3, 4],
        node_coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 10.0],
        ],
    )
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
    })
    return fem, h5


def _baseline_cut(label: str = "story 1") -> SectionCutDef:
    return SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
        label=label,
    )


# --------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------- #
def test_clean_cut_no_issues(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    report = _baseline_cut().preflight(fem, model_h5=h5)
    assert isinstance(report, PreflightReport)
    assert report.ok
    assert report.errors == ()
    assert report.warnings == ()
    assert "ok" in str(report)


def test_clean_cut_with_bounding_polygon(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
        bounding_polygon=(
            (-1.0, -1.0, 5.0),
            (2.0, -1.0, 5.0),
            (2.0, 1.0, 5.0),
            (-1.0, 1.0, 5.0),
        ),
        label="bounded",
    )
    report = cut.preflight(fem, model_h5=h5)
    assert report.ok


# --------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------- #
def test_e1_missing_ops_tag(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 999),  # 999 not in tag map
    )
    report = cut.preflight(fem, model_h5=h5)
    codes = [i.code for i in report.errors]
    assert "E1" in codes
    e1 = next(i for i in report.errors if i.code == "E1")
    assert e1.detail is not None
    assert 999 in e1.detail["missing_ops_tags"]
    # Cut still has surviving fem_eids (10, 11 resolve), so no E4.
    assert "E4" not in codes


def test_e2_dropped_fem_eid(tmp_path: Path) -> None:
    """Tag map references a fem_eid the current FEM doesn't have."""
    fem, _ = _baseline_setup(tmp_path)
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11, 12]),
            "fem_eids": np.array([101, 102, 999]),  # 999 not in fem
        },
    })
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 12),
    )
    report = cut.preflight(fem, model_h5=h5)
    codes = [i.code for i in report.errors]
    assert "E2" in codes
    e2 = next(i for i in report.errors if i.code == "E2")
    assert e2.detail is not None
    assert (12, 999) in e2.detail["missing"]
    # Surviving (10→101, 11→102) means no E4.
    assert "E4" not in codes


def test_e3_polygon_off_plane(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
        bounding_polygon=(
            (-1.0, -1.0, 5.0),
            (2.0, -1.0, 5.0),
            (2.0, 1.0, 5.001),  # off-plane by 0.001
        ),
    )
    report = cut.preflight(fem, model_h5=h5, tol=1e-6)
    codes = [i.code for i in report.errors]
    assert "E3" in codes
    e3 = next(i for i in report.errors if i.code == "E3")
    assert e3.detail is not None
    assert e3.detail["n_off_plane"] == 1
    # Looser tolerance accepts the same polygon.
    relaxed = cut.preflight(fem, model_h5=h5, tol=1e-2)
    assert "E3" not in [i.code for i in relaxed.errors]


def test_e4_all_filter_elements_dropped(tmp_path: Path) -> None:
    """All ops tags map to fem_eids that no longer exist in fem.elements."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
    })
    # FEM no longer contains 101 or 102 — different element ids.
    fem = _StubFEM(
        groups=[_StubGroup(ids=[201, 202], connectivity=[[1, 2], [3, 4]])],
        node_ids=[1, 2, 3, 4],
        node_coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 10.0],
        ],
    )
    report = _baseline_cut().preflight(fem, model_h5=h5)
    codes = [i.code for i in report.errors]
    assert "E2" in codes
    assert "E4" in codes
    assert not report.ok


# --------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------- #
def test_w1_filter_nodes_all_one_side(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    # Plane at z=20 — all beam nodes (z ∈ [0, 10]) lie below it.
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 20.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
    )
    report = cut.preflight(fem, model_h5=h5)
    codes = [i.code for i in report.warnings]
    assert "W1" in codes
    w1 = next(i for i in report.warnings if i.code == "W1")
    assert w1.detail is not None
    assert w1.detail["side"] == "negative"   # all nodes on negative side
    # No errors — W1 doesn't block.
    assert report.ok


def test_w1_skipped_when_filter_straddles(tmp_path: Path) -> None:
    """The clean baseline has filter spanning z=0 to z=10 across z=5 plane."""
    fem, h5 = _baseline_setup(tmp_path)
    report = _baseline_cut().preflight(fem, model_h5=h5)
    assert all(i.code != "W1" for i in report.issues)


# --------------------------------------------------------------------- #
# Input validation / fem-only mode
# --------------------------------------------------------------------- #
def test_both_inputs_raises(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    tag_map = FemToOpsTagMap.from_h5(h5)
    with pytest.raises(ValueError, match="not both"):
        _baseline_cut().preflight(fem, model_h5=h5, tag_map=tag_map)


def test_cached_tag_map_matches_model_h5(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    via_h5 = _baseline_cut().preflight(fem, model_h5=h5)
    tag_map = FemToOpsTagMap.from_h5(h5)
    via_map = _baseline_cut().preflight(fem, tag_map=tag_map)
    assert via_h5 == via_map


def test_fem_only_skips_ops_checks(tmp_path: Path) -> None:
    """Without model_h5 or tag_map, only E3 (polygon-on-plane) can fire."""
    fem, _ = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 999),  # would trigger E1 — but no tag map here
        bounding_polygon=(
            (-1.0, -1.0, 5.0),
            (2.0, -1.0, 5.0),
            (2.0, 1.0, 5.5),       # off-plane → E3
        ),
    )
    report = cut.preflight(fem)
    codes = [i.code for i in report.issues]
    assert "E3" in codes
    assert "E1" not in codes
    assert "E2" not in codes
    assert "E4" not in codes
    assert "W1" not in codes      # AABB scan requires resolved filter


# --------------------------------------------------------------------- #
# Report behaviour
# --------------------------------------------------------------------- #
def test_raise_for_errors_on_error(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 999),
        label="bad",
    )
    report = cut.preflight(fem, model_h5=h5)
    with pytest.raises(PreflightError, match=r"\[E1\]"):
        report.raise_for_errors()


def test_raise_for_errors_silent_on_warning_only(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 20.0),     # W1 trigger
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
    )
    report = cut.preflight(fem, model_h5=h5)
    assert not report.ok or len(report.warnings) > 0
    assert report.ok                       # warnings don't block
    report.raise_for_errors()              # no exception


def test_report_str_includes_label_and_codes(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 999),
        label="story-3",
    )
    s = str(cut.preflight(fem, model_h5=h5))
    assert "story-3" in s
    assert "E1" in s
    assert "ERROR" in s


# --------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------- #
def test_sweep_preflight_one_report_per_cut(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    # Three cuts: clean, polygon-drift, and one with a bogus tag.
    clean = _baseline_cut("a")
    drift = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11),
        bounding_polygon=(
            (-1.0, -1.0, 5.0),
            (2.0, -1.0, 5.0),
            (2.0, 1.0, 5.5),               # off-plane
        ),
        label="b",
    )
    bogus = SectionCutDef(
        plane_point=(0.0, 0.0, 5.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 11, 999),
        label="c",
    )
    sweep = SectionSweepDef(cuts=(clean, drift, bogus))

    reports = sweep.preflight(fem, model_h5=h5)
    assert isinstance(reports, tuple)
    assert len(reports) == 3
    assert reports[0].ok
    assert "E3" in [i.code for i in reports[1].errors]
    assert "E1" in [i.code for i in reports[2].errors]
    # Labels propagate.
    assert reports[0].cut_label == "a"
    assert reports[1].cut_label == "b"
    assert reports[2].cut_label == "c"


def test_sweep_preflight_both_inputs_raises(tmp_path: Path) -> None:
    fem, h5 = _baseline_setup(tmp_path)
    tag_map = FemToOpsTagMap.from_h5(h5)
    sweep = SectionSweepDef(cuts=(_baseline_cut(),))
    with pytest.raises(ValueError, match="not both"):
        sweep.preflight(fem, model_h5=h5, tag_map=tag_map)


# --------------------------------------------------------------------- #
# Internal entry point
# --------------------------------------------------------------------- #
def test_run_cut_checks_direct_call(tmp_path: Path) -> None:
    """``run_cut_checks`` is what the methods dispatch through; smoke test."""
    fem, h5 = _baseline_setup(tmp_path)
    report = run_cut_checks(_baseline_cut(), fem, model_h5=h5)
    assert report.ok
