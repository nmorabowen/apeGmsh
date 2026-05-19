"""Unit tests for ``DriftDef`` / ``DriftSweepDef`` (v5 — node-pair drift).

The carriers don't need a tag map or model.h5 — they reference FEM
node ids directly. Tests build a minimal inline FEM stub exposing
the two surfaces ``DriftDef`` reaches into:

  * ``fem.nodes.get_ids(pg=...)`` — for the ``from_pgs`` builder
  * ``fem.nodes.index(nid)`` + ``fem.nodes.coords`` — for preflight
    coordinate checks and ``DriftSweepDef.elevations(...)``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.cuts import (
    DriftDef,
    DriftSweepDef,
    PreflightError,
    PreflightReport,
)


# --------------------------------------------------------------------- #
# Fixtures: inline FEM stub
# --------------------------------------------------------------------- #
class _SelResult:
    """selection-unification v2 P3-R: the ``fem.nodes.select(...)``
    terminal — exposes ``.ids`` / ``.coords`` (the only surface PROD
    reads after the ``get_ids``→``select`` migration)."""

    def __init__(self, *, ids, _idx, _coords) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.__idx = _idx
        self.__all_coords = _coords

    @property
    def coords(self) -> np.ndarray:
        if self.ids.size == 0:
            return np.empty((0, 3), dtype=float)
        rows = [self.__idx[int(n)] for n in self.ids]
        return np.asarray(self.__all_coords)[rows]


class _StubNodes:
    def __init__(
        self,
        ids: list[int] | np.ndarray,
        coords: list[list[float]] | np.ndarray,
        ids_by_pg: dict[str, np.ndarray] | None = None,
    ) -> None:
        ids_arr = np.asarray(ids, dtype=np.int64)
        coords_arr = np.asarray(coords, dtype=float)
        if ids_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("ids and coords must have matching length.")
        self._ids = ids_arr
        self.coords = coords_arr
        self._idx = {int(n): i for i, n in enumerate(ids_arr)}
        self._ids_by_pg = ids_by_pg or {}

    def get_ids(self, *, pg: str) -> np.ndarray:
        return self._ids_by_pg.get(pg, np.array([], dtype=np.int64))

    def select(self, target=None, *, pg: str | None = None, **_kw):
        """selection-unification v2 P3-R: ``fem.nodes.get_ids(pg=)`` is
        removed; ``fem.nodes.select(pg=).ids`` is the migration target
        (P-NODE).  Mirrors the broker — same ids as the (removed)
        ``get_ids`` body, exposed via the ``.ids`` terminal."""
        ids = self._ids_by_pg.get(pg, np.array([], dtype=np.int64))
        return _SelResult(ids=ids, _idx=self._idx, _coords=self.coords)

    def index(self, nid: int) -> int:
        try:
            return self._idx[int(nid)]
        except KeyError as exc:
            raise KeyError(f"node {nid} not found") from exc


class _StubFEM:
    def __init__(
        self,
        node_ids: list[int] | np.ndarray,
        node_coords: list[list[float]] | np.ndarray,
        ids_by_pg: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.nodes = _StubNodes(node_ids, node_coords, ids_by_pg)


def _three_floor_fem() -> _StubFEM:
    """Three floor reference nodes at z=0, 3, 6 m."""
    return _StubFEM(
        node_ids=[10, 20, 30],
        node_coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 6.0],
        ],
        ids_by_pg={
            "floor-0-CM": np.array([10]),
            "floor-1-CM": np.array([20]),
            "floor-2-CM": np.array([30]),
            "floor-0-corners": np.array([10, 11]),       # multi-node
            "floor-missing": np.array([], dtype=np.int64),
        },
    )


# --------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------- #
def test_construct_basic() -> None:
    d = DriftDef(top_node=20, bottom_node=10)
    assert d.top_node == 20
    assert d.bottom_node == 10
    assert d.direction is None
    assert d.story_height is None
    assert d.label is None


def test_construct_with_direction_normalizes() -> None:
    d = DriftDef(top_node=20, bottom_node=10, direction=(3.0, 0.0, 0.0))
    assert d.direction is not None
    np.testing.assert_allclose(d.direction, (1.0, 0.0, 0.0), atol=1e-12)


def test_construct_coerces_node_ids() -> None:
    d = DriftDef(top_node=np.int64(20), bottom_node=np.int64(10))  # type: ignore[arg-type]
    assert isinstance(d.top_node, int)
    assert isinstance(d.bottom_node, int)


def test_same_top_and_bottom_raises() -> None:
    with pytest.raises(ValueError, match="must differ"):
        DriftDef(top_node=10, bottom_node=10)


def test_zero_direction_raises() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        DriftDef(top_node=20, bottom_node=10, direction=(0.0, 0.0, 0.0))


def test_negative_story_height_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        DriftDef(top_node=20, bottom_node=10, story_height=-1.0)


def test_zero_story_height_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        DriftDef(top_node=20, bottom_node=10, story_height=0.0)


# --------------------------------------------------------------------- #
# from_pgs
# --------------------------------------------------------------------- #
def test_from_pgs_happy_path() -> None:
    fem = _three_floor_fem()
    d = DriftDef.from_pgs(
        top_pg="floor-2-CM",
        bottom_pg="floor-1-CM",
        fem=fem,                    # type: ignore[arg-type]
        direction=(1.0, 0.0, 0.0),
    )
    assert d.top_node == 30
    assert d.bottom_node == 20
    assert d.label == "drift top=floor-2-CM, bottom=floor-1-CM"


def test_from_pgs_custom_label_overrides_auto() -> None:
    fem = _three_floor_fem()
    d = DriftDef.from_pgs(
        top_pg="floor-2-CM",
        bottom_pg="floor-1-CM",
        fem=fem,                    # type: ignore[arg-type]
        label="story 2 drift X",
    )
    assert d.label == "story 2 drift X"


def test_from_pgs_empty_pg_raises() -> None:
    fem = _three_floor_fem()
    with pytest.raises(ValueError, match="zero nodes"):
        DriftDef.from_pgs(
            top_pg="floor-missing",
            bottom_pg="floor-1-CM",
            fem=fem,                # type: ignore[arg-type]
        )


def test_from_pgs_multi_node_pg_raises() -> None:
    fem = _three_floor_fem()
    with pytest.raises(ValueError, match="expected exactly 1"):
        DriftDef.from_pgs(
            top_pg="floor-0-corners",     # 2 nodes
            bottom_pg="floor-1-CM",
            fem=fem,                # type: ignore[arg-type]
        )


# --------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------- #
def test_preflight_clean() -> None:
    fem = _three_floor_fem()
    d = DriftDef(top_node=30, bottom_node=20, label="story 2")
    report = d.preflight(fem)       # type: ignore[arg-type]
    assert isinstance(report, PreflightReport)
    assert report.ok
    assert report.errors == ()
    assert report.warnings == ()
    assert report.cut_label == "story 2"


def test_preflight_top_node_missing_e1() -> None:
    fem = _three_floor_fem()
    d = DriftDef(top_node=999, bottom_node=20)
    report = d.preflight(fem)       # type: ignore[arg-type]
    codes = [i.code for i in report.errors]
    assert "D-E1" in codes
    assert "D-E2" not in codes


def test_preflight_bottom_node_missing_e2() -> None:
    fem = _three_floor_fem()
    d = DriftDef(top_node=20, bottom_node=888)
    report = d.preflight(fem)       # type: ignore[arg-type]
    codes = [i.code for i in report.errors]
    assert "D-E2" in codes
    assert "D-E1" not in codes


def test_preflight_both_missing() -> None:
    fem = _three_floor_fem()
    d = DriftDef(top_node=999, bottom_node=888)
    report = d.preflight(fem)       # type: ignore[arg-type]
    codes = [i.code for i in report.errors]
    assert "D-E1" in codes
    assert "D-E2" in codes
    # Coincidence check skipped because neither node resolved.
    assert "D-W1" not in [i.code for i in report.warnings]


def test_preflight_coincident_nodes_w1() -> None:
    """Two distinct node IDs at the same coordinate trigger D-W1."""
    fem = _StubFEM(
        node_ids=[10, 11],
        node_coords=[
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],        # identical to node 10
        ],
    )
    d = DriftDef(top_node=10, bottom_node=11)
    report = d.preflight(fem)       # type: ignore[arg-type]
    codes = [i.code for i in report.warnings]
    assert "D-W1" in codes
    w1 = next(i for i in report.warnings if i.code == "D-W1")
    assert w1.detail is not None
    assert w1.detail["distance"] < 1e-6
    assert report.ok                # warning doesn't block


def test_preflight_raise_for_errors() -> None:
    fem = _three_floor_fem()
    d = DriftDef(top_node=999, bottom_node=20, label="bad")
    report = d.preflight(fem)       # type: ignore[arg-type]
    with pytest.raises(PreflightError, match=r"\[D-E1\]"):
        report.raise_for_errors()


# --------------------------------------------------------------------- #
# Pickle round-trip
# --------------------------------------------------------------------- #
def test_pickle_round_trip(tmp_path: Path) -> None:
    d = DriftDef(
        top_node=30,
        bottom_node=20,
        direction=(1.0, 0.0, 0.0),
        story_height=3000.0,
        label="story 2 drift X",
    )
    pkl = tmp_path / "drift.pkl"
    d.save_pickle(pkl)
    restored = DriftDef.load_pickle(pkl)
    assert restored == d


def test_pickle_gzip(tmp_path: Path) -> None:
    d = DriftDef(top_node=20, bottom_node=10)
    pkl = tmp_path / "drift.pkl.gz"
    d.save_pickle(pkl)
    restored = DriftDef.load_pickle(pkl)
    assert restored == d


def test_load_pickle_wrong_type_raises(tmp_path: Path) -> None:
    import pickle
    p = tmp_path / "wrong.pkl"
    p.write_bytes(pickle.dumps({"hello": "world"}))
    with pytest.raises(TypeError, match="expected DriftDef"):
        DriftDef.load_pickle(p)


# --------------------------------------------------------------------- #
# DriftSweepDef
# --------------------------------------------------------------------- #
def test_sweep_from_pg_pairs() -> None:
    fem = _three_floor_fem()
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[
            ("floor-1-CM", "floor-0-CM"),
            ("floor-2-CM", "floor-1-CM"),
        ],
        fem=fem,                    # type: ignore[arg-type]
        direction=(1.0, 0.0, 0.0),
    )
    assert len(sweep) == 2
    assert sweep[0].top_node == 20
    assert sweep[0].bottom_node == 10
    assert sweep[1].top_node == 30
    assert sweep[1].bottom_node == 20


def test_sweep_container_protocol() -> None:
    fem = _three_floor_fem()
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[("floor-1-CM", "floor-0-CM"),
                  ("floor-2-CM", "floor-1-CM")],
        fem=fem,                    # type: ignore[arg-type]
    )
    assert sweep.n_drifts == 2
    assert not sweep.is_empty
    assert list(iter(sweep)) == [sweep[0], sweep[1]]


def test_empty_sweep_is_empty() -> None:
    sweep = DriftSweepDef(drifts=())
    assert sweep.n_drifts == 0
    assert sweep.is_empty
    assert list(iter(sweep)) == []


def test_sweep_preflight_one_report_per_drift() -> None:
    fem = _three_floor_fem()
    good = DriftDef(top_node=20, bottom_node=10)
    bad = DriftDef(top_node=999, bottom_node=10, label="bad")
    sweep = DriftSweepDef(drifts=(good, bad))
    reports = sweep.preflight(fem)  # type: ignore[arg-type]
    assert isinstance(reports, tuple)
    assert len(reports) == 2
    assert reports[0].ok
    assert "D-E1" in [i.code for i in reports[1].errors]


def test_sweep_elevations() -> None:
    fem = _three_floor_fem()
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[("floor-1-CM", "floor-0-CM"),
                  ("floor-2-CM", "floor-1-CM")],
        fem=fem,                    # type: ignore[arg-type]
    )
    elevs = sweep.elevations(fem=fem)  # type: ignore[arg-type]
    np.testing.assert_allclose(elevs, [3.0, 6.0])


def test_sweep_elevations_axis_x() -> None:
    fem = _StubFEM(
        node_ids=[10, 20, 30],
        node_coords=[
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        ids_by_pg={
            "a": np.array([10]),
            "b": np.array([20]),
            "c": np.array([30]),
        },
    )
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[("b", "a"), ("c", "b")],
        fem=fem,                    # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        sweep.elevations(axis="x", fem=fem),  # type: ignore[arg-type]
        [5.0, 10.0],
    )


def test_sweep_elevations_invalid_axis() -> None:
    fem = _three_floor_fem()
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[("floor-1-CM", "floor-0-CM")],
        fem=fem,                    # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must be 'x', 'y', or 'z'"):
        sweep.elevations(axis="w", fem=fem)  # type: ignore[arg-type]


def test_sweep_pickle_round_trip(tmp_path: Path) -> None:
    fem = _three_floor_fem()
    sweep = DriftSweepDef.from_pg_pairs(
        pg_pairs=[("floor-1-CM", "floor-0-CM"),
                  ("floor-2-CM", "floor-1-CM")],
        fem=fem,                    # type: ignore[arg-type]
        direction=(1.0, 0.0, 0.0),
    )
    pkl = tmp_path / "sweep.pkl"
    sweep.save_pickle(pkl)
    restored = DriftSweepDef.load_pickle(pkl)
    assert restored == sweep
