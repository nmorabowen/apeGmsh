"""Phase-5 unit tests for :class:`SectionSweepDef`.

Sweeps are pure composition over :class:`SectionCutDef`, so most of
the validation lives in Phase 4. These tests cover:

* Construction via each factory (``from_planes``,
  ``from_horizontal_grid``, ``from_pg_pattern``).
* Container protocol (``len``, indexing, iteration).
* ``plane_locators`` axis inference + explicit axis.
* Pickle round-trip.
* Label-prefix propagation.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef, SectionSweepDef, plane_horizontal


# --------------------------------------------------------------------- #
# Stubs + h5 helper — same shape as test_builders.py
# --------------------------------------------------------------------- #
class _StubPhysicalGroupSet:
    def __init__(self, names_by_dim: dict[int, list[str]] | None = None) -> None:
        self._names_by_dim = names_by_dim or {}

    def names(self, dim: int = -1) -> list[str]:
        if dim == -1:
            out: list[str] = []
            for d_names in self._names_by_dim.values():
                out.extend(d_names)
            return sorted(out)
        return list(self._names_by_dim.get(dim, []))


class _SelResult:
    """selection-unification v2 P3-R: the ``fem.<...>.select(...)``
    terminal — exposes ``.ids`` / ``.coords`` (the only surface the
    cuts PROD reads after the ``get_ids``/``get_coords``→``select``
    migration)."""

    def __init__(self, *, ids=None, coords=None) -> None:
        self._ids = ids
        self._coords = coords

    @property
    def ids(self):
        return self._ids

    @property
    def coords(self):
        return self._coords


class _StubNodes:
    def __init__(
        self,
        coords_by_pg: dict[str, np.ndarray],
        *,
        pg_names_by_dim: dict[int, list[str]] | None = None,
    ) -> None:
        self._coords_by_pg = coords_by_pg
        self.physical = _StubPhysicalGroupSet(pg_names_by_dim)

    def get_coords(self, *, pg: str) -> np.ndarray:
        return self._coords_by_pg[pg]

    def select(self, target=None, *, pg: str | None = None, **_kw):
        """selection-unification v2 P3-R: ``fem.nodes.get_coords(pg=)``
        is removed; ``fem.nodes.select(pg=).coords`` is the migration
        target (P-COORD).  Mirrors the broker — same coords as the
        (removed) ``get_coords`` body."""
        return _SelResult(coords=self._coords_by_pg[pg])


class _StubElements:
    def __init__(self, ids_by_pg: dict[str, np.ndarray]) -> None:
        self._ids_by_pg = ids_by_pg

    def get_ids(self, *, pg: str) -> np.ndarray:
        return self._ids_by_pg.get(pg, np.array([], dtype=np.int64))

    def select(self, target=None, *, pg: str | None = None, **_kw):
        """selection-unification v2 P3-R: ``fem.elements.get_ids(pg=)``
        is removed; ``fem.elements.select(pg=).ids`` is the migration
        target (P-ELEM-IDS).  Mirrors the broker — same ids as the
        (removed) ``get_ids`` body, exposed via the ``.ids`` terminal."""
        return _SelResult(
            ids=self._ids_by_pg.get(pg, np.array([], dtype=np.int64))
        )


class _StubFEM:
    def __init__(
        self,
        *,
        node_coords_by_pg: dict[str, np.ndarray] | None = None,
        element_ids_by_pg: dict[str, np.ndarray] | None = None,
        pg_names_by_dim: dict[int, list[str]] | None = None,
    ) -> None:
        self.nodes = _StubNodes(
            node_coords_by_pg or {},
            pg_names_by_dim=pg_names_by_dim,
        )
        self.elements = _StubElements(element_ids_by_pg or {})


def _write_minimal_h5(
    path: Path,
    *,
    groups: dict[str, dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "2.2.0"
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


def _fixture_h5_and_fem(tmp_path: Path) -> tuple[Path, _StubFEM]:
    """3-storey-like fixture: one element PG, no plane PGs."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11, 12]),
            "fem_eids": np.array([101, 102, 103]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={
        "cols": np.array([101, 102, 103]),
    })
    return h5, fem


# --------------------------------------------------------------------- #
# from_planes
# --------------------------------------------------------------------- #
def test_from_planes_happy_path(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    planes = [
        ((0.0, 0.0, 1000.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 2000.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 3000.0), (0.0, 0.0, 1.0)),
    ]

    sweep = SectionSweepDef.from_planes(
        planes=planes,
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )

    assert isinstance(sweep, SectionSweepDef)
    assert len(sweep) == 3
    assert sweep.n_planes == 3
    assert not sweep.is_empty
    for i, cut in enumerate(sweep):
        assert isinstance(cut, SectionCutDef)
        assert cut.element_ids == (10, 11, 12)
        assert cut.plane_point == (0.0, 0.0, 1000.0 * (i + 1))


def test_from_planes_empty(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert sweep.is_empty
    assert len(sweep) == 0
    assert list(sweep) == []


def test_from_planes_label_prefix(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[
            ((0.0, 0.0, 1000.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 2000.0), (0.0, 0.0, 1.0)),
        ],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        label_prefix="Story shear",
    )
    assert sweep[0].label == "Story shear[0]"
    assert sweep[1].label == "Story shear[1]"


def test_from_planes_propagates_side(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        side="negative",
    )
    assert sweep[0].side == "negative"


# --------------------------------------------------------------------- #
# from_horizontal_grid
# --------------------------------------------------------------------- #
def test_from_horizontal_grid(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[500.0, 1500.0, 2500.0, 3500.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert len(sweep) == 4
    np.testing.assert_allclose(
        [cut.plane_point[2] for cut in sweep],
        [500.0, 1500.0, 2500.0, 3500.0],
    )
    # Every cut should have a +z normal.
    for cut in sweep:
        assert cut.plane_normal == (0.0, 0.0, 1.0)


def test_from_horizontal_grid_accepts_numpy_array(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=np.linspace(0.0, 3000.0, 4),
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert len(sweep) == 4
    np.testing.assert_allclose(
        [cut.plane_point[2] for cut in sweep],
        [0.0, 1000.0, 2000.0, 3000.0],
    )


# --------------------------------------------------------------------- #
# from_pg_pattern
# --------------------------------------------------------------------- #
def test_from_pg_pattern(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "dia-1": np.array([
                [0.0, 0.0, 1000.0], [1.0, 0.0, 1000.0], [0.0, 1.0, 1000.0],
            ]),
            "dia-2": np.array([
                [0.0, 0.0, 2000.0], [1.0, 0.0, 2000.0], [0.0, 1.0, 2000.0],
            ]),
            "dia-3": np.array([
                [0.0, 0.0, 3000.0], [1.0, 0.0, 3000.0], [0.0, 1.0, 3000.0],
            ]),
        },
        element_ids_by_pg={"walls": np.array([201, 202, 203])},
    )

    sweep = SectionSweepDef.from_pg_pattern(
        plane_pgs=["dia-1", "dia-2", "dia-3"],
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        normal_hint=(0.0, 0.0, 1.0),
    )
    assert len(sweep) == 3
    # Centroid of each PG is at the named elevation.
    expected_z = [1000.0, 2000.0, 3000.0]
    for cut, z in zip(sweep, expected_z):
        np.testing.assert_allclose(cut.plane_point[2], z, atol=1e-9)
        np.testing.assert_allclose(cut.plane_normal, (0.0, 0.0, 1.0), atol=1e-9)
    # Labels preserve the originating PG names.
    assert sweep[0].label == "plane=dia-1, elements=walls"
    assert sweep[2].label == "plane=dia-3, elements=walls"


# --------------------------------------------------------------------- #
# plane_locators
# --------------------------------------------------------------------- #
def test_plane_locators_infers_z(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[100.0, 200.0, 300.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    np.testing.assert_allclose(sweep.plane_locators(), [100.0, 200.0, 300.0])
    np.testing.assert_allclose(sweep.plane_locators("z"), [100.0, 200.0, 300.0])


def test_plane_locators_explicit_axis(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    # Vertical sweep at x = 10, 20, 30
    planes = [
        ((10.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((20.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((30.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]
    sweep = SectionSweepDef.from_planes(
        planes=planes,
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    np.testing.assert_allclose(sweep.plane_locators("x"), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(sweep.plane_locators(), [10.0, 20.0, 30.0])


def test_plane_locators_oblique_raises(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    n = (1.0, 1.0, 1.0)
    sweep = SectionSweepDef.from_planes(
        planes=[((0.0, 0.0, 0.0), n)],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    with pytest.raises(ValueError, match="Cannot infer axis"):
        sweep.plane_locators()


def test_plane_locators_invalid_axis_raises(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[0.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    with pytest.raises(ValueError, match="axis must be"):
        sweep.plane_locators("w")


def test_plane_locators_empty_sweep() -> None:
    sweep = SectionSweepDef(cuts=())
    locators = sweep.plane_locators("z")
    assert locators.shape == (0,)


# --------------------------------------------------------------------- #
# Container protocol
# --------------------------------------------------------------------- #
def test_indexing_and_iteration(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[10.0, 20.0, 30.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert sweep[0].plane_point[2] == 10.0
    assert sweep[-1].plane_point[2] == 30.0
    collected = [cut.plane_point[2] for cut in sweep]
    assert collected == [10.0, 20.0, 30.0]


def test_repr_smoke(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[1.0, 2.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    s = repr(sweep)
    assert "SectionSweepDef" in s
    assert "n_planes=2" in s


# --------------------------------------------------------------------- #
# Pickle round-trip
# --------------------------------------------------------------------- #
def test_pickle_roundtrip(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[
            ((0.0, 0.0, 100.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 200.0), (0.0, 0.0, 1.0)),
        ],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        label_prefix="dia",
    )
    p = sweep.save_pickle(tmp_path / "sweep.pkl")
    restored = SectionSweepDef.load_pickle(p)
    assert restored == sweep
    assert len(restored) == 2
    assert restored[0].label == "dia[0]"


def test_pickle_gzip(tmp_path: Path) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_horizontal_grid(
        elevations=[1.0],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    p = sweep.save_pickle(tmp_path / "sweep.pkl.gz")
    assert p.exists()
    restored = SectionSweepDef.load_pickle(p)
    assert restored == sweep


# --------------------------------------------------------------------- #
# Smoke: to_specs raises cleanly when STKO is missing
# --------------------------------------------------------------------- #
def test_to_specs_propagates_stko_import_error(tmp_path: Path, monkeypatch) -> None:
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("STKO_to_python"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="STKO_to_python"):
        sweep.to_specs()


# --------------------------------------------------------------------- #
# Sanity: SectionCutDef.from_plane_horizontal helper chains
# --------------------------------------------------------------------- #
def test_helper_chain_with_plane_horizontal(tmp_path: Path) -> None:
    """``from_planes`` accepts the output of ``plane_horizontal``
    directly, no shape gymnastics."""
    h5, fem = _fixture_h5_and_fem(tmp_path)
    sweep = SectionSweepDef.from_planes(
        planes=[plane_horizontal(z) for z in (1.0, 2.0, 3.0)],
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert len(sweep) == 3
    assert sweep[2].plane_point == (0.0, 0.0, 3.0)


# --------------------------------------------------------------------- #
# from_pg_glob
# --------------------------------------------------------------------- #
def _glob_fixture(tmp_path: Path) -> tuple[Path, _StubFEM]:
    """Three diaphragms at z=1000/2000/3000 + a few non-matching PGs."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "diaphragm-1": np.array([
                [0.0, 0.0, 1000.0], [1.0, 0.0, 1000.0], [0.0, 1.0, 1000.0],
            ]),
            "diaphragm-2": np.array([
                [0.0, 0.0, 2000.0], [1.0, 0.0, 2000.0], [0.0, 1.0, 2000.0],
            ]),
            "diaphragm-10": np.array([
                [0.0, 0.0, 3000.0], [1.0, 0.0, 3000.0], [0.0, 1.0, 3000.0],
            ]),
            "wall-x0": np.array([
                [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            ]),
        },
        element_ids_by_pg={"walls": np.array([201, 202, 203])},
        pg_names_by_dim={
            2: ["diaphragm-1", "diaphragm-2", "diaphragm-10", "wall-x0"],
        },
    )
    return h5, fem


def test_from_pg_glob_matches_pattern(tmp_path: Path) -> None:
    h5, fem = _glob_fixture(tmp_path)
    sweep = SectionSweepDef.from_pg_glob(
        pattern="diaphragm-*",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        normal_hint=(0.0, 0.0, 1.0),
    )
    # Three diaphragms matched, wall-x0 excluded.
    assert len(sweep) == 3
    # Labels show natural-sort order (1, 2, 10 — NOT lex 1, 10, 2).
    labels = [cut.label for cut in sweep]
    assert labels == [
        "plane=diaphragm-1, elements=walls",
        "plane=diaphragm-2, elements=walls",
        "plane=diaphragm-10, elements=walls",
    ]
    # Elevations follow the labels.
    np.testing.assert_allclose(
        sweep.plane_locators("z"),
        [1000.0, 2000.0, 3000.0],
    )


def test_from_pg_glob_no_match_raises(tmp_path: Path) -> None:
    h5, fem = _glob_fixture(tmp_path)
    with pytest.raises(ValueError, match="No physical groups matched"):
        SectionSweepDef.from_pg_glob(
            pattern="rooftop-*",
            elements_pg="walls",
            fem=fem,            # type: ignore[arg-type]
            model_h5=h5,
        )


def test_from_pg_glob_dim_filter(tmp_path: Path) -> None:
    """``dim=-1`` matches across all dimensions."""
    h5, fem = _glob_fixture(tmp_path)
    # Same fixture, but try dim=-1 — should also find the dim-2 entries.
    sweep = SectionSweepDef.from_pg_glob(
        pattern="diaphragm-*",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        dim=-1,
        normal_hint=(0.0, 0.0, 1.0),
    )
    assert len(sweep) == 3


def test_from_pg_glob_with_bounding(tmp_path: Path) -> None:
    """``with_bounding=True`` propagates to every cut."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20, 21]),
            "fem_eids": np.array([201, 202]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "level-1": np.array([
                [0.0, 0.0, 100.0], [10.0, 0.0, 100.0],
                [10.0, 10.0, 100.0], [0.0, 10.0, 100.0],
                [5.0, 5.0, 100.0],   # interior — should be dropped
            ]),
            "level-2": np.array([
                [0.0, 0.0, 200.0], [10.0, 0.0, 200.0],
                [10.0, 10.0, 200.0], [0.0, 10.0, 200.0],
            ]),
        },
        element_ids_by_pg={"walls": np.array([201, 202])},
        pg_names_by_dim={2: ["level-1", "level-2"]},
    )

    sweep = SectionSweepDef.from_pg_glob(
        pattern="level-*",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        with_bounding=True,
    )
    assert len(sweep) == 2
    for cut in sweep:
        assert cut.bounding_polygon is not None
        assert len(cut.bounding_polygon) == 4
        # Square's 4 corners survived; interior (5,5,z) dropped.


def test_from_pg_pattern_with_bounding_passes_through(tmp_path: Path) -> None:
    """``from_pg_pattern`` should propagate ``with_bounding=True`` to
    every cut — same shape as the glob version but with an explicit
    list."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20]),
            "fem_eids": np.array([201]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "level-1": np.array([
                [0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [0.0, 1.0, 100.0],
            ]),
        },
        element_ids_by_pg={"walls": np.array([201])},
    )
    sweep = SectionSweepDef.from_pg_pattern(
        plane_pgs=["level-1"],
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        with_bounding=True,
    )
    assert sweep[0].bounding_polygon is not None
    assert len(sweep[0].bounding_polygon) == 3
