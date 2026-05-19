"""Phase v2.1 — tests for bounding polygon derivation.

Covers the convex-hull primitive and the
``bounding_polygon_from_physical_surface`` public helper. Also covers
the new ``with_bounding=True`` knob on
``SectionCutDef.from_planar_pg``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import (
    SectionCutDef,
    bounding_polygon_from_physical_surface,
)
from apeGmsh.cuts._polygons import (
    _convex_hull_2d_ccw,
    _embed_from_basis,
    _plane_basis,
    _project_to_basis,
)


# --------------------------------------------------------------------- #
# Stubs (same shape as test_builders.py / test_sweeps.py)
# --------------------------------------------------------------------- #
class _SelResult:
    """selection-unification v2 P3-R: the ``fem.{nodes,elements}.
    select(...)`` terminal — exposes ``.coords`` / ``.ids`` (the only
    surface PROD reads after the ``get_coords``/``get_ids``→``select``
    migration; P-COORD / P-ELEM-IDS)."""

    def __init__(self, *, coords=None, ids=None) -> None:
        if coords is not None:
            self.coords = coords
        if ids is not None:
            self.ids = ids


class _StubNodes:
    def __init__(self, coords_by_pg: dict[str, np.ndarray]) -> None:
        self._coords_by_pg = coords_by_pg

    def get_coords(self, *, pg: str) -> np.ndarray:
        return self._coords_by_pg[pg]

    def select(self, target=None, *, pg: str | None = None, **_kw):
        return _SelResult(coords=self._coords_by_pg[pg])


class _StubElements:
    def __init__(self, ids_by_pg: dict[str, np.ndarray]) -> None:
        self._ids_by_pg = ids_by_pg

    def get_ids(self, *, pg: str) -> np.ndarray:
        return self._ids_by_pg.get(pg, np.array([], dtype=np.int64))

    def select(self, target=None, *, pg: str | None = None, **_kw):
        return _SelResult(
            ids=self._ids_by_pg.get(pg, np.array([], dtype=np.int64))
        )


class _StubFEM:
    def __init__(
        self,
        *,
        node_coords_by_pg: dict[str, np.ndarray] | None = None,
        element_ids_by_pg: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.nodes = _StubNodes(node_coords_by_pg or {})
        self.elements = _StubElements(element_ids_by_pg or {})


def _write_minimal_h5(path: Path, *, groups: dict[str, dict[str, np.ndarray]]) -> None:
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


# --------------------------------------------------------------------- #
# Convex hull primitive
# --------------------------------------------------------------------- #
def test_convex_hull_square():
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    hull = _convex_hull_2d_ccw(pts)
    assert hull.shape == (4, 2)
    # CCW around origin → starts at (0,0), goes right, up, left, down.
    # Signed area positive:
    area = 0.5 * float(
        np.dot(hull[:, 0], np.roll(hull[:, 1], -1))
        - np.dot(hull[:, 1], np.roll(hull[:, 0], -1))
    )
    assert area > 0
    np.testing.assert_allclose(abs(area), 1.0, atol=1e-12)


def test_convex_hull_drops_interior_points():
    # 4 corners + 1 interior point + 1 edge-midpoint
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],     # interior
        [0.5, 0.0],     # collinear on bottom edge — also gets dropped
    ])
    hull = _convex_hull_2d_ccw(pts)
    # Only the 4 corners survive.
    assert hull.shape == (4, 2)
    hull_set = {tuple(v) for v in hull}
    assert hull_set == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}


def test_convex_hull_triangle():
    pts = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.0],
    ])
    hull = _convex_hull_2d_ccw(pts)
    assert hull.shape == (3, 2)


def test_convex_hull_collinear_raises():
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    with pytest.raises(ValueError, match="collinear|3 distinct"):
        _convex_hull_2d_ccw(pts)


def test_convex_hull_dedups_input():
    pts = np.array([
        [0.0, 0.0], [0.0, 0.0],     # duplicate
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    hull = _convex_hull_2d_ccw(pts)
    assert hull.shape == (3, 2)


def test_convex_hull_too_few_distinct_raises():
    pts = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    with pytest.raises(ValueError, match="3 distinct"):
        _convex_hull_2d_ccw(pts)


def test_convex_hull_missing_scipy_raises_clean_error(monkeypatch):
    """If scipy isn't installed, the hull function must raise
    ImportError with the install hint — not a cryptic
    ModuleNotFoundError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    ])
    with pytest.raises(ImportError, match=r"scipy"):
        _convex_hull_2d_ccw(pts)


# --------------------------------------------------------------------- #
# Plane basis + projection round-trip
# --------------------------------------------------------------------- #
def test_plane_basis_orthonormal_to_normal():
    for n in [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]:
        e1, e2 = _plane_basis(n)
        np.testing.assert_allclose(np.linalg.norm(e1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(e2), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(e1, n), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(e2, n), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(e1, e2), 0.0, atol=1e-12)


def test_project_and_embed_roundtrip():
    n = np.array([0.0, 0.0, 1.0])
    e1, e2 = _plane_basis(n)
    point = np.array([5.0, 5.0, 10.0])
    pts_3d = np.array([
        [4.0, 4.0, 10.0],
        [6.0, 4.0, 10.0],
        [6.0, 6.0, 10.0],
        [4.0, 6.0, 10.0],
    ])
    pts_2d = _project_to_basis(pts_3d, point, e1, e2)
    back_3d = _embed_from_basis(pts_2d, point, e1, e2)
    np.testing.assert_allclose(back_3d, pts_3d, atol=1e-12)


# --------------------------------------------------------------------- #
# bounding_polygon_from_physical_surface
# --------------------------------------------------------------------- #
def test_bounding_polygon_horizontal_square():
    # 4 corners + several mesh-interior nodes at z=100
    coords = np.array([
        [0.0, 0.0, 100.0],
        [1.0, 0.0, 100.0],
        [1.0, 1.0, 100.0],
        [0.0, 1.0, 100.0],
        [0.5, 0.5, 100.0],
        [0.25, 0.25, 100.0],
        [0.75, 0.75, 100.0],
    ])
    fem = _StubFEM(node_coords_by_pg={"diaphragm": coords})

    poly = bounding_polygon_from_physical_surface(fem, "diaphragm")  # type: ignore[arg-type]

    # Hull = 4 corners.
    assert len(poly) == 4
    poly_set = {(round(v[0], 9), round(v[1], 9), round(v[2], 9)) for v in poly}
    assert poly_set == {
        (0.0, 0.0, 100.0),
        (1.0, 0.0, 100.0),
        (1.0, 1.0, 100.0),
        (0.0, 1.0, 100.0),
    }
    # All vertices sit on z=100.
    for v in poly:
        assert v[2] == 100.0


def test_bounding_polygon_oblique_plane_lies_on_plane():
    # Square in a tilted plane, normal (1, 1, 1)/sqrt(3).
    n_true = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v = np.cross(n_true, u)
    corners = np.array([
        +u + v,
        -u + v,
        -u - v,
        +u - v,
    ]) + np.array([3.0, 3.0, 3.0])
    # Add a few interior points to make the test non-trivial.
    interior = np.mean(corners, axis=0)
    coords = np.vstack([corners, [interior]])
    fem = _StubFEM(node_coords_by_pg={"tilted": coords})

    poly = bounding_polygon_from_physical_surface(
        fem, "tilted", normal_hint=n_true,    # type: ignore[arg-type]
    )

    assert len(poly) == 4
    # Each polygon vertex lies on the original plane (signed distance ≈ 0).
    for vert in poly:
        d = float(np.dot(np.asarray(vert) - np.array([3.0, 3.0, 3.0]), n_true))
        np.testing.assert_allclose(d, 0.0, atol=1e-9)


def test_bounding_polygon_empty_pg_raises():
    fem = _StubFEM(node_coords_by_pg={"empty": np.zeros((0, 3))})
    with pytest.raises(ValueError, match="zero nodes"):
        bounding_polygon_from_physical_surface(fem, "empty")  # type: ignore[arg-type]


def test_bounding_polygon_non_coplanar_pg_raises():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],     # off-plane
    ])
    fem = _StubFEM(node_coords_by_pg={"bumpy": coords})
    with pytest.raises(ValueError, match="not coplanar"):
        bounding_polygon_from_physical_surface(fem, "bumpy", tol=1e-9)  # type: ignore[arg-type]


def test_bounding_polygon_collinear_pg_raises():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    fem = _StubFEM(node_coords_by_pg={"line": coords})
    # plane_from_coords catches this first (its collinearity check), so
    # the user sees the plane-fit error, not the hull error. Either way
    # it's a ValueError mentioning collinear.
    with pytest.raises(ValueError, match="collinear"):
        bounding_polygon_from_physical_surface(fem, "line")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Integration with SectionCutDef.from_planar_pg(with_bounding=True)
# --------------------------------------------------------------------- #
def _make_fixture(tmp_path: Path) -> tuple[Path, _StubFEM]:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "diaphragm": np.array([
                [0.0, 0.0, 100.0],
                [10.0, 0.0, 100.0],
                [10.0, 10.0, 100.0],
                [0.0, 10.0, 100.0],
                [5.0, 5.0, 100.0],    # interior
            ]),
        },
        element_ids_by_pg={"walls": np.array([201, 202, 203])},
    )
    return h5, fem


def test_from_planar_pg_with_bounding_true(tmp_path: Path) -> None:
    h5, fem = _make_fixture(tmp_path)

    d = SectionCutDef.from_planar_pg(
        plane_pg="diaphragm",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        with_bounding=True,
    )

    assert d.bounding_polygon is not None
    assert len(d.bounding_polygon) == 4
    # 4 corners of the square (interior point dropped).
    poly_set = {tuple(round(c, 9) for c in v) for v in d.bounding_polygon}
    assert poly_set == {
        (0.0, 0.0, 100.0),
        (10.0, 0.0, 100.0),
        (10.0, 10.0, 100.0),
        (0.0, 10.0, 100.0),
    }


def test_from_planar_pg_with_bounding_false_default(tmp_path: Path) -> None:
    h5, fem = _make_fixture(tmp_path)

    d = SectionCutDef.from_planar_pg(
        plane_pg="diaphragm",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert d.bounding_polygon is None


def test_from_planar_pg_with_bounding_and_explicit_polygon_raises(
    tmp_path: Path,
) -> None:
    h5, fem = _make_fixture(tmp_path)
    explicit = ((0.0, 0.0, 100.0), (1.0, 0.0, 100.0), (0.0, 1.0, 100.0))

    with pytest.raises(ValueError, match="bounding_polygon.*OR.*with_bounding"):
        SectionCutDef.from_planar_pg(
            plane_pg="diaphragm",
            elements_pg="walls",
            fem=fem,            # type: ignore[arg-type]
            model_h5=h5,
            with_bounding=True,
            bounding_polygon=explicit,
        )


def test_from_planar_pg_with_bounding_passes_stko_validation(tmp_path: Path) -> None:
    """The derived polygon must satisfy STKO's SectionCutSpec.__post_init__:
    on-plane within 1e-6, non-degenerate, convex. Auto-generated hulls
    are convex by construction and lie exactly on the plane, so this is
    really a smoke test that the bytes survive."""
    stko_cuts = pytest.importorskip("STKO_to_python.cuts")
    h5, fem = _make_fixture(tmp_path)

    d = SectionCutDef.from_planar_pg(
        plane_pg="diaphragm",
        elements_pg="walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        with_bounding=True,
    )
    spec = d.to_spec()
    assert isinstance(spec, stko_cuts.SectionCutSpec)
    assert spec.bounding_polygon is not None
    assert len(spec.bounding_polygon) == 4
