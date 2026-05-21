"""Phase-4 unit tests for ``SectionCutDef.from_plane_and_pg`` /
``.from_planar_pg``.

Tests build minimal ``model.h5`` files inline (like
:mod:`tests.cuts.test_tag_map`) and stub the ``FEMData``: we don't run
a full mesh here, since Phases 2 and 3 already cover their respective
inputs. Phase 4's job is the composition, so these tests verify
``from_plane_and_pg`` correctly chains FEMData → tag map → spec, and
``from_planar_pg`` correctly chains plane fit → ``from_plane_and_pg``.

The end-to-end integration test (real Gmsh model, real ``apeSees``
build, real MPCO output, round-trip through STKO_to_python) is the
"north star" described in ``ARCHITECTURE.md`` — that lives separately
and runs only when STKO_to_python is installed with its cuts module.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef


# --------------------------------------------------------------------- #
# Inline fixtures: stub FEMData + minimal model.h5
# --------------------------------------------------------------------- #
class _SelResult:
    """selection-unification v2 P3-R: the ``fem.<...>.select(...)``
    terminal — exposes ``.ids`` / ``.coords`` (the only surface the
    cuts PROD reads after the ``get_ids``/``get_coords``→``select``
    migration; ``cuts/_defs.py:204`` etc.)."""

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
    def __init__(self, coords_by_pg: dict[str, np.ndarray]) -> None:
        self._coords_by_pg = coords_by_pg

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
    ) -> None:
        self.nodes = _StubNodes(node_coords_by_pg or {})
        self.elements = _StubElements(element_ids_by_pg or {})


def _write_minimal_h5(
    path: Path,
    *,
    groups: dict[str, dict[str, np.ndarray]],
    schema_version: str = "2.6.0",
) -> None:
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = schema_version
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
# from_plane_and_pg — happy path
# --------------------------------------------------------------------- #
def test_from_plane_and_pg_basic(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11, 12]),
            "fem_eids": np.array([101, 102, 103]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={
        "tower-cols": np.array([101, 102, 103]),
    })

    d = SectionCutDef.from_plane_and_pg(
        plane=((0.0, 0.0, 100.0), (0.0, 0.0, 1.0)),
        elements_pg="tower-cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        label="story 3",
    )

    assert d.plane_point == (0.0, 0.0, 100.0)
    assert d.plane_normal == (0.0, 0.0, 1.0)
    assert d.element_ids == (10, 11, 12)         # FEM eids -> ops tags
    assert d.label == "story 3"
    assert d.side == "positive"


def test_from_plane_and_pg_with_bounding_polygon(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={"a": np.array([101])})
    poly = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0))

    d = SectionCutDef.from_plane_and_pg(
        plane=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        elements_pg="a",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        bounding_polygon=poly,
    )
    assert d.bounding_polygon == poly


def test_from_plane_and_pg_propagates_side(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={"a": np.array([101])})

    d = SectionCutDef.from_plane_and_pg(
        plane=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        elements_pg="a",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        side="negative",
    )
    assert d.side == "negative"


# --------------------------------------------------------------------- #
# from_plane_and_pg — error paths
# --------------------------------------------------------------------- #
def test_from_plane_and_pg_empty_pg_raises(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={})  # PG resolves to empty

    with pytest.raises(ValueError, match="zero.*elements"):
        SectionCutDef.from_plane_and_pg(
            plane=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            elements_pg="missing",
            fem=fem,            # type: ignore[arg-type]
            model_h5=h5,
        )


def test_from_plane_and_pg_unmapped_fem_eid_raises(tmp_path: Path) -> None:
    """FEMData reports FEM eids that the tag map doesn't know about
    (e.g. element added after the bridge emit). Must raise loudly so
    the user notices."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),       # only 101 is mapped
        },
    })
    fem = _StubFEM(element_ids_by_pg={
        "tower": np.array([101, 999]),         # 999 has no ops_tag
    })

    with pytest.raises(KeyError, match="not in tag map"):
        SectionCutDef.from_plane_and_pg(
            plane=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            elements_pg="tower",
            fem=fem,            # type: ignore[arg-type]
            model_h5=h5,
        )


def test_from_plane_and_pg_mixed_type_filter(tmp_path: Path) -> None:
    """A PG spanning beams + shells should yield element_ids spanning
    both type groups in the tag map."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
        "ASDShellQ4": {
            "ids":      np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={
        "story-3-and-walls": np.array([101, 201, 102, 202, 203]),
    })

    d = SectionCutDef.from_plane_and_pg(
        plane=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        elements_pg="story-3-and-walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )
    assert set(d.element_ids) == {10, 11, 20, 21, 22}


# --------------------------------------------------------------------- #
# from_planar_pg — happy path
# --------------------------------------------------------------------- #
def test_from_planar_pg_happy_path(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "ASDShellQ4": {
            "ids":      np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })
    # Diaphragm surface — 4 coplanar nodes at z=100.
    fem = _StubFEM(
        node_coords_by_pg={
            "diaphragm-3": np.array([
                [0.0, 0.0, 100.0],
                [1.0, 0.0, 100.0],
                [1.0, 1.0, 100.0],
                [0.0, 1.0, 100.0],
            ]),
        },
        element_ids_by_pg={
            "story-3-walls": np.array([201, 202, 203]),
        },
    )

    d = SectionCutDef.from_planar_pg(
        plane_pg="diaphragm-3",
        elements_pg="story-3-walls",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
    )

    # Plane: centroid is (0.5, 0.5, 100); normal is along ±z.
    np.testing.assert_allclose(d.plane_point, (0.5, 0.5, 100.0), atol=1e-12)
    assert abs(d.plane_normal[2]) == 1.0
    assert abs(d.plane_normal[0]) < 1e-9
    assert abs(d.plane_normal[1]) < 1e-9
    assert d.element_ids == (20, 21, 22)
    # Auto-label includes both PG names for traceability.
    assert d.label == "plane=diaphragm-3, elements=story-3-walls"


def test_from_planar_pg_respects_normal_hint(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "wall-x0": np.array([
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        },
        element_ids_by_pg={"a": np.array([101])},
    )

    d = SectionCutDef.from_planar_pg(
        plane_pg="wall-x0",
        elements_pg="a",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        normal_hint=(-1.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(d.plane_normal, (-1.0, 0.0, 0.0), atol=1e-9)


def test_from_planar_pg_custom_label_overrides_auto(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "dia": np.array([
                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            ]),
        },
        element_ids_by_pg={"cols": np.array([101])},
    )

    d = SectionCutDef.from_planar_pg(
        plane_pg="dia",
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        label="Story 3 base shear",
    )
    assert d.label == "Story 3 base shear"


def test_from_planar_pg_non_coplanar_plane_pg_raises(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10]),
            "fem_eids": np.array([101]),
        },
    })
    fem = _StubFEM(
        node_coords_by_pg={
            "curved": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],          # off-plane
            ]),
        },
        element_ids_by_pg={"a": np.array([101])},
    )

    with pytest.raises(ValueError, match="not coplanar"):
        SectionCutDef.from_planar_pg(
            plane_pg="curved",
            elements_pg="a",
            fem=fem,            # type: ignore[arg-type]
            model_h5=h5,
            tol=1e-9,
        )


# --------------------------------------------------------------------- #
# Integration with _planes helpers
# --------------------------------------------------------------------- #
def test_chains_with_plane_horizontal(tmp_path: Path) -> None:
    """The 'I just want story shear at z=2500' workflow — uses
    ``plane_horizontal`` instead of a PG-derived plane."""
    from apeGmsh.cuts import plane_horizontal

    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids":      np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
    })
    fem = _StubFEM(element_ids_by_pg={"cols": np.array([101, 102])})

    d = SectionCutDef.from_plane_and_pg(
        plane=plane_horizontal(z=2500.0),
        elements_pg="cols",
        fem=fem,            # type: ignore[arg-type]
        model_h5=h5,
        label="Story shear @ 2500",
    )
    assert d.plane_point == (0.0, 0.0, 2500.0)
    assert d.plane_normal == (0.0, 0.0, 1.0)
    assert d.element_ids == (10, 11)
