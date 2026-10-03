"""Healing keeps free curves; ``load_brep`` imports OCC's native format.

A frame + shell model (a slab with a column standing on its corner) is
the smallest case of what STKO exports.  OCC's heal-everything call sews
the faces and drops the free column curve, so ``heal_shapes()`` heals
such a model without sewing and says so.  A model with no free curves
still sews exactly as before.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import gmsh
import pytest

from apeGmsh.core._geometry_errors import WarnGeomHealSkipsSewing


def _slab_and_column() -> None:
    occ = gmsh.model.occ
    slab = occ.addRectangle(0, 0, 0, 1, 1)
    column = occ.addLine(occ.addPoint(0, 0, 0), occ.addPoint(0, 0, 1))
    occ.fragment([(2, slab)], [(1, column)])
    occ.synchronize()


def _free_curves() -> list[int]:
    return [
        t for _, t in gmsh.model.getEntities(1)
        if len(gmsh.model.getAdjacencies(1, t)[0]) == 0
    ]


def _shares_a_vertex_with_the_slab(curve: int) -> bool:
    ends = gmsh.model.getAdjacencies(1, curve)[1]
    return any(len(gmsh.model.getAdjacencies(0, int(p))[0]) > 1 for p in ends)


def _counts() -> list[int]:
    return [len(gmsh.model.getEntities(d)) for d in range(3)]


def _save_brep(tmp_path: Path) -> Path:
    """Write the slab + column from a scratch gmsh model, leaving the
    session's own model empty."""
    session_model = gmsh.model.getCurrent()
    gmsh.model.add("scratch")
    _slab_and_column()
    out = tmp_path / "frame.brep"
    gmsh.write(str(out))
    gmsh.model.remove()
    gmsh.model.setCurrent(session_model)
    return out


# ── heal_shapes() ────────────────────────────────────────────────────

def test_heal_everything_keeps_the_free_column_and_warns(g) -> None:
    _slab_and_column()

    with pytest.warns(WarnGeomHealSkipsSewing, match="1 free curve"):
        g.model.io.heal_shapes()

    free = _free_curves()
    assert len(free) == 1
    assert _shares_a_vertex_with_the_slab(free[0])
    assert len(gmsh.model.getEntities(2)) == 1


def test_heal_everything_still_sews_a_model_without_free_curves(g) -> None:
    """Two touching, unsewn faces: heal-everything merges the shared
    edge, as it always did, and stays silent."""
    occ = gmsh.model.occ
    occ.addRectangle(0, 0, 0, 1, 1)
    occ.addRectangle(1, 0, 0, 1, 1)
    occ.synchronize()
    assert _counts() == [8, 8, 2]

    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGeomHealSkipsSewing)
        g.model.io.heal_shapes(tolerance=1e-6)

    assert _counts() == [6, 7, 2]


def test_no_warning_when_sewing_is_already_off(g) -> None:
    _slab_and_column()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGeomHealSkipsSewing)
        g.model.io.heal_shapes(sew_faces=False)
    assert len(_free_curves()) == 1


def test_no_warning_on_a_wire_model(g) -> None:
    """No faces, nothing to sew: free curves are not at risk."""
    occ = gmsh.model.occ
    occ.addLine(occ.addPoint(0, 0, 0), occ.addPoint(1, 0, 0))
    occ.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGeomHealSkipsSewing)
        g.model.io.heal_shapes()
    assert len(gmsh.model.getEntities(1)) == 1


# ── import path + load_brep (default: every shape comes in) ─────────

def test_import_heal_keeps_the_free_column(g, tmp_path: Path) -> None:
    brep = _save_brep(tmp_path)

    g.model.io.load_brep(brep, heal=True)

    free = _free_curves()
    assert len(free) == 1
    assert _shares_a_vertex_with_the_slab(free[0])


def test_load_brep_default_imports_the_free_column(g, tmp_path: Path) -> None:
    brep = _save_brep(tmp_path)

    imported = g.model.io.load_brep(brep, label="frame")

    assert len(imported[2]) == 1
    assert len(imported[1]) == 5            # 4 slab edges + the column, once each
    assert len(imported[0]) == 5
    assert len(_free_curves()) == 1


def test_label_covers_the_shapes_not_their_sub_entities(
    g, tmp_path: Path,
) -> None:
    brep = _save_brep(tmp_path)

    g.model.io.load_brep(brep, label="frame")

    assert list(g.labels.entities("frame", dim=2)) == [1]
    assert list(g.labels.entities("frame", dim=1)) == _free_curves()
    with pytest.raises(KeyError):           # no vertex carries it
        g.labels.entities("frame", dim=0)


def test_solid_import_label_stays_on_the_volume(g, tmp_path: Path) -> None:
    """A solid's faces bound it, so only the volume is a shape: the
    label resolves without dim= exactly as before the default flip."""
    session_model = gmsh.model.getCurrent()
    gmsh.model.add("scratch")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    step = tmp_path / "box.step"
    gmsh.write(str(step))
    gmsh.model.remove()
    gmsh.model.setCurrent(session_model)

    imported = g.model.io.load_step(step, label="block")

    assert set(imported) == {0, 1, 2, 3}
    assert list(g.labels.entities("block")) == imported[3]


def test_load_brep_highest_dim_only_drops_the_free_column(
    g, tmp_path: Path,
) -> None:
    """Opt-in: only the top dimension is imported."""
    brep = _save_brep(tmp_path)

    imported = g.model.io.load_brep(brep, highest_dim_only=True)

    assert set(imported) == {2}
    assert _free_curves() == []


# ── g.parts: the free column is imported, moved and labelled ───────

def test_parts_import_moves_and_labels_the_free_column(
    g, tmp_path: Path,
) -> None:
    brep = _save_brep(tmp_path)

    g.parts.import_step(brep, label="frame", translate=(10.0, 0.0, 0.0))

    free = _free_curves()
    assert len(free) == 1
    assert _shares_a_vertex_with_the_slab(free[0])
    xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(1, free[0])
    assert (xmin, xmax) == pytest.approx((10.0, 10.0), abs=1e-6)  # moved with the slab
    assert list(g.labels.entities("frame", dim=1)) == free
