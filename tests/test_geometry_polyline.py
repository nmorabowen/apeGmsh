"""Session tests for g.model.geometry.add_polyline (ADR 0097)."""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh.core._geometry_errors import WarnGeomSharpPolylineCorner


SQUARE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]


def test_add_polyline_open(g):
    tags = g.model.geometry.add_polyline(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
    )
    assert len(tags) == 2
    live = {t for _, t in gmsh.model.getEntities(1)}
    assert set(tags) <= live
    assert g.model._metadata[(1, tags[0])]["kind"] == "polyline"


def test_add_polyline_closed(g):
    tags = g.model.geometry.add_polyline(SQUARE, closed=True)
    assert len(tags) == 4


def test_add_polyline_label_groups_curves(g):
    tags = g.model.geometry.add_polyline(
        SQUARE, closed=True, label="opening_profile",
    )
    assert g.labels.has("opening_profile")
    ents = set(g.labels.entities("opening_profile"))
    assert ents == set(tags)


def test_add_polyline_fillet_inserts_arcs(g):
    tags = g.model.geometry.add_polyline(
        SQUARE, closed=True, fillet={1: 0.1, 3: 0.1},
    )
    # 4 lines + 2 arcs
    assert len(tags) == 6
    types = [gmsh.model.getType(1, t) for t in tags]
    assert types.count("Circle") == 2


def test_add_polyline_chamfer_stays_lines(g):
    tags = g.model.geometry.add_polyline(
        SQUARE, closed=True, chamfer={1: 0.1},
    )
    assert len(tags) == 5
    assert all(gmsh.model.getType(1, t) == "Line" for t in tags)


def test_add_polyline_sharp_corner_warns(g):
    with pytest.warns(WarnGeomSharpPolylineCorner, match="vertex 1"):
        g.model.geometry.add_polyline(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        )


def test_add_polyline_fillet_suppresses_warning(g):
    import warnings
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        g.model.geometry.add_polyline(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
            fillet={1: 0.1},
        )
    sharp = [
        w for w in recorded
        if issubclass(w.category, WarnGeomSharpPolylineCorner)
    ]
    assert sharp == []


def test_filleted_path_pipes(g):
    """OCC addPipe succeeds on an L-path whose corner is filleted."""
    path = g.model.geometry.add_polyline(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0)],
        fillet={1: 0.3},
    )
    profile = g.model.geometry.add_polyline(
        [
            (0.0, -0.1, -0.1),
            (0.0,  0.1, -0.1),
            (0.0,  0.1,  0.1),
            (0.0, -0.1,  0.1),
        ],
        closed=True,
    )
    loop = g.model.geometry.add_curve_loop(profile)
    face = g.model.geometry.add_plane_surface(loop)
    out = g.model.geometry.sweep(face, path, label="pipe")
    assert out["volume"] is not None
    vols = [t for _, t in gmsh.model.getEntities(3)]
    assert out["volume"] in vols
